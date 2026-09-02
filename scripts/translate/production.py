"""Production translate-and-verify runner. Resumable, gated, decon-enforced.

Usage: production.py <shard.jsonl> <out_dir> [workers]
Shard rows: {"id", "src", "question", "cot" (optional), "gold", "answer_type": "numeric"|"mc"|"freeform"}
Stages per item:
  D1 English-side decon (13-gram vs mmlu_pro_en_test)  [index built once]
  T  translate (flash-lite; repair x2 w/ feedback; escalate gpt-5.5 x1)
  G0 placeholder integrity   G1 GlotLID Armenian
  G2 answer verification: numeric/mc -> o4-mini re-solve exact match
     (on fail: o4-mini solves ENGLISH original; if that also fails -> keep, tag solver_limited)
     freeform -> 3-judge panel (flash-lite, o4-mini, gpt-5.5), 2/3 majority
  D2 Armenian-side decon (13-gram vs armbench_hy)
Output: <out_dir>/accepted.jsonl (EN+HY parallel), rejected.jsonl, stats.json.
Resume: ids present in accepted/rejected are skipped on restart.
"""
import os
import json
import sys
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import openai
import fasttext
from huggingface_hub import hf_hub_download
from openai import OpenAI

from armweb.translation import mask, unmask, gate_integrity, gate_language, extract_final_number
from armweb.decontaminate import _ngrams

KEYS_FILE = Path.home() / ".llm_api_keys.txt"
KEYS = (KEYS_FILE.read_text().split() if KEYS_FILE.exists()
        else [Path.home().joinpath(".nvidia_inference_key").read_text().strip()])
TRANSLATOR = "gcp/google/gemini-3.1-flash-lite"
ESCALATOR = "azure/openai/gpt-5.5"
SOLVER = "azure/openai/o4-mini"
PANEL = [TRANSLATOR, SOLVER, ESCALATOR]
BENCH = Path(__file__).resolve().parent.parent.parent.parent / "armweb_data" / "benchmarks"

CLIENTS = [OpenAI(api_key=k, base_url=os.environ.get("LLM_API_BASE", "https://api.openai.com/v1"),
                  timeout=240, max_retries=2) for k in KEYS]
_rr = itertools.count()
lid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
_lock = threading.Lock()

TR_PROMPT = """Translate the following {kind} from English to Eastern Armenian.
Rules:
- Tokens like ⟦N1⟧, ⟦EQ2⟧ and the separator ⟦SEP⟧ are placeholders. Copy each EXACTLY as-is, in its position. Keep exactly one ⟦SEP⟧ line.
- Do not add, remove, or reorder placeholders. Do not write any digits.
- Translate naturally and precisely.{extra}
- Output ONLY the translation.

{doc}"""

SOLVE_HY = "Լուծիր հետևյալ խնդիրը քայլ առ քայլ։ Վերջում գրիր ՄԻԱՅՆ վերջնական պատասխանը վերջին տողում՝ «ՊԱՏԱՍԽԱՆ: <պատասխան>» ձևաչափով։ Եթե հարցը բազմընտրանի է, պատասխանիր ընտրանքի տառով։\n\n{q}"
SOLVE_EN = "Solve step by step. Final line must be exactly 'ANSWER: <answer>' (letter if multiple-choice).\n\n{q}"
JUDGE = """You are checking an Armenian translation of a QA item. Question (Armenian): {q}
Reference answer (English): {gold}
Candidate answer (Armenian): {cand}
Does the candidate express the same answer as the reference? Reply with exactly YES or NO."""


def call(model, prompt, max_tokens=6000):
    """Round-robin over key pool; rotate key on throttle, back off only after a
    full rotation has been throttled."""
    start = next(_rr)
    for attempt, wait in enumerate((0, 0, 0, 0, 15, 30, 60, 90, 120, 120, 120, 120)):
        if wait:
            time.sleep(wait)
        c = CLIENTS[(start + attempt) % len(CLIENTS)]
        try:
            r = c.chat.completions.create(model=model,
                messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
            return (r.choices[0].message.content or "").strip()
        except (openai.RateLimitError, openai.APIConnectionError):
            continue
    raise openai.RateLimitError("rate limit persisted through rotation+backoff", response=None, body=None)


def load_gram_index(path):
    grams = set()
    for line in open(path, encoding="utf-8"):
        grams.update(_ngrams(line))
    return grams


def final_token(s):
    lines = [l for l in s.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def accept(rec, files):
    with _lock:
        files["acc"].write(json.dumps(rec, ensure_ascii=False) + "\n")
        files["acc"].flush()
    return ("PASS", rec["stage"])


def reject(rec, files):
    with _lock:
        files["rej"].write(json.dumps(rec, ensure_ascii=False) + "\n")
        files["rej"].flush()
    return ("REJ", rec["stage"])


def process(item, en_index, hy_index, files):
    rid = item["id"]
    rec = {"id": rid, "src": item["src"], "answer_type": item["answer_type"], "stage": "T"}
    doc_en = item["question"] + "\n⟦SEP⟧\n" + (item.get("cot") or str(item["gold"]))
    if set(_ngrams(item["question"])) & en_index:
        rec["stage"] = "D1_en_contaminated"
        return reject(rec, files)
    masked, mp = mask(doc_en)
    extra = " Keep the '####' line format." if "####" in doc_en else ""
    feedback = ""
    for n, m in enumerate([TRANSLATOR, TRANSLATOR, ESCALATOR]):
        try:
            out = call(m, TR_PROMPT.format(kind="problem and its solution", extra=extra, doc=masked) + feedback)
        except Exception as e:
            rec["stage"] = f"ERR_{type(e).__name__}"
            continue
        if out.count("⟦SEP⟧") != 1:
            feedback = "\n\nYour previous output had a structural error: keep exactly one ⟦SEP⟧ line."
            rec["stage"] = "G0_structure"
            continue
        tq_m, ta_m = (x.strip() for x in out.split("⟦SEP⟧"))
        ok, why = gate_integrity(tq_m + " " + ta_m, mp)
        if not ok:
            feedback = f"\n\nYour previous output failed a check: {why}. Copy every placeholder exactly once."
            rec["stage"] = "G0_" + why[:30]
            continue
        ok, why = gate_language(tq_m + " " + ta_m, lid)
        if not ok:
            feedback = "\n\nYour previous output was not (fully) in Armenian. Translate everything to Eastern Armenian."
            rec["stage"] = "G1_" + why
            continue
        tq, ta = unmask(tq_m, mp), unmask(ta_m, mp)
        gold = str(item["gold"]).strip()
        if item["answer_type"] in ("numeric", "mc"):
            try:
                sol = call(SOLVER, SOLVE_HY.format(q=tq))
            except Exception as e:
                rec["stage"] = f"ERR_{type(e).__name__}"
                continue
            if item["answer_type"] == "numeric":
                pred = extract_final_number(final_token(sol))
                want = extract_final_number(gold)
            else:
                pred = (final_token(sol).split(":")[-1].strip().strip("()").upper()[:1] or None)
                want = gold.upper()[:1]
            if pred == want:
                rec["stage"] = "PASS"
            else:
                try:
                    en_sol = call(SOLVER, SOLVE_EN.format(q=item["question"]))
                except Exception:
                    en_sol = ""
                if item["answer_type"] == "numeric":
                    en_pred = extract_final_number(final_token(en_sol))
                else:
                    en_pred = (final_token(en_sol).split(":")[-1].strip().strip("()").upper()[:1] or None)
                if en_pred != want:
                    rec["stage"] = "PASS_solver_limited"
                else:
                    rec["stage"] = "G2_mismatch"
                    feedback = "\n\nYour previous translation may have altered the problem's meaning. Re-translate faithfully."
                    continue
        else:
            votes = 0
            for jm in PANEL:
                try:
                    v = call(jm, JUDGE.format(q=tq, gold=gold, cand=ta), max_tokens=10)
                    votes += v.strip().upper().startswith("YES")
                except Exception:
                    pass
            if votes >= 2:
                rec["stage"] = "PASS"
            else:
                rec["stage"] = "G2_panel_reject"
                continue
        if set(_ngrams(tq + " " + ta)) & hy_index:
            rec["stage"] = "D2_armbench_hit"
            return reject(rec, files)
        rec.update({"question_en": item["question"], "cot_en": item.get("cot"),
                    "gold": gold, "question_hy": tq, "cot_hy": ta,
                    "translator": m, "attempts": n + 1})
        return accept(rec, files)
    return reject(rec, files)


def main():
    shard, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for name in ("accepted.jsonl", "rejected.jsonl"):
        p = out_dir / name
        if p.exists():
            for line in open(p, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    items = [json.loads(l) for l in open(shard, encoding="utf-8")]
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to go", flush=True)
    en_path = BENCH / "mmlu_pro_en_test.txt"
    en_index = load_gram_index(en_path) if en_path.exists() else set()
    hy_index = load_gram_index(BENCH / "armbench_hy.txt")
    print(f"decon indices: en={len(en_index)} hy={len(hy_index)} grams", flush=True)
    files = {"acc": open(out_dir / "accepted.jsonl", "a", encoding="utf-8"),
             "rej": open(out_dir / "rejected.jsonl", "a", encoding="utf-8")}
    t0, counts = time.time(), {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, (kind, stage) in enumerate(
                ex.map(lambda it: process(it, en_index, hy_index, files), todo)):
            counts[stage] = counts.get(stage, 0) + 1
            if n % 200 == 0:
                rate = (n + 1) / (time.time() - t0)
                print(f"{n}/{len(todo)} ({rate:.1f}/s) {counts}", flush=True)
    (out_dir / "stats.json").write_text(json.dumps(counts, indent=2))
    print("DONE", json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
