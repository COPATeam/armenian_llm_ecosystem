"""Automated translation-adequacy audit of ArmSTEM, stratified by
solver-limited status. Judges EN->HY adequacy on a 1-5 scale with GPT-5.5.
Usage: adequacy_audit.py <out.json> [n_per_stratum]"""
import os
import json
import random
import sys
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from openai import OpenAI

KEYS = Path.home().joinpath(".llm_api_keys.txt").read_text().split()
CLIENTS = [OpenAI(api_key=k, base_url=os.environ.get("LLM_API_BASE", "https://api.openai.com/v1"),
                  timeout=240, max_retries=2) for k in KEYS]
_rr = itertools.count()
JUDGE = "azure/openai/gpt-5.5"
DATA = Path(__file__).resolve().parent.parent.parent.parent / "armweb_data" / "translated"

PROMPT = """You are auditing an English-to-Armenian translation of a STEM problem and its solution.

ENGLISH ORIGINAL:
{en}

ARMENIAN TRANSLATION:
{hy}

Rate the translation's ADEQUACY (meaning preservation, including all mathematical/scientific content) on a 1-5 scale:
5 = fully adequate, meaning and technical content preserved
4 = minor wording issues, meaning intact
3 = noticeable issues, core meaning mostly intact
2 = significant meaning distortion
1 = wrong or unusable
End your reply with the digit alone on the final line."""


def call(prompt):
    start = next(_rr)
    for attempt, wait in enumerate((0, 0, 0, 15, 30, 60, 120)):
        if wait:
            time.sleep(wait)
        c = CLIENTS[(start + attempt) % len(CLIENTS)]
        try:
            r = c.chat.completions.create(model=JUDGE,
                messages=[{"role": "user", "content": prompt}], max_tokens=4000)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            continue
    return ""


def main():
    out_path, n_per = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150
    rng = random.Random(42)
    strata = {"PASS": [], "PASS_solver_limited": []}
    for shard in ("shard1_math", "shard2_science"):
        for line in open(DATA / shard / "accepted.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d["stage"] in strata:
                strata[d["stage"]].append(d)
    sample = []
    for stage, items in strata.items():
        for d in rng.sample(items, n_per):
            sample.append(d)
    print(f"judging {len(sample)} items", flush=True)

    def judge(d):
        en = d["question_en"] + "\n" + (d.get("cot_en") or "")
        hy = d["question_hy"] + "\n" + (d.get("cot_hy") or "")
        v = call(PROMPT.format(en=en[:6000], hy=hy[:6000]))
        digits = [ch for ch in v if ch in "12345"]
        score = int(digits[-1]) if digits else None
        return {"id": d["id"], "stage": d["stage"], "score": score}

    with ThreadPoolExecutor(max_workers=24) as ex:
        results = list(ex.map(judge, sample))
    summary = {}
    for stage in strata:
        sc = [r["score"] for r in results if r["stage"] == stage and r["score"]]
        summary[stage] = {
            "n": len(sc), "mean": round(sum(sc) / len(sc), 2),
            "dist": {s: sc.count(s) for s in range(1, 6)},
            "adequate_ge4": round(sum(1 for s in sc if s >= 4) / len(sc), 3),
        }
    json.dump({"summary": summary, "results": results}, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
