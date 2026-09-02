"""Build science shard2: MMLU-Pro ref + parse OS/OSR2 -> production schema -> zst."""
import os
import json, os, re
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
ER = os.environ["ARMWEB_ROOT"]

# 1) MMLU-Pro English decon reference
ref = f"{ER}/code/armweb_data/benchmarks/mmlu_pro_en_test.txt"
if not os.path.exists(ref):
    from datasets import load_dataset
    mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    with open(ref, "w", encoding="utf-8") as f:
        for row in mp:
            opts = " ".join(str(o) for o in (row.get("options") or []))
            f.write(((row.get("question") or "") + " " + opts).replace("\n", " ") + "\n")
    print("MMLUPRO_REF_DONE", len(mp), flush=True)

ANS_RE = re.compile(r"answer is \(?([A-J])\)?", re.I)
BOX_RE = re.compile(r"boxed\{?\(?([A-J])\)?\}?")

def final_answer(output, expected=None):
    if expected:
        m = re.match(r"^\(?([A-J])\)?$", str(expected).strip())
        if m: return m.group(1).upper()
        return None
    tail = output[-400:]
    m = ANS_RE.search(tail) or BOX_RE.search(tail)
    return m.group(1).upper() if m else None

def post_think(output):
    if "</think>" in output:
        t = output.split("</think>", 1)[1].strip()
    else:
        t = output[-500:]
    return t[:1200]

n_out = skipped = 0
with open(f"{ER}/data_translate/shard2_science.jsonl", "w", encoding="utf-8") as out:
    for name, expected_field in [("openscience", None), ("osr2", "expected_answer")]:
        kept = 0
        for i, line in enumerate(open(f"{ER}/data_translate/{name}.jsonl", encoding="utf-8")):
            r = json.loads(line)
            if name == "openscience" and "OS-Q3-235B-4" in r.get("src", ""):
                continue  # hold 4-choice config in reserve
            gold = final_answer(r.get("output", ""), r.get(expected_field) if expected_field else None)
            if not gold:
                skipped += 1
                continue
            out.write(json.dumps({"id": f"{name}_{i}", "src": r.get("src", name),
                                  "question": r["input"], "cot": post_think(r.get("output", "")),
                                  "gold": gold, "answer_type": "mc"}, ensure_ascii=False) + "\n")
            kept += 1
            n_out += 1
        print(name, "kept", kept, flush=True)
print("SHARD2_DONE total", n_out, "skipped", skipped, flush=True)
os.system(f"cd {ER}/data_translate && zstd -f -3 shard2_science.jsonl")
print("ZST_DONE", flush=True)
