"""cluster login: stream-sample science sources -> jsonl for decon+translation.
OpenScience (6M pool -> 400k sample, prefer 10-choice), OSR2 -> 150k sample."""
import os
import json, os, random
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
from datasets import load_dataset

OUT = os.environ["ARMWEB_ROOT"] + "/data_translate"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(42)

def dump(name, it, cap, fields):
    n = 0
    with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
        for row in it:
            rec = {k: row.get(k) for k in fields if row.get(k) is not None}
            rec["src"] = name
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n >= cap:
                break
    print(f"{name}: {n}", flush=True)

try:
    ds = load_dataset("nvidia/OpenScience", split="train", streaming=True)
    first = next(iter(ds))
    print("OpenScience fields:", sorted(first.keys()), flush=True)
    dump("openscience", ds.shuffle(seed=42, buffer_size=50000), 400_000, list(first.keys()))
except Exception as e:
    print("OPENSCIENCE_FAIL", type(e).__name__, str(e)[:300], flush=True)

try:
    ds = load_dataset("nvidia/OpenScienceReasoning-2", split="train", streaming=True)
    first = next(iter(ds))
    print("OSR2 fields:", sorted(first.keys()), flush=True)
    dump("osr2", ds.shuffle(seed=42, buffer_size=50000), 150_000, list(first.keys()))
except Exception as e:
    print("OSR2_FAIL", type(e).__name__, str(e)[:300], flush=True)

# decon references: English MMLU-Pro test -> benchmark txt for the 13-gram index
try:
    mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    bdir = os.environ["ARMWEB_ROOT"] + "/code/armweb_data/benchmarks"
    with open(f"{bdir}/mmlu_pro_en_test.txt", "w", encoding="utf-8") as f:
        for row in mp:
            opts = " ".join(str(o) for o in (row.get("options") or []))
            f.write((row.get("question") or "") + " " + opts + "\n")
    print("MMLUPRO_REF_DONE", len(mp), flush=True)
except Exception as e:
    print("MMLUPRO_FAIL", type(e).__name__, str(e)[:300], flush=True)
print("STAGE_SCIENCE_DONE", flush=True)
