import os
import json, os
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
from datasets import load_dataset
OUT = os.environ["ARMWEB_ROOT"] + "/data_translate"
plan = [("OS-Q2.5-72B-10", 300_000), ("OS-Q3-235B-4", 100_000)]
with open(f"{OUT}/openscience.jsonl", "w", encoding="utf-8") as f:
    for cfg, cap in plan:
        ds = load_dataset("nvidia/OpenScience", cfg, split="train", streaming=True)
        n = 0
        for row in ds.shuffle(seed=42, buffer_size=50000):
            rec = dict(row); rec["src"] = f"openscience_{cfg}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n >= cap: break
        print(cfg, n, flush=True)
print("OS_STAGE_DONE", flush=True)
