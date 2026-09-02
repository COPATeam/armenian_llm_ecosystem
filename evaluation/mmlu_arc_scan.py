import os
import json, sys
sys.path.insert(0, os.environ["ARMWEB_ROOT"] + "/code/armweb")
from armweb.decontaminate import _ngrams
from datasets import load_dataset

idx = set(); n_items = 0
for repo in ("alexandrainst/m_mmlu", "alexandrainst/m_arc"):
    ds = load_dataset(repo, "hy", split="test")
    for r in ds:
        text = r["instruction"] + " " + " ".join(str(r.get(k,"")) for k in ("option_a","option_b","option_c","option_d"))
        idx.update(_ngrams(text)); n_items += 1
print(f"index: {len(idx):,} grams from {n_items:,} items", flush=True)
h = n = 0
for line in open(os.environ["ARMWEB_ROOT"] + "/data_train/cpt/train_armbench_clean.jsonl", encoding="utf-8"):
    d = json.loads(line); n += 1
    if set(_ngrams(d["text"])) & idx: h += 1
    if n % 1000000 == 0: print(f"{n:,} scanned, {h} hits", flush=True)
print(f"FINAL armweb_train_canonical: {h} hit docs of {n:,}", flush=True)
