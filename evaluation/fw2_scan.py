import os
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
import pyarrow.parquet as pq
from armweb.decontaminate import build_ngram_index, _ngrams

ER = Path(os.environ["ARMWEB_ROOT"])
index = build_ngram_index(ER / "code/armweb_data/benchmarks")
stats = {"hits": {k: 0 for k in index}, "dropped": 0, "kept": 0}
for p in sorted((ER / "data_train/baselines/fineweb2_hy").rglob("*.parquet")):
    for text in pq.read_table(p, columns=["text"]).column("text").to_pylist():
        doc = set(_ngrams(text))
        hit = False
        for bench, grams in index.items():
            if doc & grams:
                stats["hits"][bench] += 1
                hit = True
        if hit: stats["dropped"] += 1
        else: stats["kept"] += 1
json.dump(stats, open(ER / "data_train/baselines_clean/fineweb2_hy_scan_stats.json", "w"), indent=1)
print("DONE", stats["dropped"], stats["kept"])
