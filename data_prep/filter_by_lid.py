"""Cluster-side: reconstruct arm_hy.jsonl from arm_meta.jsonl + lid_labels.tsv.gz.

Produces byte-identical output to the local run_lid.py keep-file: same row order,
same JSON serialization (ensure_ascii=False), same appended "lid" field.
Env: META (arm_meta.jsonl), LABELS (lid_labels.tsv.gz), OUT (arm_hy.jsonl).
"""
import gzip
import json
import os

KEEP = {"hye_Armn", "hyw_Armn"}

META = os.environ["META"]
LABELS = os.environ["LABELS"]
OUT = os.environ["OUT"]

labels = {}
with gzip.open(LABELS, "rt", encoding="utf-8") as f:
    for line in f:
        doc_id, label = line.rstrip("\n").split("\t")
        labels[doc_id] = label
print(f"loaded {len(labels):,} labels", flush=True)

kept = skipped = missing = 0
with open(OUT, "w", encoding="utf-8") as out:
    for line in open(META, encoding="utf-8"):
        r = json.loads(line)
        label = labels.get(r["id"])
        if label is None:
            missing += 1
            continue
        if label in KEEP:
            r["lid"] = label
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
        else:
            skipped += 1
print(json.dumps({"kept": kept, "quarantined": skipped, "label_missing": missing}))
