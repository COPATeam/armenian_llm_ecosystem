import os
"""NeMo-Curator (1.x Ray API) fuzzy-dedup cross-check on ArmWeb post-LID data.
Usage: curator_probe.py <input.jsonl> <workdir>
Matched geometry: 112 MinHash perms as 14 bands x 8 rows (J~0.72), on the
same NFKC signature view our engine used."""
import json
import shutil
import sys
import time
from pathlib import Path

ER = os.environ["ARMWEB_ROOT"]
sys.path.insert(0, f"{ER}/code/armweb")
from armweb.normalize import signature_view

inp, work = sys.argv[1], Path(sys.argv[2])
indir = work / "input"
indir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
prep = indir / "PREP_DONE"
if not prep.exists():
    import pandas as pd
    rows, part, total = [], 0, 0
    def flush_rows(rows, part):
        pd.DataFrame(rows).to_parquet(indir / f"prep_{part:04d}.parquet", index=False)
    for line in open(inp, encoding="utf-8"):
        d = json.loads(line)
        rows.append({"id": d["id"], "text": signature_view(d["text"])})
        if len(rows) >= 500000:
            flush_rows(rows, part); total += len(rows); rows, part = [], part + 1
    if rows:
        flush_rows(rows, part); total += len(rows)
    (indir / "prepped.parquet").touch() if False else None
    prep.write_bytes(b"") if False else None
    open(indir / "PREP_DONE", "w").write(str(total))
    print(f"prepped {total} rows in {time.time()-t0:.0f}s", flush=True)

import nemo_curator
print("nemo_curator", nemo_curator.__version__, flush=True)
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

for sub in ("cache", "out"):
    shutil.rmtree(work / sub, ignore_errors=True)

client = RayClient()
client.start()
kwargs = dict(
    input_path=str(indir),
    cache_path=str(work / "cache"),
    output_path=str(work / "out"),
    text_field="text",
    perform_removal=False,
    seed=42,
    char_ngrams=30,
    num_bands=14,
    minhashes_per_band=8,
    use_64_bit_hash=False,
    input_blocksize="256MiB",
)
try:
    wf = FuzzyDeduplicationWorkflow(id_field="id", **kwargs)
except TypeError as e:
    print("no id_field kwarg:", e, flush=True)
    wf = FuzzyDeduplicationWorkflow(**kwargs)
result = wf.run()
try:
    print("metadata:", json.dumps(result.metadata, default=str), flush=True)
except Exception:
    pass

import pandas as pd
import glob as _g
dup_candidates = _g.glob(str(work / "out" / "*")) + _g.glob(str(work / "out" / "**" / "*"), recursive=True)
print("out tree:", dup_candidates[:20], flush=True)
dup_dir = work / "out" / "FuzzyDuplicateIds"
df = pd.read_parquet(dup_dir)
print("columns:", list(df.columns), "rows:", len(df), flush=True)
print(df.head(5).to_string(), flush=True)
df.to_json(work / "duplicate_ids.jsonl", orient="records", lines=True)
n_docs = int(open(prep).read())
print(json.dumps({"docs": n_docs, "to_remove": len(df),
                  "removal_rate": len(df) / n_docs}), flush=True)

# try official removal workflow to obtain kept output with original ids
removed_ok = False
for path in ("nemo_curator.stages.deduplication.removal_workflow",
             "nemo_curator.stages.text.deduplication.removal_workflow",
             "nemo_curator.stages.deduplication.text_duplicates_removal"):
    try:
        mod = __import__(path, fromlist=["TextDuplicatesRemovalWorkflow"])
        RW = getattr(mod, "TextDuplicatesRemovalWorkflow")
        rw = RW(input_path=str(indir), ids_to_remove_path=str(dup_dir),
                output_path=str(work / "kept"), text_field="text")
        rw.run()
        removed_ok = True
        print("removal workflow OK via", path, flush=True)
        break
    except Exception as e:
        print("removal try", path, "->", type(e).__name__, str(e)[:200], flush=True)
if removed_ok:
    kept = pd.read_parquet(work / "kept")
    print("kept rows:", len(kept), "columns:", list(kept.columns), flush=True)
    kept[["id"]].to_json(work / "kept_ids.jsonl", orient="records", lines=True)
print(f"CURATOR_DONE {time.time()-t0:.0f}s", flush=True)
