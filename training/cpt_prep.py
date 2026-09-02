import os
"""Convert CPT mix sources to text-jsonl for Megatron preprocessing (gemma tokenizer)."""
import json
from pathlib import Path
import pyarrow.parquet as pq

ER = Path(os.environ["ARMWEB_ROOT"])
OUT = ER / "data_train/cpt/jsonl"
OUT.mkdir(exist_ok=True)

def dump(rows_iter, name):
    n = b = 0
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for text in rows_iter:
            if not text or len(text) < 200:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1; b += len(text.encode())
    print(f"{name}: {n} docs, {b/2**30:.2f} GiB", flush=True)

def parquet_texts(root, col):
    for p in sorted(Path(root).rglob("*.parquet")):
        t = pq.read_table(p, columns=[col])
        yield from (x for x in t.column(col).to_pylist() if x)

# hy: already jsonl with text field — symlink-equivalent copy of decontaminated file
import shutil
src = ER / "data_train/cpt/train_armbench_clean.jsonl"
dst = OUT / "hy_clean.jsonl"
if not dst.exists():
    dst.symlink_to(src)
print("hy_clean: linked", flush=True)

dump(parquet_texts(ER / "data_train/cpt/fineweb_edu_10bt", "text"), "fineweb_edu")
def stack_texts():
    for jf in sorted((ER / "data_train/cpt/stack_smol/data").glob("*/data.json")):
        for line in jf.open(encoding="utf-8", errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            c = d.get("content")
            if c:
                yield c
dump(stack_texts(), "stack_smol")
