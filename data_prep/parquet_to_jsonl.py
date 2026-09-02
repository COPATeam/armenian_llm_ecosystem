"""Convert release parquet splits to text-only jsonl for Megatron preprocessing."""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

release = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
for split_dir in sorted((release / "data").iterdir()):
    split = split_dir.name
    out = out_dir / f"{split}.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for part in sorted(split_dir.glob("part-*.parquet")):
            table = pq.read_table(part, columns=["text"])
            for text in table.column("text").to_pylist():
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{split}: {n} docs -> {out}", flush=True)
