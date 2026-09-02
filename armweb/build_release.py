"""Final release parquet writer with schema validation + manifest."""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .io_utils import iter_jsonl

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("source", pa.string()),
    ("url", pa.string()),
    ("topic", pa.string()),
    ("post_date", pa.string()),
    ("scrape_date", pa.string()),
    ("dedup_cluster_id", pa.string()),
    ("cluster_size", pa.int32()),
    ("split", pa.string()),
])

COLS = [f.name for f in SCHEMA]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(splits: dict, out_dir, shard_rows: int = 500_000) -> dict:
    out_dir = Path(out_dir)
    manifest = {"schema": [f"{f.name}:{f.type}" for f in SCHEMA], "splits": {}, "files": {}}
    seen_ids = set()
    for split, in_path in splits.items():
        split_dir = out_dir / "data" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        buf, part, rows_total, bytes_total = [], 0, 0, 0
        for r in iter_jsonl(in_path):
            if not r.get("text") or len(r["text"]) < 100:
                raise ValueError(f"bad text in {split}: id={r.get('id')}")
            if r["split"] != split:
                raise ValueError(f"split mismatch: row says {r['split']}, file is {split}")
            if r["id"] in seen_ids:
                raise ValueError(f"duplicate id across release: {r['id']}")
            seen_ids.add(r["id"])
            buf.append({c: r.get(c) for c in COLS})
            bytes_total += len(r["text"].encode())
            rows_total += 1
            if len(buf) >= shard_rows:
                _write_part(split_dir, part, buf, manifest)
                buf, part = [], part + 1
        if buf:
            _write_part(split_dir, part, buf, manifest)
        manifest["splits"][split] = {"rows": rows_total, "text_bytes": bytes_total}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _write_part(split_dir: Path, part: int, buf: list, manifest: dict) -> None:
    p = split_dir / f"part-{part:04d}.parquet"
    table = pa.Table.from_pylist(buf, schema=SCHEMA)
    pq.write_table(table, p, compression="zstd")
    manifest["files"][str(p.relative_to(split_dir.parent.parent))] = {
        "rows": len(buf), "sha256": _sha256(p)}
