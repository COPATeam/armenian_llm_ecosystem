"""Streaming extraction of Arm.bson keeping url/source/date/topic metadata (no author)."""
import json
import struct
from pathlib import Path

import bson

from .normalize import clean_document


def iter_bson_documents(bson_path: Path):
    """Stream BSON docs; resilient to corruption (resyncs on next valid boundary)."""
    corrupt = 0
    with open(bson_path, "rb") as f:
        while True:
            start_pos = f.tell()
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break
            doc_size = struct.unpack("<i", size_bytes)[0]
            if doc_size < 5 or doc_size > 64 * 1024 * 1024:
                corrupt += 1
                f.seek(start_pos + 1)
                continue
            doc_bytes = size_bytes + f.read(doc_size - 4)
            if len(doc_bytes) < doc_size:
                break
            if doc_bytes[-1:] != b"\x00":
                corrupt += 1
                f.seek(start_pos + 1)
                continue
            try:
                yield bson.decode(doc_bytes)
            except Exception:
                corrupt += 1
                f.seek(start_pos + 1)
    if corrupt:
        print(f"WARNING: {corrupt} corrupt reads in {bson_path.name}")


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else (str(v) if v else None)


def _str_or_none(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def extract(bson_path: Path, out_path: Path, min_len: int = 100) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"read": 0, "kept": 0, "skipped_clean": 0}
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_bson_documents(Path(bson_path)):
            stats["read"] += 1
            text = clean_document(doc.get("Title"), doc.get("Text"), min_len)
            if text is None:
                stats["skipped_clean"] += 1
                continue
            out.write(json.dumps({
                "id": str(doc.get("_id")),
                "text": text,
                "source": _str_or_none(doc.get("Source")),
                "url": _str_or_none(doc.get("Href")),
                "topic": _str_or_none(doc.get("Topic")),
                "post_date": _iso(doc.get("PostDate")),
                "scrape_date": _iso(doc.get("ScrapeDate")),
            }, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    return stats
