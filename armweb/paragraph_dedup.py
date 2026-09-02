"""Dolma-style boilerplate removal: exact paragraphs >=min_tokens repeated across >=min_docs docs."""
from collections import Counter
from pathlib import Path

import xxhash

from .io_utils import iter_jsonl, append_jsonl
from .normalize import signature_view


def _para_hash(p: str) -> str:
    return xxhash.xxh64_hexdigest(signature_view(p))


def _paras(text: str):
    return [p for p in text.split("\n\n") if p.strip()]


def count_paragraphs(in_path, min_tokens: int = 13) -> Counter:
    c = Counter()
    for r in iter_jsonl(in_path):
        seen_in_doc = set()
        for p in _paras(r["text"]):
            if len(p.split()) >= min_tokens:
                h = _para_hash(p)
                if h not in seen_in_doc:
                    c[h] += 1
                    seen_in_doc.add(h)
    return c


def strip_boilerplate(in_path, out_path, min_tokens: int = 13, min_docs: int = 10) -> dict:
    counts = count_paragraphs(in_path, min_tokens)
    bad = {h for h, n in counts.items() if n >= min_docs}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).unlink(missing_ok=True)
    stats = {"paragraphs_removed": 0, "docs_touched": 0, "docs_dropped": 0}
    examples = Counter()
    for r in iter_jsonl(in_path):
        kept, removed = [], 0
        for p in _paras(r["text"]):
            if len(p.split()) >= min_tokens and _para_hash(p) in bad:
                removed += 1
                examples[p[:80]] += 1
            else:
                kept.append(p)
        if removed:
            stats["docs_touched"] += 1
            stats["paragraphs_removed"] += removed
            r["text"] = "\n\n".join(kept)
        if len(r["text"]) < 100:
            stats["docs_dropped"] += 1
            continue
        append_jsonl(out_path, r)
    stats["top_boilerplate"] = examples.most_common(20)
    return stats
