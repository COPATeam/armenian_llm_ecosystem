"""13-gram decontamination of train against benchmark passages (OLMo decon-style)."""
from pathlib import Path

import xxhash

from .io_utils import iter_jsonl, append_jsonl
from .normalize import signature_view

N = 13


def _ngrams(text: str):
    words = signature_view(text).split()
    for i in range(len(words) - N + 1):
        yield xxhash.xxh64_hexdigest(" ".join(words[i:i + N]))


def build_ngram_index(bench_dir) -> dict:
    index = {}
    for f in sorted(Path(bench_dir).glob("*.txt")):
        if f.name.startswith("._"):  # macOS AppleDouble junk from tar transfers
            continue
        grams = set()
        for line in f.read_text(encoding="utf-8", errors="strict").splitlines():
            grams.update(_ngrams(line))
        index[f.stem] = grams
    return index


def scan(train_path, out_path, index: dict) -> dict:
    Path(out_path).unlink(missing_ok=True)
    stats = {"hits": {k: 0 for k in index}, "dropped": 0, "kept": 0}
    for r in iter_jsonl(train_path):
        doc_grams = set(_ngrams(r["text"]))
        hit = False
        for bench, grams in index.items():
            if doc_grams & grams:
                stats["hits"][bench] += 1
                hit = True
        if hit:
            stats["dropped"] += 1
        else:
            stats["kept"] += 1
            append_jsonl(out_path, r)
    return stats
