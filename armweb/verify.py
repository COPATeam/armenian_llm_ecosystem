"""Cross-split leakage gates: exact, MinHash near-dup, paragraph overlap.

Memory-light: the small splits (val/test, ~20k docs each) are held in RAM;
train (~4.5M docs) is always STREAMED against them, never loaded.
"""
import random

import xxhash
from datasketch import MinHash, MinHashLSH

from .io_utils import iter_jsonl, write_stats
from .normalize import signature_view


def _doc_hash(text):
    return xxhash.xxh128_hexdigest(signature_view(text))


def _minhash(text, num_perm=112):
    m = MinHash(num_perm=num_perm, seed=42)
    words = signature_view(text).split()
    for i in range(max(len(words) - 4, 1)):
        m.update(" ".join(words[i:i + 5]).encode())
    return m


def verify_exact(paths) -> dict:
    """Pairwise exact-hash overlaps. Small splits' hash sets in RAM; train streamed."""
    small = {s: {_doc_hash(r["text"]) for r in iter_jsonl(p)}
             for s, p in paths.items() if s != "train"}
    out = {}
    names = sorted(small)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = len(small[a] & small[b])
    if "train" in paths:
        hit_hashes = {s: set() for s in small}
        for r in iter_jsonl(paths["train"]):
            h = _doc_hash(r["text"])
            for s, hset in small.items():
                if h in hset:
                    hit_hashes[s].add(h)
        for s in sorted(small):
            key = (s, "train") if s < "train" else ("train", s)
            out[key] = len(hit_hashes[s])
    return out


def verify_near(paths, threshold=0.75, num_perm=112, sample_train=100000,
                sample_per_split=20000, seed=42) -> dict:
    """LSH-index a streamed reservoir sample of train; query val/test docs."""
    rng = random.Random(seed)
    reservoir = []
    n_seen = 0
    for r in iter_jsonl(paths["train"]):
        n_seen += 1
        if len(reservoir) < sample_train:
            reservoir.append(r["text"])
        else:
            j = rng.randrange(n_seen)
            if j < sample_train:
                reservoir[j] = r["text"]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, text in enumerate(reservoir):
        lsh.insert(f"t{i}", _minhash(text, num_perm))
    train_sampled = len(reservoir)
    del reservoir

    out = {"train_sampled": train_sampled, "train_total": n_seen}
    for split, p in paths.items():
        if split == "train":
            continue
        rows = [r["text"] for r in iter_jsonl(p)]
        if len(rows) > sample_per_split:
            rows = rng.sample(rows, sample_per_split)
        hits = sum(bool(lsh.query(_minhash(t, num_perm))) for t in rows)
        out[split] = {"checked": len(rows), "near_dups": hits,
                      "rate": hits / max(len(rows), 1)}
    return out


def verify_paragraphs(paths, min_tokens=13) -> dict:
    """Paragraph-hash leakage: small splits' paragraph sets in RAM; train streamed."""
    def paras(text):
        for para in text.split("\n\n"):
            if len(para.split()) >= min_tokens:
                yield xxhash.xxh64_hexdigest(signature_view(para))

    small = {}
    for s, p in paths.items():
        if s == "train":
            continue
        pset = set()
        for r in iter_jsonl(p):
            pset.update(paras(r["text"]))
        small[s] = pset
    leaks = {s: 0 for s in small}
    if "train" in paths:
        for r in iter_jsonl(paths["train"]):
            for h in paras(r["text"]):
                for s, pset in small.items():
                    if h in pset:
                        leaks[s] += 1
    return leaks


def run_all(paths) -> dict:
    exact = {f"{a}|{b}": v for (a, b), v in verify_exact(paths).items()}
    near = verify_near(paths)
    paras = verify_paragraphs(paths)
    report = {"exact_overlaps": exact, "near_dup": near, "paragraph_leaks": paras}
    exact_ok = all(v == 0 for v in exact.values())
    near_ok = all(v["rate"] < 0.001 for v in near.values() if isinstance(v, dict))
    ok = exact_ok and near_ok
    report["pass"] = ok
    write_stats("stage8_gates", report)
    if not ok:
        print("GATE FAILURE", report)
        raise SystemExit(1)
    return report
