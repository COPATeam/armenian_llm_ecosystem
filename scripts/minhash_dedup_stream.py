"""Streaming/memory-light variant of minhash_dedup.py — identical recipe & output.

Differences from the in-memory engine (minhash_dedup.py):
  - Pass A streams the file, keeping only offsets/lengths/hashes (no text in RAM)
  - Band hashes are collected into a (n_docs x 14) uint64 numpy array; per-band
    grouping via np.unique instead of a python dict of 80M bucket entries
  - Pass C re-reads winning lines by offset and writes survivors
RAM: ~3-4 GB for 5.9M docs. Same seed/recipe -> identical clusters as the
in-memory engine (validated on the synthetic smoke test).

Env: IN, OUT, WORKERS.
NOTE: signature_view duplicated from armweb/normalize.py — keep in sync.
"""
import json
import os
import re
import unicodedata
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xxhash

IN = Path(os.environ["IN"])
OUT = Path(os.environ["OUT"])
WORKERS = int(os.environ.get("WORKERS", max((os.cpu_count() or 8) - 4, 4)))

SEED = 42
NUM_PERM = 112
BANDS, ROWS = 14, 8
MERSENNE = (1 << 61) - 1
_rng = np.random.RandomState(SEED)
A = _rng.randint(1, MERSENNE, size=NUM_PERM, dtype=np.uint64)
B = _rng.randint(0, MERSENNE, size=NUM_PERM, dtype=np.uint64)

_WS, _DIGIT = re.compile(r"\s+"), re.compile(r"\d")


def signature_view(text):
    text = text.replace("և", "եւ")
    t = unicodedata.normalize("NFKC", text).lower()
    t = "".join(c for c in t if not unicodedata.category(c).startswith("P"))
    return _WS.sub(" ", _DIGIT.sub("0", t)).strip()


def band_hashes(text: str):
    words = signature_view(text).split()
    n = len(words)
    if n < 5:
        shingles = {" ".join(words)} if words else {""}
    else:
        shingles = {" ".join(words[i:i + 5]) for i in range(n - 4)}
    h = np.fromiter((xxhash.xxh64_intdigest(s) for s in shingles),
                    dtype=np.uint64, count=len(shingles))
    sig = np.empty(NUM_PERM, dtype=np.uint64)
    for p in range(NUM_PERM):
        sig[p] = ((A[p] * h + B[p]) % MERSENNE).min()
    return [xxhash.xxh64_intdigest(sig[b * ROWS:(b + 1) * ROWS].tobytes())
            for b in range(BANDS)]


def _worker(chunk):
    """chunk: list of (rep_idx, offset). Reads lines itself; returns band hashes."""
    out = []
    with open(IN, "rb") as f:
        for rep_idx, off in chunk:
            f.seek(off)
            r = json.loads(f.readline())
            out.append((rep_idx, band_hashes(r["text"])))
    return out


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Pass A: stream; exact dedup on signature hash; keep offsets only ──
    exact_best = {}   # hash -> rep slot
    reps_off, reps_len, reps_count = [], [], []
    input_docs = input_bytes = exact_removed = 0
    with open(IN, "rb") as f:
        off = 0
        for line in f:
            r = json.loads(line)
            tlen = len(r["text"])
            input_bytes += len(r["text"].encode())
            h = xxhash.xxh128_hexdigest(signature_view(r["text"]))
            slot = exact_best.get(h)
            if slot is None:
                exact_best[h] = len(reps_off)
                reps_off.append(off)
                reps_len.append(tlen)
                reps_count.append(1)
            else:
                exact_removed += 1
                reps_count[slot] += 1
                if tlen > reps_len[slot]:
                    reps_off[slot] = off
                    reps_len[slot] = tlen
            input_docs += 1
            off += len(line)
    del exact_best
    n = len(reps_off)
    print(f"exact: {input_docs:,} in, {exact_removed:,} removed, {n:,} kept", flush=True)

    # ── Pass B: band hashes -> numpy matrix (parallel) ────────────────────
    bh = np.empty((n, BANDS), dtype=np.uint64)
    items = list(enumerate(reps_off))
    chunks = [items[i:i + 4096] for i in range(0, n, 4096)]
    with Pool(WORKERS) as pool:
        done = 0
        for res in pool.imap_unordered(_worker, chunks):
            for rep_idx, bands in res:
                bh[rep_idx] = bands
            done += len(res)
            if done % 500_000 < 4096:
                print(f"minhash: {done:,}/{n:,}", flush=True)

    # ── union same-bucket docs per band (numpy grouping) ──────────────────
    uf = UnionFind(n)
    for b in range(BANDS):
        order = np.argsort(bh[:, b], kind="stable")
        col = bh[order, b]
        boundaries = np.flatnonzero(np.diff(col)) + 1
        start = 0
        for end in list(boundaries) + [n]:
            if end - start > 1:
                first = int(order[start])
                for j in range(start + 1, end):
                    uf.union(first, int(order[j]))
            start = end
        print(f"band {b + 1}/{BANDS} unioned", flush=True)
    del bh

    # ── components; keep longest ──────────────────────────────────────────
    best, cluster_size = {}, defaultdict(int)
    for i in range(n):
        g = uf.find(i)
        cluster_size[g] += reps_count[i]
        if g not in best or reps_len[i] > reps_len[best[g]]:
            best[g] = i

    # ── Pass C: write survivors by offset ─────────────────────────────────
    survivors_path = OUT / "survivors.jsonl"
    n_surv = surv_bytes = 0
    with open(IN, "rb") as src, open(survivors_path, "w", encoding="utf-8") as out:
        for g, i in best.items():
            src.seek(reps_off[i])
            r = json.loads(src.readline())
            r["dedup_cluster_id"] = str(g)
            r["cluster_size"] = cluster_size[g]
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_surv += 1
            surv_bytes += len(r["text"].encode())

    sizes = sorted(cluster_size.values(), reverse=True)
    stats = {"input_docs": input_docs, "input_bytes": input_bytes,
             "exact_removed": exact_removed,
             "fuzzy_removed": n - n_surv,
             "survivors": n_surv, "survivor_bytes": surv_bytes,
             "n_clusters": len(cluster_size), "top50_cluster_sizes": sizes[:50],
             "engine": "armweb-minhash-stream (numpy/xxhash, word 5-grams, 112 perms, 14x8, seed 42)"}
    (OUT / "dedup_stats.json").write_text(json.dumps(stats, indent=2))

    top_groups = sorted((g for g, s in cluster_size.items() if s > 1),
                        key=lambda g: -cluster_size[g])[:20]
    top_set = set(top_groups)
    with open(IN, "rb") as src, open(OUT / "clusters_sample.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            g = uf.find(i)
            if g in top_set:
                src.seek(reps_off[i])
                r = json.loads(src.readline())
                f.write(json.dumps({"group": str(g), "id": r["id"],
                                    "source": r.get("source"),
                                    "text": r["text"][:2000]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "top50_cluster_sizes"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
