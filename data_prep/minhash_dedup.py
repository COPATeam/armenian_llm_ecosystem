"""Exact + fuzzy MinHash dedup — FineWeb recipe, self-contained (no dask/RAPIDS).

Replaces the NeMo-Curator driver after its dask stack failed two different ways
(FP-check shuffle assertion; cudf serializer dispatch) — recorded in
cluster_usage/README.md. The RECIPE is unchanged and is exactly FineWeb's:
  - signatures on the NFKC signature view (և folded, punct stripped, digits->0)
  - word 5-gram shingles
  - 112 MinHash permutations (universal hashing (a*h+b) mod p over xxhash64)
  - LSH: 14 bands x 8 rows  ->  Jaccard threshold ~ (1/14)^(1/8) ≈ 0.72
  - same-bucket docs unioned (no FP verification, like FineWeb);
    independent datasketch audit quantifies implementation agreement
  - connected components -> clusters; keep longest doc; cluster_size recorded
Stage A exact dedup first: xxh128 of signature view, keep longest.

Runs anywhere with numpy+xxhash (cluster batch node: 224 cores; ~minutes for 5.8M docs).
Env: IN, OUT, WORKERS (default cpu_count-4).

NOTE: signature_view duplicated from armweb/normalize.py — keep in sync.
"""
import json
import os
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xxhash

ROOT = Path(os.environ.get("ARMWEB_ROOT",
                           os.environ["ARMWEB_ROOT"] + ""))
IN = Path(os.environ.get("IN", ROOT / "data/arm_hy.jsonl"))
OUT = Path(os.environ.get("OUT", ROOT / "out"))
WORKERS = int(os.environ.get("WORKERS", max((os.cpu_count() or 8) - 4, 4)))

SEED = 42
NUM_PERM = 112
BANDS, ROWS = 14, 8
assert BANDS * ROWS == NUM_PERM
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
    """14 uint64 band-hashes of the doc's 112-perm MinHash signature."""
    words = signature_view(text).split()
    n = len(words)
    if n < 5:
        shingles = [" ".join(words)] if words else [""]
    else:
        shingles = [" ".join(words[i:i + 5]) for i in range(n - 4)]
    h = np.fromiter((xxhash.xxh64_intdigest(s) for s in set(shingles)),
                    dtype=np.uint64, count=len(set(shingles)))
    # universal hashing: sig[p] = min((a_p * h + b_p) mod prime)
    sig = np.empty(NUM_PERM, dtype=np.uint64)
    for p in range(NUM_PERM):
        sig[p] = ((A[p] * h + B[p]) % MERSENNE).min()
    return [xxhash.xxh64_intdigest(sig[b * ROWS:(b + 1) * ROWS].tobytes())
            for b in range(BANDS)]


def _worker(args):
    idx, text = args
    return idx, band_hashes(text)


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

    # ── Stage A: exact dedup (streaming) ──────────────────────────────────
    seen, exact_count = {}, {}
    rows_kept, exact_removed, input_docs, input_bytes = [], 0, 0, 0
    with open(IN, encoding="utf-8") as f:
        for line in f:
            input_docs += 1
            r = json.loads(line)
            input_bytes += len(r["text"].encode())
            h = xxhash.xxh128_hexdigest(signature_view(r["text"]))
            if h in seen:
                exact_removed += 1
                exact_count[h] += 1
                if len(r["text"]) > len(rows_kept[seen[h]]["text"]):
                    rows_kept[seen[h]] = r
            else:
                seen[h] = len(rows_kept)
                exact_count[h] = 1
                rows_kept.append(r)
    for h, idx in seen.items():
        rows_kept[idx]["exact_group_size"] = exact_count[h]
    del seen
    print(f"exact: {input_docs:,} in, {exact_removed:,} removed, "
          f"{len(rows_kept):,} kept", flush=True)

    # ── Stage B: MinHash band hashes (parallel) ───────────────────────────
    buckets = {}  # (band_idx, hash) -> first doc idx  (union on collision)
    uf = UnionFind(len(rows_kept))
    with Pool(WORKERS) as pool:
        it = pool.imap_unordered(
            _worker, ((i, r["text"]) for i, r in enumerate(rows_kept)),
            chunksize=256)
        for done, (idx, bands) in enumerate(it):
            for b, bh in enumerate(bands):
                key = (b, bh)
                if key in buckets:
                    uf.union(buckets[key], idx)
                else:
                    buckets[key] = idx
            if done % 500_000 == 0:
                print(f"minhash: {done:,}/{len(rows_kept):,}", flush=True)
    del buckets
    print("minhash done, resolving components", flush=True)

    # ── components -> clusters, keep longest ──────────────────────────────
    best, cluster_size = {}, {}
    for i, r in enumerate(rows_kept):
        g = uf.find(i)
        cluster_size[g] = cluster_size.get(g, 0) + r.get("exact_group_size", 1)
        if g not in best or len(r["text"]) > len(rows_kept[best[g]]["text"]):
            best[g] = i

    survivors_path = OUT / "survivors.jsonl"
    n_surv, surv_bytes = 0, 0
    with open(survivors_path, "w", encoding="utf-8") as f:
        for g, i in best.items():
            r = dict(rows_kept[i])
            r["dedup_cluster_id"] = str(g)
            r["cluster_size"] = cluster_size[g]
            r.pop("exact_group_size", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_surv += 1
            surv_bytes += len(r["text"].encode())

    sizes = sorted(cluster_size.values(), reverse=True)
    stats = {"input_docs": input_docs, "input_bytes": input_bytes,
             "exact_removed": exact_removed,
             "fuzzy_removed": len(rows_kept) - n_surv,
             "survivors": n_surv, "survivor_bytes": surv_bytes,
             "n_clusters": len(cluster_size), "top50_cluster_sizes": sizes[:50],
             "engine": "armweb-minhash (numpy/xxhash, word 5-grams, 112 perms, 14x8, seed 42)"}
    (OUT / "dedup_stats.json").write_text(json.dumps(stats, indent=2))

    # top-20 multi-doc clusters' texts for manual categorization
    top_groups = sorted((g for g, s in cluster_size.items() if s > 1),
                        key=lambda g: -cluster_size[g])[:20]
    top_set = set(top_groups)
    with open(OUT / "clusters_sample.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows_kept):
            g = uf.find(i)
            if g in top_set:
                f.write(json.dumps({"group": str(g), "id": r["id"],
                                    "source": r.get("source"),
                                    "text": r["text"][:2000]},
                                   ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
