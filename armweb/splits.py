"""Stratified val/test-iid + temporal test-tail splits over deduped survivors.

Memory-light: pass 1 records (offset, stratum) per row; assignment works on
those tuples; pass 2 re-reads winning lines by offset and writes each split.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

MIN_VALID_DATE = "1998-01"  # earlier PostDates are sentinel junk (schema audit)


def _month_of(d):
    m = d[:7] if d and len(d) >= 7 else "unknown"
    return m if m >= MIN_VALID_DATE else "unknown"


def assign_splits(in_path, out_dir, val_n=20000, test_iid_n=20000,
                  tail_months=2, tail_cap=20000, seed=1337) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pass 1: offsets + strata only
    recs = []  # (offset, source, month)
    with open(in_path, "rb") as f:
        off = 0
        for line in f:
            r = json.loads(line)
            recs.append((off, r.get("source") or "unknown",
                         _month_of(r.get("post_date"))))
            off += len(line)

    months = sorted({m for _, _, m in recs if m != "unknown"})
    tail_set = set(months[-tail_months:]) if months else set()

    rng = random.Random(seed)
    tail, pool = [], []
    for rec in recs:
        (tail if rec[2] in tail_set else pool).append(rec)
    rng.shuffle(tail)
    overflow = tail[tail_cap:]  # tail overflow returns to train
    tail = tail[:tail_cap]

    strata = defaultdict(list)
    for rec in pool:
        strata[(rec[1], rec[2])].append(rec)
    for v in strata.values():
        rng.shuffle(v)

    def take(n):
        taken, keys = [], sorted(strata.keys())
        total = sum(len(strata[k]) for k in keys)
        if total == 0:
            return taken
        shares = {k: int(n * len(strata[k]) / total) for k in keys}
        rem = n - sum(shares.values())
        for k in keys:
            if rem <= 0:
                break
            if len(strata[k]) > shares[k]:
                shares[k] += 1
                rem -= 1
        for k in keys:
            for _ in range(min(shares[k], len(strata[k]))):
                taken.append(strata[k].pop())
        return taken

    val = take(val_n)
    test_iid = take(test_iid_n)
    train = [rec for v in strata.values() for rec in v] + overflow

    # pass 2: write splits by offset
    out = {"train": train, "val": val, "test_iid": test_iid, "test_tail": tail}
    stats = {}
    with open(in_path, "rb") as src:
        for split, split_recs in out.items():
            p = out_dir / f"{split}.jsonl"
            with open(p, "w", encoding="utf-8") as dst:
                for rec in split_recs:
                    src.seek(rec[0])
                    r = json.loads(src.readline())
                    r["split"] = split
                    dst.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[split] = len(split_recs)
    stats["tail_months"] = sorted(tail_set)
    return stats
