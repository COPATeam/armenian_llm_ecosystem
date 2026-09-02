"""Stage 5: independent dedup audit — datasketch MinHash (word 5-grams, 14x8)
on a 1% sample; agreement vs curator cluster assignments.

(datatrove's pipeline is heavyweight for a sample audit; datasketch with
word-shingles IS the FineWeb-faithful recipe — 5-gram words, 112 perms, 14x8.
The point is independence from NeMo-Curator's char-24 shingles + implementation.)
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasketch import MinHash, MinHashLSH

from armweb.io_utils import iter_jsonl, write_stats, DATA_ROOT
from armweb.normalize import signature_view

SEED = 42
SAMPLE_RATE = 0.01


def minhash(text):
    m = MinHash(num_perm=112, seed=SEED)
    words = signature_view(text).split()
    for i in range(max(len(words) - 4, 1)):
        m.update(" ".join(words[i:i + 5]).encode())
    return m


def main():
    rng = random.Random(SEED)
    # sample ids from the LID output (dedup input), keep curator cluster map for them
    sample = [r for r in iter_jsonl(DATA_ROOT / "lid" / "arm_hy.jsonl")
              if rng.random() < SAMPLE_RATE]
    print(f"sampled {len(sample)} docs")

    lsh = MinHashLSH(threshold=0.72, num_perm=112)
    mh = {}
    for i, r in enumerate(sample):
        mh[i] = minhash(r["text"])
        lsh.insert(str(i), mh[i])
    # datasketch verdict: doc is 'dup' if it collides with another sample doc
    ds_dup = set()
    for i in mh:
        hits = [h for h in lsh.query(mh[i]) if h != str(i)]
        if hits:
            ds_dup.add(i)

    # curator verdict on the same ids: dup iff cluster_size>1 for the SURVIVOR of
    # its cluster, or the doc is absent from survivors (i.e. was removed as dup)
    surv = {}
    for r in iter_jsonl(DATA_ROOT / "dedup" / "survivors.jsonl"):
        surv[r["id"]] = r.get("cluster_size", 1)
    cur_dup = set()
    for i, r in enumerate(sample):
        cs = surv.get(r["id"])
        if cs is None or cs > 1:  # removed, or survivor of a multi-doc cluster
            cur_dup.add(i)

    both = len(ds_dup & cur_dup)
    agree = (len(sample) - len(ds_dup ^ cur_dup)) / max(len(sample), 1)
    report = {"sample": len(sample), "datasketch_dups": len(ds_dup),
              "curator_dups": len(cur_dup), "both": both,
              "datasketch_only": len(ds_dup - cur_dup),
              "curator_only": len(cur_dup - ds_dup),
              "doc_verdict_agreement": round(agree, 4)}
    # dump up to 20 disagreements for manual inspection
    dis = list(ds_dup ^ cur_dup)[:20]
    report["disagreement_examples"] = [
        {"id": sample[i]["id"], "in": "datasketch" if i in ds_dup else "curator",
         "text": sample[i]["text"][:200]} for i in dis]
    write_stats("stage5_audit", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "disagreement_examples"}, indent=2))


if __name__ == "__main__":
    main()
