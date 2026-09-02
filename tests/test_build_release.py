import json

import pyarrow.parquet as pq
import pytest

from armweb.build_release import build


def write_split(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row(i, split):
    return {"id": f"{split}{i}", "text": "բավական երկար տեքստ այստեղ " * 10,
            "source": "news.am", "url": f"https://news.am/{i}", "topic": None,
            "post_date": "2025-05-01T00:00:00", "scrape_date": None,
            "dedup_cluster_id": f"c{split}{i}", "cluster_size": 1, "split": split,
            "lid": "hye_Armn"}


def test_build_schema_and_manifest(tmp_path):
    splits = {}
    for split in ("train", "val"):
        p = tmp_path / f"{split}.jsonl"
        write_split(p, [row(i, split) for i in range(5)])
        splits[split] = p
    out = tmp_path / "release"
    manifest = build(splits, out, shard_rows=3)
    t = pq.read_table(out / "data" / "train" / "part-0000.parquet")
    assert t.schema.names == ["id", "text", "source", "url", "topic",
                              "post_date", "scrape_date", "dedup_cluster_id",
                              "cluster_size", "split"]
    assert manifest["splits"]["train"]["rows"] == 5
    assert (out / "manifest.json").exists()


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "train.jsonl"
    rows = [row(1, "train"), row(1, "train")]
    write_split(p, rows)
    with pytest.raises(ValueError):
        build({"train": p}, tmp_path / "release")
