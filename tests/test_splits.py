import json

from armweb.splits import assign_splits


def make(tmp_path, n=1000):
    p = tmp_path / "in.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            month = 1 + (i % 6)  # 2025-01 .. 2025-06
            f.write(json.dumps({"id": str(i), "text": f"տեքստ {i} " * 30,
                                "source": f"src{i % 4}",
                                "post_date": f"2025-0{month}-15T00:00:00",
                                "dedup_cluster_id": str(i), "cluster_size": 1}) + "\n")
    return p


def test_splits_disjoint_and_tail_temporal(tmp_path):
    stats = assign_splits(make(tmp_path), tmp_path, val_n=100, test_iid_n=100,
                          tail_months=2, tail_cap=500, seed=1337)
    ids = {}
    for split in ("train", "val", "test_iid", "test_tail"):
        rows = [json.loads(l) for l in open(tmp_path / f"{split}.jsonl", encoding="utf-8")]
        ids[split] = {r["id"] for r in rows}
        assert all(r["split"] == split for r in rows)
    all_ids = sum(len(v) for v in ids.values())
    assert all_ids == len(set().union(*ids.values())) == 1000  # disjoint, complete
    tail_rows = [json.loads(l) for l in open(tmp_path / "test_tail.jsonl", encoding="utf-8")]
    assert tail_rows and all(r["post_date"][5:7] in ("05", "06") for r in tail_rows)
    assert len(ids["val"]) == 100 and len(ids["test_iid"]) == 100
    assert stats["val"] == 100


def test_deterministic(tmp_path):
    p = make(tmp_path)
    assign_splits(p, tmp_path / "a", val_n=100, test_iid_n=100, seed=1337)
    assign_splits(p, tmp_path / "b", val_n=100, test_iid_n=100, seed=1337)
    a = open(tmp_path / "a" / "val.jsonl", encoding="utf-8").read()
    b = open(tmp_path / "b" / "val.jsonl", encoding="utf-8").read()
    assert a == b


def test_junk_dates_not_in_tail(tmp_path):
    p = tmp_path / "in.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(200):
            date = "1900-01-01T00:00:00" if i < 50 else f"2025-0{1 + i % 6}-10T00:00:00"
            f.write(json.dumps({"id": str(i), "text": "տեքստ " * 40,
                                "source": "s", "post_date": date}) + "\n")
    assign_splits(p, tmp_path, val_n=10, test_iid_n=10, tail_months=1, seed=1337)
    tail = [json.loads(l) for l in open(tmp_path / "test_tail.jsonl", encoding="utf-8")]
    assert all(not r["post_date"].startswith("1900") for r in tail)
