import json

import pytest

import armweb.io_utils as io_utils
from armweb.verify import verify_exact, verify_near, run_all


@pytest.fixture(autouse=True)
def isolated_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(io_utils, "REPORTS", tmp_path / "reports")


def write(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_gates_catch_leaks(tmp_path):
    base = "սա շատ երկար հոդված է որը կրկնվում է " * 8
    train = [{"id": f"t{i}", "text": base + f"հատուկ վերջաբան համար {i} " * 5} for i in range(50)]
    leak_exact = dict(train[3])
    leak_exact["id"] = "v0"
    near_text = train[5]["text"].replace("հատուկ", "յուրահատուկ", 2)  # near-dup
    val = [leak_exact, {"id": "v1", "text": near_text},
           {"id": "v2", "text": "բոլորովին այլ նյութ այստեղ " * 20}]
    paths = {"train": tmp_path / "train.jsonl", "val": tmp_path / "val.jsonl"}
    write(paths["train"], train)
    write(paths["val"], val)
    ex = verify_exact(paths)
    assert ex[("train", "val")] == 1
    near = verify_near(paths, threshold=0.75)
    assert near["val"]["near_dups"] >= 1
    with pytest.raises(SystemExit):
        run_all(paths)


def test_clean_splits_pass(tmp_path):
    train = [{"id": f"t{i}", "text": f"հոդված համար {i} " + f"զանազան բովանդակություն {i} " * 20}
             for i in range(30)]
    val = [{"id": "v", "text": "միանգամայն տարբեր տեքստ առանց կրկնության " * 10}]
    paths = {"train": tmp_path / "train.jsonl", "val": tmp_path / "val.jsonl"}
    write(paths["train"], train)
    write(paths["val"], val)
    report = run_all(paths)
    assert report["pass"] is True
