import json

from armweb.decontaminate import build_ngram_index, scan


def test_planted_overlap_detected(tmp_path):
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    phrase = "մեկ երկու երեք չորս հինգ վեց յոթ ութ ինը տաս տասնմեկ տասներկու տասներեք"
    (bench_dir / "flores.txt").write_text(phrase + "\n", encoding="utf-8")
    train = tmp_path / "train.jsonl"
    with open(train, "w", encoding="utf-8") as f:
        for i in range(20):
            text = ("անկապ բառեր այստեղ գրված են շարունակ " * 6) + (phrase if i == 7 else "")
            f.write(json.dumps({"id": str(i), "text": text}, ensure_ascii=False) + "\n")
    idx = build_ngram_index(bench_dir)
    stats = scan(train, tmp_path / "out.jsonl", idx)
    assert stats["dropped"] == 1 and stats["hits"]["flores"] == 1
    kept_ids = {json.loads(l)["id"] for l in open(tmp_path / "out.jsonl", encoding="utf-8")}
    assert "7" not in kept_ids and len(kept_ids) == 19
