import json

from armweb.bson_extract import extract


def test_extract_keeps_metadata_drops_author(tiny_bson, tmp_path):
    out = tmp_path / "out.jsonl"
    stats = extract(tiny_bson, out)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert stats["read"] == 4 and stats["kept"] == 2 and len(rows) == 2
    r1 = rows[0]
    assert r1["id"] == "1" and r1["source"] == "armenpress.am"
    assert r1["post_date"].startswith("2025-05-01")
    assert r1["url"] == "https://armenpress.am/arm/news/1?utm_source=fb"
    assert r1["topic"] == "Քաղաքական"
    assert "Author" not in r1 and "author" not in r1
    assert r1["text"].startswith("Վերնագիր մեկ\n")
    assert rows[1]["post_date"] is None and rows[1]["url"] is None
