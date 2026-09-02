import json

from armweb.paragraph_dedup import count_paragraphs, strip_boilerplate

FOOTER = "Հետևեք մեզ սոցիալական ցանցերում ամեն օր ամեն ժամ բոլոր նորությունների համար միշտ անվճար"


WORDS = ["արեւ", "լուսին", "աստղ", "ծով", "լեռ", "անտառ",
         "գետ", "քաղաք", "գյուղ", "ճանապարհ", "երկինք", "հող"]


def make(tmp_path, n=12):
    p = tmp_path / "in.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            # unique content per doc via distinct WORDS (digits would be zeroed
            # by signature_view and collapse all docs into one "paragraph")
            text = (f"Հոդված {WORDS[i]}\n"
                    + (f"բովանդակություն {WORDS[i]} միայն այս հոդվածում " * 10) + "\n\n"
                    + FOOTER)
            f.write(json.dumps({"id": str(i), "text": text}, ensure_ascii=False) + "\n")
    return p


def test_boilerplate_stripped_content_kept(tmp_path):
    inp = make(tmp_path)
    counts = count_paragraphs(inp)
    assert max(counts.values()) == 12  # footer appears in all docs
    out = tmp_path / "out.jsonl"
    stats = strip_boilerplate(inp, out, min_docs=10)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 12 and stats["docs_touched"] == 12
    assert all(FOOTER not in r["text"] for r in rows)
    assert all("բովանդակություն" in r["text"] for r in rows)
