from armweb.normalize import release_view, signature_view, clean_document


def test_release_view_is_nfc_and_collapses_newlines():
    s = "Tituló\n\n\n\nbody  \nline"  # combining accent + 4 newlines + trailing spaces
    out = release_view(s)
    assert "Tituló" in out  # NFC composed
    assert "\n\n\n" not in out
    assert "body\nline" in out


def test_signature_view_folds_armenian():
    # U+0587 (och-yiwn ligature) must fold to two letters; Armenian punctuation stripped; digits zeroed
    s = "Երևան 2026։ Ի՞նչ կա"
    out = signature_view(s)
    assert "եւ" in out and "և" not in out
    assert "։" not in out and "՞" not in out
    assert "2026" not in out and "0000" in out


def test_signature_view_case_and_ws():
    assert signature_view("ԱԲԳ   աբգ") == signature_view("աբգ աբգ")


def test_clean_document_rules():
    assert clean_document(None, "body") is None
    assert clean_document("t", "") is None
    assert clean_document("short", "x") is None  # < 100 chars
    long_body = "մեծ տեքստ " * 30
    out = clean_document("Վերնագիր", long_body)
    assert out is not None
    assert out.startswith("Վերնագիր\n") and len(out) >= 100
