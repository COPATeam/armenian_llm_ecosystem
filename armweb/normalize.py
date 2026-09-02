"""Text normalization: NFC release view + aggressive NFKC signature view.

NOTE: signature_view is intentionally duplicated in scripts/curator_dedup.py
(cluster-side jobs may run without this package installed). Keep the two in sync.
"""
import re
import unicodedata

_TRAIL = re.compile(r"[ \t]+$", re.MULTILINE)
_NL3 = re.compile(r"\n{3,}")
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def release_view(text: str) -> str:
    """NFC-normalized text as released: trailing spaces stripped, 3+ newlines -> 2."""
    text = unicodedata.normalize("NFC", text)
    text = _TRAIL.sub("", text)
    return _NL3.sub("\n\n", text)


def signature_view(text: str) -> str:
    """Aggressively normalized view used ONLY for dedup signatures."""
    text = text.replace("և", "եւ")  # և -> եւ (explicit; NFKC may not fold it)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    text = _DIGIT.sub("0", text)
    return _WS.sub(" ", text).strip()


def clean_document(title, body, min_len: int = 100):
    """Title + newline + body, cleaned; None if unusable (same rules as armweb-golf)."""
    if not title or not isinstance(title, str) or not title.strip():
        return None
    if not body or not isinstance(body, str) or not body.strip():
        return None
    combined = release_view(title.strip() + "\n" + body.strip())
    return combined if len(combined) >= min_len else None
