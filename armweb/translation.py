"""Translate-and-verify pipeline core: placeholder masking + gates.

Masking (LaTeXTrans-style): numbers, LaTeX spans and code spans are replaced by
⟦N1⟧/⟦EQ1⟧ tokens before translation and restored after — kills the ~10%
number/LaTeX corruption mode and makes gate G0 (integrity) deterministic.
"""
import re

NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
EQ_RE = re.compile(r"(\$\$.*?\$\$|\$(?![\d\s])[^$\n]+?(?<!\s)\$|\\\[.*?\\\]|\\\(.*?\\\))", re.S)
PH_RE = re.compile(r"⟦(?:N|EQ)\d+⟧")


def mask(text: str):
    """Replace LaTeX spans then numbers with placeholder tokens; return (masked, mapping)."""
    mapping = {}

    def sub_eq(m):
        tok = f"⟦EQ{len(mapping) + 1}⟧"
        mapping[tok] = m.group(0)
        return tok

    masked = EQ_RE.sub(sub_eq, text)

    def sub_num(m):
        tok = f"⟦N{len(mapping) + 1}⟧"
        mapping[tok] = m.group(0)
        return tok

    # mask numbers only OUTSIDE existing placeholder tokens
    parts = PH_RE.split(masked)
    tokens = PH_RE.findall(masked)
    out = []
    for i, part in enumerate(parts):
        out.append(NUM_RE.sub(sub_num, part))
        if i < len(tokens):
            out.append(tokens[i])
    masked = "".join(out)
    return masked, mapping


def unmask(text: str, mapping: dict):
    for tok, val in mapping.items():
        text = text.replace(tok, val)
    return text


def gate_integrity(translated_masked: str, mapping: dict) -> tuple[bool, str]:
    """All placeholders present exactly once; no stray digits introduced."""
    for tok in mapping:
        if translated_masked.count(tok) != 1:
            return False, f"placeholder {tok} count != 1"
    stray = NUM_RE.findall(PH_RE.sub("", translated_masked))
    if stray:
        return False, f"stray digits introduced: {stray[:3]}"
    return True, "ok"


def gate_language(text: str, lid_model, min_prob=0.6) -> tuple[bool, str]:
    """Translated prose must be Armenian (placeholders stripped before check)."""
    clean = PH_RE.sub(" ", text)
    labels, probs = lid_model.predict(" ".join(clean.split())[:800], k=1)
    lab = labels[0].replace("__label__", "")
    ok = lab in ("hye_Armn", "hyw_Armn") and probs[0] >= min_prob
    return ok, f"{lab}:{probs[0]:.2f}"


def extract_final_number(s: str):
    """Last number in a string, normalized (commas stripped)."""
    nums = NUM_RE.findall(s.replace("−", "-"))
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip("%").rstrip(".")
