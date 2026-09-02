"""Tokenizer fertility on Armenian text: tokens/word, tokens/byte per tokenizer."""
import random

from .io_utils import iter_jsonl

CANDIDATE_HF = [
    # (label, repo) — gated repos record the error instead of failing.
    # CPT-base candidates (verified on HF 2026-08-11): Qwen3.5 9B / Gemma-4 12B
    # for the 8B-class run; Qwen3.5-2B / Gemma-4-E2B for the tokenizer ablation.
    ("llama-3.1", "meta-llama/Llama-3.1-8B"),
    ("qwen3", "Qwen/Qwen3-8B"),
    ("qwen3.5-9b", "Qwen/Qwen3.5-9B"),
    ("gemma-3", "google/gemma-3-12b-pt"),
    ("gemma-4-12b", "google/gemma-4-12B"),
]


def sample_docs(val_path, n=10000, seed=42):
    rng = random.Random(seed)
    docs = [r["text"] for r in iter_jsonl(val_path)]
    return rng.sample(docs, min(n, len(docs)))


def measure_sp(model_path, docs):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return _rates(docs, lambda t: len(sp.encode(t)))


def measure_hf(repo, docs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    return _rates(docs, lambda t: len(tok.encode(t, add_special_tokens=False)))


def _rates(docs, count_fn):
    tokens = sum(count_fn(t) for t in docs)
    words = sum(len(t.split()) for t in docs)
    byts = sum(len(t.encode("utf-8")) for t in docs)
    return {"tokens": tokens, "tokens_per_word": round(tokens / words, 4),
            "tokens_per_byte": round(tokens / byts, 4),
            "bytes_per_token": round(byts / tokens, 3)}


def fertility_table(val_path, sp_models: dict, out_csv) -> dict:
    docs = sample_docs(val_path)
    results = {}
    for label, path in sp_models.items():
        try:
            results[label] = measure_sp(path, docs)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    for label, repo in CANDIDATE_HF:
        try:
            results[label] = measure_hf(repo, docs)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("tokenizer,tokens_per_word,tokens_per_byte,bytes_per_token,note\n")
        for label, r in results.items():
            if "error" in r:
                f.write(f"{label},,,,{r['error']}\n")
            else:
                f.write(f"{label},{r['tokens_per_word']},{r['tokens_per_byte']},"
                        f"{r['bytes_per_token']},\n")
    return results
