"""Build vocab-extended Gemma-4-E2B snapshots (+8k / +16k Armenian tokens).

New tokens = highest-scored pieces from armweb_hy_32k SentencePiece vocab that
are absent from Gemma's vocab. New embedding rows are mean-initialized from each
piece's tokenization under the ORIGINAL Gemma tokenizer (SambaLingo-style).
"""
import os
import sys

os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ER = os.environ["ARMWEB_ROOT"]
BASE = f"{ER}/hf_cache/hub/models--google--gemma-4-E2B/snapshots/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
N_ADD = int(sys.argv[1])
OUT = f"{ER}/models/e2b_ext{N_ADD // 1000}k"

tok = AutoTokenizer.from_pretrained(BASE)
vocab = set(tok.get_vocab().keys())

candidates = []
with open(f"{ER}/tokenizers/armweb_hy_32k.vocab", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            continue
        piece, score = parts
        if piece.startswith("<") or len(piece) < 2:
            continue
        if piece in vocab:
            continue
        candidates.append((float(score), piece))
candidates.sort(reverse=True)
new_tokens = [p for _, p in candidates[:N_ADD]]
print(f"adding {len(new_tokens)} tokens (requested {N_ADD}); sample: {new_tokens[:5]}", flush=True)

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
# constituent ids under ORIGINAL tokenizer, before extension
constituents = [tok(p.replace("▁", " "), add_special_tokens=False)["input_ids"] for p in new_tokens]
n_added = tok.add_tokens(new_tokens)
model.resize_token_embeddings(len(tok), mean_resizing=False)

emb_in = model.get_input_embeddings().weight
emb_out = model.get_output_embeddings().weight
tied = emb_out.data_ptr() == emb_in.data_ptr()
with torch.no_grad():
    for i, ids in enumerate(constituents):
        row = len(tok) - n_added + i
        if ids:
            emb_in[row] = emb_in[torch.tensor(ids)].mean(0)
            if not tied:
                emb_out[row] = emb_out[torch.tensor(ids)].mean(0)
print(f"mean-initialized {n_added} rows (tied={tied})", flush=True)

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
t = AutoTokenizer.from_pretrained(OUT)
s = "Երևանում այսօր արև է և տաք եղանակ"
print(f"verify: base={len(AutoTokenizer.from_pretrained(BASE)(s)['input_ids'])} tokens, "
      f"extended={len(t(s)['input_ids'])} tokens", flush=True)
print("EXTEND_DONE", OUT, flush=True)
