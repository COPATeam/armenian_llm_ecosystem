"""bpb panel eval for HF checkpoints (CPT arms + base), tokenizer-independent."""
import os
import json, math, os, sys
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ER = os.environ["ARMWEB_ROOT"]
BASE = f"{ER}/hf_cache/hub/models--google--gemma-4-E2B/snapshots/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
MODELS = {
    "base-e2b": BASE,
    "cpt-stock": f"{ER}/train/ckpt/cpt_e2b-stock4/checkpoint-1910",
    "cpt-ext8k": f"{ER}/train/ckpt/cpt_e2b-ext8k/checkpoint-1910",
    "cpt-ext16k": f"{ER}/train/ckpt/cpt_e2b-ext16k/checkpoint-1910",
}
PANELS = {
    "test_iid": f"{ER}/data_train/jsonl/test_iid.jsonl",
    "test_tail": f"{ER}/data_train/jsonl/test_tail.jsonl",
    "fineweb2_test": f"{ER}/data_eval/fineweb2_hy_test.jsonl",
    "hywiki": f"{ER}/data_eval/hywiki.jsonl",
    "lrsum": f"{ER}/data_eval/lrsum_hy.jsonl",
    "flores": f"{ER}/data_eval/flores200_hy.jsonl",
}
MAX_DOCS = 1500
SEQ = 4096
results = {}
for name, path in MODELS.items():
    tok_path = path if os.path.exists(os.path.join(path, "tokenizer_config.json")) else \
        (f"{ER}/models/e2b_ext8k" if "ext8k" in name else f"{ER}/models/e2b_ext16k" if "ext16k" in name else BASE)
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    results[name] = {}
    for pname, ppath in PANELS.items():
        nll_nats, nbytes = 0.0, 0
        with open(ppath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= MAX_DOCS:
                    break
                text = json.loads(line)["text"]
                nbytes += len(text.encode())
                ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ).input_ids.cuda()
                if ids.shape[1] < 2:
                    continue
                with torch.no_grad():
                    out = model(ids, labels=ids)
                nll_nats += out.loss.item() * (ids.shape[1] - 1)
        bpb = nll_nats / math.log(2) / max(nbytes, 1)
        results[name][pname] = round(bpb, 4)
        print(f"{name} {pname} bpb={bpb:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
means = {m: round(sum(v.values()) / len(v), 4) for m, v in results.items()}
print(json.dumps({"panels": results, "means": means}, indent=2), flush=True)
with open(f"{ER}/train/cpt_ablation_bpb.json", "w") as f:
    json.dump({"panels": results, "means": means}, f, indent=2)
print("HF_BPB_DONE", flush=True)
