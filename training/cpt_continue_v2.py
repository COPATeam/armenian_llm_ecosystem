"""Plain-HF CPT trainer: Gemma-4 on the ArmWeb 75/20/5 mix (streaming, packed)."""
import os
import sys

import torch
from datasets import interleave_datasets, load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

ER = os.environ["ARMWEB_ROOT"]
MODEL = sys.argv[1] if len(sys.argv) > 1 else f"{ER}/hf_cache/hub/models--google--gemma-4-E2B/snapshots/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{ER}/train/ckpt/cpt_e2b_stock"
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1910
SEQ = 4096

tok = AutoTokenizer.from_pretrained(MODEL)

def stream(path):
    return load_dataset("json", data_files=path, split="train", streaming=True)

mix = interleave_datasets(
    [stream(f"{ER}/data_train/cpt/jsonl/hy_clean.jsonl"),
     stream(f"{ER}/data_train/cpt/jsonl/stem_hy.jsonl"),
     stream(f"{ER}/data_train/cpt/jsonl/stem_en.jsonl"),
     stream(f"{ER}/data_train/cpt/jsonl/fineweb_edu.jsonl"),
     stream(f"{ER}/data_train/cpt/jsonl/stack_smol.jsonl")],
    probabilities=[0.69, 0.04, 0.02, 0.20, 0.05], seed=42, stopping_strategy="all_exhausted")

def tok_fn(batch):
    return tok(batch["text"])

def pack(batch):
    ids = []
    for x in batch["input_ids"]:
        ids.extend(x + [tok.eos_token_id])
    n = (len(ids) // SEQ) * SEQ
    chunks = [ids[i:i + SEQ] for i in range(0, n, SEQ)]
    return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

SKIP_DOCS = int(os.environ.get("SKIP_DOCS", "0"))
if SKIP_DOCS:
    mix = mix.skip(SKIP_DOCS)
    print(f"skipped {SKIP_DOCS} mix docs", flush=True)
ds = (mix.map(tok_fn, batched=True, remove_columns=["text"])
         .map(pack, batched=True, batch_size=64, remove_columns=["input_ids", "attention_mask"])
         .shuffle(seed=42, buffer_size=200))

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="sdpa")
model.gradient_checkpointing_enable()

args = TrainingArguments(
    output_dir=OUT, max_steps=MAX_STEPS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "8")),
    learning_rate=float(os.environ.get("CONT_LR", "1e-4")),
    lr_scheduler_type="cosine", warmup_steps=0,
    weight_decay=0.1, adam_beta2=0.95, bf16=True,
    logging_steps=25, save_steps=500, save_total_limit=3,
    dataloader_num_workers=4, max_grad_norm=1.0,
    report_to=(["wandb"] if os.environ.get("WANDB_API_KEY") else []),
    run_name=os.environ.get("WANDB_RUN_NAME", os.path.basename(OUT)),
    fsdp="full_shard auto_wrap",
    fsdp_config={"min_num_params": 100_000_000, "activation_checkpointing": False},
)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
trainer.train()
trainer.save_model(f"{OUT}/final")
print("CPT_TRAIN_COMPLETE", flush=True)
