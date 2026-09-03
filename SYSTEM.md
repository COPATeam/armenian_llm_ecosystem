# SYSTEM — architecture and run book

This document explains how the pieces of the study fit together and gives
the exact commands to run every stage. `SETUP.md` covers environment
preparation; this file assumes it is done. Section numbers below follow
the order of the study, which is also the dependency order.

```
                    ┌─────────────────────────────────────────────┐
 raw Armenian crawl │ 1. ArmWeb pipeline (CPU)                    │──► ArmWeb corpus
                    │    extract → LID → dedup → splits → decon   │    (parquet splits)
                    └─────────────────────────────────────────────┘
 EN STEM sources    ┌─────────────────────────────────────────────┐
 (GSM8K, AceReason, │ 2. ArmSTEM pipeline (CPU + LLM API)         │──► ArmSTEM pairs
  OpenScience, …)   │    stage → decon → translate+verify (G0–G2) │    (EN–HY jsonl)
                    └─────────────────────────────────────────────┘
                    ┌─────────────────────────────────────────────┐
 ArmWeb splits      │ 3. Small-scale ablations (Megatron, 8 GPUs) │──► corpus quality
                    │    410M grid, ladder 70M–1B, 1.3B confirm   │    evidence (bpb)
                    └─────────────────────────────────────────────┘
                    ┌─────────────────────────────────────────────┐
                    │ 4. Tokenizer ablation (16–32 GPUs)          │──► stock vs +8k/+16k
                    └─────────────────────────────────────────────┘
 ArmWeb + ArmSTEM   ┌─────────────────────────────────────────────┐
 + FineWeb-Edu      │ 5. CPT of Gemma-4-E4B (128 GPUs, chained)   │──► arm-gemma-e4b
 + Stack-smol       └─────────────────────────────────────────────┘
                    ┌─────────────────────────────────────────────┐
                    │ 6. Evaluation (lm-eval, ArmBench, scans)    │──► paper tables
                    └─────────────────────────────────────────────┘
```

Determinism notes that apply everywhere: dedup and splits use seed 42 and
fixed hash functions (xxhash, fixed MinHash permutations), so pipeline
outputs are bit-reproducible; Megatron runs pin `--seed`; the CPT recipe
pins seed 42 and a fixed interleave order. The only non-deterministic
stage is translation (LLM sampling); its *verification* gates are
deterministic given the outputs, and the released ArmSTEM file is the
frozen artifact of record.

---

## 1. ArmWeb corpus pipeline (single machine, CPU)

Stages are separate scripts so each can be inspected and re-run; every
stage writes a JSON report into `$DATA/reports/`.

```bash
export DATA=/data/armweb_pipeline

# 1) Extraction from raw article DB exports (streaming, corruption-tolerant)
python -m armweb.bson_extract  $DATA/raw  $DATA/extracted.jsonl

# 2) Language ID: GlotLID, keep hye/hyw above threshold
python -m armweb.lid  $DATA/extracted.jsonl  $DATA/lid/arm_hy.jsonl

# 3) Exact + fuzzy dedup (the pipeline's core; global, BEFORE splitting)
python scripts/minhash_dedup_stream.py  $DATA/lid/arm_hy.jsonl  $DATA/dedup/
#    word 5-gram shingles on the NFKC signature view, 112 permutations,
#    14 bands x 8 rows (Jaccard ~ 0.72), seed 42, keep-longest per cluster

# 4) Paragraph-level dedup (Dolma-style boilerplate removal)
python -m armweb.paragraph_dedup  $DATA/dedup/survivors.jsonl  $DATA/dedup/para.jsonl

# 5) Splits: outlet x month stratified train/val/test_iid + temporal test_tail
python -m armweb.splits  $DATA/dedup/para.jsonl  $DATA/splits/

# 6) Decontamination: 13-gram overlap vs every benchmark in $DATA/benchmarks/
python -m armweb.decontaminate  $DATA/splits/train.jsonl  $DATA/benchmarks/  \
       $DATA/splits/train_decon.jsonl

# 7) Verification gates (hard asserts: zero cross-split exact dups,
#    near-dup rate <= 0.05%, zero post-decon benchmark 13-grams)
python -m armweb.verify  $DATA/splits/

# 8) Release build (parquet + stats)
python -m armweb.build_release  $DATA/splits/  $DATA/release/
```

Dedup validation (both reported in the paper appendix):

```bash
# Independent reimplementation agreement on a 1% sample (datasketch)
python scripts/audit_datatrove.py  $DATA/lid/arm_hy.jsonl  $DATA/dedup/

# NeMo-Curator cross-check at matched geometry (GPU node; official container)
sbatch evaluation/curator_probe2.sbatch     # runs evaluation/curator_probe.py
```

For the full-corpus Curator run, pass parquet input and, on large corpora,
set `UCX_TLS=tcp,cuda_copy,sm` and `input_blocksize=256MiB` (both already
in the template; without them Ray's UCX transport times out).

Approximate cost: the full pipeline over ~6M raw documents runs in a few
hours on a modern workstation; dedup is the longest stage.

## 2. ArmSTEM staging and translation (single machine + LLM API)

Source staging (downloads, sampling, schema normalization):

```bash
python data_prep/stage_science.py    # OpenScience 6M -> 400K sample, OSR2 -> 150K
python data_prep/stage_os2.py        # second OpenScience tranche
python data_prep/build_shard2.py     # science shard -> production schema (.zst)
# math shard 1 (GSM8K, AceReason-Math) follows the same schema:
# {"id", "src", "question", "cot", "gold", "answer_type"}
```

Translation with verification gates:

```bash
python scripts/translate/production.py  shard.jsonl  $DATA/translated/shard1  16
```

Per item: English-side 13-gram decontamination → placeholder masking
(numbers, LaTeX, separators) → translation with up to two feedback repair
rounds and one escalation to the stronger model → **G0** placeholder
integrity → **G1** GlotLID language check → **G2** blind re-solving of the
Armenian problem by an independent model, with an English-control
arbitration arm that distinguishes translation damage from solver
weakness → Armenian-side decontamination against ArmBench. Output is
append-only (`accepted.jsonl` / `rejected.jsonl` / `stats.json`) and the
runner is resumable — re-invoking with the same arguments skips finished
ids, so it tolerates being killed at any point.

Audits after a shard completes:

```bash
python scripts/translate/adequacy_audit.py  $DATA/translated/shard1/accepted.jsonl
```

Cost calibration from the study: 372,907 accepted pairs consumed roughly
low-single-digit millions of LLM calls end to end; acceptance rates were
91.5–99.1% depending on source, and G2 failures were solver-limited
(10.9% math, 27.6% science) rather than translation-limited.

## 3. Small-scale Megatron ablations (1 node × 8 GPUs per run)

These establish corpus quality *before* spending the CPT budget. All use
the 32K Armenian SentencePiece tokenizer trained on ArmWeb
(`$ARMWEB_ROOT/tokenizers/armweb_hy_32k.model`) and bits-per-byte (bpb)
on a fixed panel, which is tokenizer-independent.

### 3.1 Preprocess corpora to Megatron format

```bash
sbatch training/megatron_ablations/preprocess_megatron.sbatch  # ArmWeb splits
sbatch training/megatron_ablations/prep_variants.sbatch        # baselines: FineWeb-2, HPLT, CulturaX (decontaminated)
sbatch training/megatron_ablations/prep_panel.sbatch           # eval panel: fw2-test, hywiki, LRSum, FLORES
```

`prep_variants` expects the decontaminated baseline corpora produced by
`data_prep/dl_baselines.py` + `evaluation/decon_baselines.sbatch` (decon
the baselines with the SAME 13-gram procedure, or the comparison is
unfair in ArmWeb's favor).

### 3.2 The 410M grid

One job per (corpus variant, seed). Chinchilla-ish budget: 4,400 iters ×
512 × 2048 tokens ≈ 4.6B tokens, ~3.5 h on 8× H100.

```bash
cd $ARMWEB_ROOT
sbatch --export=ALL,VARIANT=armweb,SEED=42,ROOT=$ARMWEB_ROOT,IMG=/path/to/megatron-container.sqsh,MOUNT=/shared,DATA_ARGS="--train-data-path 1.0 $ARMWEB_ROOT/data_train/megatron/armweb_train_text_document" \
  training/megatron_ablations/grid_template.sbatch
# repeat with VARIANT=fineweb2 / culturax / hplt / union (blend via weighted
# --train-data-path "0.5 pathA 0.5 pathB") and additional seeds
```

### 3.3 Scaling ladder and repetition curve

```bash
# ladder: SIZE in {70m,160m,410m,1b}; ITERS chosen for ~20 tokens/param
sbatch --export=ALL,SIZE=410m,ITERS=4400,SEED=42,DATA_ARGS="..." \
  training/megatron_ablations/ladder.sbatch
```

The ladder script self-resumes from its checkpoint directory
(`--save`/`--load` point at the same path and
`--exit-duration-in-mins 225` bounds each window), so for runs longer
than one wall-clock window simply submit the same command again with
`--dependency=afterany:<jobid>`.

### 3.4 1.3B confirmation

```bash
sbatch --export=ALL,VARIANT=union,ROOT=...,IMG=...,MOUNT=...,DATA_ARGS="..." \
  training/megatron_ablations/confirm_1p3b.sbatch
```

Multi-window by design: submit a chain of identical jobs; Megatron
resumes from the latest checkpoint and trailing jobs exit immediately
once `--train-iters` is reached.

### 3.5 bpb evaluation of any Megatron checkpoint

```bash
sbatch --export=ALL,CKPT=$ARMWEB_ROOT/train/ckpt/grid410m_armweb_s42,TAG=armweb_s42,SIZE=410m,ROOT=...,IMG=...,MOUNT=... \
  evaluation/bpb/eval_bpb.sbatch
```

Reports token-level loss per panel; convert to bpb with
loss × (token_count / text_bytes) / ln 2 using the counts in
`data_eval/panel_manifest.json`. Accuracy tasks at this scale are
near-chance; MultiBLiMP is the only accuracy task with signal at 70M.

## 4. Tokenizer ablation (Gemma-4-E2B)

Question answered: does extending Gemma's vocabulary with Armenian pieces
help at a 10B-token CPT budget? (Study answer: no — sharply harmful.)

```bash
# Build +8k and +16k extended snapshots (SambaLingo-style mean-init rows)
python training/tokenizer_ablation/extend_tokenizer.py 8000
python training/tokenizer_ablation/extend_tokenizer.py 16000
# note: extend_tokenizer.py resolves the base snapshot inside $ARMWEB_ROOT/hf_cache;
# run data_prep/dl_gemma.py first and update the snapshot hash in the script.

# Stock-tokenizer arm (2 nodes; config in cpt_e2b_stock.yaml)
sbatch training/tokenizer_ablation/cpt_e2b_stock.sbatch

# Extended arms (4 nodes; EXT_TAG names the run, EXT_DIR the model dir)
sbatch --export=ALL,EXT_TAG=e2b-ext8k,EXT_DIR=e2b_ext8k,CPT_LR=1e-4 \
  training/tokenizer_ablation/cpt_ext.sbatch

# Compare with the tokenizer-independent HF bpb harness
python evaluation/bpb/hf_bpb.py     # edit MODELS dict to your checkpoint paths
```

`hf_bpb.py` computes bits-per-byte directly from summed log-likelihoods
over the raw text panels, so models with different vocabularies are
directly comparable.

## 5. Continued pretraining of Gemma-4-E4B (16 nodes × 8 GPUs)

The headline recipe (`training/cpt_train_v2.py`): 10B tokens, sequences
packed at 4096, global batch 512 sequences, LR 3e-5 cosine with 100
warmup steps, seed 42, streaming five-way mixture
`[ArmWeb 0.69, ArmSTEM-HY 0.04, ArmSTEM-EN 0.02, FineWeb-Edu 0.20,
Stack-smol 0.05]`. Controls: `cpt_train.py` (news-only 75/20/5, run at
LR 1e-4 and 3e-5) and `cpt_train_v3a.py` (identical recipe, STEM share drawn from the
full ArmSTEM corpus; STEM-CPT-full in the paper). LR is overridable via the `CPT_LR` environment variable.

```bash
# Stage inputs (login node)
python data_prep/dl_cpt_mix.py      # FineWeb-Edu sample + Stack-smol
python data_prep/dl_gemma.py        # base checkpoints
python data_prep/parquet_to_jsonl.py $ARMWEB_ROOT/release $ARMWEB_ROOT/data_train/cpt/jsonl

# Launch (16-node template)
sbatch training/cpt_e4b.sbatch
```

**Wall-clock chaining.** A 10B-token run is ~10 h on 128 H100s; with a
4 h limit that is three windows. Each window past the first uses the
matching `cpt_continue_*.py` with three values you must compute:

1. the last checkpoint path of the previous window,
2. `CONT_LR` — the cosine schedule's value at the boundary step
   (`lr = min_lr + 0.5*(peak-min_lr)*(1+cos(pi*(step-warmup)/(total-warmup)))`),
3. `SKIP_DOCS` — documents already consumed, ≈ 2.6M per 1000 steps at
   this batch geometry (the exact count is printed in the previous
   window's log; use it).

```bash
sbatch --export=ALL,CKPT=.../checkpoint-1500,CONT_LR=1.87e-5,SKIP_DOCS=3900000 \
  training/cpt_e4b_cont.sbatch
```

**Packaging.** After the final step, assemble the release model: copy the
checkpoint weights, set `tie_word_embeddings: false` in `config.json`
(the trainer saves untied weights), and copy the base tokenizer files.
Sequence length 4096 during CPT does **not** change the architectural
context window (`max_position_embeddings` stays 131072).

## 6. Evaluation

### 6.1 Likelihood suite (lm-eval)

```bash
sbatch evaluation/final_eval_baselines.sbatch
```

Zero-shot `acc` on belebele_hye_Armn, m_mmlu_hy, hellaswag_hy,
multiblimp_hye, include_base_44_armenian, arc_hy. The script resolves
task availability at runtime (task names shift between lm-eval versions)
and evaluates every model listed in its loop. ~1–2 h per E4B-size model
on one node.

### 6.2 ArmBench (generative)

```bash
sbatch evaluation/armbench_eval12.sbatch     # main 12-task pass
sbatch evaluation/ab_fanout.sbatch           # long judge-metric tasks, split across jobs
```

Uses the ArmBench-LLM lighteval fork (github.com/Metricam/ArmBench-LLM),
task ids `armenian:<name>|0`. Two hard-won constraints: pass
`batch_size=1` inside `--model_args` (the fork has no
`--override-batch-size`), and run judge-metric tasks single-process.
For the ArmenianGPT-1.0-3B baseline, first extract the text backbone:

```bash
python evaluation/extract_armgpt.py
```

then verify the extraction by checking one likelihood task exactly
matches the multimodal loader (the study's check: Belebele 0.6544,
identical) before trusting any downstream number.

### 6.3 Contamination scans

```bash
sbatch evaluation/decon_baselines.sbatch     # 13-gram scan of public baselines vs 10 Armenian eval sets
python evaluation/fw2_scan.py                # FineWeb-2 self-leak scan
python evaluation/mmlu_arc_scan.py           # post-hoc m-MMLU/ARC translated-benchmark scan
```

### 6.4 Curator cross-check

```bash
sbatch evaluation/curator_probe2.sbatch
```

Reruns fuzzy dedup with NeMo-Curator at matched geometry
(num_bands=14, minhashes_per_band=8, char_ngrams=30) and reports removal
overlap and disagreement Jaccard distribution against the release engine.

---

## Compute budget summary

| Stage | Hardware | Wall clock |
|---|---|---|
| ArmWeb pipeline (full) | 1 workstation, CPU | hours |
| Translation (373K accepted pairs) | API-bound | days, resumable |
| 410M grid (per run) | 8× H100 | ~3.5 h |
| Ladder 70M–1B (per run) | 8× H100 | 1–8 h |
| 1.3B confirmation | 8× H100 | ~3 windows × 4 h |
| Tokenizer ablation (per arm) | 16–32× H100 | ~4–8 h |
| E4B CPT, 10B tokens (per arm) | 128× H100 | ~10 h (~1,250 H100-h) |
| Likelihood suite (per model) | 8× H100 | 1–2 h |
| ArmBench (per model) | 8× H100 | 2–8 h (judge tasks long) |

## Order of operations for a full reproduction

1. `SETUP.md` A + B, then the smoke test (B.5).
2. Stage 1 (corpus) and stage 2 (translation) in parallel — they are
   independent until CPT.
3. Stage 3 ablations to validate the corpus (optional for reproducing
   only the released model, required for reproducing the paper).
4. Stage 4 tokenizer ablation (optional; justifies the stock tokenizer).
5. Stage 5 CPT: the v2 mixture arm is the released model; news-only and
   full-corpus (STEM-CPT-full) arms reproduce the factorial table.
6. Stage 6 evaluations over every arm plus the public baselines.
