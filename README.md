# Armenian LLM Ecosystem — reproduction code

Code for the paper **["From Zero to Hero: An Open LLM Ecosystem for Armenian"](https://arxiv.org/abs/2609.03350)**
(arXiv:2609.03350). This repository contains everything needed to
reproduce the study: the ArmWeb corpus pipeline, the ArmSTEM
translate-and-verify pipeline, the continued-pretraining (CPT) recipes, and
the evaluation harness configurations.

**Released artifacts** (Hugging Face):
- [`COPA-AI/armweb`](https://huggingface.co/datasets/COPA-AI/armweb) — 4.37M-document Armenian news corpus
- [`COPA-AI/armstem`](https://huggingface.co/datasets/COPA-AI/armstem) — 373K verified parallel EN–HY STEM problems (324K with step-by-step solutions)
- [`COPA-AI/arm-gemma-e4b`](https://huggingface.co/COPA-AI/arm-gemma-e4b) — Armenian-adapted Gemma-4-E4B

**Start here**: [SETUP.md](SETUP.md) (environment preparation, machine and
cluster) and [SYSTEM.md](SYSTEM.md) (architecture and stage-by-stage run
book). The sections below are a condensed overview.

## Repository layout

```
SETUP.md         Environment preparation (single machine + Slurm cluster)
SYSTEM.md        Architecture and rigorous stage-by-stage run book
armweb/          Corpus pipeline package (extraction, LID, dedup, splits,
                 decontamination, verification, release building)
scripts/         Standalone pipeline drivers
  minhash_dedup_stream.py   Streaming MinHash dedup engine (word 5-grams,
                            112 perms, 14x8 bands, seed 42)
  audit_datatrove.py        Independent datasketch dedup audit
  translate/production.py   ArmSTEM translate-and-verify runner (gates G0-G2)
  translate/adequacy_audit.py  Automated adequacy audit
data_prep/       Source staging: baseline corpora, CPT mixture streams,
                 STEM shard building, parquet->jsonl, LID reconstruction
training/        CPT trainers and Slurm templates (HF Trainer, 128 GPUs)
  megatron_ablations/   410M grid, 70M-1B ladder, 1.3B confirmation,
                        Megatron preprocessing, eval-panel prep, smoke test
  tokenizer_ablation/   Vocab extension (+8k/+16k) and E2B CPT arms
evaluation/      Eval jobs: likelihood suite (lm-eval), ArmBench (lighteval
                 fork), contamination scans, NeMo-Curator cross-check
  bpb/           Bits-per-byte panel eval (Megatron and HF harnesses)
tests/           Unit tests for the pipeline package
```

## Environment variables

All site-specific values are parameterized. Set these before running:

| Variable | Meaning |
|---|---|
| `ARMWEB_ROOT` | Root directory for data, checkpoints, and logs on your filesystem |
| `LLM_API_BASE` | OpenAI-compatible endpoint for translation/judging (any provider) |
| `~/.llm_api_keys.txt` | One API key per line (rotated on rate limits); `chmod 600` |
| `HF_TOKEN` / `HF_HOME` | Hugging Face access for gated models (Gemma) |

Slurm templates additionally contain `YOUR_SLURM_ACCOUNT`, `YOUR_PARTITION`,
and `/path/to/*.sqsh` container placeholders — replace with your site's
values. Any CUDA 12 PyTorch container with `transformers>=5` works for
training/eval; the NeMo-Curator cross-check uses NVIDIA's official
`nvcr.io/nvidia/nemo-curator` image.

## Reproducing the study

### 1. ArmWeb corpus pipeline

The pipeline runs in stages over a raw article database (we cannot
redistribute the raw crawl; the released corpus is the pipeline output):

```
extraction (armweb/bson_extract.py)          streaming, corruption-resilient
language ID (armweb/lid.py)                  GlotLID, keep hye/hyw
exact dedup + MinHash (scripts/minhash_dedup_stream.py)
paragraph dedup (armweb/paragraph_dedup.py)  Dolma-style
splits (armweb/splits.py)                    outlet x month strata + temporal tail
decontamination (armweb/decontaminate.py)    13-gram vs ten Armenian eval sets
verification gates (armweb/verify.py)        leakage gates, asserted
release build (armweb/build_release.py)
```

Key design constants (paper §2.1): word 5-gram shingles on an NFKC
"signature view" (Armenian ligature folding, punctuation stripping,
digit-zeroing), 112 permutations, 14 bands × 8 rows (Jaccard ≈ 0.72),
seed 42, global dedup **before** splitting, keep-longest.

Validation: `scripts/audit_datatrove.py` (datasketch agreement) and
`evaluation/curator_probe.py` + `curator_probe2.sbatch` (NeMo-Curator
cross-check at matched geometry; paper Appendix).

### 2. ArmSTEM translate-and-verify

```
python scripts/translate/production.py <shard.jsonl> <out_dir> [workers]
```

Input rows: `{"id", "src", "question", "cot", "gold", "answer_type"}`.
Stages per item: English-side 13-gram decontamination → placeholder masking
(numbers, LaTeX, separator) → translation with two feedback repair rounds +
escalation → G0 placeholder integrity → G1 GlotLID → G2 blind re-solving
with an English-control arbitration arm → Armenian-side decontamination
against ArmBench. The runner is append-only and resumable. Models are set
at the top of the file; any OpenAI-compatible endpoint works.

Audits: `scripts/translate/adequacy_audit.py` (stratified automated audit;
the paper adds a two-annotator native-speaker audit).

### 3. Continued pretraining

`training/cpt_train_v2.py` is the released-model recipe: Gemma-4-E4B,
10B tokens, sequence length 4096, global batch 512 sequences, cosine LR
3e-5 with 100 warmup steps, streaming five-way mixture
`[ArmWeb 0.69, ArmSTEM-HY 0.04, ArmSTEM-EN 0.02, FineWeb-Edu 0.20,
Stack-smol 0.05]`, seed 42. `cpt_train.py` is the news-only control
(75/20/5) and `cpt_train_v3a.py` the full-corpus control (STEM-CPT-full in the paper). On clusters with
wall-clock limits, chain segments with `cpt_continue_*.py` (pass the last
checkpoint, remaining steps, the cosine-schedule LR at the boundary via
`CONT_LR`, and consumed documents via `SKIP_DOCS`; see the paper's
reproducibility appendix for the exact arithmetic). Each 10B-token run
takes ~10 h on 128 H100s.

### 4. Evaluation

- **Likelihood suite** (`evaluation/final_eval_baselines.sbatch`):
  lm-eval, zero-shot `acc` on belebele_hye_Armn, m_mmlu_hy, hellaswag_hy,
  multiblimp_hye, include_base_44_armenian, arc_hy.
- **ArmBench** (`evaluation/armbench_eval12.sbatch`, `ab_fanout.sbatch`):
  the ArmBench-LLM lighteval fork (github.com/Metricam/ArmBench-LLM),
  task ids `armenian:<name>|0`. Pin `batch_size=1` in the model args for
  single-process tasks; some judge-metric tasks require long walls — the
  fanout template splits them across jobs.
- **Contamination scans** (`evaluation/fw2_scan.py`,
  `mmlu_arc_scan.py`, `decon_baselines.sbatch`): 13-gram scans of
  corpora against evaluation sets, with per-benchmark attribution.
- **ArmenianGPT baseline** (`evaluation/extract_armgpt.py`): extracts the
  text backbone from the Mistral-3 vision wrapper so standard causal-LM
  harnesses can load it; verify the extraction reproduces the multimodal
  loader's likelihood scores before trusting downstream numbers.

### 5. Small-scale ablations

The 410M grid, repetition study, 70M–1B scaling ladder, and 1.3B
confirmation are Megatron-LM trainings at Chinchilla-optimal budgets
(paper §2.1 and appendices); job templates are in
`training/megatron_ablations/` and the tokenizer-extension study in
`training/tokenizer_ablation/`. Evaluation is bits-per-byte over a fixed
panel (`evaluation/bpb/`), which is tokenizer-independent. See SYSTEM.md
§3–4 for exact submission commands and budgets.

## Citation

```bibtex
@article{arakelyan2026armweb,
  title  = {From Zero to Hero: An Open LLM Ecosystem for Armenian},
  author = {Arakelyan, Erik and Avetisyan, Khatun and Davtyan, Meri and Grigoryan, Heghine and Khachatryan, Nane and Shahsuvaryan, Hayk and Sergoyan, Henrik and Martirosyan, Vahan},
  year   = {2026},
  journal = {arXiv preprint arXiv:2609.03350}
}
```
