# SETUP — environment preparation

This document describes, step by step, how to prepare (A) a single machine
for the data pipelines and (B) a Slurm GPU cluster for training and
evaluation. See `SYSTEM.md` for what to run once set up.

---

## A. Single machine (corpus pipeline, translation, audits)

The ArmWeb pipeline and the ArmSTEM translation pipeline are CPU-only and
run on a laptop or workstation. Requirements: Python ≥ 3.11, ~32 GB RAM
(the streaming dedup engine needs ~3–4 GB per 6M documents), and disk for
your corpus (raw + intermediate ≈ 4× final corpus size).

### A.1 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pinned constraints that matter (already in `requirements.txt`):

- `numpy<2` — fasttext (GlotLID) wheels are incompatible with NumPy 2.
- `datasketch<2` — the independent dedup audit targets the v1 API.
- `xxhash` — the dedup engine and decontamination hashing.
- `openai`, `fasttext`, `huggingface_hub` — translation runner.
- `datasets`, `pandas`, `pyarrow` — contamination scans and staging.

Run the unit tests to verify the pipeline package:

```bash
python -m pytest tests/ -q
```

### A.2 Directory layout

Choose a data root and export it in every shell (or your `~/.zshrc`):

```bash
export ARMWEB_ROOT=/data/armweb        # any path with enough disk
```

The pipeline stages expect / produce this layout under a *local* data
directory (paths are arguments to each stage; this is the convention used
throughout):

```
$DATA/raw/            raw article database exports (BSON/JSONL)
$DATA/lid/            post-language-ID corpus (arm_hy.jsonl)
$DATA/dedup/          dedup survivors + cluster sample + stats
$DATA/splits/         train / val / test_iid / test_tail (+ decon train)
$DATA/benchmarks/     one .txt per decontamination target (line = item)
$DATA/translated/     ArmSTEM shards (accepted.jsonl / rejected.jsonl / stats)
$DATA/reports/        per-stage JSON/CSV reports (funnel, gates, audits)
```

### A.3 Translation API access

The translate-and-verify runner speaks the OpenAI chat-completions API and
rotates over multiple keys on rate limits:

```bash
export LLM_API_BASE=https://your-endpoint/v1     # any OpenAI-compatible URL
printf '%s\n' KEY1 KEY2 ... > ~/.llm_api_keys.txt
chmod 600 ~/.llm_api_keys.txt
```

Model identifiers (translator, escalation model, blind re-solver, judge
panel) are constants at the top of `scripts/translate/production.py` —
set them to models available on your endpoint. The study used a fast
translator, a stronger escalation model, and an independent reasoning
model as re-solver; any comparable trio works, but re-solver and
translator should be different model families to keep the G2 gate
independent.

### A.4 Hugging Face access

Gemma checkpoints are gated; accept the license on the Hugging Face model
page, then:

```bash
hf auth login              # or: export HF_TOKEN=hf_...
export HF_HOME=$ARMWEB_ROOT/hf_cache   # optional; keeps caches off $HOME
```

Note: if you redirect `HF_HOME`, the token must live at `$HF_HOME/token`.

---

## B. Slurm cluster (training and evaluation)

### B.1 Assumptions

- Slurm with container support (pyxis/enroot `--container-image`), nodes
  with 8× H100-class GPUs, and a shared filesystem visible from all nodes.
- Jobs have wall-clock limits (the templates assume ~4 h): every training
  script supports checkpoint chaining (see `SYSTEM.md` §3).
- Compute nodes can reach the internet for `pip` and Hugging Face
  downloads (or pre-stage caches on the shared filesystem).

### B.2 Site variables

Every `*.sbatch` template contains three placeholders — replace them once,
globally:

```bash
cd armenian_llm_ecosystem
grep -rl "YOUR_SLURM_ACCOUNT" --include="*.sbatch" . | xargs sed -i \
  -e "s/YOUR_SLURM_ACCOUNT/<your account>/" \
  -e "s/YOUR_PARTITION/<your partition>/"
grep -rl "/path/to/" --include="*.sbatch" .   # set container paths by hand
```

Containers used:

| Purpose | Requirement |
|---|---|
| CPT training + lm-eval + ArmBench | any CUDA 12 PyTorch image; jobs `pip install transformers==5.15.1 accelerate lm-eval` at start |
| Megatron ablations (410M grid, ladder) | an image with Megatron-LM and Apex |
| NeMo-Curator cross-check | `docker://nvcr.io#nvidia/nemo-curator:26.02` (pulled directly by pyxis) |

### B.3 Data root on the shared filesystem

```bash
export ARMWEB_ROOT=/shared/<you>/armweb
mkdir -p $ARMWEB_ROOT/{data_train/cpt/jsonl,models,train/ckpt,eval_results,logs,jobs,hf_cache,.tmp,.pipcache}
```

Expected contents before training (see `SYSTEM.md` §2 for producing them):

```
$ARMWEB_ROOT/data_train/cpt/jsonl/
  hy_clean.jsonl        ArmWeb train split, decontaminated ({"text": ...})
  stem_hy.jsonl         ArmSTEM Armenian side  ({"text": question+\n+solution})
  stem_en.jsonl         ArmSTEM English side
  fineweb_edu.jsonl     English replay stream
  stack_smol.jsonl      code stream
$ARMWEB_ROOT/models/e4b_base/   the base Gemma checkpoint (HF format)
```

`data_prep/dl_*.py` scripts download and convert the replay/code streams
and the base model; `data_prep/parquet_to_jsonl.py` flattens parquet
corpora to the `{"text": ...}` format.

### B.4 Optional: Weights & Biases

Training scripts export W&B settings only if `${HOME}/.wandb_key` exists;
absent the file, runs proceed without logging. No other telemetry is used.

### B.5 Sanity smoke test

Before committing to full runs, submit the 70M smoke job:

```bash
cd $ARMWEB_ROOT && sbatch jobs/smoke_70m.sbatch
```

It preprocesses a small corpus sample, trains a 70M Megatron model for a
few hundred steps, and validates the requeue/continuation machinery.
