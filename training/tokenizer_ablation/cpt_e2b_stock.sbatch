#!/bin/bash
#SBATCH --job-name=armweb-cpt.e2b.stock
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00
#SBATCH --output=${ARMWEB_ROOT}/logs/cpt_e2b_stock_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/pytorch-cuda-container.sqsh
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR
srun --ntasks-per-node=1 --container-image="$IMG" \
     --container-mounts=/path/to/shared-fs:/path/to/shared-fs \
     bash -c "
set -euo pipefail
export PIP_CACHE_DIR=$ER/.pipcache TMPDIR=$ER/.tmp HF_HOME=$ER/hf_cache
( /opt/venv/bin/python -m pip --version >/dev/null 2>&1 || /opt/venv/bin/python -m ensurepip -q --upgrade >/dev/null 2>&1 || true ) && ( /opt/venv/bin/python -m pip -q install -U transformers tokenizers 2>&1 | tail -1 || uv pip install -q -p /opt/venv/bin/python -U transformers tokenizers 2>&1 | tail -1 )
cd /opt/Automodel
torchrun --nnodes=$SLURM_NNODES --nproc-per-node=8 \
  --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29511 \
  examples/llm_pretrain/pretrain.py --config $ER/jobs/cpt_e2b_stock.yaml
"
echo CPT_E2B_STOCK_DONE
