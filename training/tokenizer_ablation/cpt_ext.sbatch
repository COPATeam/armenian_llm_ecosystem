#!/bin/bash
#SBATCH --job-name=armweb-cpt.$EXT_TAG
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00
#SBATCH --output=${ARMWEB_ROOT}/logs/cpt_%x_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/pytorch-cuda-container.sqsh
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR
srun --ntasks-per-node=1 --container-image="$IMG" \
     --container-mounts=/path/to/shared-fs:/path/to/shared-fs \
     bash -c "
set -euo pipefail
KF=${HOME}/.wandb_key
[ -f \$KF ] && export WANDB_API_KEY=\$(cat \$KF) WANDB_PROJECT=armweb WANDB_RUN_NAME=cpt-$EXT_TAG GRAD_ACCUM=4 CPT_LR=${CPT_LR:-1e-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PIP_CACHE_DIR=$ER/.pipcache TMPDIR=$ER/.tmp HF_HOME=$ER/hf_cache
VP=/opt/venv/bin/python
\$VP -m ensurepip --upgrade >/dev/null 2>&1 || true
if ! \$VP -m pip --version >/dev/null 2>&1; then
  curl -sS https://bootstrap.pypa.io/get-pip.py | \$VP >/dev/null
fi
\$VP -m pip install -q -U transformers tokenizers datasets accelerate
\$VP -c 'import transformers; from transformers import AutoConfig; AutoConfig.from_pretrained(\"google/gemma-4-E2B\"); print(\"venv transformers\", transformers.__version__, \"gemma4 OK\")'
torchrun --nnodes=$SLURM_NNODES --nproc-per-node=8 \
  --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29512 \
  $ER/jobs/cpt_train.py $ER/models/$EXT_DIR $ER/train/ckpt/cpt_$EXT_TAG 1910
"
echo CPT_JOB_DONE
