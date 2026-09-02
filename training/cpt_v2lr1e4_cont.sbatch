#!/bin/bash
#SBATCH --job-name=armweb-cpt.e4b-cont2
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00
#SBATCH --output=${ARMWEB_ROOT}/logs/cpt_e4b_cont2_%j.out
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
[ -f \$KF ] && export WANDB_API_KEY=\$(cat \$KF) WANDB_PROJECT=armweb WANDB_RUN_NAME=cpt-e4b-headline-cont2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PIP_CACHE_DIR=$ER/.pipcache TMPDIR=$ER/.tmp HF_HOME=$ER/hf_cache
export GRAD_ACCUM=2 CONT_LR=6.4e-5 SKIP_DOCS=5200000
VP=/opt/venv/bin/python
\$VP -m pip --version >/dev/null 2>&1 || curl -sS https://bootstrap.pypa.io/get-pip.py | \$VP >/dev/null
\$VP -m pip install -q transformers==5.15.1 tokenizers datasets accelerate
torchrun --nnodes=$SLURM_NNODES --nproc-per-node=8 \
  --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29513 \
  $ER/jobs/cpt_continue_v2.py $ER/train/ckpt/cpt_e4b-v2lr1e4/checkpoint-2000 $ER/train/ckpt/cpt_e4b-v2lr1e4-cont 2770
"
echo CPT_E4B_CONT_DONE
