#!/bin/bash
#SBATCH --job-name=armweb-smoke70m
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=${ARMWEB_ROOT}/logs/smoke70m_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/megatron-container.sqsh
CKPT=$ER/train/ckpt/smoke70m
mkdir -p $CKPT
srun -N1 --container-image="$IMG" \
     --container-mounts=/path/to/shared-fs:/path/to/shared-fs \
     bash -c "
set -euo pipefail
cd /opt/Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --num-layers 12 --hidden-size 512 --num-attention-heads 8 \
  --ffn-hidden-size 1408 --swiglu --normalization RMSNorm \
  --position-embedding-type rope --rotary-base 10000 \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 16 --global-batch-size 256 \
  --lr 6e-4 --min-lr 6e-5 --lr-decay-style cosine \
  --lr-warmup-iters 100 --train-iters 600 --clip-grad 1.0 \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --init-method-std 0.02 --bf16 \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model $ER/tokenizers/armweb_hy_32k.model \
  --data-path $ER/data_train/megatron/armweb_train_text_document \
  --split 998,1,1 \
  --save $CKPT --load $CKPT --save-interval 200 \
  --log-interval 20 --eval-interval 300 --eval-iters 10 \
  --attention-softmax-in-fp32 \
  --seed 1337
"
echo SMOKE_TRAIN_DONE
