#!/bin/bash
# 1.3B confirmation run window. Env: VARIANT, DATA_ARGS, ROOT, IMG, MOUNT.
# Multi-window: same args every window; Megatron resumes from latest ckpt and
# exits fast once train-iters is reached, so trailing chain jobs are no-ops.
#SBATCH --nodes=1
#SBATCH --time=03:55:00
set -euo pipefail
CKPT=$ROOT/train/ckpt/confirm1p3b_${VARIANT}
mkdir -p "$CKPT"
srun -N1 --container-image="$IMG" --container-mounts="$MOUNT:$MOUNT" bash -c "
set -euo pipefail
cd /opt/Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --num-layers 24 --hidden-size 2048 --num-attention-heads 16 \
  --ffn-hidden-size 5632 --swiglu --normalization RMSNorm \
  --position-embedding-type rope --rotary-base 10000 \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 8 --global-batch-size 1024 \
  --lr 2.5e-4 --min-lr 2.5e-5 --lr-decay-style cosine \
  --lr-warmup-iters 500 --train-iters 13500 --clip-grad 1.0 \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --init-method-std 0.014 --bf16 --use-distributed-optimizer \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model $ROOT/tokenizers/armweb_hy_32k.model \
  $DATA_ARGS \
  --split 998,1,1 \
  --save $CKPT --load $CKPT --save-interval 500 \
  --log-interval 100 --eval-interval 2700 --eval-iters 20 \
  --attention-softmax-in-fp32 \
  --seed 1337
"
echo WINDOW_DONE variant=$VARIANT
