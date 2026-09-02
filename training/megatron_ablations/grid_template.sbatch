#!/bin/bash
# ArmWeb 410M ablation-grid run. Parameterized via --export:
#   VARIANT, SEED, DATA_ARGS, ROOT, IMG, MOUNT
#SBATCH --nodes=1
#SBATCH --time=03:50:00
set -euo pipefail
CKPT=$ROOT/train/ckpt/grid410m_${VARIANT}_s${SEED}
mkdir -p "$CKPT" "$ROOT/train/logs"
srun -N1 --container-image="$IMG" --container-mounts="$MOUNT:$MOUNT" bash -c "
set -euo pipefail
cd /opt/Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --num-layers 24 --hidden-size 1024 --num-attention-heads 16 \
  --ffn-hidden-size 2816 --swiglu --normalization RMSNorm \
  --position-embedding-type rope --rotary-base 10000 \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 16 --global-batch-size 512 \
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine \
  --lr-warmup-iters 220 --train-iters 4400 --clip-grad 1.0 \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --init-method-std 0.02 --bf16 \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model $ROOT/tokenizers/armweb_hy_32k.model \
  $DATA_ARGS \
  --split 998,1,1 \
  --save $CKPT --load $CKPT --save-interval 1100 \
  --log-interval 50 --eval-interval 1100 --eval-iters 20 \
  --attention-softmax-in-fp32 \
  --seed $SEED
"
echo GRID_RUN_DONE variant=$VARIANT seed=$SEED
