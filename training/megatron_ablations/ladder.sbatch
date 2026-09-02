#!/bin/bash
# Ladder run. Env: SIZE, ITERS, SEED, DATA_ARGS
#SBATCH --nodes=1
#SBATCH --time=03:55:00
set -euo pipefail
ER=${ARMWEB_ROOT}
CKPT=$ER/train/ckpt/ladder_${SIZE}_s${SEED}
mkdir -p "$CKPT"
case "$SIZE" in
  70m)  ARCH="--num-layers 12 --hidden-size 512 --num-attention-heads 8 --ffn-hidden-size 1408"; GBS=256 ;;
  160m) ARCH="--num-layers 12 --hidden-size 768 --num-attention-heads 12 --ffn-hidden-size 2112"; GBS=512 ;;
  410m) ARCH="--num-layers 24 --hidden-size 1024 --num-attention-heads 16 --ffn-hidden-size 2816"; GBS=512 ;;
  1b)   ARCH="--num-layers 16 --hidden-size 2048 --num-attention-heads 32 --ffn-hidden-size 5632"; GBS=1024 ;;
esac
srun -N1 --container-image=/path/to/megatron-container.sqsh \
     --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -euo pipefail
cd /opt/Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
  $ARCH --swiglu --normalization RMSNorm \
  --position-embedding-type rope --rotary-base 10000 \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 16 --global-batch-size $GBS \
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine \
  --lr-warmup-iters 200 --train-iters $ITERS --exit-duration-in-mins 225 --clip-grad 1.0 \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --init-method-std 0.02 --bf16 \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model $ER/tokenizers/armweb_hy_32k.model \
  $DATA_ARGS \
  --split 998,1,1 \
  --save $CKPT --load $CKPT --save-interval 2000 \
  --log-interval 100 --eval-interval 100000 --eval-iters 10 \
  --attention-softmax-in-fp32 \
  --seed $SEED
"
echo LADDER_RUN_DONE size=$SIZE seed=$SEED
