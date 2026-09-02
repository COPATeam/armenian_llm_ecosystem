#!/bin/bash
# bpb eval of one checkpoint over the full panel. Env: CKPT (dir), TAG (label), ROOT, IMG, MOUNT
#SBATCH --nodes=1
#SBATCH --time=01:30:00
set -euo pipefail
SIZE="${SIZE:-410m}"
case "$SIZE" in
  70m)  ARCH="--num-layers 12 --hidden-size 512 --num-attention-heads 8 --ffn-hidden-size 1408" ;;
  160m) ARCH="--num-layers 12 --hidden-size 768 --num-attention-heads 12 --ffn-hidden-size 2112" ;;
  1b)   ARCH="--num-layers 16 --hidden-size 2048 --num-attention-heads 32 --ffn-hidden-size 5632" ;;
  410m) ARCH="--num-layers 24 --hidden-size 1024 --num-attention-heads 16 --ffn-hidden-size 2816" ;;
  1p3b) ARCH="--num-layers 24 --hidden-size 2048 --num-attention-heads 16 --ffn-hidden-size 5632" ;;
esac
export ARCH
srun -N1 --container-image="$IMG" --container-mounts="$MOUNT:$MOUNT" bash -c "
set -euo pipefail
cd /opt/Megatron-LM
for NAME in fineweb2_hy_test hywiki lrsum_hy flores200_hy armweb_test_iid armweb_test_tail; do
  case \$NAME in armweb_*) PREFIX=$ROOT/data_train/megatron/\${NAME}_text_document;; *) PREFIX=$ROOT/data_eval/panel_\${NAME}_text_document;; esac
  TOK=\$(python -c \"import os; print(os.path.getsize('\$PREFIX.bin')//2)\")
  ITERS=\$(python -c \"print(max((\$TOK)//(128*2049)+1, 2))\")
  echo \"=== PANEL \$NAME tokens=\$TOK iters=\$ITERS ===\"
  torchrun --nproc_per_node=8 pretrain_gpt.py \
    $ARCH --swiglu --normalization RMSNorm \
    --position-embedding-type rope --rotary-base 10000 \
    --untie-embeddings-and-output-weights --disable-bias-linear \
    --seq-length 2048 --max-position-embeddings 2048 \
    --micro-batch-size 16 --global-batch-size 128 \
    --lr 1e-5 --train-iters 1 --skip-train \
    --bf16 --attention-softmax-in-fp32 \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $ROOT/tokenizers/armweb_hy_32k.model \
    --train-data-path 1.0 \$PREFIX --valid-data-path 1.0 \$PREFIX --test-data-path 1.0 \$PREFIX \
    --load $CKPT --finetune --no-load-optim --no-load-rng --exit-on-missing-checkpoint \
    --eval-iters \$ITERS --eval-interval 1 \
    --seed 1337
done
"
echo BPB_EVAL_DONE tag=$TAG
