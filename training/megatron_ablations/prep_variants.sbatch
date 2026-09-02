#!/bin/bash
#SBATCH --job-name=armweb-prepvar
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=03:50:00
#SBATCH --output=${ARMWEB_ROOT}/logs/prepvar_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/megatron-container.sqsh
srun -N1 --container-image="$IMG" --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -euo pipefail
declare -A SRC=(
  [fineweb2]=$ER/data_train/staging/fineweb2_hy_decon.jsonl
  [hplt]=$ER/data_train/baselines_clean/hplt_hy_decon.jsonl
  [culturax]=$ER/data_train/baselines_clean/culturax_hy_decon.jsonl
  [ru]=$ER/data_train/staging/ru_dedup.jsonl
)
for NAME in fineweb2 hplt culturax ru; do
  test -s $ER/data_train/megatron/\${NAME}_text_document.bin && { echo SKIP_\$NAME; continue; }
  python /opt/Megatron-LM/tools/preprocess_data.py \
    --input \${SRC[\$NAME]} \
    --output-prefix $ER/data_train/megatron/\$NAME \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $ER/tokenizers/armweb_hy_32k.model \
    --append-eod --workers 200
  echo PREP_\${NAME}_DONE
done
ls -lh $ER/data_train/megatron/
"
echo PREPVAR_COMPLETE
