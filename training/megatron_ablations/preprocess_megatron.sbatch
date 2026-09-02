#!/bin/bash
#SBATCH --job-name=armweb-prep
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=03:30:00
#SBATCH --output=${ARMWEB_ROOT}/logs/prep_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/megatron-container.sqsh
srun -N1 --container-image="$IMG" --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -euo pipefail
python $ER/jobs/parquet_to_jsonl.py $ER/release $ER/data_train/jsonl
mkdir -p $ER/data_train/megatron
for SPLIT in train val test_iid test_tail; do
  test -s $ER/data_train/megatron/armweb_\${SPLIT}_text_document.bin && { echo SKIP_\${SPLIT}; continue; }
  python /opt/Megatron-LM/tools/preprocess_data.py \
    --input $ER/data_train/jsonl/\$SPLIT.jsonl \
    --output-prefix $ER/data_train/megatron/armweb_\$SPLIT \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $ER/tokenizers/armweb_hy_32k.model \
    --append-eod --workers 200
  echo PREP_\${SPLIT}_DONE
done
ls -lh $ER/data_train/megatron/
"
echo PREPROCESS_COMPLETE
