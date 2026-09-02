#!/bin/bash
#SBATCH --job-name=armweb-armbench7
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=03:55:00
#SBATCH --output=${ARMWEB_ROOT}/logs/armbench12_%x_%j.out
set -uo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/pytorch-cuda-container.sqsh
srun -N1 --container-image="$IMG" --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -uo pipefail
export HF_HOME=$ER/hf_cache PIP_CACHE_DIR=$ER/.pipcache TMPDIR=$ER/.tmp NLTK_DATA=$ER/nltk_data
VP=/opt/venv/bin/python
\$VP -m pip install -q -e $ER/jobs/lighteval-armbench transformers==5.15.1 accelerate nltk 2>&1 | tail -1
CT=$ER/jobs/lighteval-armbench/src/lighteval/tasks/multilingual/tasks/armenian.py
TAG=\$(basename \$MODELS)
LOGD=$ER/eval_results/armbench/logs_\$TAG
mkdir -p \$LOGD
P8='mmlu_pro exam_math scientific squad ms_marco exam_history exam_literature topic-14class space_fix punctuation finer pioner pos arak'
P1='sentiment email conversation paraphrase short_sentences_translation belebele include syndarin dream hartak'
for N in \$P8; do
  echo \"=== TASK8 \$N \$(date +%H:%M) ===\"
  \$VP -m accelerate.commands.launch --num_processes 8 -m lighteval accelerate \
    \"model_name=\$MODELS,dtype=bfloat16,trust_remote_code=True\" \
    \"armenian:\$N|0\" --custom-tasks \$CT \
    --output-dir $ER/eval_results/armbench/\$TAG > \$LOGD/\$N.log 2>&1 \
    && echo \"TASK_OK \$N\" || echo \"TASK_FAIL \$N\"
done
for N in \$P1; do
  echo \"=== TASK1 \$N \$(date +%H:%M) ===\"
  \$VP -m accelerate.commands.launch --num_processes 1 -m lighteval accelerate \
    \"model_name=\$MODELS,dtype=bfloat16,trust_remote_code=True\" \
    \"armenian:\$N|0\" --custom-tasks \$CT \
    --output-dir $ER/eval_results/armbench/\$TAG > \$LOGD/\$N.log 2>&1 \
    && echo \"TASK_OK \$N\" || echo \"TASK_FAIL \$N\"
done
echo ALL_TASKS_ATTEMPTED
"
echo ARMBENCH7_DONE
