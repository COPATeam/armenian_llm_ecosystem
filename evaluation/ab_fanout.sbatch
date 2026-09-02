#!/bin/bash
#SBATCH --job-name=armweb-abfan
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=03:50:00
#SBATCH --output=${ARMWEB_ROOT}/logs/abfan_%j.out
set -uo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/pytorch-cuda-container.sqsh
srun -N1 --container-image="$IMG" --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -uo pipefail
export HF_HOME=$ER/hf_cache PIP_CACHE_DIR=$ER/.pipcache TMPDIR=$ER/.tmp NLTK_DATA=$ER/nltk_data
export HF_TOKEN=\$(cat $ER/hf_cache/token 2>/dev/null || true)
VP=/opt/venv/bin/python
\$VP -m ensurepip --upgrade >/dev/null 2>&1 || true
\$VP -m pip --version >/dev/null 2>&1 || curl -sS https://bootstrap.pypa.io/get-pip.py | \$VP >/dev/null
\$VP -m pip install -q -e $ER/jobs/lighteval-armbench transformers==5.15.1 accelerate nltk 2>&1 | tail -1
CT=$ER/jobs/lighteval-armbench/src/lighteval/tasks/multilingual/tasks/armenian.py
TAG=\$(basename \$MODELS)
LOGD=$ER/eval_results/armbench/logs_\$TAG
mkdir -p \$LOGD
for N in \$TASKS8; do
  echo \"=== TASK8 \$N \$(date +%H:%M) ===\"
  \$VP -m accelerate.commands.launch --num_processes 8 -m lighteval accelerate \
    \"model_name=\$MODELS,dtype=bfloat16,trust_remote_code=True,batch_size=1\" \
    \"armenian:\$N|0\" --custom-tasks \$CT \
    --output-dir $ER/eval_results/armbench/\$TAG > \$LOGD/\$N.log 2>&1 \
    && echo \"TASK_OK \$N\" || echo \"TASK_FAIL \$N\"
done
for N in \$TASKS1; do
  echo \"=== TASK1 \$N \$(date +%H:%M) ===\"
  \$VP -m accelerate.commands.launch --num_processes 1 -m lighteval accelerate \
    \"model_name=\$MODELS,dtype=bfloat16,trust_remote_code=True,batch_size=1\" \
    \"armenian:\$N|0\" --custom-tasks \$CT \
    --output-dir $ER/eval_results/armbench/\$TAG > \$LOGD/\$N.log 2>&1 \
    && echo \"TASK_OK \$N\" || echo \"TASK_FAIL \$N\"
done
echo ALL_TASKS_ATTEMPTED
"
