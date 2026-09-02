#!/bin/bash
#SBATCH --job-name=armweb-finaleval
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=03:50:00
#SBATCH --output=${ARMWEB_ROOT}/logs/finaleval_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/pytorch-cuda-container.sqsh
srun -N1 --container-image="$IMG" --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -euo pipefail
export HF_HOME=$ER/hf_cache PIP_CACHE_DIR=$ER/.pipcache TMPDIR=$ER/.tmp
VP=/opt/venv/bin/python
\$VP -m ensurepip --upgrade >/dev/null 2>&1 || true
\$VP -m pip --version >/dev/null 2>&1 || curl -sS https://bootstrap.pypa.io/get-pip.py | \$VP >/dev/null
\$VP -m pip install -q transformers==5.15.1 tokenizers accelerate lm-eval 2>&1 | tail -1
mkdir -p $ER/eval_results
RUN=\$(\$VP - <<'PYEOF'
from lm_eval.tasks import TaskManager
want = ['belebele_hye_Armn', 'm_mmlu_hy', 'hellaswag_hy', 'multiblimp_hye',
        'multiblimp_hyw', 'include_base_44_armenian', 'sib200_hye_Armn', 'arc_hy']
avail = set(TaskManager().all_tasks)
print(','.join(t for t in want if t in avail))
PYEOF
)
echo \"RESOLVED_TASKS=\$RUN\"
for MODEL in $ER/train/ckpt/cpt_e4b-headline-cont2/checkpoint-500; do
  TAG=\$(basename \$MODEL)
  echo \"=== EVAL \$TAG ===\"
  \$VP -m lm_eval --model hf \
    --model_args pretrained=\$MODEL,tokenizer=$ER/models/armweb-e4b-cpt,dtype=bfloat16,trust_remote_code=True \
    --tasks \$RUN --batch_size auto \
    --output_path $ER/eval_results/\$TAG 2>&1 | tail -40
done
"
echo FINAL_EVAL_DONE
