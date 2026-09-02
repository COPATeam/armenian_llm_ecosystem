#!/bin/bash
#SBATCH --job-name=armweb-curator
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=03:50:00
#SBATCH --output=${ARMWEB_ROOT}/logs/curator_%j.out
set -uo pipefail
ER=${ARMWEB_ROOT}
srun -N1 --container-image='docker://nvcr.io#nvidia/nemo-curator:26.02' \
     --container-mounts=/path/to/shared-fs:/path/to/shared-fs \
     bash -c "
set -uo pipefail
export TMPDIR=$ER/.tmp
PY=\$(command -v python || command -v python3)
\$PY -c 'import nemo_curator; print(\"curator\", nemo_curator.__version__)'
\$PY $ER/jobs/curator_probe.py \$INPUT $ER/curator_check/\$TAG
"
