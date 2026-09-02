#!/bin/bash
#SBATCH --job-name=armweb-bdecon2
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=03:50:00
#SBATCH --output=${ARMWEB_ROOT}/logs/bdecon2_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/megatron-container.sqsh
srun -N1 --container-image="$IMG" --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -euo pipefail
export PIP_CACHE_DIR=$ER/.pipcache TMPDIR=$ER/.tmp
mkdir -p \$PIP_CACHE_DIR \$TMPDIR
pip -q install xxhash 2>&1 | tail -1
cd $ER/code/armweb
python - <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
import pyarrow.parquet as pq
from armweb.decontaminate import build_ngram_index, scan

ER = Path('${ARMWEB_ROOT}')
OUT = ER / 'data_train/baselines_clean'
OUT.mkdir(exist_ok=True)
index = build_ngram_index(ER / 'code/armweb_data/benchmarks')

for name, sub in [('hplt_hy', 'hplt_hy'), ('culturax_hy', 'culturax_hy')]:
    out = OUT / f'{name}.jsonl'
    n = 0
    with open(out, 'w', encoding='utf-8') as f:
        for p in sorted((ER / 'data_train/baselines' / sub).rglob('*.parquet')):
            for text in pq.read_table(p, columns=['text']).column('text').to_pylist():
                f.write(json.dumps({'id': f'{name}_{n}', 'text': text}, ensure_ascii=False) + '\n')
                n += 1
    print(name, 'docs:', n, flush=True)
    assert n > 500_000, f'{name}: unexpectedly few docs'
    stats = scan(out, OUT / f'{name}_decon.jsonl', index)
    (OUT / f'{name}_decon_stats.json').write_text(json.dumps(stats, indent=2))
    total = stats['dropped'] + stats['kept']
    print(name, 'dropped', stats['dropped'], 'of', total,
          f\"({100*stats['dropped']/max(total,1):.2f}%)\",
          {k: v for k, v in stats['hits'].items() if v}, flush=True)
PYEOF
"
echo BDECON2_DONE
