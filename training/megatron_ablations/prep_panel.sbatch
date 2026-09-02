#!/bin/bash
#SBATCH --job-name=armweb-panelprep
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=${ARMWEB_ROOT}/logs/panelprep_%j.out
set -euo pipefail
ER=${ARMWEB_ROOT}
IMG=/path/to/megatron-container.sqsh
srun -N1 --container-image="$IMG" --container-mounts=/path/to/shared-fs:/path/to/shared-fs bash -c "
set -euo pipefail
cd $ER
python - <<'PYEOF'
import json
from pathlib import Path

ER = Path('${ARMWEB_ROOT}')
BENCH = ER / 'code/armweb_data/benchmarks'
OUT = ER / 'data_eval'
OUT.mkdir(exist_ok=True)
manifest = {}
PANELS = ['fineweb2_hy_test', 'hywiki', 'lrsum_hy', 'flores200_hy']
for name in PANELS:
    src = BENCH / f'{name}.txt'
    dst = OUT / f'{name}.jsonl'
    byts = 0
    n = 0
    with open(dst, 'w', encoding='utf-8') as f:
        for line in src.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            f.write(json.dumps({'text': line}, ensure_ascii=False) + '\n')
            byts += len(line.encode())
            n += 1
    manifest[name] = {'docs': n, 'text_bytes': byts}
(OUT / 'panel_manifest.json').write_text(json.dumps(manifest, indent=2))
print(manifest)
PYEOF
for NAME in fineweb2_hy_test hywiki lrsum_hy flores200_hy; do
  python /opt/Megatron-LM/tools/preprocess_data.py \
    --input $ER/data_eval/\$NAME.jsonl \
    --output-prefix $ER/data_eval/panel_\$NAME \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $ER/tokenizers/armweb_hy_32k.model \
    --append-eod --workers 64
done
python - <<'PYEOF'
import json, struct
from pathlib import Path
ER = Path('${ARMWEB_ROOT}')
m = json.loads((ER / 'data_eval/panel_manifest.json').read_text())
for name in list(m):
    binf = ER / f'data_eval/panel_{name}_text_document.bin'
    m[name]['tokens'] = binf.stat().st_size // 2  # uint16
(ER / 'data_eval/panel_manifest.json').write_text(json.dumps(m, indent=2))
print(json.dumps(m, indent=2))
PYEOF
"
echo PANEL_PREP_DONE
