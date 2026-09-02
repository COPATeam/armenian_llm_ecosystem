import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "armweb_data"
REPORTS = DATA_ROOT / "reports"


def write_stats(name: str, payload: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / f"{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return p


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
