"""GlotLID routing: keep Armenian (hye/hyw), quarantine the rest."""
import json
from pathlib import Path

import fasttext
from huggingface_hub import hf_hub_download

from .io_utils import iter_jsonl

KEEP = {"hye_Armn", "hyw_Armn"}


def load_model():
    p = hf_hub_download("cis-lmu/glotlid", "model.bin")
    return fasttext.load_model(p)


def classify(model, text: str) -> tuple[str, float]:
    snippet = " ".join(text.split())[:1000]
    # call the C++ binding directly: model.predict() wraps it in
    # np.array(..., copy=False) which breaks on numpy>=2
    preds = model.f.predict(snippet, 1, 0.0, "strict")
    prob, label = preds[0]
    return label.replace("__label__", ""), float(prob)


def run_lid(in_path, keep_path, quarantine_path, model=None) -> dict:
    model = model or load_model()
    Path(keep_path).parent.mkdir(parents=True, exist_ok=True)
    stats = {"kept": 0, "quarantined": 0, "hyw_count": 0, "label_hist": {}}
    with open(keep_path, "w", encoding="utf-8") as keep_f, \
         open(quarantine_path, "w", encoding="utf-8") as quar_f:
        for row in iter_jsonl(in_path):
            label, prob = classify(model, row["text"])
            stats["label_hist"][label] = stats["label_hist"].get(label, 0) + 1
            if label in KEEP:
                row["lid"] = label
                keep_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                stats["hyw_count"] += label == "hyw_Armn"
            else:
                row["lid"] = label
                row["lid_prob"] = prob
                quar_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["quarantined"] += 1
    return stats
