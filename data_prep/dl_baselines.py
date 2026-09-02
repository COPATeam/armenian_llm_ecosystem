import os
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
from huggingface_hub import snapshot_download

BASE = os.environ["ARMWEB_ROOT"] + "/data_train/baselines"
JOBS = [
    ("HuggingFaceFW/fineweb-2", ["data/hye_Armn/train/*"], f"{BASE}/fineweb2_hy"),
    ("HPLT/HPLT2.0_cleaned", ["hye_Armn*"], f"{BASE}/hplt_hy"),
    ("uonlp/CulturaX", ["hy/*"], f"{BASE}/culturax_hy"),  # gated: works once token present
]
for repo, patterns, dest in JOBS:
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns,
                          local_dir=dest, max_workers=4)
        print(f"DONE {repo}", flush=True)
    except Exception as e:
        print(f"FAILED {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True)
