import os
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
from huggingface_hub import snapshot_download
BASE = os.environ["ARMWEB_ROOT"] + "/data_train/cpt"
JOBS = [
    ("HuggingFaceFW/fineweb-edu", "dataset", ["sample/10BT/*"], f"{BASE}/fineweb_edu_10bt"),
    ("bigcode/the-stack-smol", "dataset", None, f"{BASE}/stack_smol"),
    ("Metric-AI/ArmBench-LLM-data", "dataset", None, f"{BASE}/armbench"),
]
for repo, typ, patterns, dest in JOBS:
    try:
        kw = dict(repo_id=repo, repo_type=typ, local_dir=dest, max_workers=8)
        if patterns:
            kw["allow_patterns"] = patterns
        snapshot_download(**kw)
        print("DONE", repo, flush=True)
    except Exception as e:
        print("FAILED", repo, type(e).__name__, str(e)[:150], flush=True)
