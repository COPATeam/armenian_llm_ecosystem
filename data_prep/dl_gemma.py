import os
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
from huggingface_hub import snapshot_download
for repo in ("google/gemma-4-E2B", "google/gemma-4-E4B"):
    try:
        p = snapshot_download(repo_id=repo, max_workers=8)
        print("DONE", repo, p, flush=True)
    except Exception as e:
        print("FAILED", repo, type(e).__name__, str(e)[:150], flush=True)
