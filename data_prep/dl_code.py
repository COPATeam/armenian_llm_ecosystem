import os
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
from huggingface_hub import snapshot_download
try:
    snapshot_download(repo_id="HuggingFaceTB/smollm-corpus", repo_type="dataset",
                      allow_patterns=["python-edu/*"],
                      local_dir=os.environ["ARMWEB_ROOT"] + "/data_train/cpt/python_edu",
                      max_workers=8)
    print("DONE python-edu", flush=True)
except Exception as e:
    print("FAILED", type(e).__name__, str(e)[:200], flush=True)
