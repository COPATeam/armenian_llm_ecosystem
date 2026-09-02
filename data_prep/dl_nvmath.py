import os
os.environ.setdefault("HF_HOME", os.environ["ARMWEB_ROOT"] + "/hf_cache")
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
BASE = os.environ["ARMWEB_ROOT"] + "/data_train/cpt"
for repo, dest, cap_gb in [("nvidia/Nemotron-CC-Math-v1", f"{BASE}/nemotron_cc_math", 4.0),
                           ("nvidia/Nemotron-Math-Proofs-v2", f"{BASE}/nemotron_proofs", 2.0)]:
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith((".parquet", ".jsonl", ".json.gz", ".jsonl.zst"))]
    got = 0.0
    n = 0
    for f in sorted(files):
        if got >= cap_gb:
            break
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=dest)
        got += os.path.getsize(p) / 2**30
        n += 1
    print(f"DONE {repo}: {n} files, {got:.2f} GiB", flush=True)
