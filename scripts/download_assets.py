#!/usr/bin/env python3
"""
Step 1: Run this on a server WITH internet access.
Step 2: rsync downloaded files to GPU server (port 43038).

Downloads go to /openbayes/input/input0/hub/ — same path as GPU server,
so rsync preserves the path structure and HF cache works offline.

Usage:
    python3 download_assets.py

After download, rsync to GPU server:
    rsync -avP /openbayes/input/input0/hub/models--Qwen--Qwen3.5-9B \
        root@<gpu-server>:/openbayes/input/input0/hub/
    rsync -avP /openbayes/input/input0/datasets/ \
        root@<gpu-server>:/openbayes/input/input0/datasets/
"""

import os, sys

BASE = "/openbayes/input/input0"
HUB = f"{BASE}/hub"
DS_DIR = f"{BASE}/datasets"

os.environ.update({
    "HF_HOME": BASE,
    "HF_HUB_CACHE": HUB,
    "HF_DATASETS_CACHE": DS_DIR,
    "TRANSFORMERS_CACHE": HUB,
})

# Try China mirror first
for ep in ["https://hf-mirror.com", "https://huggingface.co"]:
    try:
        import requests
        if requests.head(f"{ep}/api/models", timeout=10).status_code < 500:
            os.environ["HF_ENDPOINT"] = ep
            print(f"Using: {ep}")
            break
    except Exception:
        continue


def dl_model(repo):
    from huggingface_hub import snapshot_download
    print(f"\n>>> Downloading model: {repo}")
    p = snapshot_download(repo, cache_dir=HUB, resume_download=True)
    print(f"    Done → {p}")
    # Verify
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=HUB)
    cfg = AutoConfig.from_pretrained(repo, cache_dir=HUB)
    print(f"    Verified: vocab={tok.vocab_size} hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")


def dl_dataset(name, config=None):
    from datasets import load_dataset
    tag = f"{name}" + (f"/{config}" if config else "")
    print(f"\n>>> Downloading dataset: {tag}")
    ds = load_dataset(name, config, cache_dir=DS_DIR)
    for s in ds:
        print(f"    {s}: {len(ds[s])} rows")


if __name__ == "__main__":
    os.makedirs(HUB, exist_ok=True)
    os.makedirs(DS_DIR, exist_ok=True)

    dl_model("Qwen/Qwen3.5-9B")
    dl_dataset("openai/gsm8k", "main")
    dl_dataset("HuggingFaceH4/MATH-500")

    # Print rsync commands for copy to GPU server
    print("\n" + "=" * 60)
    print("Download complete. Now rsync to GPU server:")
    print("=" * 60)
    print(f"""
rsync -avP {HUB}/models--Qwen--Qwen3.5-9B \\
    root@<gpu-server>:{HUB}/

rsync -avP {DS_DIR}/openai___gsm8k \\
    root@<gpu-server>:{DS_DIR}/

rsync -avP {DS_DIR}/HuggingFaceH4___MATH-500 \\
    root@<gpu-server>:{DS_DIR}/
""")
