#!/usr/bin/env python3
"""
Step 1: Run on server WITH internet (only needs huggingface_hub + datasets).
Step 2: rsync to GPU server (port 43038).

Usage:
    pip install huggingface_hub datasets
    python3 scripts/download_assets.py
"""

import os

BASE = "/openbayes/input/input0"
HUB = f"{BASE}/hub"
DS_DIR = f"{BASE}/datasets"

os.environ.update({
    "HF_HOME": BASE,
    "HF_HUB_CACHE": HUB,
    "HF_DATASETS_CACHE": DS_DIR,
    "TRANSFORMERS_CACHE": HUB,
})

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
    p = snapshot_download(repo, cache_dir=HUB)
    print(f"    Done → {p}")


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

    print("\n" + "=" * 60)
    print("Done. Rsync to GPU server:")
    print("=" * 60)
    print(f"""
rsync -avP {HUB}/models--Qwen--Qwen3.5-9B \\
    root@<gpu-server>:{HUB}/

rsync -avP {DS_DIR}/ \\
    root@<gpu-server>:{DS_DIR}/
""")
