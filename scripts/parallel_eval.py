#!/usr/bin/env python3
"""Evaluate a single LoRA adapter on GSM8K (n=100). Used by parallel wrapper."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/openbayes/input/input0")
os.environ.setdefault("HF_HUB_CACHE", "/openbayes/input/input0/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/openbayes/input/input0/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/openbayes/input/input0/hub")

import argparse, json, sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))
from analyze_gates_1_2 import eval_lora_on_gsm8k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--cache_dir", default="/openbayes/input/input0/hub")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = eval_lora_on_gsm8k(args.base_model, args.cache_dir, args.adapter, n=args.n)
    res["adapter"] = args.adapter
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
