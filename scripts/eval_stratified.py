#!/usr/bin/env python3
"""
Per-question GSM8K eval saver (Path C: difficulty-stratified analysis).

Runs one adapter OR the base model on the first N GSM8K-test questions
and saves per-question (pred, gold, correct) to <out>/per_q.json.

Afterwards, call scripts/analyze_stratified.py to compute accuracy per
difficulty band (defined by base-model correctness).
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/openbayes/input/input0")
os.environ.setdefault("HF_HUB_CACHE", "/openbayes/input/input0/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/openbayes/input/input0/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/openbayes/input/input0/hub")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.torch_compat import apply_torch_compat_patch  # noqa: E402

apply_torch_compat_patch()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

_ans_pat = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_fallback = re.compile(r"(-?[\d,]+\.?\d*)")
_think_pat = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_answer(text: str) -> str:
    text = _think_pat.sub("", text).strip()
    m = _ans_pat.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _fallback.findall(text)
    return nums[-1].replace(",", "") if nums else ""


def load_gsm8k_test(n: int, cache_dir: str, selection: str = "first_n", seed: int = 42):
    parquet_paths = glob.glob(
        f"{cache_dir}/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet"
    )
    if parquet_paths:
        import pandas as pd
        from datasets import Dataset
        df = pd.read_parquet(parquet_paths[0])
        ds = Dataset.from_pandas(df)
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")

    if selection == "full":
        return ds
    elif selection == "first_n":
        return ds.select(range(min(n, len(ds))))
    elif selection == "random":
        import random
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(n, len(ds)))
        return ds.select(sorted(indices))
    else:
        return ds.select(range(min(n, len(ds))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="LoRA adapter dir, or None for base-model eval")
    p.add_argument("--base_model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--cache_dir", default="/openbayes/input/input0/hub")
    p.add_argument("--n", type=int, default=100,
                   help="Number of questions (ignored when --selection full)")
    p.add_argument("--selection", default="first_n",
                   choices=["first_n", "full", "random"],
                   help="How to select test questions. Use 'full' for paper-grade evaluation.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"},
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter).eval()
    else:
        model = base.eval()

    ds = load_gsm8k_test(args.n, args.cache_dir, selection=args.selection)
    per_q = []
    correct = 0
    for i, ex in enumerate(ds):
        m = _ans_pat.search(ex["answer"])
        gold = m.group(1).replace(",", "") if m else ""
        msgs = [
            {"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."},
            {"role": "user", "content": f"Question: {ex['question']}"},
        ]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        is_correct = pred == gold
        correct += int(is_correct)
        per_q.append({
            "i": i,
            "question": ex["question"][:200],
            "gold": gold,
            "pred": pred[:100],
            "correct": bool(is_correct),
        })

    n_evaluated = len(per_q)
    result = {
        "adapter": args.adapter or "BASE",
        "base_model": args.base_model,
        "n": n_evaluated,
        "n_requested": args.n,
        "selection": args.selection,
        "correct": correct,
        "accuracy": correct / max(n_evaluated, 1),
        "eval_question_ids": [q["i"] for q in per_q],
        "per_q": per_q,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] acc={correct}/{n_evaluated}={correct/max(n_evaluated,1):.3f} selection={args.selection} saved to {args.out}")


if __name__ == "__main__":
    main()
