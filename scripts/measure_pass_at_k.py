#!/usr/bin/env python3
"""Measure pass@k for base model and LoRA adapters on GSM8K.

Tests the RLVR ceiling hypothesis: does GRPO narrow pass@k relative to
the base model, and does CE Replay preserve it?

Usage:
  # Base model
  python scripts/measure_pass_at_k.py --out results/pass_at_k_pilot/base.json

  # LoRA adapter
  python scripts/measure_pass_at_k.py \
    --adapter results/sage_minimal_abc/B_tasa_only/sage_tasa_only_seed42/checkpoint-final \
    --out results/pass_at_k_pilot/B_seed42.json

  # Quick smoke test
  python scripts/measure_pass_at_k.py --n-questions 10 --k 5 --out /tmp/smoke.json
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
if os.path.isdir("/openbayes/input/input0/hub"):
    os.environ.setdefault("HF_HOME", "/openbayes/input/input0")
    os.environ.setdefault("HF_HUB_CACHE", "/openbayes/input/input0/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/openbayes/input/input0/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/openbayes/input/input0/hub")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.torch_compat import apply_torch_compat_patch
apply_torch_compat_patch()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021 Codex paper)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(
        sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    )


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
    elif selection == "random":
        import random
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(n, len(ds)))
        return ds.select(sorted(indices))
    else:
        return ds.select(range(min(n, len(ds))))


def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, device):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    responses = []
    for i, out in enumerate(outputs):
        prompt_len = inputs.input_ids[i].ne(tokenizer.pad_token_id).sum()
        resp = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        responses.append(resp)
    return responses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--cache-dir", default="/openbayes/input/input0/hub")
    p.add_argument("--n-questions", type=int, default=200)
    p.add_argument("--selection", default="first_n", choices=["first_n", "full", "random"])
    p.add_argument("--k", type=int, default=25, help="Number of samples per question")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"[pass@k] adapter={args.adapter or 'BASE'} k={args.k} n_q={args.n_questions} T={args.temperature}")

    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True, device_map={"": args.device},
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter).eval()
        print(f"[pass@k] loaded adapter from {args.adapter}")
    else:
        model = base.eval()
        print("[pass@k] using base model (no adapter)")

    ds = load_gsm8k_test(args.n_questions, args.cache_dir, selection=args.selection)
    print(f"[pass@k] loaded {len(ds)} questions")

    sys_msg = "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."
    _supports_thinking = False
    try:
        tok.apply_chat_template(
            [{"role": "user", "content": "t"}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        _supports_thinking = True
    except TypeError:
        pass

    per_question = []
    t0 = time.time()

    for qi, ex in enumerate(ds):
        m = _ans_pat.search(ex["answer"])
        gold = m.group(1).replace(",", "") if m else ""

        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Question: {ex['question']}"},
        ]
        tpl_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if _supports_thinking:
            tpl_kwargs["enable_thinking"] = False
        prompt = tok.apply_chat_template(msgs, **tpl_kwargs)

        all_preds = []
        all_correct = []
        unique_correct_answers = set()

        for batch_start in range(0, args.k, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.k)
            batch_prompts = [prompt] * (batch_end - batch_start)
            responses = generate_batch(
                model, tok, batch_prompts, args.max_new_tokens, args.temperature, args.device
            )
            for resp in responses:
                pred = extract_answer(resp)
                is_correct = pred == gold
                all_preds.append(pred)
                all_correct.append(is_correct)
                if is_correct:
                    normalized = resp.strip()
                    unique_correct_answers.add(normalized)

        n_correct = sum(all_correct)
        k_values = [1, 5, 10, 25, 50, 100]
        pass_at = {}
        for kv in k_values:
            if kv <= args.k:
                pass_at[f"pass@{kv}"] = pass_at_k(args.k, n_correct, kv)

        per_question.append({
            "idx": qi,
            "question": ex["question"][:200],
            "gold": gold,
            "n_samples": args.k,
            "n_correct": n_correct,
            "n_unique_correct": len(unique_correct_answers),
            "pass_at": pass_at,
        })

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_pass1 = sum(q["pass_at"].get("pass@1", 0) for q in per_question) / len(per_question)
            avg_passk = sum(q["pass_at"].get(f"pass@{args.k}", 0) for q in per_question) / len(per_question)
            print(f"  [{qi+1}/{len(ds)}] elapsed={elapsed:.0f}s avg_pass@1={avg_pass1:.3f} avg_pass@{args.k}={avg_passk:.3f}")

    elapsed_total = time.time() - t0

    agg_pass_at = {}
    for kv in [1, 5, 10, 25, 50, 100]:
        key = f"pass@{kv}"
        vals = [q["pass_at"][key] for q in per_question if key in q["pass_at"]]
        if vals:
            agg_pass_at[key] = sum(vals) / len(vals)

    avg_unique = sum(q["n_unique_correct"] for q in per_question) / len(per_question)
    avg_correct = sum(q["n_correct"] for q in per_question) / len(per_question)

    result = {
        "adapter": args.adapter or "BASE",
        "base_model": args.base_model,
        "n_questions": len(per_question),
        "k": args.k,
        "temperature": args.temperature,
        "selection": args.selection,
        "elapsed_sec": round(elapsed_total, 1),
        "aggregate_pass_at": agg_pass_at,
        "avg_correct_per_question": round(avg_correct, 2),
        "avg_unique_correct_per_question": round(avg_unique, 2),
        "per_question": per_question,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[DONE] {args.adapter or 'BASE'}")
    print(f"  questions={len(per_question)}  k={args.k}  T={args.temperature}")
    print(f"  elapsed={elapsed_total:.0f}s")
    for key, val in sorted(agg_pass_at.items()):
        print(f"  {key} = {val:.4f}")
    print(f"  avg_unique_correct = {avg_unique:.2f}")
    print(f"  saved to {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
