#!/usr/bin/env python3
"""
Evaluate HalluZero-trained checkpoints on GSM8K and MATH.

Reports:
  - accuracy, pass@1, average generation tokens
  - zero-score diagnostic (% of generations receiving 0 reward)
  - per-sample detailed results saved to JSON

Supports multi-GPU inference via device_map="auto".
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("halluzero-eval")

ANS_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def parse_args():
    p = argparse.ArgumentParser(description="HalluZero Evaluation")
    p.add_argument("--model_path", type=str, required=True,
                    help="Path to trained checkpoint or HF model id")
    p.add_argument("--output_dir", type=str, default="./results/eval")
    p.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math"],
                    choices=["gsm8k", "math"])
    p.add_argument("--gsm8k_samples", type=int, default=-1,
                    help="Number of GSM8K test samples (-1 = all 1319)")
    p.add_argument("--math_samples", type=int, default=500,
                    help="Number of MATH test samples")
    p.add_argument("--num_generations", type=int, default=1,
                    help="Generations per problem for pass@k (k=1)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy; >0 for sampling (needed for pass@k>1)")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_gsm8k_gold(answer_text: str) -> str:
    m = ANS_RE.search(answer_text)
    if m:
        return m.group(1).replace(",", "").strip()
    return ""


def extract_model_answer(text: str) -> str:
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = ANS_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def answers_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()


# ---------------------------------------------------------------------------
# pass@k estimator  (from Chen et al., "Evaluating Large Language Models
# Trained on Code", 2021)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int, temperature: float,
                   top_p: float) -> list[str]:
    """Generate for a batch of prompts; returns list of decoded strings."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=2048,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = outputs[:, prompt_len:]

    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    token_counts = [int((g != tokenizer.pad_token_id).sum()) for g in gen_ids]
    return texts, token_counts


# ---------------------------------------------------------------------------
# GSM8K evaluation
# ---------------------------------------------------------------------------

def eval_gsm8k(model, tokenizer, args) -> dict:
    logger.info("=== GSM8K Evaluation ===")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.gsm8k_samples > 0:
        ds = ds.select(range(min(args.gsm8k_samples, len(ds))))
    logger.info("  samples: %d, generations/sample: %d", len(ds), args.num_generations)

    prompts_raw, golds = [], []
    for ex in ds:
        p = (
            "Solve the following math problem step by step. "
            "Show your work and put the final answer after ####.\n\n"
            f"Question: {ex['question']}\n\nAnswer:"
        )
        prompts_raw.append(p)
        golds.append(extract_gsm8k_gold(ex["answer"]))

    all_results = []
    total_correct_greedy = 0
    total_tokens = 0
    total_zero_reward = 0
    total_generations = 0
    per_problem_correct_counts = []

    for gen_idx in range(args.num_generations):
        for batch_start in tqdm(range(0, len(prompts_raw), args.batch_size),
                                desc=f"GSM8K gen={gen_idx}"):
            batch_end = min(batch_start + args.batch_size, len(prompts_raw))
            batch_prompts = prompts_raw[batch_start:batch_end]
            batch_golds = golds[batch_start:batch_end]

            texts, tcounts = generate_batch(
                model, tokenizer, batch_prompts,
                args.max_new_tokens, args.temperature, args.top_p,
            )

            for i, (text, tc, gold) in enumerate(zip(texts, tcounts, batch_golds)):
                pred = extract_model_answer(text)
                correct = answers_match(pred, gold)
                reward = 1.0 if correct else 0.0

                total_tokens += tc
                total_generations += 1
                if reward == 0.0:
                    total_zero_reward += 1

                idx = batch_start + i
                if gen_idx == 0:
                    total_correct_greedy += int(correct)
                    per_problem_correct_counts.append(int(correct))
                    all_results.append({
                        "idx": idx,
                        "question": ds[idx]["question"],
                        "gold": gold,
                        "generations": [{"text": text[:500], "pred": pred,
                                         "correct": correct, "tokens": tc}],
                    })
                else:
                    per_problem_correct_counts[idx] += int(correct)
                    all_results[idx]["generations"].append({
                        "text": text[:500], "pred": pred,
                        "correct": correct, "tokens": tc,
                    })

    n = len(prompts_raw)
    accuracy = total_correct_greedy / n if n else 0
    avg_tokens = total_tokens / max(total_generations, 1)
    zero_ratio = total_zero_reward / max(total_generations, 1)

    pass1_scores = [
        pass_at_k(args.num_generations, c, 1)
        for c in per_problem_correct_counts
    ]
    pass_at_1 = float(np.mean(pass1_scores))

    result = {
        "benchmark": "gsm8k",
        "num_problems": n,
        "num_generations_per_problem": args.num_generations,
        "accuracy_greedy": accuracy,
        "pass_at_1": pass_at_1,
        "avg_tokens": avg_tokens,
        "zero_score_ratio": zero_ratio,
        "total_correct_greedy": total_correct_greedy,
        "details": all_results,
    }
    logger.info("  accuracy=%.4f  pass@1=%.4f  avg_tokens=%.1f  zero_ratio=%.4f",
                accuracy, pass_at_1, avg_tokens, zero_ratio)
    return result


# ---------------------------------------------------------------------------
# MATH evaluation
# ---------------------------------------------------------------------------

def eval_math(model, tokenizer, args) -> dict:
    logger.info("=== MATH Evaluation ===")
    ds = load_dataset("hendrycks/competition_math", split="test")
    if args.math_samples > 0:
        ds = ds.select(range(min(args.math_samples, len(ds))))
    logger.info("  samples: %d, generations/sample: %d", len(ds), args.num_generations)

    prompts_raw, golds = [], []
    for ex in ds:
        p = (
            "Solve the following math problem. Show your reasoning step by step. "
            "Put your final answer in \\boxed{}.\n\n"
            f"Problem: {ex['problem']}\n\nSolution:"
        )
        prompts_raw.append(p)
        m = BOXED_RE.search(ex["solution"])
        golds.append(m.group(1).strip() if m else "")

    all_results = []
    total_correct_greedy = 0
    total_tokens = 0
    total_zero_reward = 0
    total_generations = 0
    per_problem_correct_counts = []

    for gen_idx in range(args.num_generations):
        for batch_start in tqdm(range(0, len(prompts_raw), args.batch_size),
                                desc=f"MATH gen={gen_idx}"):
            batch_end = min(batch_start + args.batch_size, len(prompts_raw))
            batch_prompts = prompts_raw[batch_start:batch_end]
            batch_golds = golds[batch_start:batch_end]

            texts, tcounts = generate_batch(
                model, tokenizer, batch_prompts,
                args.max_new_tokens, args.temperature, args.top_p,
            )

            for i, (text, tc, gold) in enumerate(zip(texts, tcounts, batch_golds)):
                pred = extract_model_answer(text)
                correct = answers_match(pred, gold)
                reward = 1.0 if correct else 0.0

                total_tokens += tc
                total_generations += 1
                if reward == 0.0:
                    total_zero_reward += 1

                idx = batch_start + i
                if gen_idx == 0:
                    total_correct_greedy += int(correct)
                    per_problem_correct_counts.append(int(correct))
                    all_results.append({
                        "idx": idx,
                        "problem": ds[idx]["problem"][:200],
                        "gold": gold,
                        "level": ds[idx].get("level", ""),
                        "type": ds[idx].get("type", ""),
                        "generations": [{"text": text[:500], "pred": pred,
                                         "correct": correct, "tokens": tc}],
                    })
                else:
                    per_problem_correct_counts[idx] += int(correct)
                    all_results[idx]["generations"].append({
                        "text": text[:500], "pred": pred,
                        "correct": correct, "tokens": tc,
                    })

    n = len(prompts_raw)
    accuracy = total_correct_greedy / n if n else 0
    avg_tokens = total_tokens / max(total_generations, 1)
    zero_ratio = total_zero_reward / max(total_generations, 1)

    pass1_scores = [
        pass_at_k(args.num_generations, c, 1)
        for c in per_problem_correct_counts
    ]
    pass_at_1 = float(np.mean(pass1_scores))

    level_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        lv = r.get("level", "unknown")
        level_acc[lv]["total"] += 1
        if r["generations"][0]["correct"]:
            level_acc[lv]["correct"] += 1
    level_breakdown = {
        k: v["correct"] / max(v["total"], 1)
        for k, v in sorted(level_acc.items())
    }

    result = {
        "benchmark": "math",
        "num_problems": n,
        "num_generations_per_problem": args.num_generations,
        "accuracy_greedy": accuracy,
        "pass_at_1": pass_at_1,
        "avg_tokens": avg_tokens,
        "zero_score_ratio": zero_ratio,
        "total_correct_greedy": total_correct_greedy,
        "level_breakdown": level_breakdown,
        "details": all_results,
    }
    logger.info("  accuracy=%.4f  pass@1=%.4f  avg_tokens=%.1f  zero_ratio=%.4f",
                accuracy, pass_at_1, avg_tokens, zero_ratio)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Loading model from %s", args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    eval_fns = {"gsm8k": eval_gsm8k, "math": eval_math}
    summary = {}
    t0 = time.time()

    for bench in args.benchmarks:
        if bench not in eval_fns:
            logger.warning("Unknown benchmark %s, skipping", bench)
            continue
        result = eval_fns[bench](model, tokenizer, args)
        detail_path = os.path.join(args.output_dir, f"{bench}_results.json")
        with open(detail_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        summary[bench] = {k: v for k, v in result.items() if k != "details"}

    elapsed = time.time() - t0
    summary["_meta"] = {
        "model_path": args.model_path,
        "elapsed_sec": round(elapsed, 1),
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info("EVALUATION SUMMARY  (%.1f s)", elapsed)
    logger.info("=" * 65)
    for bench, s in summary.items():
        if bench.startswith("_"):
            continue
        logger.info("  %-8s  acc=%.4f  pass@1=%.4f  tokens=%.1f  zero%%=%.2f%%",
                    bench, s["accuracy_greedy"], s["pass_at_1"],
                    s["avg_tokens"], s["zero_score_ratio"] * 100)
    logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
