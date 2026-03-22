#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on GSM8K and MATH datasets.
Outputs accuracy metrics for a single phase-diagram point.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_phase_point")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate phase diagram checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--positive_ratio", type=float, required=True)
    parser.add_argument("--negative_weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Max samples to evaluate per dataset")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./results/phase_diagram")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_math", action="store_true",
                        help="Also evaluate on MATH dataset")
    return parser.parse_args()


def extract_numeric_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def evaluate_dataset(model, tokenizer, dataset, prompt_template, max_samples,
                     temperature, max_new_tokens, batch_size, device):
    correct = 0
    total = 0
    results = []

    samples = dataset.select(range(min(max_samples, len(dataset))))

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i: i + batch_size]
        prompts = [prompt_template(ex) for ex in _iter_batch(batch)]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            outputs = model.generate(**inputs, **gen_kwargs)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for j, output in enumerate(outputs):
            prompt_len = (inputs["input_ids"][j] != pad_id).sum().item()
            generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)

            idx = i + j
            if idx >= len(samples):
                break

            gold_answer = str(_get_answer(batch, j))
            pred_answer = extract_numeric_answer(generated)

            is_correct = _normalize(pred_answer) == _normalize(gold_answer)
            correct += int(is_correct)
            total += 1

            results.append({
                "idx": idx,
                "gold": gold_answer,
                "pred": pred_answer,
                "correct": is_correct,
                "generated": generated[:500],
            })

    accuracy = correct / max(total, 1)
    return accuracy, results


def _iter_batch(batch):
    """Iterate over a batched dict from HuggingFace datasets."""
    keys = list(batch.keys())
    n = len(batch[keys[0]])
    for i in range(n):
        yield {k: batch[k][i] for k in keys}


def _get_answer(batch, idx):
    if "answer" in batch:
        ans = batch["answer"]
        if isinstance(ans, list):
            return ans[idx]
        return ans
    return ""


def _normalize(s: str) -> str:
    return s.strip().lower().replace(",", "").replace(" ", "")


def gsm8k_prompt(example):
    return (
        f"Solve the following math problem step by step. "
        f"End with '#### <answer>'.\n\n"
        f"Question: {example['question']}\n\nAnswer:"
    )


def math_prompt(example):
    problem = example.get("problem", example.get("question", ""))
    return (
        f"Solve the following math problem. Show your work and put your "
        f"final answer in \\boxed{{}}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


def main():
    args = parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    logger.info(f"Loading checkpoint from {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    all_metrics = {
        "positive_ratio": args.positive_ratio,
        "negative_weight": args.negative_weight,
        "seed": args.seed,
        "checkpoint_dir": args.checkpoint_dir,
    }

    # GSM8K evaluation
    logger.info("Evaluating on GSM8K...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_acc, gsm8k_results = evaluate_dataset(
        model, tokenizer, gsm8k, gsm8k_prompt, args.num_samples,
        args.temperature, args.max_new_tokens, args.batch_size, device,
    )
    all_metrics["gsm8k_accuracy"] = gsm8k_acc
    logger.info(f"GSM8K accuracy: {gsm8k_acc:.4f}")

    # MATH evaluation
    if args.eval_math:
        logger.info("Evaluating on MATH...")
        try:
            math_ds = load_dataset("lighteval/MATH", split="test")
            math_acc, math_results = evaluate_dataset(
                model, tokenizer, math_ds, math_prompt, args.num_samples,
                args.temperature, args.max_new_tokens, args.batch_size, device,
            )
            all_metrics["math_accuracy"] = math_acc
            logger.info(f"MATH accuracy: {math_acc:.4f}")
        except Exception as e:
            logger.warning(f"MATH evaluation failed: {e}")
            all_metrics["math_accuracy"] = None

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"alpha{args.positive_ratio:.2f}_beta{args.negative_weight:.2f}_seed{args.seed}"
    output_path = os.path.join(args.output_dir, f"eval_{tag}.json")
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
