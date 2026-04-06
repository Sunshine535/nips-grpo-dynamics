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

_HAS_PEFT = False
try:
    from peft import AutoPeftModelForCausalLM
    _HAS_PEFT = True
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance
apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_phase_point")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate phase diagram checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--positive_ratio", type=float, default=None)
    parser.add_argument("--negative_weight", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None,
                        help="rho value (for rho-sweep runs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Max samples to evaluate per dataset")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./results/phase_diagram")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_math", action="store_true",
                        help="Also evaluate on MATH dataset")
    parser.add_argument("--gsm8k_samples", type=int, default=None,
                        help="Override num_samples for GSM8K")
    parser.add_argument("--math_samples", type=int, default=None,
                        help="Override num_samples for MATH")
    return parser.parse_args()


def _extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...} in MATH solutions, handling nested braces."""
    idx = text.find(r"\boxed{")
    if idx == -1:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else ""


def extract_numeric_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
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

        input_len = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)

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
        raw = ans[idx] if isinstance(ans, list) else ans
        # GSM8K answers contain full solution; extract number after ####
        return extract_numeric_answer(str(raw))
    if "solution" in batch:
        sol = batch["solution"]
        sol_text = sol[idx] if isinstance(sol, list) else sol
        boxed = _extract_boxed(str(sol_text))
        if boxed:
            return boxed
        nums = re.findall(r"-?[\d,]+\.?\d*", str(sol_text))
        return nums[-1].replace(",", "") if nums else ""
    return ""


def _normalize(s: str) -> str:
    return s.strip().lower().replace(",", "").replace(" ", "")


def gsm8k_prompt(example):
    return (
        f"Solve the following math problem step by step. "
        f"Put your final numerical answer after ####.\n\n"
        f"Question: {example['question']}\n\nAnswer:"
    )

# Chat-template-aware version for instruct models
_eval_tokenizer = None

def gsm8k_prompt_chat(example):
    """Format as chat message for instruct models."""
    content = (
        f"Solve the following math problem step by step. "
        f"Put your final numerical answer after ####.\n\n"
        f"Question: {example['question']}"
    )
    messages = [{"role": "user", "content": content}]
    if _eval_tokenizer and hasattr(_eval_tokenizer, 'apply_chat_template'):
        return _eval_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return gsm8k_prompt(example)


def math_prompt(example):
    problem = example.get("problem", example.get("question", ""))
    return (
        f"Solve the following math problem. Show your work and put your "
        f"final answer in \\boxed{{}}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


def main():
    args = parse_args()

    logger.info(f"Loading checkpoint from {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    is_peft_ckpt = os.path.isfile(os.path.join(args.checkpoint_dir, "adapter_config.json"))
    if is_peft_ckpt and _HAS_PEFT:
        logger.info("Detected PEFT/LoRA checkpoint — using AutoPeftModelForCausalLM")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        if is_peft_ckpt:
            logger.warning("PEFT checkpoint found but peft not installed; loading as base model")
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    patch_model_instance(model)
    model.eval()
    device = next(model.parameters()).device

    all_metrics = {
        "seed": args.seed,
        "checkpoint_dir": args.checkpoint_dir,
    }
    if args.positive_ratio is not None:
        all_metrics["positive_ratio"] = args.positive_ratio
    if args.negative_weight is not None:
        all_metrics["negative_weight"] = args.negative_weight
    if args.rho is not None:
        all_metrics["rho"] = args.rho

    # GSM8K evaluation — use chat template for instruct models
    global _eval_tokenizer
    _eval_tokenizer = tokenizer
    use_chat = hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None
    prompt_fn = gsm8k_prompt_chat if use_chat else gsm8k_prompt
    if use_chat:
        logger.info("Using chat template for instruct model evaluation")

    gsm8k_n = args.gsm8k_samples if args.gsm8k_samples is not None else args.num_samples
    logger.info("Evaluating on GSM8K (max %d samples)...", gsm8k_n)
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_acc, gsm8k_results = evaluate_dataset(
        model, tokenizer, gsm8k, prompt_fn, gsm8k_n,
        args.temperature, args.max_new_tokens, args.batch_size, device,
    )
    all_metrics["gsm8k_accuracy"] = gsm8k_acc
    logger.info(f"GSM8K accuracy: {gsm8k_acc:.4f}")

    # MATH evaluation
    if args.eval_math:
        math_n = args.math_samples if args.math_samples is not None else args.num_samples
        logger.info("Evaluating on MATH (max %d samples)...", math_n)
        math_ds = None
        for ds_name in ["hendrycks/competition_math", "lighteval/MATH"]:
            try:
                math_ds = load_dataset(ds_name, split="test")
                logger.info("Loaded MATH dataset from %s", ds_name)
                break
            except Exception:
                continue
        if math_ds is not None:
            math_acc, math_results = evaluate_dataset(
                model, tokenizer, math_ds, math_prompt, math_n,
                args.temperature, args.max_new_tokens, args.batch_size, device,
            )
            all_metrics["math_accuracy"] = math_acc
            logger.info(f"MATH accuracy: {math_acc:.4f}")
        else:
            logger.warning("MATH dataset unavailable (tried hendrycks/competition_math, lighteval/MATH)")
            all_metrics["math_accuracy"] = None

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    if args.rho is not None:
        tag = f"rho{args.rho:.2f}_seed{args.seed}"
    elif args.positive_ratio is not None and args.negative_weight is not None:
        tag = f"alpha{args.positive_ratio:.2f}_beta{args.negative_weight:.2f}_seed{args.seed}"
    else:
        tag = f"seed{args.seed}"
    output_path = os.path.join(args.output_dir, f"eval_{tag}.json")
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
