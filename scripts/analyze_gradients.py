#!/usr/bin/env python3
"""
Diagnostic tool: compare gradient norms/directions for 0-score vs non-0-score
samples during GRPO training.

Runs a few training steps, hooks into backward pass to capture per-sample
gradient statistics, and produces analysis of gradient behavior.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.zero_score_handler import ZeroScoreConfig, ZeroScoreHandler, ZeroScoreStrategy
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grad-analysis")


def parse_args():
    parser = argparse.ArgumentParser(description="Gradient analysis for HalluZero")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output_dir", type=str, default="./results/gradient_analysis")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--strategies", nargs="+",
                        default=["clip", "temperature", "curriculum", "relabel"])
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def extract_answer(text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def collect_layer_gradients(model) -> dict:
    """Collect gradient norms per layer group."""
    layer_grads = defaultdict(float)
    layer_counts = defaultdict(int)

    for name, param in model.named_parameters():
        if param.grad is not None:
            parts = name.split(".")
            if "layers" in parts:
                idx = parts.index("layers")
                layer_key = f"layer_{parts[idx + 1]}"
            elif "embed" in name:
                layer_key = "embedding"
            elif "lm_head" in name or "output" in name:
                layer_key = "lm_head"
            else:
                layer_key = "other"

            layer_grads[layer_key] += param.grad.data.norm(2).item() ** 2
            layer_counts[layer_key] += 1

    return {k: (v ** 0.5) / max(layer_counts[k], 1)
            for k, v in layer_grads.items()}


@torch.no_grad()
def score_samples(model, tokenizer, prompts, gold_answers, max_new_tokens):
    """Generate answers and compute binary rewards."""
    rewards = []
    generated_texts = []

    for prompt, gold in zip(prompts, gold_answers):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = outputs[:, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        generated_texts.append(text)

        pred = extract_answer(text)
        gold_clean = extract_answer(gold)
        try:
            correct = abs(float(pred) - float(gold_clean)) < 1e-3 if pred and gold_clean else False
        except ValueError:
            correct = pred.strip() == gold_clean.strip()
        rewards.append(1.0 if correct else 0.0)

    return rewards, generated_texts


def compute_policy_gradient_for_sample(model, tokenizer, prompt, generated_text,
                                       reward, baseline_reward):
    """Compute the policy gradient for a single sample and return flattened grad."""
    model.zero_grad()

    full_text = prompt + generated_text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    outputs = model(**inputs, labels=inputs["input_ids"])
    nll_loss = outputs.loss

    advantage = reward - baseline_reward
    policy_loss = -advantage * nll_loss
    policy_loss.backward()

    grad_vector = []
    for p in model.parameters():
        if p.grad is not None:
            grad_vector.append(p.grad.data.flatten())
    return torch.cat(grad_vector) if grad_vector else torch.tensor([0.0])


def analyze_gradient_distribution(model, tokenizer, dataset, args):
    """Main gradient analysis: compare 0-score vs non-0-score gradient properties."""
    prompts = []
    gold_answers = []
    for i, example in enumerate(dataset):
        if i >= args.num_samples:
            break
        prompt = (
            "Solve the following math problem step by step. "
            "Show your work and put the final answer after ####.\n\n"
            f"Question: {example['question']}\n\nAnswer:"
        )
        prompts.append(prompt)
        gold_answers.append(example["answer"])

    logger.info("Scoring %d samples...", len(prompts))
    rewards, generated = score_samples(
        model, tokenizer, prompts, gold_answers, args.max_new_tokens
    )

    rewards_t = torch.tensor(rewards)
    zero_mask = rewards_t == 0.0
    baseline = rewards_t.mean().item()

    logger.info("Reward distribution: mean=%.3f, zero_ratio=%.3f",
                baseline, zero_mask.float().mean().item())

    zero_grad_norms = []
    nonzero_grad_norms = []
    zero_layer_grads = defaultdict(list)
    nonzero_layer_grads = defaultdict(list)
    zero_grads_flat = []
    nonzero_grads_flat = []

    max_analysis = min(50, len(prompts))
    logger.info("Computing per-sample gradients for %d samples...", max_analysis)

    for i in tqdm(range(max_analysis), desc="Gradient computation"):
        grad_vec = compute_policy_gradient_for_sample(
            model, tokenizer, prompts[i], generated[i],
            rewards[i], baseline
        )
        grad_norm = grad_vec.norm(2).item()

        layer_g = collect_layer_gradients(model)

        if rewards[i] == 0.0:
            zero_grad_norms.append(grad_norm)
            for k, v in layer_g.items():
                zero_layer_grads[k].append(v)
            if len(zero_grads_flat) < 10:
                zero_grads_flat.append(grad_vec.cpu())
        else:
            nonzero_grad_norms.append(grad_norm)
            for k, v in layer_g.items():
                nonzero_layer_grads[k].append(v)
            if len(nonzero_grads_flat) < 10:
                nonzero_grads_flat.append(grad_vec.cpu())

    analysis = {
        "num_samples": len(prompts),
        "num_analyzed": max_analysis,
        "reward_mean": baseline,
        "zero_score_ratio": zero_mask.float().mean().item(),
    }

    if zero_grad_norms:
        analysis["zero_score_grads"] = {
            "count": len(zero_grad_norms),
            "mean_norm": float(np.mean(zero_grad_norms)),
            "std_norm": float(np.std(zero_grad_norms)),
            "median_norm": float(np.median(zero_grad_norms)),
            "max_norm": float(np.max(zero_grad_norms)),
        }
    if nonzero_grad_norms:
        analysis["nonzero_score_grads"] = {
            "count": len(nonzero_grad_norms),
            "mean_norm": float(np.mean(nonzero_grad_norms)),
            "std_norm": float(np.std(nonzero_grad_norms)),
            "median_norm": float(np.median(nonzero_grad_norms)),
            "max_norm": float(np.max(nonzero_grad_norms)),
        }

    if zero_grad_norms and nonzero_grad_norms:
        analysis["norm_ratio"] = (
            float(np.mean(zero_grad_norms)) / max(float(np.mean(nonzero_grad_norms)), 1e-8)
        )

    if zero_grads_flat and nonzero_grads_flat:
        mean_zero = torch.stack(zero_grads_flat).mean(dim=0)
        mean_nonzero = torch.stack(nonzero_grads_flat).mean(dim=0)
        cos_sim = F.cosine_similarity(mean_zero.unsqueeze(0), mean_nonzero.unsqueeze(0))
        analysis["mean_gradient_cosine_similarity"] = cos_sim.item()

    all_layers = set(list(zero_layer_grads.keys()) + list(nonzero_layer_grads.keys()))
    layer_analysis = {}
    for layer in sorted(all_layers):
        la = {}
        if layer in zero_layer_grads:
            la["zero_mean"] = float(np.mean(zero_layer_grads[layer]))
        if layer in nonzero_layer_grads:
            la["nonzero_mean"] = float(np.mean(nonzero_layer_grads[layer]))
        if "zero_mean" in la and "nonzero_mean" in la:
            la["ratio"] = la["zero_mean"] / max(la["nonzero_mean"], 1e-8)
        layer_analysis[layer] = la
    analysis["per_layer_analysis"] = layer_analysis

    return analysis


def analyze_strategy_effects(base_analysis: dict, args) -> dict:
    """Analyze theoretical impact of each zero-score strategy on gradients."""
    results = {}
    zero_mean = base_analysis.get("zero_score_grads", {}).get("mean_norm", 0)
    nonzero_mean = base_analysis.get("nonzero_score_grads", {}).get("mean_norm", 0)

    for strategy in args.strategies:
        cfg = ZeroScoreConfig(strategy=ZeroScoreStrategy(strategy))
        handler = ZeroScoreHandler(cfg)

        sim_advantages = torch.randn(100)
        sim_rewards = torch.zeros(100)
        sim_rewards[:30] = 1.0

        modified = handler.reweight_advantages(sim_advantages, sim_rewards, global_step=250)

        zero_mask = sim_rewards == 0.0
        results[strategy] = {
            "original_zero_adv_mean": sim_advantages[zero_mask].abs().mean().item(),
            "modified_zero_adv_mean": modified[zero_mask].abs().mean().item(),
            "original_nonzero_adv_mean": sim_advantages[~zero_mask].abs().mean().item(),
            "modified_nonzero_adv_mean": modified[~zero_mask].abs().mean().item(),
            "effective_zero_grad_norm_estimate": zero_mean * (
                modified[zero_mask].abs().mean().item() /
                max(sim_advantages[zero_mask].abs().mean().item(), 1e-8)
            ),
        }

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading model: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    patch_model_instance(model)

    logger.info("Loading GSM8K dataset")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    logger.info("Starting gradient analysis")
    base_analysis = analyze_gradient_distribution(model, tokenizer, dataset, args)

    logger.info("Analyzing strategy effects")
    strategy_analysis = analyze_strategy_effects(base_analysis, args)
    base_analysis["strategy_analysis"] = strategy_analysis

    output_path = os.path.join(args.output_dir, "gradient_analysis.json")
    with open(output_path, "w") as f:
        json.dump(base_analysis, f, indent=2)

    logger.info("=" * 60)
    logger.info("GRADIENT ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info("Zero-score ratio: %.3f", base_analysis.get("zero_score_ratio", 0))
    if "zero_score_grads" in base_analysis:
        zg = base_analysis["zero_score_grads"]
        logger.info("Zero-score grad norm: mean=%.4f, std=%.4f", zg["mean_norm"], zg["std_norm"])
    if "nonzero_score_grads" in base_analysis:
        ng = base_analysis["nonzero_score_grads"]
        logger.info("Non-zero grad norm: mean=%.4f, std=%.4f", ng["mean_norm"], ng["std_norm"])
    if "norm_ratio" in base_analysis:
        logger.info("Gradient norm ratio (zero/nonzero): %.4f", base_analysis["norm_ratio"])
    if "mean_gradient_cosine_similarity" in base_analysis:
        logger.info("Gradient direction cosine sim: %.4f",
                    base_analysis["mean_gradient_cosine_similarity"])

    logger.info("\nStrategy effects:")
    for strat, info in strategy_analysis.items():
        logger.info("  %s: zero_adv %.4f -> %.4f, est_grad_norm %.4f",
                    strat,
                    info["original_zero_adv_mean"],
                    info["modified_zero_adv_mean"],
                    info["effective_zero_grad_norm_estimate"])

    logger.info("Full results saved to %s", output_path)


if __name__ == "__main__":
    main()
