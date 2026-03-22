#!/usr/bin/env python3
"""
GRPO training with configurable α (positive_ratio) and β (negative_weight).

Model: Qwen/Qwen3.5-9B on GSM8K via TRL GRPOTrainer.
Tracks per-step: accuracy, reward mean/std, KL divergence, gradient norms.
Saves checkpoint + training_metrics.json to output_dir.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.balanced_grpo import BalancedGRPOCallback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_grpo_sweep")


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training (single α,β point)")
    parser.add_argument("--positive_ratio", type=float, required=True,
                        help="α: weight on positive (correct) signals")
    parser.add_argument("--negative_weight", type=float, required=True,
                        help="β: multiplier on negative signal weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/sweep_grid.yaml")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model (default from config)")
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def build_reward_function(positive_ratio: float, negative_weight: float):
    """Reward function that applies balanced α/β weighting."""
    alpha = positive_ratio
    beta = negative_weight

    def reward_fn(completions, answer, **kwargs):
        rewards = []
        for completion, gold in zip(completions, answer):
            if isinstance(completion, list):
                text = completion[0].get("content", "") if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", str(completion))
            else:
                text = str(completion)

            pred = extract_gsm8k_answer(text)
            gold_clean = str(gold).strip()

            if pred == gold_clean:
                rewards.append(1.0 * alpha)
            else:
                rewards.append(-1.0 * (1.0 - alpha) * beta)
        return rewards

    return reward_fn


def format_gsm8k_prompt(example):
    return {
        "prompt": [
            {"role": "user", "content": (
                f"Solve the following math problem step by step. "
                f"End with '#### <answer>'.\n\n"
                f"Question: {example['question']}"
            )},
        ],
        "answer": extract_gsm8k_answer(example["answer"]),
    }


class MetricsCallback(TrainerCallback):
    """Track extended metrics per logging step."""

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.step_metrics = []
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "alpha": self.alpha,
            "beta": self.beta,
        }

        for key in ["loss", "train_loss", "reward/mean", "reward/std",
                     "kl", "reward/accuracies/mean", "lr",
                     "grad_norm", "reward/margins/mean"]:
            if key in logs:
                record[key.replace("/", "_")] = logs[key]

        self.step_metrics.append(record)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.step_metrics, f, indent=2)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    from trl import GRPOConfig, GRPOTrainer

    alpha = args.positive_ratio
    beta = args.negative_weight
    seed = args.seed
    model_name = args.model_name or cfg["model"]["name"]

    output_dir = args.output_dir or os.path.join(
        cfg["output"]["checkpoint_dir"],
        f"alpha{alpha:.2f}_beta{beta:.2f}_seed{seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Training: alpha=%.2f, beta=%.2f, seed=%d", alpha, beta, seed)
    logger.info("Model: %s", model_name)
    logger.info("Output: %s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])
    dataset = dataset.map(format_gsm8k_prompt, remove_columns=dataset.column_names)
    logger.info("Dataset: %d samples", len(dataset))

    tcfg = cfg["training"]
    training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or tcfg["num_train_epochs"],
        per_device_train_batch_size=(
            args.per_device_train_batch_size or tcfg["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps or tcfg["gradient_accumulation_steps"]
        ),
        learning_rate=args.learning_rate or tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        seed=seed,
        num_generations=tcfg["num_generations"],
        max_completion_length=tcfg["max_completion_length"],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        report_to="none",
        log_level="info",
        log_on_each_node=False,
    )

    reward_fn = build_reward_function(alpha, beta)
    metrics_cb = MetricsCallback(alpha, beta)
    balanced_cb = BalancedGRPOCallback(alpha, beta)

    trainer = GRPOTrainer(
        model=model_name,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=[metrics_cb, balanced_cb],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)

    # Save training metrics summary
    metrics = {
        "positive_ratio": alpha,
        "negative_weight": beta,
        "seed": seed,
        "model": model_name,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "total_steps": train_result.metrics.get("train_steps", trainer.state.global_step),
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save step-level logs
    metrics_cb.save(os.path.join(output_dir, "step_metrics.json"))

    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        with open(os.path.join(output_dir, "step_logs.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    # Save balanced GRPO callback metrics
    if balanced_cb.step_metrics:
        with open(os.path.join(output_dir, "balanced_grpo_logs.json"), "w") as f:
            json.dump(balanced_cb.step_metrics, f, indent=2)

    logger.info("Training complete: alpha=%.2f, beta=%.2f", alpha, beta)


if __name__ == "__main__":
    main()
