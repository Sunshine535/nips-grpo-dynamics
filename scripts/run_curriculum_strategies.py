#!/usr/bin/env python3
"""
Curriculum strategies for α/β annealing during GRPO training.

Implements three schedules:
  1. anneal-positive:  α starts high (0.9) → decays to 0.5
  2. anneal-negative:  β starts at 0.0  → increases to 1.0
  3. cosine-schedule:  both α, β follow cosine annealing

Each is compared against the best static (α, β) from the sweep.
Uses TRL GRPOTrainer with reward function that reads a mutable schedule.
"""

import argparse
import json
import logging
import math
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
from src.qwen35_compat import apply_qwen35_text_only_patch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("curriculum_strategies")


# ── Schedules ────────────────────────────────────────────────────────────

class Schedule:
    """Base class for α/β annealing schedules."""

    def __init__(self, total_steps: int):
        self.total_steps = max(total_steps, 1)
        self.current_alpha = 0.5
        self.current_beta = 1.0

    def step(self, global_step: int):
        raise NotImplementedError

    def get_alpha(self):
        return self.current_alpha

    def get_beta(self):
        return self.current_beta


class AnnealPositiveSchedule(Schedule):
    """α starts at α_max and linearly decays to α_min."""

    def __init__(self, total_steps, alpha_max=0.9, alpha_min=0.5, beta=1.0):
        super().__init__(total_steps)
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.current_beta = beta

    def step(self, global_step):
        frac = min(global_step / self.total_steps, 1.0)
        self.current_alpha = self.alpha_max - frac * (self.alpha_max - self.alpha_min)


class AnnealNegativeSchedule(Schedule):
    """β starts at β_min and linearly increases to β_max. α stays fixed."""

    def __init__(self, total_steps, alpha=0.5, beta_min=0.0, beta_max=1.0):
        super().__init__(total_steps)
        self.current_alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max

    def step(self, global_step):
        frac = min(global_step / self.total_steps, 1.0)
        self.current_beta = self.beta_min + frac * (self.beta_max - self.beta_min)


class CosineSchedule(Schedule):
    """Both α and β follow cosine annealing."""

    def __init__(self, total_steps,
                 alpha_start=0.9, alpha_end=0.5,
                 beta_start=0.0, beta_end=1.0):
        super().__init__(total_steps)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end

    def step(self, global_step):
        frac = min(global_step / self.total_steps, 1.0)
        cos_val = 0.5 * (1 + math.cos(math.pi * frac))
        self.current_alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * cos_val
        self.current_beta = self.beta_end + (self.beta_start - self.beta_end) * cos_val


class StaticSchedule(Schedule):
    """No annealing: constant α, β."""

    def __init__(self, total_steps, alpha=0.5, beta=1.0):
        super().__init__(total_steps)
        self.current_alpha = alpha
        self.current_beta = beta

    def step(self, global_step):
        pass


SCHEDULE_REGISTRY = {
    "anneal-positive": AnnealPositiveSchedule,
    "anneal-negative": AnnealNegativeSchedule,
    "cosine-schedule": CosineSchedule,
    "static": StaticSchedule,
}


# ── Reward and data ─────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def build_scheduled_reward(schedule: Schedule):
    """Reward function that reads α/β from a mutable schedule object."""

    def reward_fn(completions, answer, **kwargs):
        alpha = schedule.get_alpha()
        beta = schedule.get_beta()
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


class ScheduleUpdateCallback(TrainerCallback):
    """Update the schedule at each training step and log α/β."""

    def __init__(self, schedule: Schedule, strategy_name: str):
        self.schedule = schedule
        self.strategy_name = strategy_name
        self.log_records = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.schedule.step(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {
            "step": state.global_step,
            "alpha": round(self.schedule.get_alpha(), 4),
            "beta": round(self.schedule.get_beta(), 4),
            "strategy": self.strategy_name,
        }
        for key in ["loss", "reward/mean", "reward/std", "kl", "grad_norm"]:
            if key in logs:
                record[key.replace("/", "_")] = logs[key]
        self.log_records.append(record)
        logs[f"curriculum/alpha"] = self.schedule.get_alpha()
        logs[f"curriculum/beta"] = self.schedule.get_beta()


# ── Training function ────────────────────────────────────────────────────

def train_with_strategy(strategy_name, schedule, cfg, model_name, dataset,
                        tokenizer, output_dir, seed):
    from trl import GRPOConfig, GRPOTrainer

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Strategy: %s  (output: %s)", strategy_name, output_dir)

    tcfg = cfg["training"]
    training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=1,
        seed=seed,
        num_generations=tcfg["num_generations"],
        max_completion_length=tcfg["max_completion_length"],
        report_to="none",
        log_level="info",
    )

    reward_fn = build_scheduled_reward(schedule)
    schedule_cb = ScheduleUpdateCallback(schedule, strategy_name)

    trainer = GRPOTrainer(
        model=model_name,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=[schedule_cb],
    )

    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "strategy": strategy_name,
        "seed": seed,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "schedule_logs.json"), "w") as f:
        json.dump(schedule_cb.log_records, f, indent=2)

    return metrics, schedule_cb.log_records


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum Strategy Comparison")
    parser.add_argument("--config", type=str, default="configs/sweep_grid.yaml")
    parser.add_argument("--output_dir", type=str, default="./results/curriculum")
    parser.add_argument("--strategies", nargs="+",
                        default=["anneal-positive", "anneal-negative",
                                 "cosine-schedule", "static"],
                        choices=list(SCHEDULE_REGISTRY.keys()))
    parser.add_argument("--best_alpha", type=float, default=0.5,
                        help="Best static α from sweep (for static baseline)")
    parser.add_argument("--best_beta", type=float, default=1.0,
                        help="Best static β from sweep (for static baseline)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps_estimate", type=int, default=500,
                        help="Estimated total training steps for schedule")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])
    dataset = dataset.map(format_gsm8k_prompt, remove_columns=dataset.column_names)

    all_results = {}

    for strategy_name in args.strategies:
        strat_dir = os.path.join(args.output_dir, strategy_name)

        if os.path.exists(os.path.join(strat_dir, "training_metrics.json")):
            logger.info("Skipping %s (already trained)", strategy_name)
            with open(os.path.join(strat_dir, "training_metrics.json")) as f:
                all_results[strategy_name] = json.load(f)
            continue

        T = args.total_steps_estimate
        if strategy_name == "anneal-positive":
            schedule = AnnealPositiveSchedule(T, alpha_max=0.9, alpha_min=0.5,
                                              beta=args.best_beta)
        elif strategy_name == "anneal-negative":
            schedule = AnnealNegativeSchedule(T, alpha=args.best_alpha,
                                              beta_min=0.0, beta_max=1.0)
        elif strategy_name == "cosine-schedule":
            schedule = CosineSchedule(T, alpha_start=0.9, alpha_end=0.5,
                                      beta_start=0.0, beta_end=1.0)
        elif strategy_name == "static":
            schedule = StaticSchedule(T, alpha=args.best_alpha, beta=args.best_beta)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        metrics, logs = train_with_strategy(
            strategy_name, schedule, cfg, model_name, dataset,
            tokenizer, strat_dir, args.seed,
        )
        all_results[strategy_name] = metrics

    # Save comparison summary
    summary_path = os.path.join(args.output_dir, "curriculum_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Comparison saved to %s", summary_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("CURRICULUM STRATEGY COMPARISON")
    logger.info("=" * 60)
    for name, m in all_results.items():
        logger.info("  %-20s  loss=%.4f  runtime=%.0fs",
                     name, m.get("train_loss", 0) or 0,
                     m.get("train_runtime", 0) or 0)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
