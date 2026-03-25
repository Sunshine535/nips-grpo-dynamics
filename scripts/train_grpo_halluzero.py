#!/usr/bin/env python3
"""
GRPO training with zero-score gradient reshaping for hallucination reduction.

Uses TRL's GRPOTrainer with a custom reward function (binary correctness on GSM8K)
and overrides gradient computation to reweight 0-score samples.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.zero_score_handler import ZeroScoreConfig, ZeroScoreHandler, ZeroScoreStrategy
from src.qwen35_compat import (
    apply_qwen35_text_only_patch, patch_model_instance, ClearRopeDeltasCallback,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("halluzero")


def parse_args():
    parser = argparse.ArgumentParser(description="HalluZero GRPO Training")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--zero_score_strategy", type=str, default=None,
                        choices=["clip", "temperature", "curriculum", "relabel", "none"])
    parser.add_argument("--clip_factor", type=float, default=None)
    parser.add_argument("--temperature_boost", type=float, default=None)
    parser.add_argument("--curriculum_warmup_steps", type=int, default=None)
    parser.add_argument("--relabel_epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


ANS_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def extract_gsm8k_answer(text: str) -> str:
    """Extract the numeric answer after #### in GSM8K format."""
    match = ANS_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def extract_model_answer(text: str) -> str:
    """Extract the final numeric answer from model generation."""
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    match = ANS_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    answer_pattern = re.search(r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*(-?[\d,]+(?:\.\d+)?)", text)
    if answer_pattern:
        return answer_pattern.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def build_reward_fn(tokenizer):
    """Build a binary correctness reward function for GSM8K."""

    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [None] * len(completions))
        ground_truths = kwargs.get("ground_truth", [None] * len(completions))
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            if isinstance(completion, list):
                text = tokenizer.decode(completion, skip_special_tokens=True)
            else:
                text = completion

            pred = extract_model_answer(text)
            gold = extract_gsm8k_answer(gt) if gt else ""

            try:
                correct = abs(float(pred) - float(gold)) < 1e-3 if pred and gold else False
            except ValueError:
                correct = pred.strip() == gold.strip()

            rewards.append(1.0 if correct else 0.0)
        return rewards

    return reward_fn


def prepare_dataset(cfg: dict, tokenizer):
    """Load GSM8K and format prompts."""
    ds_cfg = cfg["dataset"]
    dataset = load_dataset(ds_cfg["name"], "main", split=ds_cfg["split"])

    def format_prompt(example):
        prompt = (
            "Solve the following math problem step by step. "
            "Show your work and put the final answer after ####.\n\n"
            f"Question: {example['question']}\n\nAnswer:"
        )
        return {"prompt": prompt, "ground_truth": example["answer"]}

    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    return dataset


class HalluZeroGRPOTrainer(GRPOTrainer):
    """GRPO Trainer with zero-score gradient reshaping."""

    def __init__(self, *args, zero_score_handler: ZeroScoreHandler = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_score_handler = zero_score_handler
        self._zero_score_enabled = zero_score_handler is not None
        self._step_rewards = None
        if self._zero_score_enabled:
            self._wrap_reward_funcs_for_step_rewards()

    def _wrap_reward_funcs_for_step_rewards(self):
        """Capture per-sample rewards when TRL calls reward funcs (inputs dict has no raw rewards)."""

        def wrap(fn):
            def wrapped(*a, **kw):
                out = fn(*a, **kw)
                self._step_rewards = torch.as_tensor(
                    out, dtype=torch.float32, device=self.accelerator.device
                )
                return out

            return wrapped

        rfs = self.reward_funcs
        if not isinstance(rfs, (list, tuple)):
            rfs = [rfs]
        wrapped = []
        for rf in rfs:
            if callable(rf) and not isinstance(rf, nn.Module):
                wrapped.append(wrap(rf))
            else:
                wrapped.append(rf)
        self.reward_funcs = wrapped

    def _compute_loss(self, model, inputs, **kwargs):
        loss = super()._compute_loss(model, inputs, **kwargs)

        if self._zero_score_enabled and self._step_rewards is not None:
            rewards = self._step_rewards
            zero_mask = rewards == 0.0
            if zero_mask.any():
                zero_ratio = zero_mask.float().mean().item()
                nonzero_ratio = 1.0 - zero_ratio

                if zero_ratio > 0.8:
                    loss = loss * 0.5
                elif zero_ratio > 0 and nonzero_ratio > 0:
                    scale = self.zero_score_handler.config.clip_factor
                    loss = loss * (nonzero_ratio + zero_ratio * scale)

        return loss


def main():
    args = parse_args()
    cfg = load_config(args.config_path)

    output_dir = args.output_dir or cfg["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading tokenizer and model: %s", cfg["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    requested_attn = cfg["model"].get("attn_implementation", "flash_attention_2")
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if requested_attn == "flash_attention_2":
            requested_attn = "sdpa"
            logger.warning("flash_attn not installed, falling back to %s", requested_attn)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        attn_implementation=requested_attn,
        trust_remote_code=True,
        device_map=None,
    )
    patch_model_instance(model)

    logger.info("Preparing dataset")
    dataset = prepare_dataset(cfg, tokenizer)

    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        try:
            import numpy as _np
            _np.random.seed(args.seed)
        except ImportError:
            pass
        logger.info("Random seed set to %d", args.seed)

    zs_cfg_raw = cfg.get("zero_score", {})
    strategy = args.zero_score_strategy or zs_cfg_raw.get("strategy", "curriculum")

    zs_handler = None
    if strategy != "none":
        zs_config = ZeroScoreConfig(
            strategy=ZeroScoreStrategy(strategy),
            clip_factor=args.clip_factor if args.clip_factor is not None
                        else zs_cfg_raw.get("clip_factor", 0.1),
            temperature_boost=args.temperature_boost if args.temperature_boost is not None
                              else zs_cfg_raw.get("temperature_boost", 1.5),
            curriculum_warmup_steps=args.curriculum_warmup_steps if args.curriculum_warmup_steps is not None
                                    else zs_cfg_raw.get("curriculum_warmup_steps", 500),
            relabel_epsilon=args.relabel_epsilon if args.relabel_epsilon is not None
                            else zs_cfg_raw.get("relabel_epsilon", 0.01),
        )
        zs_handler = ZeroScoreHandler(zs_config)
    logger.info("Zero-score strategy: %s", strategy)

    train_cfg = cfg["training"]
    grpo_cfg = cfg["grpo"]

    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg["num_epochs"]
    lr = args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"]

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=lr,
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=cfg["dataset"]["max_completion_length"],
        logging_steps=cfg["logging"]["logging_steps"],
        save_steps=cfg["logging"]["save_steps"],
        eval_steps=cfg["logging"]["eval_steps"],
        report_to=cfg["logging"].get("report_to", "none"),
        run_name=cfg["logging"].get("run_name", "halluzero"),
        dataloader_num_workers=4,
        seed=args.seed if args.seed is not None else 42,
        remove_unused_columns=False,
        log_completions=True,
    )

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    logger.info("Using LoRA: r=%d, alpha=%d", peft_config.r, peft_config.lora_alpha)

    reward_fn = build_reward_fn(tokenizer)

    trainer = HalluZeroGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        zero_score_handler=zs_handler,
        peft_config=peft_config,
        callbacks=[ClearRopeDeltasCallback()],
    )

    logger.info("Starting GRPO training with HalluZero modifications")
    train_result = trainer.train()

    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
