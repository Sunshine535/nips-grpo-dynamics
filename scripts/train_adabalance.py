#!/usr/bin/env python3
import argparse
import glob
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
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adabalance import AdaBalanceConfig, AdaBalanceController, AdaBalanceCallback
from src.rho_grpo import build_gsm8k_binary_reward_function
from src.rho_grpo_trainer import AdaBalanceGRPOTrainer
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance, ClearRopeDeltasCallback

apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_adabalance")


def find_latest_checkpoint(output_dir):
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    return ckpts[-1] if ckpts else None


def parse_args():
    parser = argparse.ArgumentParser(description="AdaBalance GRPO training")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--rho_init", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/rho_sweep.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto")
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


class AdaBalanceMetricsCallback(TrainerCallback):
    def __init__(self, controller: AdaBalanceController):
        self.controller = controller
        self.step_metrics = []
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        record = {
            "step": state.global_step,
            "elapsed_sec": round(time.time() - self.start_time, 1),
        }

        for key in [
            "loss", "reward/mean", "reward/std", "kl",
            "reward/accuracies/mean", "grad_norm",
        ]:
            if key in logs:
                record[key.replace("/", "_")] = logs[key]

        telemetry = self.controller.get_telemetry()
        record.update(telemetry)

        self.step_metrics.append(record)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.step_metrics, f, indent=2)

    def on_save(self, args, state, control, **kwargs):
        """Persist controller state inside each checkpoint for correct resume."""
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            path = os.path.join(ckpt_dir, "adabalance_state.json")
            with open(path, "w") as f:
                json.dump(self.controller.state_dict(), f, indent=2)
            logger.info(
                "Saved AdaBalance state to %s (step %d, rho=%.4f)",
                path, state.global_step, self.controller.rho,
            )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    from trl import GRPOConfig

    seed = args.seed
    model_name = args.model_name or cfg["model"]["name"]
    tcfg = cfg["training"]

    tag = f"adabalance_K{args.K}_tau{args.tau}_seed{seed}"
    output_dir = args.output_dir or os.path.join(cfg["output"]["adabalance_dir"], tag)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("AdaBalance training: K=%d, tau=%.2f, seed=%d", args.K, args.tau, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])
    dataset = dataset.map(format_gsm8k_prompt, remove_columns=dataset.column_names)

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
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        seed=seed,
        num_generations=tcfg["num_generations"],
        max_completion_length=tcfg["max_completion_length"],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        report_to="none",
        log_level="info",
    )

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map=None,
    )
    patch_model_instance(model)

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    ada_yaml = cfg.get("adabalance", {})
    ada_config = AdaBalanceConfig(
        K=args.K or ada_yaml.get("K", 50),
        tau=args.tau or ada_yaml.get("tau", 0.1),
        rho_init=args.rho_init or ada_yaml.get("rho_init", 1.0),
        warmup_steps=ada_yaml.get("warmup_steps", 50),
        rho_min_floor=ada_yaml.get("rho_min_floor", 0.1),
        rho_max_ceil=ada_yaml.get("rho_max_ceil", 10.0),
        history_window=ada_yaml.get("history_window", 200),
    )
    controller = AdaBalanceController(ada_config)

    reward_fn = build_gsm8k_binary_reward_function()
    group_size = tcfg["num_generations"]

    rope_cb = ClearRopeDeltasCallback()
    metrics_cb = AdaBalanceMetricsCallback(controller)

    trainer = AdaBalanceGRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        rho=ada_config.rho_init,
        degenerate_floor=0.0,
        controller=controller,
        group_size=group_size,
        kl_coef=tcfg["kl_coef"],
        clip_range=tcfg["clip_range"],
        callbacks=[metrics_cb, rope_cb],
    )

    resume_ckpt = None
    if args.resume_from_checkpoint != "none":
        if args.resume_from_checkpoint == "auto":
            resume_ckpt = find_latest_checkpoint(output_dir)
        else:
            resume_ckpt = args.resume_from_checkpoint

    if resume_ckpt:
        ada_state_path = os.path.join(resume_ckpt, "adabalance_state.json")
        if os.path.exists(ada_state_path):
            with open(ada_state_path) as f:
                controller.load_state_dict(json.load(f))
            logger.info("Restored AdaBalance state from %s", ada_state_path)

    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    ada_state_path = os.path.join(output_dir, "adabalance_state.json")
    with open(ada_state_path, "w") as f:
        json.dump(controller.state_dict(), f, indent=2)

    final_metrics = {
        "method": "adabalance",
        "K": args.K,
        "tau": args.tau,
        "seed": seed,
        "model": model_name,
        "train_loss": train_result.metrics.get("train_loss"),
        "total_steps": train_result.metrics.get("train_steps", trainer.state.global_step),
        "final_rho": controller.rho,
        "final_p_hat": controller.p_hat_ema,
        "final_gsr": controller.gsr_ema,
    }

    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    metrics_cb.save(os.path.join(output_dir, "adabalance_telemetry.json"))

    if controller.bounds_history:
        with open(os.path.join(output_dir, "adabalance_bounds_history.json"), "w") as f:
            json.dump(controller.bounds_history, f, indent=2)

    logger.info("AdaBalance training complete. Final rho=%.4f", controller.rho)


if __name__ == "__main__":
    main()
