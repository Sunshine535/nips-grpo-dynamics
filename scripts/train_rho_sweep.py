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
from src.torch_compat import apply_torch_compat_patch
apply_torch_compat_patch()

from src.rho_grpo import RhoGRPOCallback, build_gsm8k_binary_reward_function
from src.rho_grpo_trainer import RhoGRPOTrainer, RhoStabilityCallback
from src.stability_analysis import (
    analyze_stability,
    classify_regime,
    group_starvation_rate,
    compute_gradient_variance,
)
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance, ClearRopeDeltasCallback

apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_rho_sweep")


def find_latest_checkpoint(output_dir):
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    return ckpts[-1] if ckpts else None


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training with rho parameterization")
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/rho_sweep.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto")
    parser.add_argument("--use_vllm", action="store_true", default=False,
                        help="Use vLLM for fast generation (requires vllm package)")
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
            {"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####. Do not use <think> tags."},
            {"role": "user", "content": f"Question: {example['question']}"},
        ],
        "answer": extract_gsm8k_answer(example["answer"]),
    }


class StabilityTelemetryCallback(TrainerCallback):
    def __init__(self, rho, group_size, kl_coef, clip_range):
        self.rho = rho
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.step_metrics = []
        self.start_time = time.time()
        self.reward_history = []
        self.kl_history = []
        self.initial_kl = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "rho": self.rho,
        }

        for key in [
            "loss", "train_loss", "reward/mean", "reward/std",
            "kl", "reward/accuracies/mean", "lr",
            "grad_norm", "reward/margins/mean",
        ]:
            if key in logs:
                record[key.replace("/", "_")] = logs[key]

        reward_mean = logs.get("reward/mean", logs.get("reward_mean", None))
        if reward_mean is not None:
            p_estimate = max(0.01, min(0.99, reward_mean))
            self.reward_history.append(p_estimate)

            gsr = group_starvation_rate(p_estimate, self.group_size)
            record["p_0"] = gsr
            record["GSR"] = gsr

            bounds = analyze_stability(
                p_estimate, self.group_size,
                self.kl_coef, self.clip_range,
            )
            record["rho_min"] = bounds.rho_min
            record["rho_max"] = bounds.rho_max
            record["rho_star"] = bounds.rho_star
            record["V_plus"] = bounds.V_plus
            record["V_minus"] = bounds.V_minus
            record["C_pG"] = bounds.C_pG
            record["gradient_variance"] = compute_gradient_variance(self.rho, bounds)

            kl_val = logs.get("kl", 0)
            self.kl_history.append(kl_val)
            if self.initial_kl is None and kl_val > 0:
                self.initial_kl = kl_val

            kl_ratio = kl_val / self.initial_kl if self.initial_kl and self.initial_kl > 0 else 1.0
            regime = classify_regime(
                self.rho, bounds,
                kl_ratio=kl_ratio,
            )
            record["regime"] = regime

        self.step_metrics.append(record)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.step_metrics, f, indent=2)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    from trl import GRPOConfig

    rho = args.rho
    seed = args.seed
    model_name = args.model_name or cfg["model"]["name"]
    tcfg = cfg["training"]

    output_dir = args.output_dir or os.path.join(
        cfg["output"]["sweep_coarse_dir"],
        f"rho{rho:.2f}_seed{seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Training: rho=%.2f, seed=%d", rho, seed)
    logger.info("Model: %s", model_name)
    logger.info("Output: %s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])
    dataset = dataset.map(format_gsm8k_prompt, remove_columns=dataset.column_names)
    logger.info("Dataset: %d samples", len(dataset))

    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = cfg["sweep"].get("coarse_max_steps", -1)

    # vLLM configuration for fast generation
    vllm_kwargs = {}
    if args.use_vllm:
        vllm_server_url = os.environ.get("VLLM_SERVER_URL", "")
        if vllm_server_url:
            # Server mode: vLLM runs as external server (supports TP across GPUs)
            vllm_kwargs = {
                "use_vllm": True,
                "vllm_mode": "server",
                "vllm_server_base_url": vllm_server_url,
            }
            logger.info("Using vLLM server mode: %s", vllm_server_url)
        else:
            # Colocate mode: vLLM runs inside training process
            vllm_port = int(os.environ.get("VLLM_PORT", 51216))
            tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
            vllm_kwargs = {
                "use_vllm": True,
                "vllm_mode": "colocate",
                "vllm_gpu_memory_utilization": 0.35,
                "vllm_group_port": vllm_port,
                "vllm_tensor_parallel_size": tp_size,
            }
            logger.info("Using vLLM colocate (port=%d, tp=%d)", vllm_port, tp_size)

    training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or tcfg["num_train_epochs"],
        per_device_train_batch_size=args.per_device_train_batch_size or tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=args.gradient_accumulation_steps or tcfg["gradient_accumulation_steps"],
        learning_rate=args.learning_rate or tcfg["learning_rate"],
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
        max_steps=max_steps if max_steps > 0 else -1,
        report_to="none",
        log_level="info",
        log_on_each_node=False,
        ddp_timeout=7200,  # 2h timeout for vLLM colocate + DDP
        **vllm_kwargs,
    )

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    logger.info("Loading model: %s (bf16, attn=%s)", model_name, attn_impl)
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

    reward_fn = build_gsm8k_binary_reward_function()
    group_size = tcfg["num_generations"]

    rope_cb = ClearRopeDeltasCallback()

    trainer = RhoGRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        rho=rho,
        degenerate_floor=0.0,
        callbacks=[rope_cb],
    )

    stability_cb = RhoStabilityCallback(
        trainer, group_size, tcfg["kl_coef"], tcfg["clip_range"],
    )
    trainer.add_callback(stability_cb)

    resume_ckpt = None
    if args.resume_from_checkpoint != "none":
        if args.resume_from_checkpoint == "auto":
            resume_ckpt = find_latest_checkpoint(output_dir)
        else:
            resume_ckpt = args.resume_from_checkpoint
        if resume_ckpt:
            logger.info("Resuming from checkpoint: %s", resume_ckpt)

    logger.info("Starting training (rho=%.2f)...", rho)
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "rho": rho,
        "seed": seed,
        "model": model_name,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "total_steps": train_result.metrics.get("train_steps", trainer.state.global_step),
    }

    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if stability_cb.telemetry:
        with open(os.path.join(output_dir, "stability_telemetry.json"), "w") as f:
            json.dump(stability_cb.telemetry, f, indent=2)

    if trainer._rho_step_stats:
        with open(os.path.join(output_dir, "rho_grpo_logs.json"), "w") as f:
            json.dump(trainer._rho_step_stats, f, indent=2)

    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        with open(os.path.join(output_dir, "step_logs.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    logger.info("Training complete: rho=%.2f, seed=%d", rho, seed)


if __name__ == "__main__":
    main()
