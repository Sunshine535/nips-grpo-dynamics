#!/usr/bin/env python3
"""
GRPO variant comparison experiments for stability analysis.

Implements four GRPO variants as compute_loss modifications:
  - vanilla:  Standard GRPO (no modifications, baseline)
  - dapo:     Dynamic Sampling + Clip-Higher (asymmetric upper clip 0.28)
  - gspo:     Sequence-level Policy Optimization (sequence-level ratio approx)
  - gtpo:     Gradient Truncation PO (skip degenerate groups entirely)

Each variant modifies the advantage/loss computation in a way that maps
onto a specific region of our (rho, p, G) stability framework.

Usage:
    python scripts/train_grpo_variants.py --variant dapo --rho 1.0 --seed 42
    python scripts/train_grpo_variants.py --variant gtpo --rho 1.0 --seed 42 --max_steps 200
"""

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
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rho_grpo import build_gsm8k_binary_reward_function, compute_group_statistics
from src.stability_analysis import (
    analyze_stability,
    classify_regime,
    compute_gradient_variance,
    group_starvation_rate,
)
from src.qwen35_compat import (
    apply_qwen35_text_only_patch,
    patch_model_instance,
    ClearRopeDeltasCallback,
)

apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_grpo_variants")

VARIANT_CHOICES = ["vanilla", "dapo", "gspo", "gtpo"]


# ---------------------------------------------------------------------------
# Variant trainer
# ---------------------------------------------------------------------------

class VariantGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass that applies variant-specific modifications inside
    compute_loss.  The variant is selected once at init time; the corresponding
    transform is applied to the advantages tensor every training step.

    Variant semantics:
      vanilla  - pass-through (standard GRPO)
      dapo     - asymmetric clipping (clip_higher=0.28) + skip zero-std groups
      gspo     - approximate sequence-level importance ratio
      gtpo     - zero out advantages for degenerate groups (m=0 or m=G)
    """

    def __init__(
        self,
        *args,
        variant: str = "vanilla",
        rho: float = 1.0,
        group_size: int = 4,
        dapo_clip_higher: float = 0.28,
        dapo_clip_lower: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if variant not in VARIANT_CHOICES:
            raise ValueError(f"Unknown variant '{variant}', choose from {VARIANT_CHOICES}")
        self._variant = variant
        self._rho = rho
        self._group_size = group_size
        self._dapo_clip_higher = dapo_clip_higher
        self._dapo_clip_lower = dapo_clip_lower

        # Per-step bookkeeping for variant-specific metrics
        self._variant_step_stats: list[dict] = []

    # -- helpers -------------------------------------------------------------

    def _detect_degenerate_groups(self, advantages: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask (per sample) that is True for degenerate groups.

        A group is degenerate when all its advantages are identical (zero
        variance), which happens when m=0 or m=G in binary-reward GRPO.
        """
        G = self._group_size
        n_samples = advantages.shape[0]
        n_groups = n_samples // G
        if n_groups == 0:
            return torch.zeros(n_samples, dtype=torch.bool, device=advantages.device)

        truncated = advantages[: n_groups * G]
        grouped = truncated.view(n_groups, G)

        # A group is degenerate if all members share the same advantage value
        # (typically all zeros after normalization).
        group_std = grouped.std(dim=1)
        is_degen = group_std < 1e-7  # (n_groups,)

        # Expand back to per-sample mask
        per_sample_degen = is_degen.unsqueeze(1).expand(-1, G).reshape(-1)

        # Handle remainder samples (not part of a full group) -- mark as non-degenerate
        if n_samples > n_groups * G:
            remainder = torch.zeros(
                n_samples - n_groups * G,
                dtype=torch.bool,
                device=advantages.device,
            )
            per_sample_degen = torch.cat([per_sample_degen, remainder])

        return per_sample_degen

    # -- variant transforms --------------------------------------------------

    def _apply_vanilla(self, advantages: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, dict]:
        """Standard GRPO -- no modification."""
        return advantages, {"variant_op": "none"}

    def _apply_dapo(self, advantages: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, dict]:
        """
        DAPO: Dynamic sampling + Clip-Higher.

        1. Zero out advantages for degenerate groups (dynamic sampling approx:
           instead of re-sampling, we simply skip these groups by zeroing their
           gradient contribution).
        2. Widen the upper clipping bound from epsilon to clip_higher (0.28)
           while keeping the lower bound at clip_lower (0.2).  We encode this
           as an asymmetric rescaling of positive advantages so that the
           effective upper clip is wider.

        The asymmetric clip is applied by scaling positive advantages up by
        clip_higher / clip_lower, which has the same directional effect as
        widening the upper clip range in the surrogate objective.
        """
        degen_mask = self._detect_degenerate_groups(advantages)
        n_skipped = int(degen_mask.sum().item())

        modified = advantages.clone()
        # Zero degenerate groups (dynamic sampling proxy)
        modified[degen_mask] = 0.0

        # Asymmetric clip-higher: scale positive advantages to widen effective
        # upper clip range.  ratio > 1 means the upper clip binds less often.
        clip_ratio = self._dapo_clip_higher / max(self._dapo_clip_lower, 1e-8)
        pos_mask = modified > 0
        modified[pos_mask] = modified[pos_mask] * clip_ratio
        n_clip_scaled = int(pos_mask.sum().item())

        stats = {
            "variant_op": "dapo",
            "n_skipped_degen_groups": n_skipped,
            "clip_higher": self._dapo_clip_higher,
            "clip_lower": self._dapo_clip_lower,
            "clip_ratio": clip_ratio,
            "n_clip_higher_scaled": n_clip_scaled,
        }
        return modified, stats

    def _apply_gspo(self, advantages: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, dict]:
        """
        GSPO: Sequence-level Policy Optimization (approximation).

        Instead of per-token importance ratios, GSPO uses a sequence-level
        ratio.  We approximate this by averaging advantages within each group
        and assigning the group mean to every token in that group.  This
        smooths the per-token noise and approximates sequence-level credit.

        The effect: within a group, every token sees the same advantage signal,
        which reduces token-level variance and is analogous to using a single
        sequence-level ratio.
        """
        G = self._group_size
        n_samples = advantages.shape[0]
        n_groups = n_samples // G

        modified = advantages.clone()

        if n_groups > 0:
            grouped = modified[: n_groups * G].view(n_groups, G)
            group_means = grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)
            # Replace per-sample advantages with group mean
            modified[: n_groups * G] = group_means.expand(-1, G).reshape(-1)

        orig_std = float(advantages.std().item()) if advantages.numel() > 0 else 0.0
        new_std = float(modified.std().item()) if modified.numel() > 0 else 0.0

        stats = {
            "variant_op": "gspo",
            "n_groups": n_groups,
            "advantage_std_before": orig_std,
            "advantage_std_after": new_std,
            "variance_reduction_ratio": new_std / max(orig_std, 1e-8),
        }
        return modified, stats

    def _apply_gtpo(self, advantages: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, dict]:
        """
        GTPO: Gradient Truncation Policy Optimization.

        Skip degenerate groups entirely -- zero out their advantages so no
        gradient flows through groups where m=0 or m=G.  This is the cleanest
        approximation: degenerate groups provide no learning signal, so
        removing them reduces gradient noise.
        """
        degen_mask = self._detect_degenerate_groups(advantages)
        n_total = advantages.shape[0]
        n_degen = int(degen_mask.sum().item())
        n_active = n_total - n_degen

        modified = advantages.clone()
        modified[degen_mask] = 0.0

        # Optionally re-scale surviving advantages to keep gradient magnitude
        # stable despite dropping some groups.
        if n_active > 0 and n_degen > 0:
            scale_factor = n_total / n_active
            modified[~degen_mask] = modified[~degen_mask] * scale_factor
        else:
            scale_factor = 1.0

        stats = {
            "variant_op": "gtpo",
            "n_degenerate_samples": n_degen,
            "n_active_samples": n_active,
            "degenerate_ratio": n_degen / max(n_total, 1),
            "rescale_factor": scale_factor,
        }
        return modified, stats

    # -- main override -------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "advantages" in inputs and isinstance(inputs.get("advantages"), torch.Tensor):
            advantages = inputs["advantages"]

            # Dispatch to variant-specific transform
            dispatch = {
                "vanilla": self._apply_vanilla,
                "dapo": self._apply_dapo,
                "gspo": self._apply_gspo,
                "gtpo": self._apply_gtpo,
            }
            transform_fn = dispatch[self._variant]
            modified_advantages, variant_stats = transform_fn(advantages, inputs)

            # Collect common stats
            pos_mask = modified_advantages > 0
            neg_mask = modified_advantages < 0
            zero_mask = modified_advantages == 0
            step = self.state.global_step if self.state else 0

            common_stats = {
                "step": step,
                "variant": self._variant,
                "rho": self._rho,
                "n_positive": int(pos_mask.sum().item()),
                "n_negative": int(neg_mask.sum().item()),
                "n_zero": int(zero_mask.sum().item()),
                "mean_pos_adv": float(modified_advantages[pos_mask].mean().item()) if pos_mask.any() else 0.0,
                "mean_neg_adv": float(modified_advantages[neg_mask].mean().item()) if neg_mask.any() else 0.0,
                "adv_std": float(modified_advantages.std().item()) if modified_advantages.numel() > 1 else 0.0,
            }
            common_stats.update(variant_stats)
            self._variant_step_stats.append(common_stats)

            # Replace advantages in the inputs dict
            inputs = dict(inputs)
            inputs["advantages"] = modified_advantages

        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


# ---------------------------------------------------------------------------
# Telemetry callback (same structure as train_rho_sweep.py)
# ---------------------------------------------------------------------------

class VariantTelemetryCallback(TrainerCallback):
    """Log stability diagnostics alongside variant-specific metrics."""

    def __init__(self, variant, rho, group_size, kl_coef, clip_range):
        self.variant = variant
        self.rho = rho
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.step_metrics: list[dict] = []
        self.start_time = time.time()
        self.initial_kl = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "variant": self.variant,
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

            gsr = group_starvation_rate(p_estimate, self.group_size)
            record["GSR"] = gsr

            bounds = analyze_stability(
                p_estimate, self.group_size,
                self.kl_coef, self.clip_range,
            )
            record["rho_min"] = bounds.rho_min
            record["rho_max"] = bounds.rho_max
            record["rho_star"] = bounds.rho_star
            record["gradient_variance"] = compute_gradient_variance(self.rho, bounds)

            kl_val = logs.get("kl", 0)
            if self.initial_kl is None and kl_val > 0:
                self.initial_kl = kl_val
            kl_ratio = kl_val / self.initial_kl if self.initial_kl and self.initial_kl > 0 else 1.0
            regime = classify_regime(self.rho, bounds, kl_ratio=kl_ratio)
            record["regime"] = regime

        self.step_metrics.append(record)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.step_metrics, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_latest_checkpoint(output_dir):
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    return ckpts[-1] if ckpts else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO variant comparison experiments",
    )
    parser.add_argument(
        "--variant", type=str, required=True, choices=VARIANT_CHOICES,
        help="GRPO variant to train",
    )
    parser.add_argument("--rho", type=float, default=1.0,
                        help="Rho asymmetry parameter (applied on top of variant)")
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
    # DAPO-specific
    parser.add_argument("--dapo_clip_higher", type=float, default=0.28,
                        help="DAPO: upper clip bound (default 0.28)")
    parser.add_argument("--dapo_clip_lower", type=float, default=0.2,
                        help="DAPO: lower clip bound (default 0.2)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    import yaml
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
                f"Put your final numerical answer after ####.\n\n"
                f"Question: {example['question']}"
            )},
        ],
        "answer": extract_gsm8k_answer(example["answer"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    variant = args.variant
    rho = args.rho
    seed = args.seed
    model_name = args.model_name or cfg["model"]["name"]
    tcfg = cfg["training"]

    # Output directory: results/variants/<variant>_rho<rho>_seed<seed>
    output_dir = args.output_dir or os.path.join(
        cfg["output"]["base_dir"],
        "variants",
        f"{variant}_rho{rho:.2f}_seed{seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== GRPO Variant Experiment ===")
    logger.info("Variant: %s", variant)
    logger.info("Rho: %.2f, Seed: %d", rho, seed)
    logger.info("Model: %s", model_name)
    logger.info("Output: %s", output_dir)

    # -- tokenizer -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- dataset -------------------------------------------------------------
    dataset = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])
    dataset = dataset.map(format_gsm8k_prompt, remove_columns=dataset.column_names)
    logger.info("Dataset: %d samples", len(dataset))

    # -- training config -----------------------------------------------------
    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = cfg["sweep"].get("coarse_max_steps", -1)

    # vLLM configuration
    vllm_kwargs = {}
    if args.use_vllm:
        vllm_server_url = os.environ.get("VLLM_SERVER_URL", "")
        if vllm_server_url:
            vllm_kwargs = {
                "use_vllm": True,
                "vllm_mode": "server",
                "vllm_server_base_url": vllm_server_url,
            }
            logger.info("Using vLLM server mode: %s", vllm_server_url)
        else:
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

    group_size = tcfg["num_generations"]

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
        num_generations=group_size,
        max_completion_length=tcfg["max_completion_length"],
        max_steps=max_steps if max_steps > 0 else -1,
        report_to="none",
        log_level="info",
        log_on_each_node=False,
        ddp_timeout=7200,
        **vllm_kwargs,
    )

    # -- model ---------------------------------------------------------------
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
    rope_cb = ClearRopeDeltasCallback()

    # -- trainer (variant-specific) ------------------------------------------
    trainer = VariantGRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        variant=variant,
        rho=rho,
        group_size=group_size,
        dapo_clip_higher=args.dapo_clip_higher,
        dapo_clip_lower=args.dapo_clip_lower,
        callbacks=[rope_cb],
    )

    telemetry_cb = VariantTelemetryCallback(
        variant=variant,
        rho=rho,
        group_size=group_size,
        kl_coef=tcfg["kl_coef"],
        clip_range=tcfg["clip_range"],
    )
    trainer.add_callback(telemetry_cb)

    # -- resume from checkpoint ----------------------------------------------
    resume_ckpt = None
    if args.resume_from_checkpoint != "none":
        if args.resume_from_checkpoint == "auto":
            resume_ckpt = find_latest_checkpoint(output_dir)
        else:
            resume_ckpt = args.resume_from_checkpoint
        if resume_ckpt:
            logger.info("Resuming from checkpoint: %s", resume_ckpt)

    # -- train ---------------------------------------------------------------
    logger.info("Starting training (variant=%s, rho=%.2f)...", variant, rho)
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    # -- save ----------------------------------------------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # training_metrics.json
    metrics = {
        "variant": variant,
        "rho": rho,
        "seed": seed,
        "model": model_name,
        "dapo_clip_higher": args.dapo_clip_higher if variant == "dapo" else None,
        "dapo_clip_lower": args.dapo_clip_lower if variant == "dapo" else None,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "total_steps": train_result.metrics.get("train_steps", trainer.state.global_step),
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # variant_step_stats.json
    if trainer._variant_step_stats:
        with open(os.path.join(output_dir, "variant_step_stats.json"), "w") as f:
            json.dump(trainer._variant_step_stats, f, indent=2)

    # stability telemetry
    if telemetry_cb.step_metrics:
        telemetry_cb.save(os.path.join(output_dir, "stability_telemetry.json"))

    # step_logs.json (full trainer log history)
    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        with open(os.path.join(output_dir, "step_logs.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    logger.info(
        "Training complete: variant=%s, rho=%.2f, seed=%d, steps=%d",
        variant, rho, seed,
        train_result.metrics.get("train_steps", trainer.state.global_step),
    )


if __name__ == "__main__":
    main()
