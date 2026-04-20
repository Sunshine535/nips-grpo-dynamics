#!/usr/bin/env python3
"""
CSD Pilot Experiments — 3 pilots to validate CSD theory.

Pilot 1: CSD Equivalence — log CSD components during standard GRPO training
Pilot 2: ADQ Collapse Elimination — adaptive ρ* vs constant ρ=1.0
Pilot 3: Q_CSD Collapse Predictor — dense seeds at ρ=1.0, compare AUROC

Usage:
  # Run all 3 pilots sequentially on a single GPU
  python scripts/run_csd_pilot.py --pilot all --model Qwen/Qwen2.5-7B-Instruct

  # Run specific pilot
  python scripts/run_csd_pilot.py --pilot 1 --model Qwen/Qwen2.5-7B-Instruct
  python scripts/run_csd_pilot.py --pilot 2 --model Qwen/Qwen2.5-7B-Instruct --seeds 5
  python scripts/run_csd_pilot.py --pilot 3 --model Qwen/Qwen2.5-7B-Instruct --seeds 10

  # Quick test (fewer steps)
  QUICK=1 python scripts/run_csd_pilot.py --pilot all --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Offline + cache config for openbayes/tju-hpc (no internet)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
if os.path.isdir('/openbayes/input/input0/hub'):
    os.environ.setdefault('HF_HOME', '/openbayes/input/input0')
    os.environ.setdefault('HF_HUB_CACHE', '/openbayes/input/input0/hub')
    os.environ.setdefault('HF_DATASETS_CACHE', '/openbayes/input/input0/datasets')
    os.environ.setdefault('TRANSFORMERS_CACHE', '/openbayes/input/input0/hub')
elif os.path.isdir('/ytech_m2v4_hdd/mengzijie/.cache/hf'):
    os.environ.setdefault('HF_HOME', '/ytech_m2v4_hdd/mengzijie/.cache/hf')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.torch_compat import apply_torch_compat_patch
apply_torch_compat_patch()

from src.rho_grpo import build_gsm8k_binary_reward_function
from src.rho_grpo_trainer import RhoGRPOTrainer, AdaBalanceGRPOTrainer, RhoStabilityCallback
from src.adabalance import AdaBalanceController, AdaBalanceConfig
from src.csd_logging import CSDLoggingCallback
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance, ClearRopeDeltasCallback

apply_qwen35_text_only_patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("csd_pilot")

QUICK = os.environ.get("QUICK", "0") == "1"
DEFAULT_MAX_STEPS = 50 if QUICK else 200
DEFAULT_SEEDS_P2 = 3 if QUICK else 5
DEFAULT_SEEDS_P3 = 5 if QUICK else 10


def parse_args():
    p = argparse.ArgumentParser(description="CSD Pilot Experiments")
    p.add_argument("--pilot", type=str, default="all",
                   choices=["1", "2", "3", "all",
                            "1_single", "2_single", "3_single", "3_analyze"])
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--config", type=str, default="configs/rho_sweep.yaml")
    p.add_argument("--output_dir", type=str, default="results/csd_pilot")
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--seed_start", type=int, default=42, help="Starting seed for single-run mode")
    p.add_argument("--rho", type=float, default=1.0, help="ρ for Pilot 1 and 3")
    p.add_argument("--use_adq", action="store_true", help="Use ADQ (adaptive ρ) for single run")
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"],
                   help="Dataset to train on")
    p.add_argument("--group_size", type=int, default=None,
                   help="Override num_generations from config (must divide batch size)")
    p.add_argument("--max_completion_length", type=int, default=None,
                   help="Override max_completion_length from config")
    p.add_argument("--use_bandit", action="store_true",
                   help="Use UCB1 bandit over discrete ρ grid (Path A)")
    p.add_argument("--use_exact_rho", action="store_true",
                   help="Use exact-ρ* controller (per-group autograd.grad estimator)")
    p.add_argument("--exact_update_every", type=int, default=20,
                   help="How often (in main steps) to run the exact estimator")
    return p.parse_args()


def load_data_and_model(model_name, config_path, seed, max_steps, output_dir, use_vllm=False, dataset_name="gsm8k", group_size_override=None, max_completion_length_override=None):
    """Load dataset, model, tokenizer, and create GRPOConfig.

    dataset_name: "gsm8k" (default) or "math500".
    group_size_override: if not None, override tcfg["num_generations"] and batch size
       accordingly (we force per_device_train_batch_size so that bs × G is at least 2×G).
    max_completion_length_override: if not None, override tcfg["max_completion_length"].
    """
    from trl import GRPOConfig

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "gsm8k":
        dataset = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])
    elif dataset_name == "math500":
        # MATH-500 cached as datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl
        import glob, json as _json
        ds_paths = glob.glob(os.environ.get("HF_HUB_CACHE", "/openbayes/input/input0/hub")
                             + "/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl")
        assert ds_paths, f"MATH-500 test.jsonl not found"
        from datasets import Dataset
        rows = [_json.loads(l) for l in open(ds_paths[0])]
        dataset = Dataset.from_list(rows)
    else:
        raise ValueError(f"unknown dataset {dataset_name}")

    import re
    _pattern = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
    _boxed = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

    # Test if tokenizer supports enable_thinking=False (Qwen3.5 does)
    _supports_thinking_toggle = False
    try:
        test_msgs = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(test_msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        _supports_thinking_toggle = True
        logger.info("Tokenizer supports enable_thinking=False (Qwen3/3.5 thinking mode disabled)")
    except TypeError:
        logger.info("Tokenizer does not support enable_thinking param (standard model)")

    def format_prompt_gsm8k(example):
        answer_match = _pattern.search(example["answer"])
        answer = answer_match.group(1).replace(",", "") if answer_match else ""
        msgs = [
            {"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."},
            {"role": "user", "content": f"Question: {example['question']}"},
        ]
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if _supports_thinking_toggle:
            kwargs["enable_thinking"] = False
        prompt_str = tokenizer.apply_chat_template(msgs, **kwargs)
        return {"prompt": prompt_str, "answer": answer}

    def format_prompt_math500(example):
        # MATH-500 fields: problem, level, type, solution, answer
        gold_raw = example.get("answer") or ""
        if not gold_raw and "solution" in example:
            m = _boxed.search(example["solution"])
            gold_raw = m.group(1) if m else ""
        msgs = [
            {"role": "system", "content": "You are a math tutor. Solve the problem step by step and put the final answer in \\boxed{}."},
            {"role": "user", "content": example["problem"]},
        ]
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if _supports_thinking_toggle:
            kwargs["enable_thinking"] = False
        prompt_str = tokenizer.apply_chat_template(msgs, **kwargs)
        return {"prompt": prompt_str, "answer": str(gold_raw).strip()}

    fmt = format_prompt_gsm8k if dataset_name == "gsm8k" else format_prompt_math500
    dataset = dataset.map(fmt, remove_columns=dataset.column_names)

    vllm_kwargs = {}
    if use_vllm:
        vllm_port = int(os.environ.get("VLLM_PORT", 51216))
        tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
        vllm_kwargs = {
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": 0.35,
            "vllm_group_port": vllm_port,
            "vllm_tensor_parallel_size": tp_size,
        }

    effective_G = group_size_override if group_size_override is not None else tcfg["num_generations"]
    # per_device_train_batch_size must be >= num_generations (TRL constraint); scale up if needed.
    effective_bs = max(tcfg["per_device_train_batch_size"], effective_G)
    training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=effective_bs,
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        # CRITICAL: gradient_checkpointing=True corrupts rollout in TRL 0.14
        # → token loops (ToToTo). Always disable for GRPO with this stack.
        gradient_checkpointing=False,
        logging_steps=max(1, tcfg["logging_steps"]),
        save_steps=max_steps + 1,  # Don't save checkpoints in pilot
        save_total_limit=1,
        seed=seed,
        num_generations=effective_G,
        max_completion_length=max_completion_length_override if max_completion_length_override is not None else tcfg["max_completion_length"],
        max_steps=max_steps,
        report_to="none",
        log_level="info",
        log_on_each_node=False,
        **vllm_kwargs,
    )

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        # SDPA breaks on PyTorch 2.4 + transformers 5.x Qwen3.5 (non-contiguous bias)
        attn_impl = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True, device_map=None,
    )
    patch_model_instance(model)

    # Compat: TRL 0.14 expects `warnings_issued` dict (transformers 4.x) — stub for transformers 5.x
    if not hasattr(model, 'warnings_issued'):
        model.warnings_issued = {}

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type="CAUSAL_LM",
    )

    group_size = tcfg["num_generations"]
    kl_coef = tcfg["kl_coef"]
    clip_range = tcfg["clip_range"]

    return dataset, model, tokenizer, training_config, peft_config, group_size, kl_coef, clip_range


def run_single_training(model_name, config_path, seed, rho, max_steps, output_dir,
                        use_adq=False, use_vllm=False, dataset_name="gsm8k",
                        group_size_override=None, max_completion_length_override=None,
                        use_bandit=False, use_exact_rho=False,
                        exact_update_every=20):
    """Run a single training run with CSD logging."""
    if use_exact_rho:
        suffix = "_exact"
    elif use_bandit:
        suffix = "_bandit"
    elif use_adq:
        suffix = "_adq"
    else:
        suffix = ""
    if group_size_override:
        suffix = f"_G{group_size_override}" + suffix
    run_dir = os.path.join(output_dir, f"rho{rho:.2f}_seed{seed}{suffix}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=== Run: rho=%.2f, seed=%d, G=%s, adq=%s, dataset=%s ===",
                rho, seed, group_size_override or "cfg", use_adq, dataset_name)

    dataset, model, tokenizer, training_config, peft_config, G, kl_coef, clip_range = \
        load_data_and_model(model_name, config_path, seed, max_steps, run_dir, use_vllm,
                            dataset_name=dataset_name, group_size_override=group_size_override,
                            max_completion_length_override=max_completion_length_override)

    if dataset_name == "math500":
        from src.rho_grpo import build_math500_binary_reward_function
        reward_fn = build_math500_binary_reward_function()
    else:
        reward_fn = build_gsm8k_binary_reward_function()
    rope_cb = ClearRopeDeltasCallback()

    # Use V14 trainer (compatible with TRL 0.14's inline-advantage compute_loss)
    from src.rho_grpo_trainer_v14 import RhoGRPOTrainerV14
    controller = None
    bandit_controller = None
    if use_adq:
        ada_config = AdaBalanceConfig(
            K=10, tau=0.15, rho_init=rho, warmup_steps=10,
            rho_min_floor=0.3, rho_max_ceil=10.0,
        )
        controller = AdaBalanceController(ada_config)
    if use_bandit:
        from src.bandit_rho import UCBBanditRho, BanditRhoConfig
        bandit_controller = UCBBanditRho(BanditRhoConfig(
            rho_grid=[0.3, 0.7, 1.0, 2.0, 3.0],
            exploration_c=1.0,
            warmup_steps=5,
            update_every=1,
            reward_window=10,
        ))
    exact_controller = None
    if use_exact_rho:
        from src.exact_rho_controller import ExactRhoController, ExactRhoConfig
        exact_controller = ExactRhoController(ExactRhoConfig(
            update_every=exact_update_every, rho_init=rho,
            rho_min=0.1, rho_max=10.0, ema_alpha=0.7,
            min_groups_for_update=2,
        ))

    trainer = RhoGRPOTrainerV14(
        model=model, args=training_config, train_dataset=dataset,
        processing_class=tokenizer, reward_funcs=reward_fn,
        peft_config=peft_config,
        rho=rho,
        controller=controller,
        bandit_controller=bandit_controller,
        exact_controller=exact_controller,
        exact_update_every=exact_update_every,
        group_size_for_ada=G,
        ada_kl_coef=kl_coef,
        ada_clip_range=clip_range,
        callbacks=[rope_cb],
    )

    # Add CSD logging
    csd_cb = CSDLoggingCallback(group_size=G)
    csd_cb._trainer_ref = trainer
    trainer.add_callback(csd_cb)

    # Add stability logging
    stability_cb = RhoStabilityCallback(trainer, G, kl_coef, clip_range)
    trainer.add_callback(stability_cb)

    # Train
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save final LoRA adapter so downstream GSM8K eval has something to load.
    # The pilot config disables checkpoint saving during training, so we save
    # once here at the end.
    try:
        adapter_dir = os.path.join(run_dir, "checkpoint-final")
        trainer.save_model(adapter_dir)
    except Exception as e:
        logger.exception("save_model failed: %s", e)

    # Save results
    results = {
        "rho": rho, "seed": seed, "use_adq": use_adq,
        "max_steps": max_steps, "elapsed_sec": round(elapsed, 1),
        "model": model_name,
    }

    # Final reward from log history. TRL 0.14 logs under the plain "reward"
    # key (not "reward/mean"); fall back to "reward/mean" for older stacks.
    if trainer.state.log_history:
        reward_keys = ("reward", "reward/mean")
        final_rewards = []
        for l in trainer.state.log_history:
            for k in reward_keys:
                if k in l and l[k] is not None:
                    try:
                        final_rewards.append(float(l[k]))
                    except (TypeError, ValueError):
                        pass
                    break
        if final_rewards:
            results["final_reward_mean"] = round(final_rewards[-1], 4)
            results["max_reward_mean"] = round(max(final_rewards), 4)
            # Collapse detection: reward stays below 0.1 in last 20% of training
            late_start = max(0, len(final_rewards) - len(final_rewards) // 5)
            late_rewards = final_rewards[late_start:]
            results["collapsed"] = bool(float(np.mean(late_rewards)) < 0.1) if late_rewards else False
            results["n_reward_logs"] = len(final_rewards)

    def _coerce(o):
        """Recursively cast numpy scalars/arrays to JSON-safe Python types."""
        if isinstance(o, dict):
            return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_coerce(v) for v in o]
        if isinstance(o, np.ndarray):
            return _coerce(o.tolist())
        if isinstance(o, np.generic):
            return o.item()
        return o

    def _save_json(path, payload, label):
        try:
            with open(path, "w") as f:
                json.dump(_coerce(payload), f, indent=2)
        except Exception as e:  # never let one save kill the others
            logger.exception("[%s] save failed: %s", label, e)

    _save_json(os.path.join(run_dir, "pilot_results.json"),     results,                       "pilot_results")
    if csd_cb.csd_logs:
        _save_json(os.path.join(run_dir, "csd_logs.json"),       csd_cb.csd_logs,               "csd_logs")
    if stability_cb.telemetry:
        _save_json(os.path.join(run_dir, "stability_telemetry.json"), stability_cb.telemetry,    "stability_telemetry")
    if trainer._rho_step_stats:
        _save_json(os.path.join(run_dir, "rho_grpo_logs.json"),  trainer._rho_step_stats,       "rho_grpo_logs")
    if hasattr(trainer, "_ada_telemetry") and trainer._ada_telemetry:
        _save_json(os.path.join(run_dir, "ada_telemetry.json"),  trainer._ada_telemetry,        "ada_telemetry")
    if hasattr(trainer, "_bandit_telemetry") and trainer._bandit_telemetry:
        _save_json(os.path.join(run_dir, "bandit_telemetry.json"), trainer._bandit_telemetry,    "bandit_telemetry")
        if bandit_controller is not None:
            _save_json(os.path.join(run_dir, "bandit_dump.json"), bandit_controller.dump(),      "bandit_dump")
    if hasattr(trainer, "_exact_telemetry") and trainer._exact_telemetry:
        _save_json(os.path.join(run_dir, "exact_telemetry.json"), trainer._exact_telemetry,      "exact_telemetry")
        if exact_controller is not None:
            _save_json(os.path.join(run_dir, "exact_dump.json"), exact_controller.dump(),        "exact_dump")

    # Clean up GPU memory
    del trainer, model
    torch.cuda.empty_cache()

    return results


def pilot1_csd_verification(args):
    """Pilot 1: Log CSD components during standard GRPO at ρ=1.0."""
    logger.info("===== PILOT 1: CSD Equivalence Verification =====")
    out = os.path.join(args.output_dir, "pilot1_csd_verification")

    results = []
    for rho in [0.5, 1.0, 2.0, 3.0]:
        r = run_single_training(
            args.model, args.config, seed=42, rho=rho,
            max_steps=args.max_steps, output_dir=out, use_vllm=args.use_vllm,
        )
        results.append(r)

    with open(os.path.join(out, "pilot1_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Pilot 1 complete. Results: %s", out)
    return results


def pilot2_adq_collapse(args):
    """Pilot 2: ADQ vs constant ρ=1.0 — does ADQ eliminate collapse?"""
    logger.info("===== PILOT 2: ADQ Collapse Elimination =====")
    out = os.path.join(args.output_dir, "pilot2_adq_collapse")
    n_seeds = args.seeds or DEFAULT_SEEDS_P2

    results_const = []
    results_adq = []

    for seed in range(42, 42 + n_seeds):
        # Constant ρ=1.0
        r = run_single_training(
            args.model, args.config, seed=seed, rho=1.0,
            max_steps=args.max_steps, output_dir=out, use_adq=False,
            use_vllm=args.use_vllm,
        )
        results_const.append(r)

        # ADQ (adaptive ρ starting from 1.0)
        r = run_single_training(
            args.model, args.config, seed=seed, rho=1.0,
            max_steps=args.max_steps, output_dir=out, use_adq=True,
            use_vllm=args.use_vllm,
        )
        results_adq.append(r)

    # Summarize
    const_collapse = sum(1 for r in results_const if r.get("collapsed", False))
    adq_collapse = sum(1 for r in results_adq if r.get("collapsed", False))

    summary = {
        "n_seeds": n_seeds,
        "rho": 1.0,
        "constant_rho": {
            "collapse_count": const_collapse,
            "collapse_rate": round(const_collapse / n_seeds, 2),
            "mean_final_reward": round(np.mean([r.get("final_reward_mean", 0) for r in results_const]), 4),
        },
        "adq": {
            "collapse_count": adq_collapse,
            "collapse_rate": round(adq_collapse / n_seeds, 2),
            "mean_final_reward": round(np.mean([r.get("final_reward_mean", 0) for r in results_adq]), 4),
        },
        "killer_result": const_collapse > 0 and adq_collapse == 0,
        "results_const": results_const,
        "results_adq": results_adq,
    }

    with open(os.path.join(out, "pilot2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("===== PILOT 2 RESULT =====")
    logger.info("Constant ρ=1.0: %d/%d collapsed (%.0f%%)",
                const_collapse, n_seeds, 100 * const_collapse / n_seeds)
    logger.info("ADQ (CSD ρ*): %d/%d collapsed (%.0f%%)",
                adq_collapse, n_seeds, 100 * adq_collapse / n_seeds)
    if summary["killer_result"]:
        logger.info("🎯 KILLER RESULT: ADQ eliminates ALL collapse!")
    return summary


def pilot3_qcsd_predictor(args):
    """Pilot 3: Dense seeds at ρ=1.0, test Q_CSD as collapse predictor."""
    logger.info("===== PILOT 3: Q_CSD Collapse Predictor =====")
    out = os.path.join(args.output_dir, "pilot3_qcsd_predictor")
    n_seeds = args.seeds or DEFAULT_SEEDS_P3

    results = []
    for seed in range(42, 42 + n_seeds):
        r = run_single_training(
            args.model, args.config, seed=seed, rho=args.rho,
            max_steps=args.max_steps, output_dir=out, use_vllm=args.use_vllm,
        )
        results.append(r)

    # Analyze: load CSD logs and compute early Q_CSD vs final collapse
    early_qcsd = []
    collapsed = []

    for seed in range(42, 42 + n_seeds):
        run_dir = os.path.join(out, f"rho{args.rho:.2f}_seed{seed}")
        csd_path = os.path.join(run_dir, "csd_logs.json")
        if os.path.exists(csd_path):
            with open(csd_path) as f:
                csd_data = json.load(f)
            if len(csd_data) >= 3:
                # Use step 1-3 average as "early" Q_CSD
                early_q = np.mean([d.get("q_csd", 0) for d in csd_data[:3]])
                early_qcsd.append(early_q)
                final_collapsed = csd_data[-1].get("is_collapsed", False)
                collapsed.append(int(final_collapsed))

    # Compute AUROC if we have both collapsed and non-collapsed runs
    auroc = None
    if len(set(collapsed)) > 1 and len(early_qcsd) > 0:
        try:
            from sklearn.metrics import roc_auc_score
            # Q_CSD predicts NON-collapse (higher = better), so invert for collapse prediction
            auroc = roc_auc_score(collapsed, [-q for q in early_qcsd])
        except Exception:
            # Manual AUROC approximation
            pos = [q for q, c in zip(early_qcsd, collapsed) if c == 1]
            neg = [q for q, c in zip(early_qcsd, collapsed) if c == 0]
            if pos and neg:
                correct = sum(1 for p_val in pos for n_val in neg if n_val > p_val)
                auroc = correct / (len(pos) * len(neg))

    summary = {
        "n_seeds": n_seeds,
        "rho": args.rho,
        "n_collapsed": sum(collapsed) if collapsed else 0,
        "n_converged": len(collapsed) - sum(collapsed) if collapsed else 0,
        "early_qcsd_values": [round(q, 6) for q in early_qcsd],
        "collapse_labels": collapsed,
        "qcsd_auroc": round(auroc, 4) if auroc is not None else None,
        "results": results,
    }

    with open(os.path.join(out, "pilot3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("===== PILOT 3 RESULT =====")
    logger.info("Collapsed: %d/%d", sum(collapsed) if collapsed else 0, n_seeds)
    if auroc is not None:
        logger.info("Q_CSD AUROC for collapse prediction: %.4f", auroc)
        if auroc > 0.85:
            logger.info("🎯 STRONG: Q_CSD is an excellent collapse predictor!")
        elif auroc > 0.7:
            logger.info("✓ MODERATE: Q_CSD has predictive value")
        else:
            logger.info("⚠ WEAK: Q_CSD has limited predictive value")
    else:
        logger.info("⚠ Cannot compute AUROC (need both collapsed and non-collapsed runs)")

    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("CSD Pilot — model=%s, pilot=%s, max_steps=%d",
                args.model, args.pilot, args.max_steps)

    # Single-run modes (called by run_csd_pilot.sh for multi-GPU parallelism)
    if args.pilot == "1_single":
        run_single_training(
            args.model, args.config, seed=args.seed_start, rho=args.rho,
            max_steps=args.max_steps, output_dir=args.output_dir,
            use_vllm=args.use_vllm, dataset_name=args.dataset,
            group_size_override=args.group_size,
            max_completion_length_override=args.max_completion_length,
            use_bandit=args.use_bandit,
            use_exact_rho=args.use_exact_rho,
            exact_update_every=args.exact_update_every,
        )
        return

    if args.pilot == "2_single":
        run_single_training(
            args.model, args.config, seed=args.seed_start, rho=args.rho,
            max_steps=args.max_steps, output_dir=args.output_dir,
            use_adq=args.use_adq, use_vllm=args.use_vllm,
        )
        return

    if args.pilot == "3_single":
        run_single_training(
            args.model, args.config, seed=args.seed_start, rho=args.rho,
            max_steps=args.max_steps, output_dir=args.output_dir,
            use_vllm=args.use_vllm,
        )
        return

    if args.pilot == "3_analyze":
        # Just run the analysis part of pilot 3 (called after parallel runs)
        pilot3_qcsd_predictor(args)
        return

    # Full pilot modes (sequential, used when running without shell parallelism)
    if args.pilot in ("1", "all"):
        pilot1_csd_verification(args)

    if args.pilot in ("2", "all"):
        pilot2_adq_collapse(args)

    if args.pilot in ("3", "all"):
        pilot3_qcsd_predictor(args)

    logger.info("All requested pilots complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
