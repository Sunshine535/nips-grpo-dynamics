#!/usr/bin/env python3
"""
Round 3 — Frozen-bank SFT control.

Builds a FIXED, frozen supervised-learning dataset from GSM8K train gold
solutions (perfect supervision) and trains a LoRA adapter on it for the
same number of gradient steps as ASE-R. This is the theoretical upper bound
for any SFT-based approach — gold solutions are the cleanest possible
replay bank.

If this reaches ≥65% on GSM8K test, it means "SFT on verified solutions" is
sufficient and the SPO+Replay GRPO gradient channel isn't novel.
If it stays <50%, the GRPO credit-assignment signal in SPO+Replay is
doing real work that SFT alone cannot replicate.

Usage:
  python scripts/run_sft_gold_control.py --seed 42
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
if os.path.isdir("/openbayes/input/input0/hub"):
    os.environ.setdefault("HF_HOME", "/openbayes/input/input0")
    os.environ.setdefault("HF_HUB_CACHE", "/openbayes/input/input0/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/openbayes/input/input0/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/openbayes/input/input0/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.torch_compat import apply_torch_compat_patch
apply_torch_compat_patch()
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance

apply_qwen35_text_only_patch()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sft_gold")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--config", default="configs/aser_mvp.yaml")
    p.add_argument("--output-dir", default="results/wave13_sft_gold_control")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all GSM8K train")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    lora_cfg_dict = cfg.get("lora", {})

    run_name = f"sft_gold_seed{args.seed}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    # ─── Tokenizer + dataset ───
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])

    _ans_pat = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
    _supports_thinking_toggle = False
    try:
        tok.apply_chat_template([{"role": "user", "content": "t"}], add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
        _supports_thinking_toggle = True
    except TypeError:
        pass

    def _fmt(example):
        """Prompt = same system+user as ASE-R training; completion = GSM8K gold solution."""
        msgs = [
            {"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."},
            {"role": "user", "content": f"Question: {example['question']}"},
        ]
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if _supports_thinking_toggle:
            kwargs["enable_thinking"] = False
        prompt = tok.apply_chat_template(msgs, **kwargs)
        # GSM8K gold solution = `example["answer"]` (contains full reasoning + #### answer)
        completion = example["answer"]
        return {"text": prompt + completion + tok.eos_token}

    ds = ds.map(_fmt, remove_columns=ds.column_names)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info(f"SFT gold dataset size: {len(ds)}")

    # ─── Model ───
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True, device_map=None,
    )
    patch_model_instance(model)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    peft_cfg = LoraConfig(
        r=lora_cfg_dict.get("r", 64),
        lora_alpha=lora_cfg_dict.get("lora_alpha", 128),
        target_modules=lora_cfg_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg_dict.get("lora_dropout", 0.05),
        task_type="CAUSAL_LM",
    )

    # ─── SFTConfig with matched hyperparams ───
    training_config = SFTConfig(
        output_dir=run_dir,
        num_train_epochs=10,                 # way more than max_steps; capped by max_steps
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=False,
        logging_steps=max(1, tcfg["logging_steps"]),
        save_steps=args.max_steps + 1,
        save_total_limit=1,
        seed=args.seed,
        max_seq_length=tcfg.get("max_completion_length", 256) + 256,  # prompt + completion
        max_steps=args.max_steps,
        report_to="none",
        log_level="info",
        log_on_each_node=False,
        dataset_text_field="text",
    )

    # ─── Train ───
    t0 = time.time()
    trainer = SFTTrainer(
        model=model, args=training_config, train_dataset=ds,
        processing_class=tok, peft_config=peft_cfg,
    )
    trainer.train()
    elapsed = time.time() - t0

    # ─── Save ───
    adapter_dir = os.path.join(run_dir, "checkpoint-final")
    trainer.save_model(adapter_dir)

    # Pull final training loss
    final_loss = None
    if trainer.state.log_history:
        for lh in trainer.state.log_history[::-1]:
            if "loss" in lh:
                final_loss = float(lh["loss"])
                break

    results = {
        "method": "SFT_gold_control",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "n_train_samples": len(ds),
        "elapsed_sec": round(elapsed, 1),
        "final_loss": final_loss,
        "model": args.model,
    }
    with open(os.path.join(run_dir, "sft_gold_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[done] run_dir={run_dir}  final_loss={final_loss}  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
