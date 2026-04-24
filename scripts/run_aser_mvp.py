#!/usr/bin/env python3
"""
ASE-R MVP launcher — fresh trainer entry point (does NOT extend run_csd_pilot.py).

Stages (controlled by CLI flags, defaults give stage-3 = full MVP):
  --backbone {spo, dr_grpo}
  --no-dup       : disable AdaptiveDupBatchSampler (Stage 1)
  --lambda-rep 0 : disable replay-CE loss (Stage 1 + Stage 2)

Usage:
  python scripts/run_aser_mvp.py --seed 42 --backbone spo
"""
import argparse
import json
import logging
import os
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
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.torch_compat import apply_torch_compat_patch
apply_torch_compat_patch()

from src.rho_grpo import build_gsm8k_binary_reward_function
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance, ClearRopeDeltasCallback
from src.prompt_stats import PromptStatsStore
from src.replay_bank import VerifiedReplayBank
from src.adaptive_dup_sampler import AdaptiveDupBatchSampler
from src.aser_trainer_v14 import ASERTrainerV14

apply_qwen35_text_only_patch()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("aser_mvp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--config", default="configs/aser_mvp.yaml")
    p.add_argument("--output-dir", default="results/wave10_aser")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", choices=["spo", "dr_grpo", "tasa"], default="spo")
    p.add_argument("--no-dup", action="store_true", help="disable adaptive duplication (Stage 1)")
    p.add_argument("--lambda-rep", type=float, default=None,
                   help="override config; set to 0 to disable replay CE")
    p.add_argument("--pg-weight", type=float, default=1.0,
                   help="multiplier on the GRPO policy-gradient loss. 0.0 → pure online RFT.")
    p.add_argument("--alpha-pos", type=float, default=0.0,
                   help="(α,β) phase diagram: positive advantage weight. 0 = off.")
    p.add_argument("--beta-neg", type=float, default=0.0,
                   help="(α,β) phase diagram: negative advantage scale. 0 = off.")
    p.add_argument("--tasa-threshold", type=float, default=0.5,
                   help="TASA correctness threshold c (default 0.5)")
    p.add_argument("--zero-score-strategy", type=str, default="none",
                   choices=["clip", "temperature", "curriculum", "relabel", "none"],
                   help="HalluZero zero-score gradient reshaping strategy")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def make_run_dir(args, cfg):
    parts = [f"{args.backbone}", f"seed{args.seed}"]
    if args.no_dup:
        parts.append("nodup")
    lr = cfg["aser"]["lambda_rep"] if args.lambda_rep is None else args.lambda_rep
    if float(lr) == 0.0:
        parts.append("noreplay")
    if float(args.pg_weight) == 0.0:
        parts.append("nopg")
    name = args.run_name or ("aser_" + "_".join(parts))
    d = os.path.join(args.output_dir, name)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    return d


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    acfg = cfg["aser"]
    if args.lambda_rep is not None:
        acfg["lambda_rep"] = float(args.lambda_rep)
    run_dir = make_run_dir(args, cfg)

    # ─── Tokenizer + dataset ───
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])

    import re
    _ans_pat = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
    _supports_thinking_toggle = False
    try:
        tok.apply_chat_template([{"role": "user", "content": "t"}], add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
        _supports_thinking_toggle = True
    except TypeError:
        pass

    def _fmt(example, idx):
        m = _ans_pat.search(example["answer"])
        ans = m.group(1).replace(",", "") if m else ""
        msgs = [
            {"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."},
            {"role": "user", "content": f"Question: {example['question']}"},
        ]
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if _supports_thinking_toggle:
            kwargs["enable_thinking"] = False
        p = tok.apply_chat_template(msgs, **kwargs)
        return {"prompt": p, "answer": ans, "prompt_id": int(idx)}

    ds = ds.map(_fmt, with_indices=True, remove_columns=ds.column_names)

    # ─── Model ───
    try:
        import flash_attn   # noqa
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

    lora_cfg = cfg.get("lora", {})
    peft_cfg = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type="CAUSAL_LM",
    )

    # ─── Training config ───
    from trl import GRPOConfig
    training_config = GRPOConfig(
        output_dir=run_dir,
        num_train_epochs=1,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=False,
        logging_steps=max(1, tcfg["logging_steps"]),
        save_steps=min(250, args.max_steps + 1),
        save_total_limit=1,
        seed=args.seed,
        num_generations=tcfg["num_generations"],
        max_completion_length=tcfg["max_completion_length"],
        max_steps=args.max_steps,
        report_to="none",
        log_level="info",
        log_on_each_node=False,
    )

    # ─── ASER state objects ───
    prompt_stats = PromptStatsStore(
        alpha_baseline=acfg.get("alpha_baseline", 0.1),
        alpha_success=acfg.get("alpha_success", 0.1),
    )
    replay_bank = VerifiedReplayBank(max_per_prompt=acfg.get("replay_max_per_prompt", 2))
    rope_cb = ClearRopeDeltasCallback()

    reward_fn = build_gsm8k_binary_reward_function()

    zs_handler = None
    if args.zero_score_strategy != "none":
        from src.zero_score_handler import ZeroScoreConfig, ZeroScoreHandler, ZeroScoreStrategy
        zs_handler = ZeroScoreHandler(ZeroScoreConfig(strategy=ZeroScoreStrategy(args.zero_score_strategy)))

    trainer = ASERTrainerV14(
        model=model, args=training_config, train_dataset=ds,
        processing_class=tok, reward_funcs=reward_fn, peft_config=peft_cfg,
        backbone_mode=args.backbone,
        prompt_stats=prompt_stats,
        replay_bank=replay_bank,
        lambda_rep=acfg.get("lambda_rep", 0.05),
        replay_batch_size=acfg.get("replay_batch_size", 4),
        replay_warmup_steps=acfg.get("replay_warmup_steps", 50),
        success_threshold=acfg.get("success_threshold", 0.5),
        pg_weight=args.pg_weight,
        alpha_pos=args.alpha_pos,
        beta_neg=args.beta_neg,
        tasa_threshold=args.tasa_threshold,
        zero_score_handler=zs_handler,
        callbacks=[rope_cb],
    )

    # Monkey-patch get_train_dataloader to swap the sampler when adaptive-dup is enabled.
    if not args.no_dup:
        import torch.utils.data as td

        def _make_dataloader(self):
            sampler = AdaptiveDupBatchSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                stats_store=prompt_stats,
                dup_frac=acfg.get("dup_frac", 0.25),
                hardness_temp=acfg.get("hardness_temp", 2.0),
                warmup_steps=acfg.get("dup_warmup_steps", 100),
                seed=self.args.seed,
            )
            return td.DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=0, pin_memory=True,
            )

        trainer.get_train_dataloader = _make_dataloader.__get__(trainer, trainer.__class__)

    # ─── Train ───
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ─── Save ───
    adapter_dir = os.path.join(run_dir, "checkpoint-final")
    try:
        trainer.save_model(adapter_dir)
    except Exception as e:
        logger.exception("save_model failed: %s", e)

    results = {
        "backbone": args.backbone, "seed": args.seed,
        "no_dup": args.no_dup,
        "lambda_rep": acfg.get("lambda_rep"),
        "pg_weight": args.pg_weight,
        "max_steps": args.max_steps,
        "elapsed_sec": round(elapsed, 1),
        "model": args.model,
    }
    if trainer.state.log_history:
        rk = ("reward", "reward/mean")
        fr = []
        for lh in trainer.state.log_history:
            for k in rk:
                if k in lh and lh[k] is not None:
                    try:
                        fr.append(float(lh[k]))
                    except Exception:
                        pass
                    break
        if fr:
            results["final_reward_mean"] = round(fr[-1], 4)
            results["max_reward_mean"] = round(max(fr), 4)
            results["n_reward_logs"] = len(fr)

    def _coerce(o):
        if isinstance(o, dict): return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_coerce(v) for v in o]
        if isinstance(o, np.ndarray): return _coerce(o.tolist())
        if isinstance(o, np.generic): return o.item()
        return o

    def _save(path, payload, label):
        try:
            with open(path, "w") as f:
                json.dump(_coerce(payload), f, indent=2)
        except Exception as e:
            logger.exception("%s save failed: %s", label, e)

    _save(os.path.join(run_dir, "aser_results.json"), results, "aser_results")
    _save(os.path.join(run_dir, "aser_step_stats.json"), trainer._aser_step_stats, "step_stats")
    _save(os.path.join(run_dir, "prompt_stats.json"), prompt_stats.dump(), "prompt_stats")
    _save(os.path.join(run_dir, "replay_bank_summary.json"), {
        "size": replay_bank.size(),
        "n_prompts": replay_bank.n_prompts(),
    }, "replay_summary")

    print(f"[done] run_dir={run_dir}  final_reward={results.get('final_reward_mean')}  "
          f"replay_size={replay_bank.size()}")


if __name__ == "__main__":
    main()
