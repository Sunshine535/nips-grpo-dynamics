#!/usr/bin/env python3
"""SAGE-GRPO launcher — Signed Advantage-Guided Evidence Replay.

Modes:
  --sage-mode tasa_only         : signed PG only, no replay (B)
  --sage-mode positive_ce_only  : signed PG + positive CE replay (D)
  --sage-mode pair_only         : signed PG + pairwise contrastive replay
  --sage-mode full              : signed PG + pairwise + optional positive CE (C)
"""
import argparse, json, logging, os, random, sys, time
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

import re
from src.rho_grpo import build_gsm8k_binary_reward_function
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance, ClearRopeDeltasCallback
from src.prompt_credit_state import PromptCreditStore
from src.contrastive_evidence_bank import ContrastiveEvidenceBank
from src.sage_grpo_trainer import SageGRPOTrainer
from src.provenance import write_manifest

apply_qwen35_text_only_patch()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sage_grpo")
_ans_pat = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--config", default="configs/sage_grpo_minimal.yaml")
    p.add_argument("--output-dir", default="results/sage_grpo")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sage-mode", choices=["tasa_only", "positive_ce_only", "pair_only", "full"], default="full")
    p.add_argument("--lambda-pair", type=float, default=None)
    p.add_argument("--lambda-pos", type=float, default=None)
    p.add_argument("--pair-batch-size", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    scfg = cfg.get("sage", {})

    lambda_pair = args.lambda_pair if args.lambda_pair is not None else scfg.get("lambda_pair", 0.05)
    lambda_pos = args.lambda_pos if args.lambda_pos is not None else scfg.get("lambda_pos", 0.0)
    pair_batch_size = args.pair_batch_size if args.pair_batch_size is not None else scfg.get("pair_batch_size", 2)

    name = args.run_name or f"sage_{args.sage_mode}_seed{args.seed}"
    run_dir = os.path.join(args.output_dir, name)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    write_manifest(run_dir, kind="train", config=cfg, config_path=args.config,
                   seed=args.seed, model=args.model,
                   dataset={"name": cfg["dataset"]["name"], "split": cfg["dataset"]["split"]},
                   extra={"sage_mode": args.sage_mode, "lambda_pair": lambda_pair,
                          "lambda_pos": lambda_pos, "max_steps": args.max_steps, "phase": "start"})

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(cfg["dataset"]["name"], "main", split=cfg["dataset"]["split"])
    _supports_thinking = False
    try:
        tok.apply_chat_template([{"role": "user", "content": "t"}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
        _supports_thinking = True
    except TypeError:
        pass

    def _fmt(example, idx):
        m = _ans_pat.search(example["answer"])
        ans = m.group(1).replace(",", "") if m else ""
        msgs = [{"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."},
                {"role": "user", "content": f"Question: {example['question']}"}]
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if _supports_thinking:
            kwargs["enable_thinking"] = False
        p = tok.apply_chat_template(msgs, **kwargs)
        return {"prompt": p, "answer": ans, "prompt_id": int(idx)}
    ds = ds.map(_fmt, with_indices=True, remove_columns=ds.column_names)

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
        trust_remote_code=True, device_map=None)
    patch_model_instance(model)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    lora_cfg = cfg.get("lora", {})
    peft_cfg = LoraConfig(
        r=lora_cfg.get("r", 64), lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05), task_type="CAUSAL_LM")

    from trl import GRPOConfig
    training_config = GRPOConfig(
        output_dir=run_dir, num_train_epochs=1,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"], warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"], max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"], gradient_checkpointing=False,
        logging_steps=max(1, tcfg["logging_steps"]),
        save_steps=min(250, args.max_steps + 1), save_total_limit=1,
        seed=args.seed, num_generations=tcfg["num_generations"],
        max_completion_length=tcfg["max_completion_length"],
        max_steps=args.max_steps, report_to="none", log_level="info")

    credit_store = PromptCreditStore(
        alpha_baseline=scfg.get("alpha_baseline", 0.1),
        n_min=scfg.get("n_min", 5),
        max_exposure=scfg.get("max_exposure", 10))
    evidence_bank = ContrastiveEvidenceBank(
        max_pos_per_prompt=2, max_neg_per_prompt=2,
        age_tau=scfg.get("age_tau", 200.0))
    reward_fn = build_gsm8k_binary_reward_function()

    trainer = SageGRPOTrainer(
        model=model, args=training_config, train_dataset=ds,
        processing_class=tok, reward_funcs=reward_fn, peft_config=peft_cfg,
        evidence_bank=evidence_bank, prompt_credit_store=credit_store,
        lambda_pair=lambda_pair, lambda_pos=lambda_pos,
        pair_batch_size=pair_batch_size,
        replay_warmup_steps=scfg.get("replay_warmup_steps", 50),
        success_threshold=scfg.get("success_threshold", 0.5),
        failure_threshold=scfg.get("failure_threshold", 0.0),
        tasa_c=scfg.get("tasa_c", 0.5),
        ref_coef=scfg.get("ref_coef", 1.0),
        sage_mode=args.sage_mode,
        callbacks=[ClearRopeDeltasCallback()])

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    adapter_dir = os.path.join(run_dir, "checkpoint-final")
    try:
        trainer.save_model(adapter_dir)
    except Exception as e:
        logger.exception("save_model failed: %s", e)

    def _coerce(o):
        if isinstance(o, dict): return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_coerce(v) for v in o]
        if isinstance(o, np.ndarray): return _coerce(o.tolist())
        if isinstance(o, np.generic): return o.item()
        return o

    results = {"method": "SAGE-GRPO", "sage_mode": args.sage_mode, "seed": args.seed,
               "lambda_pair": lambda_pair, "lambda_pos": lambda_pos,
               "max_steps": args.max_steps, "elapsed_sec": round(elapsed, 1), "model": args.model}
    if trainer.state.log_history:
        fr = []
        for lh in trainer.state.log_history:
            for k in ("reward", "reward/mean"):
                if k in lh and lh[k] is not None:
                    try: fr.append(float(lh[k]))
                    except: pass
                    break
        if fr:
            results["final_reward_mean"] = round(fr[-1], 4)
            results["max_reward_mean"] = round(max(fr), 4)

    with open(os.path.join(run_dir, "sage_results.json"), "w") as f:
        json.dump(_coerce(results), f, indent=2)
    with open(os.path.join(run_dir, "sage_step_stats.json"), "w") as f:
        json.dump(_coerce(trainer._sage_step_stats), f, indent=2)
    with open(os.path.join(run_dir, "evidence_bank_summary.json"), "w") as f:
        json.dump(_coerce(evidence_bank.summary()), f, indent=2)

    write_manifest(run_dir, kind="train", config=cfg, config_path=args.config,
                   seed=args.seed, model=args.model,
                   dataset={"name": cfg["dataset"]["name"], "split": cfg["dataset"]["split"]},
                   adapter=adapter_dir,
                   extra={"sage_mode": args.sage_mode, "phase": "end",
                          "final_reward_mean": results.get("final_reward_mean"),
                          "bank_summary": evidence_bank.summary()},
                   manifest_name="run_manifest.json")

    print(f"[done] {name} sage_mode={args.sage_mode} "
          f"reward={results.get('final_reward_mean')} bank={evidence_bank.summary()}")


if __name__ == "__main__":
    main()
