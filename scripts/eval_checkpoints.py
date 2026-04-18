#!/usr/bin/env python3
"""
快速 eval 4 个 checkpoint 在 GSM8K 上的 reward。
对比 const-ρ vs ADQ 是否有任何差异。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/openbayes/input/input0'
os.environ['HF_HUB_CACHE'] = '/openbayes/input/input0/hub'
os.environ['HF_DATASETS_CACHE'] = '/openbayes/input/input0/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/openbayes/input/input0/hub'
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.torch_compat import apply_torch_compat_patch
apply_torch_compat_patch()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3.5-9B"
CACHE_DIR = "/openbayes/input/input0/hub"
PILOT_DIR = "/openbayes/input/input0/nips-grpo-dynamics/results/csd_pilot/pilot2_adq_collapse"
N_EVAL = 50  # 快速 eval

# Pattern for GSM8K answer matching
_ans_pat = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_fallback = re.compile(r"(-?[\d,]+\.?\d*)")
_think_pat = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_answer(text):
    text = _think_pat.sub("", text).strip()
    m = _ans_pat.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _fallback.findall(text)
    return nums[-1].replace(",", "") if nums else ""


def load_base_and_lora(ckpt_path):
    print(f"  loading base...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"},
    )
    print(f"  loading lora from {ckpt_path}...", flush=True)
    model = PeftModel.from_pretrained(base, ckpt_path)
    model.eval()
    return tok, model


def eval_on_gsm8k(tok, model, n=N_EVAL):
    # Load GSM8K directly from parquet (bypass hub lookup)
    import glob
    parquet_paths = glob.glob("/openbayes/input/input0/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")
    if parquet_paths:
        from datasets import Dataset
        import pandas as pd
        df = pd.read_parquet(parquet_paths[0])
        ds = Dataset.from_pandas(df).select(range(min(n, len(df))))
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))

    correct = 0
    n_pos_answers = 0
    for i, ex in enumerate(ds):
        gold = _ans_pat.search(ex["answer"]).group(1).replace(",", "") if _ans_pat.search(ex["answer"]) else ""
        msgs = [
            {"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."},
            {"role": "user", "content": f"Question: {ex['question']}"},
        ]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        if pred:
            n_pos_answers += 1
        is_correct = (pred == gold)
        if is_correct:
            correct += 1
        if i < 3:
            print(f"    Q{i}: pred='{pred[:30]}' gold='{gold}' correct={is_correct}", flush=True)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "n_any_answer": n_pos_answers,
    }


def main():
    results = {}
    ckpt_dirs = sorted([d for d in os.listdir(PILOT_DIR) if d.startswith("rho")])
    print(f"Found {len(ckpt_dirs)} checkpoints: {ckpt_dirs}")

    for name in ckpt_dirs:
        ckpt_path = os.path.join(PILOT_DIR, name, "checkpoint-50")
        if not os.path.isdir(ckpt_path):
            print(f"  SKIP {name}: no checkpoint-50")
            continue
        print(f"\n=== Eval: {name} ===", flush=True)
        tok, model = load_base_and_lora(ckpt_path)
        res = eval_on_gsm8k(tok, model)
        print(f"  → {res}", flush=True)
        results[name] = res
        del model, tok
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    const_acc, adq_acc = [], []
    for name, r in results.items():
        group = "ADQ" if "_adq" in name else "CONST"
        print(f"  {group} {name}: acc={r['accuracy']:.3f} ({r['correct']}/{r['n']}), answered={r['n_any_answer']}/{r['n']}")
        (adq_acc if "_adq" in name else const_acc).append(r['accuracy'])

    if const_acc and adq_acc:
        print(f"\nCONST mean accuracy: {sum(const_acc)/len(const_acc):.3f}")
        print(f"ADQ   mean accuracy: {sum(adq_acc)/len(adq_acc):.3f}")
        diff = sum(adq_acc)/len(adq_acc) - sum(const_acc)/len(const_acc)
        print(f"Δ (ADQ - CONST):     {diff:+.3f}")

    with open(os.path.join(PILOT_DIR, "checkpoint_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {PILOT_DIR}/checkpoint_eval.json")


if __name__ == "__main__":
    main()
