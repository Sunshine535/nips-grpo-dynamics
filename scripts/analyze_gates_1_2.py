#!/usr/bin/env python3
"""
Analyze Gate 1 + Gate 2 runs and close the two remaining review blockers.

Input  : results/gates_1_2/{gate1_adq_seed42, gate2_rho*_seed*}/
         produced by run_gates_1_2.sh.
Outputs: results/gates_1_2/analysis.json
         results/gates_1_2/analysis_summary.txt
         results/gates_1_2/rho_trajectory.png (Gate 1 only, if matplotlib)

It does NOT re-run training; it reads the JSON artifacts plus the final
LoRA checkpoint (if present) for a GSM8K accuracy eval (n=100).
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_run(run_dir: Path) -> dict:
    """Gather JSON artifacts from a single training run directory."""
    out = {"run_dir": str(run_dir)}
    for fname in (
        "pilot_results.json",
        "ada_telemetry.json",
        "rho_grpo_logs.json",
        "csd_logs.json",
        "stability_telemetry.json",
    ):
        p = run_dir / fname
        if p.is_file():
            try:
                with p.open() as f:
                    out[fname.replace(".json", "")] = json.load(f)
            except Exception as e:
                out[f"{fname}_error"] = str(e)
    return out


def summarize_rho_trajectory(ada_telemetry: list) -> dict:
    """Extract rho(t) stats for Gate 1."""
    if not ada_telemetry:
        return {"error": "ada_telemetry empty"}
    rhos = [rec.get("rho") for rec in ada_telemetry if rec.get("rho") is not None]
    if not rhos:
        return {"error": "no rho values in telemetry"}
    rhos = np.asarray(rhos, dtype=float)
    return {
        "n_steps": len(rhos),
        "rho_initial": float(rhos[0]),
        "rho_final": float(rhos[-1]),
        "rho_min": float(rhos.min()),
        "rho_max": float(rhos.max()),
        "rho_mean": float(rhos.mean()),
        "rho_std": float(rhos.std()),
        "rho_nontrivial": bool(rhos.std() > 1e-3),
    }


def plot_rho_trajectory(ada_telemetry: list, out_path: Path):
    """Try to save a rho(t) plot. Silent on missing matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    steps = [rec.get("step", i) for i, rec in enumerate(ada_telemetry)]
    rhos = [rec.get("rho") for rec in ada_telemetry]
    p_hats = [rec.get("p_hat") for rec in ada_telemetry]
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(steps, rhos, label="ρ(t)", color="C0")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("ρ", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    if any(p is not None for p in p_hats):
        ax2 = ax1.twinx()
        ax2.plot(steps, p_hats, label="p̂(t)", color="C3", alpha=0.6)
        ax2.set_ylabel("p̂ (batch success rate)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
    plt.title("ADQ: ρ(t) and p̂(t) during training (Gate 1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────
# GSM8K eval (reuse the same logic as scripts/eval_checkpoints.py)
# ──────────────────────────────────────────────────────────────
_ans_pat = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_fallback = re.compile(r"(-?[\d,]+\.?\d*)")
_think_pat = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_answer(text: str) -> str:
    text = _think_pat.sub("", text).strip()
    m = _ans_pat.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _fallback.findall(text)
    return nums[-1].replace(",", "") if nums else ""


def _load_base_model_once(base_model: str, cache_dir: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/openbayes/input/input0")
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", "/openbayes/input/input0/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    from src.torch_compat import apply_torch_compat_patch
    apply_torch_compat_patch()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, cache_dir=cache_dir,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"},
    )
    return tok, base, torch


def eval_lora_on_gsm8k(base_model: str, cache_dir: str, lora_path: str, n: int = 100) -> dict:
    """Evaluate a LoRA adapter on GSM8K test set, return accuracy."""
    tok, base, torch = _load_base_model_once(base_model, cache_dir)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, lora_path).eval()

    parquet_paths = glob.glob(
        f"{cache_dir}/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet"
    )
    if parquet_paths:
        import pandas as pd
        from datasets import Dataset
        df = pd.read_parquet(parquet_paths[0])
        ds = Dataset.from_pandas(df).select(range(min(n, len(df))))
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))

    correct = 0
    n_any = 0
    for ex in ds:
        m = _ans_pat.search(ex["answer"])
        gold = m.group(1).replace(",", "") if m else ""
        msgs = [
            {"role": "system", "content": "You are a math tutor. Solve problems step by step. Write your final numerical answer after ####."},
            {"role": "user", "content": f"Question: {ex['question']}"},
        ]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
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
            n_any += 1
        if pred == gold:
            correct += 1
    del model, base, tok
    torch.cuda.empty_cache()
    return {"n": n, "correct": correct, "accuracy": correct / n, "n_any": n_any}


# ──────────────────────────────────────────────────────────────
# Gate-level aggregation
# ──────────────────────────────────────────────────────────────
def aggregate_gate2(runs: list) -> dict:
    """Group Gate 2 runs by rho, compute mean/std of final_reward + accuracy."""
    by_rho: dict = {}
    for r in runs:
        tag = Path(r["run_dir"]).name
        m = re.match(r"rho([0-9.]+)_seed(\d+)", tag)
        if not m:
            continue
        rho = float(m.group(1))
        rho_key = f"{rho:.2f}"
        pilot = r.get("pilot_results", {})
        rec = {
            "seed": int(m.group(2)),
            "final_reward_mean": pilot.get("final_reward_mean"),
            "max_reward_mean": pilot.get("max_reward_mean"),
            "collapsed": pilot.get("collapsed"),
            "accuracy": r.get("accuracy"),
        }
        by_rho.setdefault(rho_key, []).append(rec)

    agg = {}
    for rho_key, entries in sorted(by_rho.items()):
        fr = [e["final_reward_mean"] for e in entries if e["final_reward_mean"] is not None]
        mx = [e["max_reward_mean"] for e in entries if e["max_reward_mean"] is not None]
        acc = [e["accuracy"] for e in entries if e["accuracy"] is not None]
        agg[rho_key] = {
            "n_seeds": len(entries),
            "final_reward_mean": float(np.mean(fr)) if fr else None,
            "final_reward_std": float(np.std(fr)) if len(fr) >= 2 else None,
            "max_reward_mean": float(np.mean(mx)) if mx else None,
            "accuracy_mean": float(np.mean(acc)) if acc else None,
            "accuracy_std": float(np.std(acc)) if len(acc) >= 2 else None,
            "per_seed": entries,
        }
    return agg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gates_dir", default="results/gates_1_2")
    p.add_argument("--base_model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--cache_dir", default="/openbayes/input/input0/hub")
    p.add_argument("--eval_n", type=int, default=100, help="GSM8K eval size per run")
    p.add_argument("--skip_eval", action="store_true", help="Skip GSM8K accuracy eval")
    return p.parse_args()


def main():
    args = parse_args()
    gates_dir = Path(args.gates_dir)
    assert gates_dir.is_dir(), f"gates_dir not found: {gates_dir}"

    run_dirs = sorted(
        d for d in gates_dir.iterdir()
        if d.is_dir() and (d.name.startswith("rho") or d.name.startswith("gate"))
    )
    if not run_dirs:
        # Pilot script saves under "rho{:.2f}_seed{seed}[_adq]" inside --output_dir.
        run_dirs = sorted(
            d for d in gates_dir.iterdir()
            if d.is_dir() and re.match(r"rho[0-9.]+_seed\d+", d.name)
        )
    print(f"Found {len(run_dirs)} run directories", flush=True)

    runs = []
    for d in run_dirs:
        r = load_run(d)
        # Opportunistic LoRA eval on final checkpoint
        if not args.skip_eval:
            ckpts = sorted(glob.glob(str(d / "checkpoint-*")))
            if ckpts:
                last_ckpt = ckpts[-1]
                try:
                    print(f"  eval {d.name} @ {Path(last_ckpt).name} (n={args.eval_n}) ...",
                          flush=True)
                    ev = eval_lora_on_gsm8k(args.base_model, args.cache_dir,
                                             last_ckpt, n=args.eval_n)
                    r.update(ev)
                    print(f"    acc={ev['accuracy']:.3f} ({ev['correct']}/{ev['n']})",
                          flush=True)
                except Exception as e:
                    r["eval_error"] = str(e)
                    print(f"    eval failed: {e}", flush=True)
        runs.append(r)

    analysis = {"runs": runs}

    # Gate 1: rho trajectory (ADQ run)
    gate1_runs = [r for r in runs if "_adq" in Path(r["run_dir"]).name
                                        or "adq" in Path(r["run_dir"]).name.lower()]
    if gate1_runs:
        g1 = gate1_runs[0]
        ada_tel = g1.get("ada_telemetry", [])
        analysis["gate1"] = {
            "run_dir": g1["run_dir"],
            "rho_trajectory_summary": summarize_rho_trajectory(ada_tel),
            "final_reward_mean": g1.get("pilot_results", {}).get("final_reward_mean"),
            "accuracy": g1.get("accuracy"),
        }
        plot_path = gates_dir / "rho_trajectory.png"
        if ada_tel:
            ok = plot_rho_trajectory(ada_tel, plot_path)
            if ok:
                analysis["gate1"]["rho_plot"] = str(plot_path)

    # Gate 2: aggregate across seeds
    gate2_runs = [r for r in runs if "_adq" not in Path(r["run_dir"]).name
                                        and not Path(r["run_dir"]).name.startswith("gate1")]
    if gate2_runs:
        analysis["gate2"] = aggregate_gate2(gate2_runs)

    out = gates_dir / "analysis.json"
    with out.open("w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Wrote: {out}")

    # Human-readable summary
    lines = ["# Gate 1 + Gate 2 Analysis Summary\n"]
    g1 = analysis.get("gate1")
    if g1:
        rt = g1["rho_trajectory_summary"]
        lines.append("## Gate 1: V14 ADQ real-model run")
        if "error" in rt:
            lines.append(f"  ERROR: {rt['error']}")
        else:
            lines.append(f"  ρ(t): init={rt['rho_initial']:.3f}, final={rt['rho_final']:.3f}, "
                         f"min={rt['rho_min']:.3f}, max={rt['rho_max']:.3f}, "
                         f"std={rt['rho_std']:.3f}, nontrivial={rt['rho_nontrivial']}")
            lines.append(f"  ρ moved during training: {'YES' if rt['rho_nontrivial'] else 'NO'}")
        if g1.get("final_reward_mean") is not None:
            lines.append(f"  final reward mean: {g1['final_reward_mean']:.3f}")
        if g1.get("accuracy") is not None:
            lines.append(f"  final GSM8K acc: {g1['accuracy']:.3%}")
        lines.append("")
    g2 = analysis.get("gate2")
    if g2:
        lines.append("## Gate 2: 3-seed confirmatory sweep")
        lines.append(f"  {'ρ':>6} {'n_seeds':>8} {'final_reward':>18} {'acc_mean±std':>20}")
        for rho_key, e in g2.items():
            fr = e.get("final_reward_mean")
            fr_s = e.get("final_reward_std")
            fr_disp = f"{fr:.3f}±{fr_s:.3f}" if fr is not None and fr_s is not None else (
                f"{fr:.3f}" if fr is not None else "—")
            a = e.get("accuracy_mean")
            a_s = e.get("accuracy_std")
            a_disp = f"{a:.3%}±{a_s:.3%}" if a is not None and a_s is not None else (
                f"{a:.3%}" if a is not None else "—")
            lines.append(f"  {rho_key:>6} {e['n_seeds']:>8} {fr_disp:>18} {a_disp:>20}")
    summary_path = gates_dir / "analysis_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
