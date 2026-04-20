#!/usr/bin/env python3
"""
Round 2 analysis: combine Wave 10 ASE-R + Wave 11 (ρ-boost + RFT control) eval JSONs.

Outputs:
- Overall accuracy table per arm (mean ± std)
- Matched-seed Welch's t-test: SPO+Replay (ASE-R) vs fixed-ρ=0.70 at n=9
- Novelty-control Welch's t-test: SPO+Replay vs online RFT at n=3
- Stratified (easy/hard by base-model correctness) per arm
- Per-seed table for audit

Run after Wave 11 evals finish and are synced to local.
"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np


def load_eval(p: Path):
    d = json.load(open(p))
    if "per_q" in d and isinstance(d["per_q"], list):
        return d["per_q"]
    if "correct" in d and isinstance(d["correct"], list):
        pred = d.get("pred", [""] * len(d["correct"]))
        gold = d.get("gold", [""] * len(d["correct"]))
        return [
            {"pred": p, "gold": g, "correct": bool(c)}
            for p, g, c in zip(pred, gold, d["correct"])
        ]
    return []


def acc(rows):
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r.get("correct")) / len(rows)


def welch_t(a, b):
    a, b = np.array(a), np.array(b)
    mA, mB = a.mean(), b.mean()
    vA, vB = a.var(ddof=1), b.var(ddof=1)
    nA, nB = len(a), len(b)
    if nA < 2 or nB < 2:
        return float("nan"), float("nan")
    t = (mA - mB) / np.sqrt(vA / nA + vB / nB)
    df_num = (vA / nA + vB / nB) ** 2
    df_den = (vA / nA) ** 2 / (nA - 1) + (vB / nB) ** 2 / (nB - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    return float(t), float(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave10-dir", default="results/stratified_eval_aser",
                    help="dir with Wave 10 ASE-R eval JSONs (spo_full_*, spo_only_*, spo_dup_*)")
    ap.add_argument("--wave11-dir", default="results/stratified_eval_wave11",
                    help="dir with Wave 11 eval JSONs (rho0.70_seed* + rft_seed*)")
    ap.add_argument("--prior-dir", default="results/stratified_eval",
                    help="dir with prior baseline eval JSONs (rho0.70_seed42/43/44.json)")
    ap.add_argument("--base-eval", default=None,
                    help="base-model eval JSON (per-question) for stratification")
    ap.add_argument("--out", default="results/analysis_wave11.json")
    args = ap.parse_args()

    arms = {}
    # ASE-R (SPO + Replay, dup never fired)
    arms["spo_replay (ASE-R, n=9)"] = {
        p.stem.replace("spo_full_seed", ""): p
        for p in Path(args.wave10_dir).glob("spo_full_seed*.json")
    }
    # SPO only (no replay, no dup) — shows what backbone alone achieves
    arms["spo_only (n=3)"] = {
        p.stem.replace("spo_only_seed", ""): p
        for p in Path(args.wave10_dir).glob("spo_only_seed*.json")
    }
    # Wave 11 RFT control (pg-weight 0, bank-only SFT on verified successes)
    arms["rft_only (n=3)"] = {
        p.stem.replace("rft_seed", ""): p
        for p in Path(args.wave11_dir).glob("rft_seed*.json")
    }
    # Fixed-ρ=0.70 baseline: Wave 11 {46..51} + prior {42,43,44}
    rho070_files = {}
    for p in Path(args.prior_dir).glob("rho0.70_seed*.json"):
        s = p.stem.replace("rho0.70_seed", "")
        if s.isdigit():
            rho070_files[s] = p
    for p in Path(args.wave11_dir).glob("rho0.70_seed*.json"):
        s = p.stem.replace("rho0.70_seed", "")
        if s.isdigit():
            rho070_files[s] = p
    arms[f"fixed_rho=0.70 (n={len(rho070_files)})"] = rho070_files

    # Optional stratification
    base_correct_idxs = None
    if args.base_eval and os.path.exists(args.base_eval):
        base_rows = load_eval(Path(args.base_eval))
        base_correct_idxs = {i for i, r in enumerate(base_rows) if r.get("correct")}

    summary = {"arms": {}}
    for name, seed_files in arms.items():
        per_seed = {}
        for seed, p in seed_files.items():
            rows = load_eval(p)
            per_seed[seed] = {
                "overall": acc(rows),
                "n_questions": len(rows),
            }
            if base_correct_idxs is not None and rows:
                easy = [rows[i] for i in range(len(rows)) if i in base_correct_idxs]
                hard = [rows[i] for i in range(len(rows)) if i not in base_correct_idxs]
                per_seed[seed]["easy"] = acc(easy)
                per_seed[seed]["hard"] = acc(hard)
                per_seed[seed]["n_easy"] = len(easy)
                per_seed[seed]["n_hard"] = len(hard)

        accs = [d["overall"] for d in per_seed.values() if not np.isnan(d["overall"])]
        n = len(accs)
        summary["arms"][name] = {
            "n_seeds": n,
            "overall_mean": mean(accs) if n else float("nan"),
            "overall_std": stdev(accs) if n >= 2 else float("nan"),
            "per_seed": per_seed,
        }
        if base_correct_idxs is not None:
            easy_accs = [d["easy"] for d in per_seed.values() if "easy" in d and not np.isnan(d["easy"])]
            hard_accs = [d["hard"] for d in per_seed.values() if "hard" in d and not np.isnan(d["hard"])]
            if easy_accs:
                summary["arms"][name]["easy_mean"] = mean(easy_accs)
                summary["arms"][name]["easy_std"] = stdev(easy_accs) if len(easy_accs) >= 2 else 0.0
            if hard_accs:
                summary["arms"][name]["hard_mean"] = mean(hard_accs)
                summary["arms"][name]["hard_std"] = stdev(hard_accs) if len(hard_accs) >= 2 else 0.0

    # Matched-seed t-tests
    comparisons = []
    def _run_t(a_name, b_name):
        a = arms[a_name]; b = arms[b_name]
        shared = sorted(set(a.keys()) & set(b.keys()), key=lambda s: int(s) if s.isdigit() else 9999)
        a_accs = [acc(load_eval(a[s])) for s in shared]
        b_accs = [acc(load_eval(b[s])) for s in shared]
        t, df = welch_t(a_accs, b_accs)
        comparisons.append({
            "a": a_name, "b": b_name, "shared_seeds": shared,
            "a_mean": mean(a_accs) if a_accs else float("nan"),
            "b_mean": mean(b_accs) if b_accs else float("nan"),
            "delta_pp": (mean(a_accs) - mean(b_accs)) * 100 if a_accs and b_accs else float("nan"),
            "welch_t": t, "welch_df": df,
        })
    a_name = [k for k in arms if k.startswith("spo_replay")][0]
    rho_name = [k for k in arms if k.startswith("fixed_rho=0.70")][0]
    _run_t(a_name, rho_name)
    if "rft_only (n=3)" in arms:
        _run_t(a_name, "rft_only (n=3)")
    if "spo_only (n=3)" in arms:
        _run_t(a_name, "spo_only (n=3)")

    summary["comparisons"] = comparisons

    # Markdown table
    print("\n## Overall accuracy (GSM8K test n=200)\n")
    print("| Arm                          | n seeds | mean ± std       |")
    print("|------------------------------|---------|-------------------|")
    for name, d in summary["arms"].items():
        if "overall_mean" in d:
            m = d["overall_mean"] * 100
            s = d["overall_std"] * 100 if not np.isnan(d.get("overall_std", float("nan"))) else 0
            print(f"| {name:<29s}| {d['n_seeds']:^7d} | {m:5.1f} ± {s:4.1f}%      |")

    print("\n## Matched-seed comparisons")
    for c in comparisons:
        print(f"\n**{c['a']}** vs **{c['b']}**: shared seeds = {c['shared_seeds']}")
        print(f"  Δ = {c['delta_pp']:+.1f}pp ({c['a_mean']*100:.1f}% vs {c['b_mean']*100:.1f}%)")
        print(f"  Welch's t = {c['welch_t']:.2f}, df = {c['welch_df']:.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, default=str)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
