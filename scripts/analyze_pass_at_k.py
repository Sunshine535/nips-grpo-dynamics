#!/usr/bin/env python3
"""Analyze pass@k pilot results — test the RLVR ceiling hypothesis.

Reads JSON outputs from measure_pass_at_k.py and produces:
1. Aggregate pass@k comparison table
2. Per-question scatter: base pass@k vs trained pass@k
3. Verdict: does CE Replay preserve pass@k?

Usage:
  python scripts/analyze_pass_at_k.py \
    --base results/pass_at_k_pilot/base.json \
    --grpo results/pass_at_k_pilot/B_seed42.json \
    --replay results/pass_at_k_pilot/D_seed42.json \
    --out results/pass_at_k_pilot/analysis.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def per_question_pass_at(results: dict, k_key: str) -> List[float]:
    return [q["pass_at"].get(k_key, 0.0) for q in results["per_question"]]


def compute_verdict(base: dict, grpo: dict, replay: dict, k: int) -> dict:
    k_key = f"pass@{k}"
    base_vals = per_question_pass_at(base, k_key)
    grpo_vals = per_question_pass_at(grpo, k_key)
    replay_vals = per_question_pass_at(replay, k_key)

    n = min(len(base_vals), len(grpo_vals), len(replay_vals))
    base_vals, grpo_vals, replay_vals = base_vals[:n], grpo_vals[:n], replay_vals[:n]

    base_mean = sum(base_vals) / n
    grpo_mean = sum(grpo_vals) / n
    replay_mean = sum(replay_vals) / n

    grpo_drop = base_mean - grpo_mean
    replay_drop = base_mean - replay_mean
    replay_vs_grpo = replay_mean - grpo_mean

    grpo_better = sum(1 for b, g in zip(base_vals, grpo_vals) if g > b)
    replay_better = sum(1 for g, r in zip(grpo_vals, replay_vals) if r > g)

    return {
        "k": k,
        "base_mean": round(base_mean, 4),
        "grpo_mean": round(grpo_mean, 4),
        "replay_mean": round(replay_mean, 4),
        "grpo_drop_from_base": round(grpo_drop, 4),
        "replay_drop_from_base": round(replay_drop, 4),
        "replay_gain_over_grpo": round(replay_vs_grpo, 4),
        "n_questions_grpo_beats_base": grpo_better,
        "n_questions_replay_beats_grpo": replay_better,
        "n_questions": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model pass@k JSON")
    p.add_argument("--grpo", required=True, help="GRPO-only (B) pass@k JSON")
    p.add_argument("--replay", required=True, help="GRPO+CE Replay (D) pass@k JSON")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = load_result(args.base)
    grpo = load_result(args.grpo)
    replay = load_result(args.replay)

    print("=" * 70)
    print("RLVR CEILING HYPOTHESIS TEST")
    print("=" * 70)
    print(f"  Base:   {base['adapter']}  ({base['n_questions']} questions, k={base['k']})")
    print(f"  GRPO:   {grpo['adapter']}  ({grpo['n_questions']} questions, k={grpo['k']})")
    print(f"  Replay: {replay['adapter']}  ({replay['n_questions']} questions, k={replay['k']})")
    print()

    k_max = min(base["k"], grpo["k"], replay["k"])
    k_values = [kv for kv in [1, 5, 10, 25] if kv <= k_max]

    verdicts = []
    print(f"{'k':>5} | {'Base':>8} | {'GRPO':>8} | {'Replay':>8} | {'GRPO drop':>10} | {'Replay drop':>12} | {'R vs G':>8}")
    print("-" * 80)
    for kv in k_values:
        v = compute_verdict(base, grpo, replay, kv)
        verdicts.append(v)
        print(f"{kv:>5} | {v['base_mean']:>8.4f} | {v['grpo_mean']:>8.4f} | {v['replay_mean']:>8.4f} | "
              f"{v['grpo_drop_from_base']:>+10.4f} | {v['replay_drop_from_base']:>+12.4f} | {v['replay_gain_over_grpo']:>+8.4f}")

    print()

    diversity_base = base.get("avg_unique_correct_per_question", 0)
    diversity_grpo = grpo.get("avg_unique_correct_per_question", 0)
    diversity_replay = replay.get("avg_unique_correct_per_question", 0)
    print(f"Diversity (avg unique correct solutions per question):")
    print(f"  Base:   {diversity_base:.2f}")
    print(f"  GRPO:   {diversity_grpo:.2f}")
    print(f"  Replay: {diversity_replay:.2f}")
    print()

    top_k = verdicts[-1]
    hypothesis_1 = top_k["grpo_drop_from_base"] > 0.02
    hypothesis_2 = top_k["replay_gain_over_grpo"] > 0.02
    hypothesis_3 = abs(top_k["replay_drop_from_base"]) < top_k["grpo_drop_from_base"] * 0.5

    print("HYPOTHESIS TESTS:")
    h1_str = "CONFIRMED" if hypothesis_1 else "NOT CONFIRMED"
    h2_str = "CONFIRMED" if hypothesis_2 else "NOT CONFIRMED"
    h3_str = "CONFIRMED" if hypothesis_3 else "NOT CONFIRMED"
    print(f"  H1 (GRPO narrows pass@{top_k['k']} vs base):        {h1_str} (drop={top_k['grpo_drop_from_base']:+.4f})")
    print(f"  H2 (Replay preserves more than GRPO):     {h2_str} (gain={top_k['replay_gain_over_grpo']:+.4f})")
    print(f"  H3 (Replay within 50% of base ceiling):   {h3_str}")
    print()

    if hypothesis_1 and hypothesis_2:
        signal = "GREEN LIGHT"
        verdict_text = "CE Replay preserves reasoning diversity. Proceed to full study."
    elif hypothesis_1 and not hypothesis_2:
        signal = "YELLOW"
        verdict_text = "GRPO narrows as expected, but Replay doesn't help. Need more seeds/models to confirm."
    elif not hypothesis_1:
        signal = "ORANGE"
        verdict_text = "GRPO doesn't narrow pass@k in this setup. NeurIPS 2025 finding may not replicate at this scale."
    else:
        signal = "INCONCLUSIVE"
        verdict_text = "Mixed signals. Expand pilot before committing."

    print(f"VERDICT: [{signal}] {verdict_text}")
    print("=" * 70)

    analysis = {
        "signal": signal,
        "verdict": verdict_text,
        "verdicts_by_k": verdicts,
        "diversity": {
            "base": diversity_base,
            "grpo": diversity_grpo,
            "replay": diversity_replay,
        },
        "hypotheses": {
            "H1_grpo_narrows": hypothesis_1,
            "H2_replay_preserves": hypothesis_2,
            "H3_replay_near_base": hypothesis_3,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
