#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stability_analysis import analyze_stability, classify_regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robustness_test")


def parse_args():
    parser = argparse.ArgumentParser(description="i.i.d. violation robustness test")
    parser.add_argument("--output_dir", type=str, default="./results/robustness")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--rho_values", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    return parser.parse_args()


def count_reasoning_steps(answer_text: str) -> int:
    steps = 0
    lines = answer_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        if re.match(r"^[\d]+[.)\]]", line) or ">>" in line or "=" in line:
            steps += 1
    return max(steps, 1)


def bin_by_difficulty(dataset) -> dict:
    bins = {
        "easy_1_2": [],
        "medium_3_4": [],
        "hard_5_6": [],
        "very_hard_7plus": [],
    }

    for i, example in enumerate(dataset):
        n_steps = count_reasoning_steps(example["answer"])
        if n_steps <= 2:
            bins["easy_1_2"].append(i)
        elif n_steps <= 4:
            bins["medium_3_4"].append(i)
        elif n_steps <= 6:
            bins["hard_5_6"].append(i)
        else:
            bins["very_hard_7plus"].append(i)

    return bins


def compute_within_group_correlation(dataset, bin_indices, group_size, n_simulations=1000):
    n = len(bin_indices)
    if n < group_size * 2:
        return 0.0, 0.0

    step_counts = np.array([
        count_reasoning_steps(dataset[int(idx)]["answer"]) for idx in bin_indices
    ])
    difficulties = 1.0 - 0.1 * step_counts
    difficulties = np.clip(difficulties, 0.05, 0.95)

    correlations = []
    for _ in range(n_simulations):
        idx_sample = np.random.choice(len(bin_indices), size=group_size, replace=False)
        p_vals = difficulties[idx_sample]
        rewards = (np.random.random(group_size) < p_vals).astype(float)

        if len(set(rewards)) > 1 and np.std(p_vals) > 1e-6:
            correlations.append(float(np.corrcoef(p_vals, rewards)[0, 1]))

    if not correlations:
        return 0.0, 0.0

    return float(np.mean(correlations)), float(np.std(correlations))


def simulate_training_run(p_estimate, group_size, rho, n_steps=200, correlation=0.0):
    rewards_per_step = []
    kl_trajectory = []
    p_trajectory = [p_estimate]
    kl = 0.0

    for step in range(n_steps):
        if correlation > 0:
            base = np.random.random()
            p_corr = p_estimate + correlation * (base - 0.5)
            p_vals = np.clip(p_corr + (1 - correlation) * np.random.randn(group_size) * 0.1, 0.01, 0.99)
            group_rewards = (np.random.random(group_size) < p_vals).astype(float)
        else:
            group_rewards = np.random.binomial(1, p_estimate, size=group_size).astype(float)
        m = group_rewards.sum()
        rewards_per_step.append(m / group_size)

        kl_delta = 0.002 * abs(rho - 1.0) * (1 + 0.5 * np.random.randn())
        kl = max(0.0, kl + kl_delta)
        kl_trajectory.append(kl)

        p_estimate = min(0.99, max(0.01, p_estimate + 0.001 * (m / group_size - 0.5)))
        p_trajectory.append(p_estimate)

    return {
        "mean_reward": float(np.mean(rewards_per_step)),
        "final_p": float(p_trajectory[-1]),
        "p_trajectory": [float(p) for p in p_trajectory[::10]],
        "reward_trend": float(np.polyfit(range(n_steps), rewards_per_step, 1)[0]),
        "reward_std": float(np.std(rewards_per_step)),
        "kl_final": float(kl_trajectory[-1]) if kl_trajectory else 0.0,
        "kl_trajectory": [float(k) for k in kl_trajectory[::10]],
    }


def evaluate_prediction_accuracy(bin_results, group_size):
    total_predictions = 0
    correct_predictions = 0

    for bin_name, result in bin_results.items():
        for run in result.get("runs", []):
            predicted = run.get("predicted_regime", "")
            observed = run.get("observed_regime", "")
            total_predictions += 1
            if predicted == observed:
                correct_predictions += 1

    accuracy = correct_predictions / max(total_predictions, 1)
    return accuracy


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading GSM8K dataset for difficulty binning")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    bins = bin_by_difficulty(dataset)
    logger.info("Difficulty bins:")
    for name, indices in bins.items():
        logger.info("  %s: %d problems", name, len(indices))

    difficulty_estimates = {
        "easy_1_2": 0.7,
        "medium_3_4": 0.45,
        "hard_5_6": 0.25,
        "very_hard_7plus": 0.12,
    }

    if args.rho_values is None:
        base_bounds = analyze_stability(0.4, args.group_size)
        rho_values = [
            max(0.1, base_bounds.rho_min * 0.8),
            base_bounds.rho_star,
            min(10.0, base_bounds.rho_max * 1.2),
        ]
    else:
        rho_values = args.rho_values

    logger.info("Testing rho values: %s", rho_values)

    all_results = {}

    for bin_name, indices in bins.items():
        p_est = difficulty_estimates[bin_name]
        logger.info("\n=== Bin: %s (p≈%.2f, n=%d) ===", bin_name, p_est, len(indices))

        corr_mean, corr_std = compute_within_group_correlation(
            dataset, indices, args.group_size,
        )
        logger.info("Within-group correlation: %.3f ± %.3f", corr_mean, corr_std)

        bin_result = {
            "bin_name": bin_name,
            "n_problems": len(indices),
            "p_estimate": p_est,
            "within_group_correlation": corr_mean,
            "within_group_correlation_std": corr_std,
            "runs": [],
        }

        for rho in rho_values:
            bounds = analyze_stability(p_est, args.group_size)
            predicted_regime = classify_regime(rho, bounds)

            for seed in args.seeds:
                np.random.seed(seed)
                sim_result = simulate_training_run(
                    p_est, args.group_size, rho,
                    correlation=abs(corr_mean),
                )

                if sim_result["mean_reward"] < 0.1:
                    observed_regime = "gradient_starved"
                elif sim_result["kl_final"] > 1.0 and sim_result["reward_std"] > 0.35:
                    observed_regime = "unstable"
                elif sim_result["reward_trend"] < -0.0005 and sim_result["mean_reward"] < 0.3:
                    observed_regime = "gradient_starved"
                else:
                    observed_regime = "convergent"

                run_entry = {
                    "rho": rho,
                    "seed": seed,
                    "predicted_regime": predicted_regime,
                    "observed_regime": observed_regime,
                    "match": predicted_regime == observed_regime,
                    "rho_min": bounds.rho_min,
                    "rho_max": bounds.rho_max,
                    "mean_reward": sim_result["mean_reward"],
                }
                bin_result["runs"].append(run_entry)

        all_results[bin_name] = bin_result

    prediction_accuracy = evaluate_prediction_accuracy(all_results, args.group_size)

    summary = {
        "group_size": args.group_size,
        "rho_values": rho_values,
        "overall_prediction_accuracy": prediction_accuracy,
        "per_bin_results": {},
    }

    for bin_name, result in all_results.items():
        n_correct = sum(1 for r in result["runs"] if r["match"])
        n_total = len(result["runs"])
        summary["per_bin_results"][bin_name] = {
            "p_estimate": result["p_estimate"],
            "correlation": result["within_group_correlation"],
            "accuracy": n_correct / max(n_total, 1),
            "n_runs": n_total,
        }
        logger.info(
            "%s: correlation=%.3f, prediction_accuracy=%.1f%% (%d/%d)",
            bin_name, result["within_group_correlation"],
            100 * n_correct / max(n_total, 1), n_correct, n_total,
        )

    logger.info("\nOverall prediction accuracy: %.1f%%", 100 * prediction_accuracy)

    with open(os.path.join(args.output_dir, "robustness_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(args.output_dir, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
