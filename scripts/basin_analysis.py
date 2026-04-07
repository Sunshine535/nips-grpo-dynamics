"""
Basin geometry analysis for GRPO training dynamics.

Computes statistical-physics order parameters from sweep data:
- Binder cumulant U4 = 1 - <R^4> / (3 <R^2>^2)
- Susceptibility chi = N * (<R^2> - <R>^2)
- Finite-size scaling with group size G

Usage:
    python scripts/basin_analysis.py --results-dir results/ --output results/basin_analysis/
    python scripts/basin_analysis.py --results-dir results/ --output results/basin_analysis/ --plot
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class RunOutcome:
    rho: float
    seed: int
    model: str
    group_size: int
    final_reward: float
    reward_trajectory: list
    converged: bool  # reward > 0.5 at end


def load_sweep_results(results_dir: str, model_tag: str = "qwen35") -> list[RunOutcome]:
    """Load all sweep runs from a results directory."""
    outcomes = []
    sweep_dir = os.path.join(results_dir, model_tag, "sweep_coarse")
    if not os.path.isdir(sweep_dir):
        sweep_dir = results_dir

    for dirname in sorted(os.listdir(sweep_dir)):
        match = re.match(r"rho([\d.]+)_seed(\d+)", dirname)
        if not match:
            continue

        rho = float(match.group(1))
        seed = int(match.group(2))
        log_path = os.path.join(sweep_dir, dirname, "step_logs.json")

        if not os.path.exists(log_path):
            continue

        with open(log_path) as f:
            data = json.load(f)

        reward_entries = [e for e in data if "reward" in e and "train_runtime" not in e]
        if not reward_entries:
            continue

        trajectory = [e["reward"] for e in reward_entries]
        final_reward = trajectory[-1]

        outcomes.append(RunOutcome(
            rho=rho,
            seed=seed,
            model=model_tag,
            group_size=4,  # default from config
            final_reward=final_reward,
            reward_trajectory=trajectory,
            converged=final_reward > 0.5,
        ))

    return outcomes


def binder_cumulant(rewards: np.ndarray) -> float:
    """
    Binder cumulant U4 = 1 - <R^4> / (3 <R^2>^2).

    U4 → 2/3 for Gaussian (single basin)
    U4 → 0 for bimodal (two basins)
    U4 < 0 for strongly bimodal
    """
    if len(rewards) < 2:
        return np.nan
    r2 = np.mean(rewards ** 2)
    r4 = np.mean(rewards ** 4)
    if r2 < 1e-12:
        return np.nan
    return 1.0 - r4 / (3.0 * r2 ** 2)


def susceptibility(rewards: np.ndarray) -> float:
    """
    Susceptibility chi = N * (<R^2> - <R>^2).
    Peaks at phase transitions.
    """
    n = len(rewards)
    if n < 2:
        return np.nan
    return n * (np.mean(rewards ** 2) - np.mean(rewards) ** 2)


def bimodality_coefficient(rewards: np.ndarray) -> float:
    """
    Bimodality coefficient b = (skewness^2 + 1) / kurtosis.
    b > 5/9 suggests bimodality.
    """
    n = len(rewards)
    if n < 4:
        return np.nan
    m = np.mean(rewards)
    s = np.std(rewards, ddof=1)
    if s < 1e-12:
        return np.nan
    skew = np.mean(((rewards - m) / s) ** 3)
    kurt = np.mean(((rewards - m) / s) ** 4)
    if kurt < 1e-12:
        return np.nan
    return (skew ** 2 + 1) / kurt


def analyze_basins(outcomes: list[RunOutcome]) -> dict:
    """Compute order parameters for each rho value."""
    by_rho = defaultdict(list)
    for o in outcomes:
        by_rho[o.rho].append(o.final_reward)

    results = {}
    for rho in sorted(by_rho.keys()):
        rewards = np.array(by_rho[rho])
        n = len(rewards)
        results[rho] = {
            "n_seeds": n,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "binder_U4": float(binder_cumulant(rewards)),
            "susceptibility": float(susceptibility(rewards)),
            "bimodality_coeff": float(bimodality_coefficient(rewards)),
            "convergence_rate": float(np.mean(rewards > 0.5)),
            "individual_rewards": rewards.tolist(),
        }

    return results


def find_critical_rho(basin_results: dict) -> dict:
    """
    Identify critical rho from order parameter analysis.

    Critical point: where U4 dips below 2/3 (Gaussian reference)
    or susceptibility peaks.
    """
    rhos = sorted(basin_results.keys())
    u4_values = [basin_results[r]["binder_U4"] for r in rhos]
    chi_values = [basin_results[r]["susceptibility"] for r in rhos]

    # Find susceptibility peak
    chi_peak_idx = int(np.nanargmax(chi_values))
    rho_c_chi = rhos[chi_peak_idx]

    # Find U4 minimum (most bimodal point)
    u4_min_idx = int(np.nanargmin(u4_values))
    rho_c_u4 = rhos[u4_min_idx]

    return {
        "rho_c_susceptibility": rho_c_chi,
        "chi_peak": chi_values[chi_peak_idx],
        "rho_c_binder": rho_c_u4,
        "u4_min": u4_values[u4_min_idx],
        "gaussian_ref": 2.0 / 3.0,
    }


def trajectory_divergence_time(outcomes: list[RunOutcome], rho: float) -> dict:
    """
    For a given rho, find when seed trajectories start diverging.
    Returns the step at which inter-seed variance exceeds a threshold.
    """
    runs = [o for o in outcomes if o.rho == rho]
    if len(runs) < 2:
        return {"divergence_step": None, "n_runs": len(runs)}

    max_len = max(len(r.reward_trajectory) for r in runs)
    min_len = min(len(r.reward_trajectory) for r in runs)

    variances = []
    for t in range(min_len):
        rewards_t = [r.reward_trajectory[t] for r in runs]
        variances.append(np.var(rewards_t))

    variances = np.array(variances)
    # Divergence = first step where variance exceeds 10% of max variance
    max_var = np.max(variances) if len(variances) > 0 else 0
    threshold = 0.1 * max_var if max_var > 0.01 else 0.01

    divergence_step = None
    for t, v in enumerate(variances):
        if v > threshold:
            divergence_step = t
            break

    return {
        "divergence_step": divergence_step,
        "n_runs": len(runs),
        "max_variance": float(max_var),
        "variance_trajectory": variances.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Basin geometry analysis")
    parser.add_argument("--results-dir", default="results/", help="Results directory")
    parser.add_argument("--model", default="qwen35", help="Model tag")
    parser.add_argument("--output", default="results/basin_analysis/", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    outcomes = load_sweep_results(args.results_dir, args.model)
    print(f"Loaded {len(outcomes)} runs from {args.results_dir}")

    if not outcomes:
        print("No data found. Check --results-dir and --model.")
        return

    # Basin analysis
    basin_results = analyze_basins(outcomes)
    critical = find_critical_rho(basin_results)

    print("\n=== Basin Analysis ===")
    print(f"{'rho':>6} {'n':>3} {'mean':>8} {'std':>8} {'U4':>8} {'chi':>8} {'bimod':>8} {'conv%':>6}")
    for rho in sorted(basin_results.keys()):
        r = basin_results[rho]
        print(f"{rho:6.2f} {r['n_seeds']:3d} {r['mean_reward']:8.3f} {r['std_reward']:8.3f} "
              f"{r['binder_U4']:8.3f} {r['susceptibility']:8.3f} {r['bimodality_coeff']:8.3f} "
              f"{r['convergence_rate']:6.1%}")

    print(f"\nCritical rho (susceptibility peak): {critical['rho_c_susceptibility']}")
    print(f"Critical rho (Binder minimum): {critical['rho_c_binder']}")
    print(f"Gaussian reference U4 = {critical['gaussian_ref']:.3f}")

    # Trajectory divergence for critical rho
    for rho in sorted(basin_results.keys()):
        div = trajectory_divergence_time(outcomes, rho)
        if div["divergence_step"] is not None:
            print(f"rho={rho:.2f}: trajectories diverge at step {div['divergence_step']}")

    # Save results
    output = {
        "basin_analysis": {str(k): v for k, v in basin_results.items()},
        "critical_point": critical,
        "model": args.model,
        "n_total_runs": len(outcomes),
    }
    with open(os.path.join(args.output, "basin_analysis.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}/basin_analysis.json")

    if args.plot:
        try:
            plot_basin_analysis(basin_results, critical, args.output)
        except ImportError:
            print("matplotlib not available, skipping plots")


def plot_basin_analysis(basin_results, critical, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = sorted(basin_results.keys())
    u4 = [basin_results[r]["binder_U4"] for r in rhos]
    chi = [basin_results[r]["susceptibility"] for r in rhos]
    means = [basin_results[r]["mean_reward"] for r in rhos]
    stds = [basin_results[r]["std_reward"] for r in rhos]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Mean reward ± std
    ax = axes[0, 0]
    ax.errorbar(rhos, means, yerr=stds, fmt="o-", capsize=4)
    ax.set_xlabel("ρ")
    ax.set_ylabel("Final Reward")
    ax.set_title("Reward Landscape")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)

    # Panel 2: Binder cumulant
    ax = axes[0, 1]
    ax.plot(rhos, u4, "s-", color="red")
    ax.axhline(2/3, ls="--", color="gray", alpha=0.5, label="Gaussian ref")
    ax.set_xlabel("ρ")
    ax.set_ylabel("U₄")
    ax.set_title("Binder Cumulant")
    ax.legend()

    # Panel 3: Susceptibility
    ax = axes[1, 0]
    ax.plot(rhos, chi, "^-", color="blue")
    ax.set_xlabel("ρ")
    ax.set_ylabel("χ")
    ax.set_title("Susceptibility")

    # Panel 4: Individual seed outcomes
    ax = axes[1, 1]
    for rho in rhos:
        rewards = basin_results[rho]["individual_rewards"]
        ax.scatter([rho] * len(rewards), rewards, alpha=0.6, s=50)
    ax.set_xlabel("ρ")
    ax.set_ylabel("Final Reward")
    ax.set_title("Per-Seed Outcomes")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "basin_analysis.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {output_dir}/basin_analysis.png")


if __name__ == "__main__":
    main()
