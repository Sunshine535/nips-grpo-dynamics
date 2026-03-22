#!/usr/bin/env python3
"""
Build complete phase diagram analysis from sweep results.

Outputs:
  phase_diagram.pdf          — 2D heatmap of accuracy at each (α, β)
  pareto_frontier.pdf        — accuracy vs compute Pareto front
  training_dynamics.pdf      — per-step reward/KL/loss for selected points
  curriculum_comparison.pdf  — curriculum vs static strategies
  phase_analysis.json        — best point, Pareto, boundary detection
"""

import argparse
import glob
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_phase_diagram")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Phase Diagram Analysis")
    parser.add_argument("--results_dir", type=str, default="./results/phase_diagram")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--curriculum_dir", type=str, default="./results/curriculum")
    parser.add_argument("--output_dir", type=str, default="./results/analysis")
    parser.add_argument("--metric", type=str, default="gsm8k_accuracy")
    parser.add_argument("--grid_resolution", type=int, default=50)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────

def load_eval_results(results_dir: str) -> list[dict]:
    files = glob.glob(os.path.join(results_dir, "eval_*.json"))
    logger.info("Found %d eval result files", len(files))
    results = []
    for fpath in files:
        with open(fpath) as f:
            results.append(json.load(f))
    return results


def load_training_dynamics(checkpoint_dir: str) -> dict:
    """Load step_metrics.json from each checkpoint directory."""
    dynamics = {}
    if not os.path.isdir(checkpoint_dir):
        return dynamics
    for entry in os.listdir(checkpoint_dir):
        metrics_path = os.path.join(checkpoint_dir, entry, "step_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                dynamics[entry] = json.load(f)
    return dynamics


def aggregate_by_point(results: list[dict], metric: str) -> dict:
    """Average metric across seeds for each (α, β)."""
    grouped = {}
    for r in results:
        key = (r["positive_ratio"], r["negative_weight"])
        if key not in grouped:
            grouped[key] = []
        val = r.get(metric)
        if val is not None:
            grouped[key].append(val)

    aggregated = {}
    for key, vals in grouped.items():
        aggregated[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "values": vals,
        }
    return aggregated


# ── Phase diagram heatmap ────────────────────────────────────────────────

def plot_phase_diagram(aggregated, metric, output_path, grid_res, smooth_sigma):
    alphas = np.array([k[0] for k in aggregated])
    betas = np.array([k[1] for k in aggregated])
    means = np.array([aggregated[k]["mean"] for k in aggregated])

    ai = np.linspace(alphas.min(), alphas.max(), grid_res)
    bi = np.linspace(betas.min(), betas.max(), grid_res)
    AI, BI = np.meshgrid(ai, bi)
    ZI = griddata((alphas, betas), means, (AI, BI), method="cubic")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # (a) Raw heatmap
    im0 = axes[0].pcolormesh(ai, bi, ZI, cmap="RdYlGn", shading="auto")
    axes[0].scatter(alphas, betas, c="black", s=25, marker="x", zorder=5)
    axes[0].set_xlabel("α (positive ratio)", fontsize=12)
    axes[0].set_ylabel("β (negative weight)", fontsize=12)
    axes[0].set_title("(a) Accuracy Heatmap", fontsize=13)
    fig.colorbar(im0, ax=axes[0], label=metric.replace("_", " ").title())

    # Mark best point
    best_key = max(aggregated, key=lambda k: aggregated[k]["mean"])
    axes[0].scatter(*best_key, c="blue", s=120, marker="*", zorder=10,
                    label=f"best: α={best_key[0]}, β={best_key[1]}")
    axes[0].legend(fontsize=9)

    # (b) Phase boundaries (gradient magnitude)
    ZI_filled = np.nan_to_num(ZI, nan=np.nanmean(ZI))
    ZI_smooth = gaussian_filter(ZI_filled, sigma=smooth_sigma)
    grad_x = sobel(ZI_smooth, axis=1)
    grad_y = sobel(ZI_smooth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold_90 = np.percentile(grad_mag, 90)

    im1 = axes[1].pcolormesh(ai, bi, grad_mag, cmap="hot", shading="auto")
    axes[1].contour(ai, bi, grad_mag, levels=[threshold_90], colors="cyan",
                    linewidths=2, linestyles="--")
    axes[1].set_xlabel("α (positive ratio)", fontsize=12)
    axes[1].set_ylabel("β (negative weight)", fontsize=12)
    axes[1].set_title("(b) Phase Boundaries (gradient magnitude)", fontsize=13)
    fig.colorbar(im1, ax=axes[1], label="∇ magnitude")

    # (c) Stability (std across seeds)
    stds = np.array([aggregated[k].get("std", 0) for k in aggregated])
    if stds.max() > 0:
        ZI_std = griddata((alphas, betas), stds, (AI, BI), method="cubic")
        im2 = axes[2].pcolormesh(ai, bi, ZI_std, cmap="Reds", shading="auto")
        fig.colorbar(im2, ax=axes[2], label="Std across seeds")
    else:
        axes[2].text(0.5, 0.5, "Single seed\n(no variance data)",
                     transform=axes[2].transAxes, ha="center", fontsize=14)
    axes[2].scatter(alphas, betas, c="black", s=25, marker="x", zorder=5)
    axes[2].set_xlabel("α (positive ratio)", fontsize=12)
    axes[2].set_ylabel("β (negative weight)", fontsize=12)
    axes[2].set_title("(c) Stability (cross-seed std)", fontsize=13)

    plt.suptitle("Phase Diagram of RL Signal Balance", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Phase diagram saved to %s", output_path)

    return grad_mag, ai, bi, threshold_90


# ── Pareto frontier ──────────────────────────────────────────────────────

def plot_pareto_frontier(aggregated, metric, output_path):
    """Pareto front: accuracy vs effective negative pressure."""
    points = []
    for (a, b), v in aggregated.items():
        effective_neg = (1.0 - a) * b
        points.append((effective_neg, v["mean"], a, b))
    points.sort(key=lambda p: p[0])

    neg_pressures = [p[0] for p in points]
    accs = [p[1] for p in points]

    # Extract Pareto front
    pareto = []
    best_acc = -1
    for p in sorted(points, key=lambda x: x[0]):
        if p[1] > best_acc:
            best_acc = p[1]
            pareto.append(p)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(neg_pressures, accs, s=50, alpha=0.6, c="#2196F3", edgecolor="black",
               linewidth=0.5, label="All points")

    if pareto:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.plot(px, py, "r-o", linewidth=2, markersize=8, label="Pareto frontier",
                zorder=10)
        for p in pareto:
            ax.annotate(f"α={p[2]},β={p[3]}", (p[0], p[1]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Effective Negative Pressure: (1-α)·β", fontsize=13)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=13)
    ax.set_title("Pareto Frontier: Accuracy vs Negative Signal Strength", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Pareto frontier saved to %s", output_path)

    return pareto


# ── Training dynamics ────────────────────────────────────────────────────

def plot_training_dynamics(dynamics, output_path, max_traces=9):
    """Plot reward/loss/KL curves for a sample of (α,β) points."""
    if not dynamics:
        logger.warning("No training dynamics data found, skipping plot")
        return

    keys = sorted(dynamics.keys())[:max_traces]
    n = len(keys)
    if n == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, n))

    for idx, tag in enumerate(keys):
        records = dynamics[tag]
        if not records:
            continue
        steps = [r.get("step", i) for i, r in enumerate(records)]
        losses = [r.get("loss", r.get("train_loss", None)) for r in records]
        rewards = [r.get("reward_mean", None) for r in records]
        kls = [r.get("kl", None) for r in records]

        label = tag.replace("alpha", "α=").replace("_beta", " β=").replace("_seed", " s=")

        if any(v is not None for v in losses):
            axes[0].plot(steps, [v or 0 for v in losses], color=cmap[idx],
                         alpha=0.7, linewidth=1.2, label=label)
        if any(v is not None for v in rewards):
            axes[1].plot(steps, [v or 0 for v in rewards], color=cmap[idx],
                         alpha=0.7, linewidth=1.2, label=label)
        if any(v is not None for v in kls):
            axes[2].plot(steps, [v or 0 for v in kls], color=cmap[idx],
                         alpha=0.7, linewidth=1.2, label=label)

    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Dynamics Across (α, β) Points", fontsize=14)
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")

    axes[1].set_ylabel("Reward (mean)", fontsize=12)

    axes[2].set_ylabel("KL Divergence", fontsize=12)
    axes[2].set_xlabel("Training Step", fontsize=12)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Training dynamics saved to %s", output_path)


# ── Curriculum comparison ────────────────────────────────────────────────

def plot_curriculum_comparison(curriculum_dir, output_path):
    """Compare curriculum strategies by loading their schedule logs."""
    summary_path = os.path.join(curriculum_dir, "curriculum_comparison.json")
    if not os.path.exists(summary_path):
        logger.warning("No curriculum_comparison.json found, skipping")
        return

    with open(summary_path) as f:
        comparison = json.load(f)

    strategies = list(comparison.keys())
    if not strategies:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"anneal-positive": "#2196F3", "anneal-negative": "#FF9800",
              "cosine-schedule": "#4CAF50", "static": "#9C27B0"}

    # (a) Training loss comparison
    for strat in strategies:
        log_path = os.path.join(curriculum_dir, strat, "schedule_logs.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            logs = json.load(f)
        steps = [r.get("step", i) for i, r in enumerate(logs)]
        losses = [r.get("loss", r.get("train_loss", 0)) or 0 for r in logs]
        axes[0].plot(steps, losses, color=colors.get(strat, "gray"),
                     linewidth=2, label=strat, alpha=0.8)

    axes[0].set_xlabel("Step", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("(a) Training Loss", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # (b) α/β schedule curves
    for strat in strategies:
        log_path = os.path.join(curriculum_dir, strat, "schedule_logs.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            logs = json.load(f)
        steps = [r.get("step", i) for i, r in enumerate(logs)]
        alphas = [r.get("alpha", 0.5) for r in logs]
        betas = [r.get("beta", 1.0) for r in logs]
        c = colors.get(strat, "gray")
        axes[1].plot(steps, alphas, color=c, linewidth=2, linestyle="-",
                     label=f"{strat} α", alpha=0.8)
        axes[1].plot(steps, betas, color=c, linewidth=2, linestyle="--",
                     alpha=0.5)

    axes[1].set_xlabel("Step", fontsize=12)
    axes[1].set_ylabel("Value", fontsize=12)
    axes[1].set_title("(b) α (solid) / β (dashed) Schedules", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Curriculum comparison saved to %s", output_path)


# ── Classify regions ─────────────────────────────────────────────────────

def classify_regions(aggregated, metric):
    """Classify each (α,β) point into stable/unstable/collapsed."""
    accs = [v["mean"] for v in aggregated.values()]
    global_mean = np.mean(accs)
    global_std = np.std(accs)

    regions = {}
    for (a, b), v in aggregated.items():
        acc = v["mean"]
        seed_std = v.get("std", 0)

        if acc < global_mean - 1.5 * global_std:
            region = "collapsed"
        elif seed_std > 0.05 or acc < global_mean - 0.5 * global_std:
            region = "unstable"
        else:
            region = "stable"

        regions[f"alpha{a}_beta{b}"] = {
            "alpha": a, "beta": b,
            "accuracy": acc, "seed_std": seed_std,
            "region": region,
        }
    return regions


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_eval_results(args.results_dir)
    if not results:
        logger.error("No eval results found in %s", args.results_dir)
        return

    aggregated = aggregate_by_point(results, args.metric)
    logger.info("Loaded %d unique (α, β) points", len(aggregated))

    # Best static point
    best_key = max(aggregated, key=lambda k: aggregated[k]["mean"])
    best_info = aggregated[best_key]
    logger.info("Best static: α=%.2f, β=%.2f → %.4f ± %.4f",
                 best_key[0], best_key[1], best_info["mean"], best_info["std"])

    # 1. Phase diagram
    grad_mag, ai, bi, threshold = plot_phase_diagram(
        aggregated, args.metric,
        os.path.join(args.output_dir, "phase_diagram.pdf"),
        args.grid_resolution, args.smooth_sigma,
    )

    # 2. Pareto frontier
    pareto = plot_pareto_frontier(
        aggregated, args.metric,
        os.path.join(args.output_dir, "pareto_frontier.pdf"),
    )

    # 3. Training dynamics
    dynamics = load_training_dynamics(args.checkpoint_dir)
    plot_training_dynamics(
        dynamics,
        os.path.join(args.output_dir, "training_dynamics.pdf"),
    )

    # 4. Curriculum comparison
    plot_curriculum_comparison(
        args.curriculum_dir,
        os.path.join(args.output_dir, "curriculum_comparison.pdf"),
    )

    # 5. Region classification
    regions = classify_regions(aggregated, args.metric)

    # Save comprehensive analysis
    analysis = {
        "metric": args.metric,
        "num_points": len(aggregated),
        "best_static": {
            "alpha": best_key[0],
            "beta": best_key[1],
            "accuracy": best_info["mean"],
            "std": best_info["std"],
        },
        "pareto_frontier": [
            {"neg_pressure": p[0], "accuracy": p[1], "alpha": p[2], "beta": p[3]}
            for p in pareto
        ],
        "regions": regions,
        "all_points": {
            f"alpha{k[0]}_beta{k[1]}": {
                "alpha": k[0], "beta": k[1],
                "mean": v["mean"], "std": v["std"], "n_seeds": v["n_seeds"],
            }
            for k, v in sorted(aggregated.items())
        },
    }

    analysis_path = os.path.join(args.output_dir, "phase_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Print summary table
    logger.info("=" * 70)
    logger.info("PHASE DIAGRAM ANALYSIS")
    logger.info("=" * 70)
    logger.info("%-8s %-8s %-10s %-10s %-6s %-10s",
                "Alpha", "Beta", "Accuracy", "Std", "Seeds", "Region")
    logger.info("-" * 70)
    for key in sorted(aggregated.keys()):
        v = aggregated[key]
        rkey = f"alpha{key[0]}_beta{key[1]}"
        region = regions.get(rkey, {}).get("region", "?")
        logger.info("%-8.2f %-8.2f %-10.4f %-10.4f %-6d %-10s",
                     key[0], key[1], v["mean"], v["std"], v["n_seeds"], region)
    logger.info("")
    logger.info("Best: α=%.2f, β=%.2f → %.4f", best_key[0], best_key[1], best_info["mean"])
    logger.info("Pareto frontier: %d points", len(pareto))
    logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
