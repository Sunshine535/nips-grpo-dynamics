#!/usr/bin/env python3
"""
Aggregate all sweep results and plot 3D phase diagram.
x=positive_ratio, y=negative_weight, z=accuracy.
Identifies phase transition boundaries via gradient analysis.
"""

import argparse
import glob
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plot_phase_diagram")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot phase diagram from sweep results")
    parser.add_argument("--results_dir", type=str, default="./results/phase_diagram",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output_dir", type=str, default="./results/plots")
    parser.add_argument("--metric", type=str, default="gsm8k_accuracy",
                        choices=["gsm8k_accuracy", "math_accuracy"])
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma for boundary detection")
    parser.add_argument("--grid_resolution", type=int, default=50)
    return parser.parse_args()


def load_results(results_dir: str) -> list[dict]:
    pattern = os.path.join(results_dir, "eval_*.json")
    files = glob.glob(pattern)
    logger.info(f"Found {len(files)} result files in {results_dir}")

    results = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
            results.append(data)
    return results


def aggregate_by_seeds(results: list[dict], metric: str) -> dict:
    """Average metric across seeds for each (alpha, beta) pair."""
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
            "mean": np.mean(vals),
            "std": np.std(vals),
            "n": len(vals),
        }
    return aggregated


def plot_3d_surface(aggregated: dict, metric: str, output_path: str, grid_res: int):
    alphas = np.array([k[0] for k in aggregated.keys()])
    betas = np.array([k[1] for k in aggregated.keys()])
    means = np.array([v["mean"] for v in aggregated.values()])

    ai = np.linspace(alphas.min(), alphas.max(), grid_res)
    bi = np.linspace(betas.min(), betas.max(), grid_res)
    AI, BI = np.meshgrid(ai, bi)

    ZI = griddata((alphas, betas), means, (AI, BI), method="cubic")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(AI, BI, ZI, cmap=cm.viridis, alpha=0.85,
                           edgecolor="none", antialiased=True)

    ax.scatter(alphas, betas, means, c="red", s=30, zorder=10, label="data points")

    ax.set_xlabel("Positive Ratio (α)", fontsize=12, labelpad=10)
    ax.set_ylabel("Negative Weight (β)", fontsize=12, labelpad=10)
    ax.set_zlabel(metric.replace("_", " ").title(), fontsize=12, labelpad=10)
    ax.set_title("Phase Diagram: RL Signal Balance", fontsize=14, pad=20)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=metric)
    ax.legend()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"3D surface saved to {output_path}")


def plot_heatmap(aggregated: dict, metric: str, output_path: str, grid_res: int):
    alphas = np.array([k[0] for k in aggregated.keys()])
    betas = np.array([k[1] for k in aggregated.keys()])
    means = np.array([v["mean"] for v in aggregated.values()])

    ai = np.linspace(alphas.min(), alphas.max(), grid_res)
    bi = np.linspace(betas.min(), betas.max(), grid_res)
    AI, BI = np.meshgrid(ai, bi)
    ZI = griddata((alphas, betas), means, (AI, BI), method="cubic")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(ai, bi, ZI, cmap="RdYlGn", shading="auto")
    ax.scatter(alphas, betas, c="black", s=20, zorder=10, marker="x")

    ax.set_xlabel("Positive Ratio (α)", fontsize=13)
    ax.set_ylabel("Negative Weight (β)", fontsize=13)
    ax.set_title(f"Phase Diagram Heatmap: {metric}", fontsize=14)
    fig.colorbar(im, ax=ax, label=metric.replace("_", " ").title())

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to {output_path}")


def find_phase_boundaries(aggregated: dict, metric: str, output_path: str,
                          grid_res: int, smooth_sigma: float):
    """Detect phase transition boundaries via gradient magnitude."""
    alphas = np.array([k[0] for k in aggregated.keys()])
    betas = np.array([k[1] for k in aggregated.keys()])
    means = np.array([v["mean"] for v in aggregated.values()])

    ai = np.linspace(alphas.min(), alphas.max(), grid_res)
    bi = np.linspace(betas.min(), betas.max(), grid_res)
    AI, BI = np.meshgrid(ai, bi)
    ZI = griddata((alphas, betas), means, (AI, BI), method="cubic")

    ZI_filled = np.nan_to_num(ZI, nan=np.nanmean(ZI))
    ZI_smooth = gaussian_filter(ZI_filled, sigma=smooth_sigma)

    grad_x = sobel(ZI_smooth, axis=1)
    grad_y = sobel(ZI_smooth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    threshold = np.percentile(grad_mag, 90)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Gradient magnitude
    im0 = axes[0].pcolormesh(ai, bi, grad_mag, cmap="hot", shading="auto")
    axes[0].set_title("Gradient Magnitude (Phase Transition Signal)", fontsize=13)
    axes[0].set_xlabel("Positive Ratio (α)")
    axes[0].set_ylabel("Negative Weight (β)")
    fig.colorbar(im0, ax=axes[0])

    # Boundary overlay on heatmap
    im1 = axes[1].pcolormesh(ai, bi, ZI, cmap="RdYlGn", shading="auto")
    axes[1].contour(ai, bi, grad_mag, levels=[threshold], colors="red", linewidths=2)
    axes[1].set_title(f"Phase Boundaries (top 10% gradient)", fontsize=13)
    axes[1].set_xlabel("Positive Ratio (α)")
    axes[1].set_ylabel("Negative Weight (β)")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Phase boundaries saved to {output_path}")

    # Return boundary points for further analysis
    boundary_mask = grad_mag >= threshold
    boundary_points = []
    for i in range(len(bi)):
        for j in range(len(ai)):
            if boundary_mask[i, j]:
                boundary_points.append({
                    "alpha": float(ai[j]),
                    "beta": float(bi[i]),
                    "grad_magnitude": float(grad_mag[i, j]),
                })
    return boundary_points


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        logger.error("No results found. Run training and evaluation first.")
        return

    aggregated = aggregate_by_seeds(results, args.metric)
    logger.info(f"Aggregated {len(aggregated)} unique (alpha, beta) points")

    # Print summary table
    logger.info("=" * 60)
    logger.info(f"{'Alpha':>8} {'Beta':>8} {'Mean':>10} {'Std':>10} {'N':>4}")
    logger.info("-" * 60)
    for (a, b), v in sorted(aggregated.items()):
        logger.info(f"{a:8.2f} {b:8.2f} {v['mean']:10.4f} {v['std']:10.4f} {v['n']:4d}")

    # Generate all plots
    plot_3d_surface(
        aggregated, args.metric,
        os.path.join(args.output_dir, "phase_diagram_3d.png"),
        args.grid_resolution,
    )

    plot_heatmap(
        aggregated, args.metric,
        os.path.join(args.output_dir, "phase_diagram_heatmap.png"),
        args.grid_resolution,
    )

    boundary_points = find_phase_boundaries(
        aggregated, args.metric,
        os.path.join(args.output_dir, "phase_boundaries.png"),
        args.grid_resolution, args.smooth_sigma,
    )

    boundary_path = os.path.join(args.output_dir, "boundary_points.json")
    with open(boundary_path, "w") as f:
        json.dump(boundary_points, f, indent=2)
    logger.info(f"Found {len(boundary_points)} boundary points, saved to {boundary_path}")

    # Save summary
    summary = {
        "metric": args.metric,
        "num_points": len(aggregated),
        "best_point": max(aggregated.items(), key=lambda x: x[1]["mean"]),
        "worst_point": min(aggregated.items(), key=lambda x: x[1]["mean"]),
        "num_boundary_points": len(boundary_points),
    }
    summary["best_point"] = {
        "alpha": summary["best_point"][0][0],
        "beta": summary["best_point"][0][1],
        **summary["best_point"][1],
    }
    summary["worst_point"] = {
        "alpha": summary["worst_point"][0][0],
        "beta": summary["worst_point"][0][1],
        **summary["worst_point"][1],
    }

    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
