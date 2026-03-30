#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stability_analysis import (
    build_stability_map,
    analyze_stability,
    compute_gradient_variance,
    group_starvation_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stability_map")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and visualize stability map")
    parser.add_argument("--output_dir", type=str, default="./results/stability_analysis")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--rho_min", type=float, default=0.05)
    parser.add_argument("--rho_max", type=float, default=6.0)
    parser.add_argument("--p_min", type=float, default=0.02)
    parser.add_argument("--p_max", type=float, default=0.98)
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--empirical_results", type=str, default=None)
    return parser.parse_args()


def plot_stability_map(smap, output_dir, G, kl_coef, empirical=None):
    rho_range = smap["rho_range"]
    p_range = smap["p_range"]
    regime_int_map = smap["regime_int_map"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    regime_colors = ["#2ecc71", "#e74c3c", "#e67e22", "#f39c12", "#9b59b6"]
    regime_labels = ["Convergent", "Gradient Starved", "Unstable", "Mildly Unstable", "At Risk"]
    cmap = ListedColormap(regime_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    im0 = axes[0].pcolormesh(
        rho_range, p_range, regime_int_map,
        cmap=cmap, norm=norm, shading="auto",
    )

    axes[0].plot(smap["rho_min_curve"], p_range, "k--", linewidth=2, label=r"$\rho_{\min}$ (Thm 3)")
    axes[0].plot(smap["rho_max_curve"], p_range, "k-.", linewidth=2, label=r"$\rho_{\max}$ (Prop 1)")
    axes[0].plot(smap["rho_star_curve"], p_range, "w-", linewidth=2.5, label=r"$\rho^*$ (AdaBalance)")

    if empirical is not None:
        for point in empirical:
            marker = "o" if point["regime"] == "convergent" else ("x" if point["regime"] == "gradient_starved" else "s")
            color = "#2ecc71" if point["regime"] == "convergent" else ("#e74c3c" if point["regime"] == "gradient_starved" else "#e67e22")
            axes[0].scatter(
                point["rho"], point["p_hat"],
                c=color, s=80, marker=marker, edgecolors="black", linewidths=1, zorder=10,
            )

    axes[0].set_xlabel(r"$\rho$ (effective balance ratio)", fontsize=14)
    axes[0].set_ylabel(r"$p$ (success probability)", fontsize=14)
    axes[0].set_title(f"(a) Stability Map (G={G})", fontsize=15)
    axes[0].legend(fontsize=10, loc="upper left")

    legend_patches = [Patch(facecolor=c, label=l) for c, l in zip(regime_colors[:3], regime_labels[:3])]
    axes[0].legend(handles=legend_patches + axes[0].get_legend_handles_labels()[0][-3:],
                   fontsize=9, loc="upper left")

    variance_map = smap["variance_map"]
    im1 = axes[1].pcolormesh(
        rho_range, p_range, np.log10(variance_map + 1e-10),
        cmap="viridis", shading="auto",
    )
    axes[1].plot(smap["rho_star_curve"], p_range, "r-", linewidth=2, label=r"$\rho^*$ (min variance)")
    axes[1].set_xlabel(r"$\rho$", fontsize=14)
    axes[1].set_ylabel(r"$p$", fontsize=14)
    axes[1].set_title("(b) Log Gradient Variance", fontsize=15)
    fig.colorbar(im1, ax=axes[1], label=r"$\log_{10} \mathrm{Var}(\nabla \hat{L}_\rho)$")
    axes[1].legend(fontsize=10)

    p_slices = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(p_slices)))
    for p_val, color in zip(p_slices, colors):
        p_idx = np.argmin(np.abs(p_range - p_val))
        var_slice = variance_map[p_idx, :]
        axes[2].plot(rho_range, var_slice, color=color, linewidth=2, label=f"p={p_val:.1f}")

        bounds = analyze_stability(p_val, G, kl_coef)
        axes[2].axvline(bounds.rho_min, color=color, linestyle="--", alpha=0.5)
        axes[2].axvline(bounds.rho_max, color=color, linestyle="-.", alpha=0.5)
        axes[2].plot(bounds.rho_star, compute_gradient_variance(bounds.rho_star, bounds),
                     "o", color=color, markersize=8)

    axes[2].set_xlabel(r"$\rho$", fontsize=14)
    axes[2].set_ylabel(r"$\mathrm{Var}(\nabla \hat{L}_\rho)$", fontsize=14)
    axes[2].set_title("(c) Variance Slices", fontsize=15)
    axes[2].legend(fontsize=9)
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        r"Stability Analysis of $\rho$-Weighted GRPO Under Binary Rewards",
        fontsize=16, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_map.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "stability_map.png"), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Stability map saved")


def plot_gsr_analysis(p_range, G_values, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for G in G_values:
        gsr = [group_starvation_rate(p, G) for p in p_range]
        axes[0].plot(p_range, gsr, linewidth=2, label=f"G={G}")

    axes[0].axhline(0.8, color="red", linestyle="--", alpha=0.5, label=r"$\tau_{star}=0.8$")
    axes[0].set_xlabel(r"$p$ (success probability)", fontsize=13)
    axes[0].set_ylabel("Group Starvation Rate", fontsize=13)
    axes[0].set_title("(a) GSR vs Success Probability", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    for G in G_values:
        rho_min_vals = []
        for p in p_range:
            bounds = analyze_stability(p, G)
            rho_min_vals.append(bounds.rho_min)
        axes[1].plot(p_range, rho_min_vals, linewidth=2, label=f"G={G}")

    axes[1].set_xlabel(r"$p$ (success probability)", fontsize=13)
    axes[1].set_ylabel(r"$\rho_{\min}$", fontsize=13)
    axes[1].set_title(r"(b) $\rho_{\min}$ vs Success Probability", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gsr_analysis.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rho_range = np.linspace(args.rho_min, args.rho_max, args.resolution)
    p_range = np.linspace(args.p_min, args.p_max, args.resolution)

    logger.info("Computing stability map: %d x %d grid, G=%d", len(rho_range), len(p_range), args.group_size)
    smap = build_stability_map(rho_range, p_range, args.group_size, args.kl_coef, args.clip_range)

    empirical = None
    if args.empirical_results and os.path.exists(args.empirical_results):
        with open(args.empirical_results) as f:
            empirical = json.load(f)
        logger.info("Loaded %d empirical points", len(empirical))

    plot_stability_map(smap, args.output_dir, args.group_size, args.kl_coef, empirical)

    plot_gsr_analysis(p_range, [2, 4, 8, 16], args.output_dir)

    analysis = {
        "group_size": args.group_size,
        "kl_coef": args.kl_coef,
        "clip_range": args.clip_range,
        "rho_range": [float(r) for r in rho_range],
        "p_range": [float(p) for p in p_range],
        "sample_bounds": [],
    }

    for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        bounds = analyze_stability(p, args.group_size, args.kl_coef, args.clip_range)
        analysis["sample_bounds"].append({
            "p": p,
            "rho_min": bounds.rho_min,
            "rho_max": bounds.rho_max,
            "rho_star": bounds.rho_star,
            "V_plus": bounds.V_plus,
            "V_minus": bounds.V_minus,
            "C_pG": bounds.C_pG,
            "GSR": bounds.GSR,
        })

    with open(os.path.join(args.output_dir, "stability_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info("Stability analysis complete")
    for entry in analysis["sample_bounds"]:
        logger.info(
            "p=%.2f: rho_min=%.3f, rho_max=%.3f, rho*=%.3f, GSR=%.3f",
            entry["p"], entry["rho_min"], entry["rho_max"], entry["rho_star"], entry["GSR"],
        )


if __name__ == "__main__":
    main()
