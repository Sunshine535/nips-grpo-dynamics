#!/usr/bin/env python3
"""
Diagnostic analysis: aggregate training logs and evaluation results across
all runs, produce comparison tables and publication-quality figures.

Outputs:
  figures/zero_ratio_curves.pdf
  figures/grad_norms.pdf
  figures/accuracy_curves.pdf
  figures/strategy_comparison.pdf
  tables/strategy_comparison.csv
  tables/strategy_comparison.tex
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("diagnostic")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not found; will skip figure generation")

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False
    logger.warning("pandas not found; tables will be JSON-only")


def parse_args():
    p = argparse.ArgumentParser(description="HalluZero Diagnostic Analysis")
    p.add_argument("--results_dir", type=str, default="./results",
                    help="Root results directory containing baseline/ and sweep/")
    p.add_argument("--output_dir", type=str, default="./results/analysis")
    p.add_argument("--figure_format", type=str, default="pdf",
                    choices=["pdf", "png", "svg"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_trainer_state(run_dir: str) -> list[dict] | None:
    """Parse trainer_state.json for per-step metrics."""
    path = os.path.join(run_dir, "trainer_state.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    return state.get("log_history", [])


def parse_train_log(run_dir: str) -> list[dict]:
    """Fallback: parse train.log for metrics if trainer_state unavailable."""
    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        return []

    entries = []
    step_re = re.compile(r"'(loss|grad_norm|learning_rate|epoch)':\s*([\d.eE+-]+)")
    with open(log_path) as f:
        for line in f:
            matches = step_re.findall(line)
            if matches:
                entry = {k: float(v) for k, v in matches}
                entries.append(entry)
    return entries


def load_eval_summary(run_dir: str) -> dict | None:
    """Load evaluation summary.json from a run's eval subdirectory."""
    for candidate in [
        os.path.join(run_dir, "eval", "summary.json"),
        os.path.join(run_dir, "summary.json"),
        os.path.join(run_dir, "eval_results", "summary.json"),
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


def identify_run(run_dir: str) -> dict:
    """Extract strategy/hparam/seed from directory structure or config."""
    info = {"path": run_dir, "strategy": "unknown", "hparam": "", "seed": ""}

    config_path = os.path.join(run_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        zs = cfg.get("zero_score", {})
        info["strategy"] = zs.get("strategy", "unknown")

    parts = Path(run_dir).parts
    for p in parts:
        if p.startswith("seed_"):
            info["seed"] = p.replace("seed_", "")
        if p == "baseline":
            info["strategy"] = "baseline"

    if "sweep" in str(run_dir):
        for i, p in enumerate(parts):
            if p in ("clip", "temperature", "curriculum", "relabel"):
                info["strategy"] = p
                if i + 1 < len(parts) and not parts[i + 1].startswith("seed"):
                    info["hparam"] = parts[i + 1]

    return info


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_all_runs(results_dir: str) -> list[dict]:
    """Walk results_dir and collect metadata + metrics for every completed run."""
    runs = []
    for dirpath, dirnames, files in os.walk(results_dir):
        if "training_config.json" in files or "trainer_state.json" in files:
            info = identify_run(dirpath)
            log_history = parse_trainer_state(dirpath) or parse_train_log(dirpath)
            eval_summary = load_eval_summary(dirpath)

            info["log_history"] = log_history
            info["eval_summary"] = eval_summary
            info["has_logs"] = len(log_history) > 0
            info["has_eval"] = eval_summary is not None
            runs.append(info)

    logger.info("Collected %d runs from %s", len(runs), results_dir)
    return runs


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def extract_metric_curve(log_history: list[dict], key: str) -> tuple:
    """Return (steps, values) for a given metric key."""
    steps, values = [], []
    for i, entry in enumerate(log_history):
        if key in entry:
            step = entry.get("step", i)
            steps.append(step)
            values.append(entry[key])
    return np.array(steps), np.array(values)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "baseline": "#888888",
    "clip": "#e74c3c",
    "temperature": "#3498db",
    "curriculum": "#2ecc71",
    "relabel": "#9b59b6",
}

STRATEGY_LABELS = {
    "baseline": "GRPO Baseline",
    "clip": "HalluZero-Clip",
    "temperature": "HalluZero-Temp",
    "curriculum": "HalluZero-Curr",
    "relabel": "HalluZero-Soft",
}


def _style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)


def plot_zero_ratio_curves(runs: list[dict], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = set()
    for run in runs:
        strat = run["strategy"]
        steps, vals = extract_metric_curve(run["log_history"], "zero_score_ratio")
        if len(steps) == 0:
            steps, vals = extract_metric_curve(run["log_history"], "loss")
            if len(steps) == 0:
                continue

        label = STRATEGY_LABELS.get(strat, strat) if strat not in plotted else None
        color = STRATEGY_COLORS.get(strat, "#666666")
        ax.plot(steps, vals, color=color, alpha=0.6, linewidth=1.5, label=label)
        plotted.add(strat)

    _style_axis(ax, "Zero-Score Ratio Over Training",
                "Training Step", "Zero-Score Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_grad_norms(runs: list[dict], output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plotted = set()
    for run in runs:
        strat = run["strategy"]
        steps, vals = extract_metric_curve(run["log_history"], "grad_norm")
        if len(steps) == 0:
            continue

        label = STRATEGY_LABELS.get(strat, strat) if strat not in plotted else None
        color = STRATEGY_COLORS.get(strat, "#666666")
        axes[0].plot(steps, vals, color=color, alpha=0.6, linewidth=1.2, label=label)
        plotted.add(strat)

    _style_axis(axes[0], "Gradient Norm Over Training",
                "Training Step", "Gradient Norm")

    strat_norms = defaultdict(list)
    for run in runs:
        strat = run["strategy"]
        _, vals = extract_metric_curve(run["log_history"], "grad_norm")
        if len(vals) > 0:
            strat_norms[strat].append(float(np.mean(vals)))

    if strat_norms:
        strats = sorted(strat_norms.keys())
        means = [np.mean(strat_norms[s]) for s in strats]
        stds = [np.std(strat_norms[s]) if len(strat_norms[s]) > 1 else 0 for s in strats]
        colors = [STRATEGY_COLORS.get(s, "#666666") for s in strats]
        labels = [STRATEGY_LABELS.get(s, s) for s in strats]
        bars = axes[1].bar(range(len(strats)), means, yerr=stds,
                           color=colors, alpha=0.8, capsize=4)
        axes[1].set_xticks(range(len(strats)))
        axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        _style_axis(axes[1], "Mean Gradient Norm by Strategy", "", "Mean Grad Norm")
        axes[1].get_legend().remove() if axes[1].get_legend() else None

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_accuracy_curves(runs: list[dict], output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plotted = set()
    for run in runs:
        strat = run["strategy"]
        steps, vals = extract_metric_curve(run["log_history"], "eval_accuracy")
        if len(steps) == 0:
            steps, vals = extract_metric_curve(run["log_history"], "eval_loss")
            if len(steps) == 0:
                continue
            vals = -vals  # invert for comparison

        label = STRATEGY_LABELS.get(strat, strat) if strat not in plotted else None
        color = STRATEGY_COLORS.get(strat, "#666666")
        axes[0].plot(steps, vals, color=color, alpha=0.6, linewidth=1.5,
                     label=label, marker="o", markersize=3)
        plotted.add(strat)

    _style_axis(axes[0], "Accuracy Over Training",
                "Training Step", "Accuracy / -Loss")

    strat_acc = defaultdict(list)
    for run in runs:
        ev = run.get("eval_summary")
        if ev and "gsm8k" in ev:
            strat_acc[run["strategy"]].append(ev["gsm8k"].get("accuracy_greedy", 0))

    if strat_acc:
        strats = sorted(strat_acc.keys())
        means = [np.mean(strat_acc[s]) for s in strats]
        stds = [np.std(strat_acc[s]) if len(strat_acc[s]) > 1 else 0 for s in strats]
        colors = [STRATEGY_COLORS.get(s, "#666666") for s in strats]
        labels = [STRATEGY_LABELS.get(s, s) for s in strats]
        axes[1].bar(range(len(strats)), means, yerr=stds,
                    color=colors, alpha=0.8, capsize=4)
        axes[1].set_xticks(range(len(strats)))
        axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        _style_axis(axes[1], "Final GSM8K Accuracy by Strategy", "", "Accuracy")
        if axes[1].get_legend():
            axes[1].get_legend().remove()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_strategy_comparison(comparison: list[dict], output_path: str):
    if not comparison:
        logger.warning("No comparison data for strategy_comparison plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("gsm8k_accuracy", "GSM8K Accuracy"),
        ("math_accuracy", "MATH Accuracy"),
        ("zero_score_ratio", "Zero-Score Ratio"),
        ("mean_grad_norm", "Mean Gradient Norm"),
    ]

    for ax, (key, title) in zip(axes.flat, metrics):
        strats = []
        vals = []
        errs = []
        for entry in comparison:
            if key in entry and entry[key] is not None:
                strats.append(entry["strategy_label"])
                vals.append(entry[key])
                errs.append(entry.get(f"{key}_std", 0))

        if not strats:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#999")
            ax.set_title(title, fontsize=12, fontweight="bold")
            continue

        colors = [STRATEGY_COLORS.get(e.get("strategy", ""), "#666")
                  for e in comparison if key in e and e[key] is not None]
        ax.barh(range(len(strats)), vals, xerr=errs, color=colors, alpha=0.85,
                capsize=4, height=0.6)
        ax.set_yticks(range(len(strats)))
        ax.set_yticklabels(strats, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.3)

        for i, v in enumerate(vals):
            fmt = f"{v:.4f}" if v < 1 else f"{v:.2f}"
            ax.text(v + max(vals) * 0.02, i, fmt, va="center", fontsize=9)

    fig.suptitle("HalluZero Strategy Comparison", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(runs: list[dict]) -> list[dict]:
    """Aggregate per-strategy metrics across seeds into a comparison table."""
    strat_data = defaultdict(lambda: {
        "gsm8k_acc": [], "math_acc": [], "zero_ratio": [], "grad_norm": [],
        "pass_at_1": [], "avg_tokens": [], "hparams": set(),
    })

    for run in runs:
        key = run["strategy"]
        hp = run.get("hparam", "")
        if hp:
            key = f"{key}_{hp}"
        bucket = strat_data[key]
        bucket["strategy"] = run["strategy"]
        bucket["hparams"].add(hp)

        ev = run.get("eval_summary")
        if ev:
            if "gsm8k" in ev:
                bucket["gsm8k_acc"].append(ev["gsm8k"].get("accuracy_greedy", 0))
                bucket["zero_ratio"].append(ev["gsm8k"].get("zero_score_ratio", 0))
                bucket["pass_at_1"].append(ev["gsm8k"].get("pass_at_1", 0))
                bucket["avg_tokens"].append(ev["gsm8k"].get("avg_tokens", 0))
            if "math" in ev:
                bucket["math_acc"].append(ev["math"].get("accuracy_greedy", 0))

        _, gnorms = extract_metric_curve(run["log_history"], "grad_norm")
        if len(gnorms) > 0:
            bucket["grad_norm"].append(float(np.mean(gnorms)))

    rows = []
    for key, data in sorted(strat_data.items()):
        row = {
            "strategy": data["strategy"],
            "strategy_label": STRATEGY_LABELS.get(data["strategy"], data["strategy"]),
            "hparam": ", ".join(sorted(data["hparams"])) if data["hparams"] else "",
            "config_key": key,
        }

        for metric, field in [
            ("gsm8k_accuracy", "gsm8k_acc"),
            ("math_accuracy", "math_acc"),
            ("zero_score_ratio", "zero_ratio"),
            ("mean_grad_norm", "grad_norm"),
            ("pass_at_1", "pass_at_1"),
            ("avg_tokens", "avg_tokens"),
        ]:
            vals = data[field]
            if vals:
                row[metric] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                row[metric] = None
                row[f"{metric}_std"] = None

        rows.append(row)

    return rows


def save_tables(comparison: list[dict], output_dir: str):
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(tables_dir, "strategy_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Saved JSON table")

    if HAS_PD:
        df = pd.DataFrame(comparison)
        display_cols = [c for c in df.columns if not c.endswith("_std")]
        df_display = df[display_cols]
        df_display.to_csv(os.path.join(tables_dir, "strategy_comparison.csv"), index=False)
        logger.info("Saved CSV table")

        latex = df_display.to_latex(index=False, float_format="%.4f", na_rep="-")
        with open(os.path.join(tables_dir, "strategy_comparison.tex"), "w") as f:
            f.write(latex)
        logger.info("Saved LaTeX table")
    else:
        logger.info("Skipping CSV/LaTeX (pandas not available)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Scanning results in %s", args.results_dir)
    runs = collect_all_runs(args.results_dir)

    if not runs:
        logger.error("No completed runs found in %s", args.results_dir)
        sys.exit(1)

    logger.info("Runs by strategy:")
    strat_counts = defaultdict(int)
    for r in runs:
        strat_counts[r["strategy"]] += 1
    for s, c in sorted(strat_counts.items()):
        logger.info("  %-15s %d runs", s, c)

    comparison = build_comparison_table(runs)

    save_tables(comparison, args.output_dir)

    if HAS_MPL:
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fmt = args.figure_format

        plot_zero_ratio_curves(
            runs, os.path.join(fig_dir, f"zero_ratio_curves.{fmt}"))
        plot_grad_norms(
            runs, os.path.join(fig_dir, f"grad_norms.{fmt}"))
        plot_accuracy_curves(
            runs, os.path.join(fig_dir, f"accuracy_curves.{fmt}"))
        plot_strategy_comparison(
            comparison, os.path.join(fig_dir, f"strategy_comparison.{fmt}"))
    else:
        logger.warning("Skipping figures (matplotlib not installed)")

    logger.info("=" * 65)
    logger.info("STRATEGY COMPARISON SUMMARY")
    logger.info("=" * 65)
    header = f"{'Strategy':<25s} {'GSM8K':>8s} {'MATH':>8s} {'Zero%':>8s} {'GNorm':>8s} {'P@1':>8s}"
    logger.info(header)
    logger.info("-" * 65)
    for row in comparison:
        gsm = f"{row['gsm8k_accuracy']:.4f}" if row['gsm8k_accuracy'] is not None else "   -   "
        math_a = f"{row['math_accuracy']:.4f}" if row['math_accuracy'] is not None else "   -   "
        zr = f"{row['zero_score_ratio']:.4f}" if row['zero_score_ratio'] is not None else "   -   "
        gn = f"{row['mean_grad_norm']:.4f}" if row['mean_grad_norm'] is not None else "   -   "
        p1 = f"{row['pass_at_1']:.4f}" if row['pass_at_1'] is not None else "   -   "
        logger.info(f"{row['config_key']:<25s} {gsm:>8s} {math_a:>8s} {zr:>8s} {gn:>8s} {p1:>8s}")

    logger.info("Analysis complete. Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
