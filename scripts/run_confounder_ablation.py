#!/usr/bin/env python3
"""Confounder ablation: test whether rho stability analysis remains predictive
when confounders (group size G, KL coefficient lambda_KL) are varied.

Addresses NeurIPS reviewer concern that stability boundaries may be confounded
by hyperparameters other than rho. Runs three ablation axes:

  Ablation 1 - Group size G:    G in {2, 4, 8, 16}
  Ablation 2 - KL coefficient:  lambda_KL in {0.01, 0.05, 0.1, 0.2}
  Ablation 3 - Interaction:     2 extreme (G, lambda_KL) combos, full rho sweep

For each (G, lambda_KL) combination the script:
  1. Computes theoretical stability boundaries (rho_min, rho_max, rho_star)
  2. Picks 3 representative rho values (below boundary, optimal, above boundary)
  3. Launches GRPO training via train_rho_sweep.py with modified config
  4. Compares theoretical predictions to empirical training outcomes
  5. Saves structured JSON results

Usage:
  python scripts/run_confounder_ablation.py --output_dir results/confounder_ablation
  python scripts/run_confounder_ablation.py --quick --max_steps 100
"""
import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stability_analysis import (
    analyze_stability,
    classify_regime,
    group_starvation_rate,
    compute_gradient_variance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("confounder_ablation")

# ---------------------------------------------------------------------------
# Defaults matching the paper
# ---------------------------------------------------------------------------
DEFAULT_GROUP_SIZES = [2, 4, 8, 16]
DEFAULT_KL_COEFS = [0.01, 0.05, 0.1, 0.2]
DEFAULT_CLIP_RANGE = 0.2
DEFAULT_P_ESTIMATE = 0.4  # moderate difficulty baseline for boundary computation

QUICK_GROUP_SIZES = [2, 8]
QUICK_KL_COEFS = [0.01, 0.1]

# Extreme combos for interaction ablation
INTERACTION_COMBOS = [
    {"G": 2, "kl_coef": 0.01},   # small group, weak regularization
    {"G": 16, "kl_coef": 0.2},   # large group, strong regularization
]
QUICK_INTERACTION_COMBOS = [
    {"G": 2, "kl_coef": 0.01},
    {"G": 16, "kl_coef": 0.2},
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Confounder ablation for rho stability analysis"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/confounder_ablation",
        help="Directory for all ablation outputs",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced grids and fewer steps for fast validation",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override max training steps per run (default: 200, quick: 100)",
    )
    parser.add_argument(
        "--base_config", type=str, default="configs/rho_sweep.yaml",
        help="Base config file (will be modified per ablation condition)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for training runs",
    )
    parser.add_argument(
        "--p_estimate", type=float, default=DEFAULT_P_ESTIMATE,
        help="Task difficulty estimate for theoretical boundary computation",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Compute theoretical predictions only, skip actual training",
    )
    parser.add_argument(
        "--interaction_rho_values", nargs="+", type=float, default=None,
        help="Full rho sweep values for interaction ablation (default: 7 points)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_base_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_temp_config(base_cfg: dict, group_size: int, kl_coef: float,
                     max_steps: int, output_subdir: str) -> str:
    """Create a temporary YAML config with modified G and lambda_KL."""
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["num_generations"] = group_size
    cfg["training"]["kl_coef"] = kl_coef
    cfg["sweep"]["coarse_max_steps"] = max_steps
    cfg["output"]["sweep_coarse_dir"] = output_subdir
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="confounder_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


# ---------------------------------------------------------------------------
# Theoretical predictions
# ---------------------------------------------------------------------------
def compute_theoretical_boundaries(p: float, G: int, kl_coef: float,
                                   clip_range: float = DEFAULT_CLIP_RANGE) -> dict:
    """Compute stability boundaries for a given (G, kl_coef) pair."""
    bounds = analyze_stability(p, G, kl_coef, clip_range)
    gsr = group_starvation_rate(p, G)
    return {
        "p": p,
        "G": G,
        "kl_coef": kl_coef,
        "clip_range": clip_range,
        "rho_min": float(bounds.rho_min),
        "rho_max": float(bounds.rho_max),
        "rho_star": float(bounds.rho_star),
        "V_plus": float(bounds.V_plus),
        "V_minus": float(bounds.V_minus),
        "C_pG": float(bounds.C_pG),
        "GSR": float(gsr),
        "gradient_variance_at_star": float(
            compute_gradient_variance(bounds.rho_star, bounds)
        ),
    }


def pick_representative_rhos(bounds_dict: dict) -> list[dict]:
    """Pick 3 representative rho values: below stability, optimal, above stability.

    Returns list of dicts with 'rho' and 'expected_regime'.
    """
    rho_min = bounds_dict["rho_min"]
    rho_max = bounds_dict["rho_max"]
    rho_star = bounds_dict["rho_star"]

    rhos = []

    # Low rho: below rho_min (should be gradient-starved)
    rho_low = max(0.05, rho_min * 0.5) if rho_min > 0.1 else 0.05
    rhos.append({"rho": round(rho_low, 3), "label": "below_boundary",
                 "expected_regime": "gradient_starved"})

    # Optimal rho: at rho_star (should be convergent)
    rhos.append({"rho": round(rho_star, 3), "label": "optimal",
                 "expected_regime": "convergent"})

    # High rho: above rho_max (should be unstable)
    rho_high = min(rho_max * 1.5, 10.0)
    rhos.append({"rho": round(rho_high, 3), "label": "above_boundary",
                 "expected_regime": "unstable"})

    return rhos


# ---------------------------------------------------------------------------
# Training launch
# ---------------------------------------------------------------------------
def launch_training_run(rho: float, seed: int, config_path: str,
                        output_dir: str, max_steps: int) -> dict:
    """Launch a single train_rho_sweep.py run and collect results.

    Respects CUDA_VISIBLE_DEVICES from the environment for GPU assignment.
    Returns dict with training outcome metrics.
    """
    run_dir = os.path.join(output_dir, f"rho{rho:.3f}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(Path(__file__).resolve().parent, "train_rho_sweep.py"),
        "--rho", str(rho),
        "--seed", str(seed),
        "--config", config_path,
        "--output_dir", run_dir,
        "--max_steps", str(max_steps),
        "--resume_from_checkpoint", "none",
    ]

    env = os.environ.copy()
    logger.info("Launching: rho=%.3f seed=%d -> %s", rho, seed, run_dir)
    logger.info("  cmd: %s", " ".join(cmd))

    start = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=7200,
        )
        elapsed = time.time() - start
        success = result.returncode == 0

        if not success:
            logger.warning("Run failed (rho=%.3f): %s", rho,
                           result.stderr[-500:] if result.stderr else "no stderr")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        success = False
        logger.warning("Run timed out (rho=%.3f) after %.0fs", rho, elapsed)

    # Collect output metrics
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    telemetry_path = os.path.join(run_dir, "stability_telemetry.json")
    step_logs_path = os.path.join(run_dir, "step_logs.json")

    training_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            training_metrics = json.load(f)

    telemetry = []
    if os.path.exists(telemetry_path):
        with open(telemetry_path) as f:
            telemetry = json.load(f)

    step_logs = []
    if os.path.exists(step_logs_path):
        with open(step_logs_path) as f:
            step_logs = json.load(f)

    return {
        "rho": rho,
        "seed": seed,
        "success": success,
        "elapsed_sec": round(elapsed, 1),
        "run_dir": run_dir,
        "training_metrics": training_metrics,
        "n_telemetry_steps": len(telemetry),
        "telemetry_summary": _summarize_telemetry(telemetry),
        "step_log_summary": _summarize_step_logs(step_logs),
    }


def _summarize_telemetry(telemetry: list) -> dict:
    """Extract key statistics from stability telemetry."""
    if not telemetry:
        return {}
    rewards = [t.get("reward_mean", 0) for t in telemetry if "reward_mean" in t]
    kls = [t.get("kl", 0) for t in telemetry if "kl" in t]
    regimes = [t.get("regime", "") for t in telemetry if "regime" in t]

    summary = {}
    if rewards:
        summary["reward_mean"] = float(np.mean(rewards))
        summary["reward_final"] = float(rewards[-1])
        summary["reward_trend"] = float(
            np.polyfit(range(len(rewards)), rewards, 1)[0]
        ) if len(rewards) >= 2 else 0.0
    if kls:
        summary["kl_mean"] = float(np.mean(kls))
        summary["kl_max"] = float(np.max(kls))
        summary["kl_final"] = float(kls[-1])
    if regimes:
        from collections import Counter
        regime_counts = Counter(regimes)
        summary["dominant_regime"] = regime_counts.most_common(1)[0][0]
        summary["regime_distribution"] = dict(regime_counts)

    return summary


def _summarize_step_logs(step_logs: list) -> dict:
    """Extract key statistics from step-level training logs."""
    if not step_logs:
        return {}

    losses = [s.get("loss", s.get("train_loss")) for s in step_logs
              if s.get("loss") is not None or s.get("train_loss") is not None]
    grad_norms = [s.get("grad_norm", 0) for s in step_logs if "grad_norm" in s]
    reward_means = [s.get("reward/mean", s.get("reward_mean"))
                    for s in step_logs
                    if s.get("reward/mean") is not None or s.get("reward_mean") is not None]

    summary = {}
    if losses:
        valid_losses = [l for l in losses if l is not None]
        if valid_losses:
            summary["loss_mean"] = float(np.mean(valid_losses))
            summary["loss_final"] = float(valid_losses[-1])
            summary["loss_nan_count"] = sum(
                1 for l in valid_losses if np.isnan(l) or np.isinf(l)
            )
    if grad_norms:
        summary["grad_norm_mean"] = float(np.mean(grad_norms))
        summary["grad_norm_max"] = float(np.max(grad_norms))
    if reward_means:
        valid_rewards = [r for r in reward_means if r is not None]
        if valid_rewards:
            summary["reward_mean"] = float(np.mean(valid_rewards))
            summary["reward_final"] = float(valid_rewards[-1])

    return summary


# ---------------------------------------------------------------------------
# Empirical regime classification from training results
# ---------------------------------------------------------------------------
def classify_empirical_regime(run_result: dict) -> str:
    """Classify the observed training regime from collected metrics.

    Uses the same heuristics as run_robustness_test.py but adapted for
    real training telemetry (not simulation).
    """
    tel = run_result.get("telemetry_summary", {})
    logs = run_result.get("step_log_summary", {})

    # Check for explicit regime from telemetry
    dominant = tel.get("dominant_regime", "")
    if dominant in ("gradient_starved", "unstable"):
        return dominant

    # Heuristic classification from metrics
    reward_final = tel.get("reward_final", logs.get("reward_final", None))
    reward_mean = tel.get("reward_mean", logs.get("reward_mean", None))
    reward_trend = tel.get("reward_trend", 0.0)
    kl_max = tel.get("kl_max", 0.0)
    kl_final = tel.get("kl_final", 0.0)
    loss_nan = logs.get("loss_nan_count", 0)
    grad_norm_max = logs.get("grad_norm_max", 0.0)

    # NaN losses or extreme grad norms -> unstable
    if loss_nan > 0 or grad_norm_max > 50.0:
        return "unstable"

    # KL divergence explosion -> unstable
    if kl_max > 1.0 or kl_final > 0.5:
        return "unstable"

    # Very low reward + negative trend -> gradient starved
    if reward_mean is not None and reward_mean < 0.1:
        return "gradient_starved"
    if reward_final is not None and reward_final < 0.1 and reward_trend < -0.0005:
        return "gradient_starved"

    # Training did not complete
    if not run_result.get("success", False):
        return "unstable"

    return "convergent"


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------
def run_ablation_group_size(args, base_cfg: dict) -> dict:
    """Ablation 1: Vary group size G, keeping lambda_KL at default."""
    G_values = QUICK_GROUP_SIZES if args.quick else DEFAULT_GROUP_SIZES
    kl_coef = base_cfg["training"]["kl_coef"]
    max_steps = args.max_steps or (100 if args.quick else 200)

    logger.info("=" * 70)
    logger.info("ABLATION 1: Group Size G  (kl_coef=%.3f fixed)", kl_coef)
    logger.info("  G values: %s", G_values)
    logger.info("=" * 70)

    results = {"ablation": "group_size", "kl_coef": kl_coef, "conditions": []}

    for G in G_values:
        logger.info("\n--- G=%d ---", G)

        # Theoretical predictions
        theory = compute_theoretical_boundaries(args.p_estimate, G, kl_coef)
        rho_picks = pick_representative_rhos(theory)
        logger.info("  Theory: rho_min=%.3f, rho_star=%.3f, rho_max=%.3f, GSR=%.4f",
                     theory["rho_min"], theory["rho_star"], theory["rho_max"],
                     theory["GSR"])

        condition = {
            "G": G,
            "kl_coef": kl_coef,
            "theory": theory,
            "rho_picks": rho_picks,
            "runs": [],
        }

        if not args.dry_run:
            subdir = os.path.join(args.output_dir, "ablation_G", f"G{G}")
            config_path = make_temp_config(
                base_cfg, G, kl_coef, max_steps, subdir,
            )
            try:
                for pick in rho_picks:
                    run_result = launch_training_run(
                        pick["rho"], args.seed, config_path, subdir, max_steps,
                    )
                    observed = classify_empirical_regime(run_result)
                    run_result["expected_regime"] = pick["expected_regime"]
                    run_result["observed_regime"] = observed
                    run_result["label"] = pick["label"]
                    run_result["prediction_match"] = (
                        pick["expected_regime"] == observed
                    )
                    condition["runs"].append(run_result)
                    logger.info(
                        "  G=%d rho=%.3f [%s]: expected=%s, observed=%s -> %s",
                        G, pick["rho"], pick["label"],
                        pick["expected_regime"], observed,
                        "MATCH" if run_result["prediction_match"] else "MISMATCH",
                    )
            finally:
                os.unlink(config_path)

        results["conditions"].append(condition)

    return results


def run_ablation_kl_coef(args, base_cfg: dict) -> dict:
    """Ablation 2: Vary KL coefficient lambda_KL, keeping G at default."""
    kl_values = QUICK_KL_COEFS if args.quick else DEFAULT_KL_COEFS
    G = base_cfg["training"]["num_generations"]
    max_steps = args.max_steps or (100 if args.quick else 200)

    logger.info("=" * 70)
    logger.info("ABLATION 2: KL Coefficient lambda_KL  (G=%d fixed)", G)
    logger.info("  lambda_KL values: %s", kl_values)
    logger.info("=" * 70)

    results = {"ablation": "kl_coef", "G": G, "conditions": []}

    for kl_coef in kl_values:
        logger.info("\n--- lambda_KL=%.3f ---", kl_coef)

        theory = compute_theoretical_boundaries(args.p_estimate, G, kl_coef)
        rho_picks = pick_representative_rhos(theory)
        logger.info("  Theory: rho_min=%.3f, rho_star=%.3f, rho_max=%.3f",
                     theory["rho_min"], theory["rho_star"], theory["rho_max"])

        condition = {
            "G": G,
            "kl_coef": kl_coef,
            "theory": theory,
            "rho_picks": rho_picks,
            "runs": [],
        }

        if not args.dry_run:
            subdir = os.path.join(args.output_dir, "ablation_kl", f"kl{kl_coef:.3f}")
            config_path = make_temp_config(
                base_cfg, G, kl_coef, max_steps, subdir,
            )
            try:
                for pick in rho_picks:
                    run_result = launch_training_run(
                        pick["rho"], args.seed, config_path, subdir, max_steps,
                    )
                    observed = classify_empirical_regime(run_result)
                    run_result["expected_regime"] = pick["expected_regime"]
                    run_result["observed_regime"] = observed
                    run_result["label"] = pick["label"]
                    run_result["prediction_match"] = (
                        pick["expected_regime"] == observed
                    )
                    condition["runs"].append(run_result)
                    logger.info(
                        "  kl=%.3f rho=%.3f [%s]: expected=%s, observed=%s -> %s",
                        kl_coef, pick["rho"], pick["label"],
                        pick["expected_regime"], observed,
                        "MATCH" if run_result["prediction_match"] else "MISMATCH",
                    )
            finally:
                os.unlink(config_path)

        results["conditions"].append(condition)

    return results


def run_ablation_interaction(args, base_cfg: dict) -> dict:
    """Ablation 3: Interaction -- extreme (G, lambda_KL) combos with full rho sweep."""
    combos = QUICK_INTERACTION_COMBOS if args.quick else INTERACTION_COMBOS
    max_steps = args.max_steps or (100 if args.quick else 200)

    # Full rho sweep for interaction ablation
    if args.interaction_rho_values:
        rho_sweep = args.interaction_rho_values
    elif args.quick:
        rho_sweep = [0.1, 0.5, 1.0, 2.0, 5.0]
    else:
        rho_sweep = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

    logger.info("=" * 70)
    logger.info("ABLATION 3: Interaction (G x lambda_KL)")
    logger.info("  Combos: %s", combos)
    logger.info("  Rho sweep: %s", rho_sweep)
    logger.info("=" * 70)

    results = {"ablation": "interaction", "rho_sweep": rho_sweep, "conditions": []}

    for combo in combos:
        G = combo["G"]
        kl_coef = combo["kl_coef"]
        logger.info("\n--- G=%d, lambda_KL=%.3f ---", G, kl_coef)

        # Theoretical boundaries
        theory = compute_theoretical_boundaries(args.p_estimate, G, kl_coef)
        logger.info("  Theory: rho_min=%.3f, rho_star=%.3f, rho_max=%.3f, GSR=%.4f",
                     theory["rho_min"], theory["rho_star"], theory["rho_max"],
                     theory["GSR"])

        # Predict regime for each rho
        bounds = analyze_stability(args.p_estimate, G, kl_coef)
        rho_predictions = []
        for rho in rho_sweep:
            regime = classify_regime(rho, bounds)
            rho_predictions.append({
                "rho": rho,
                "predicted_regime": regime,
            })

        condition = {
            "G": G,
            "kl_coef": kl_coef,
            "theory": theory,
            "rho_predictions": rho_predictions,
            "runs": [],
        }

        if not args.dry_run:
            subdir = os.path.join(
                args.output_dir, "ablation_interaction", f"G{G}_kl{kl_coef:.3f}",
            )
            config_path = make_temp_config(
                base_cfg, G, kl_coef, max_steps, subdir,
            )
            try:
                for pred in rho_predictions:
                    rho = pred["rho"]
                    run_result = launch_training_run(
                        rho, args.seed, config_path, subdir, max_steps,
                    )
                    observed = classify_empirical_regime(run_result)
                    run_result["predicted_regime"] = pred["predicted_regime"]
                    run_result["observed_regime"] = observed
                    run_result["prediction_match"] = (
                        pred["predicted_regime"] == observed
                    )
                    condition["runs"].append(run_result)
                    logger.info(
                        "  G=%d kl=%.3f rho=%.3f: predicted=%s, observed=%s -> %s",
                        G, kl_coef, rho,
                        pred["predicted_regime"], observed,
                        "MATCH" if run_result["prediction_match"] else "MISMATCH",
                    )
            finally:
                os.unlink(config_path)

        results["conditions"].append(condition)

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def compute_summary(ablation_results: dict) -> dict:
    """Compute per-ablation and overall accuracy summaries."""
    total_runs = 0
    total_matches = 0

    for condition in ablation_results.get("conditions", []):
        for run in condition.get("runs", []):
            total_runs += 1
            if run.get("prediction_match", False):
                total_matches += 1

    return {
        "n_runs": total_runs,
        "n_matches": total_matches,
        "accuracy": total_matches / max(total_runs, 1),
    }


def compute_theory_trends(ablation_results: dict) -> dict:
    """Extract how boundaries shift across the ablation axis."""
    conditions = ablation_results.get("conditions", [])
    if not conditions:
        return {}

    trends = []
    for cond in conditions:
        t = cond.get("theory", {})
        trends.append({
            "G": t.get("G"),
            "kl_coef": t.get("kl_coef"),
            "rho_min": t.get("rho_min"),
            "rho_max": t.get("rho_max"),
            "rho_star": t.get("rho_star"),
            "GSR": t.get("GSR"),
            "stability_width": (
                t.get("rho_max", 0) - t.get("rho_min", 0)
            ),
        })

    return {
        "trends": trends,
        "ablation_type": ablation_results.get("ablation"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Confounder Ablation Study")
    logger.info("  output_dir: %s", args.output_dir)
    logger.info("  quick: %s", args.quick)
    logger.info("  dry_run: %s", args.dry_run)
    logger.info("  p_estimate: %.2f", args.p_estimate)

    base_cfg = load_base_config(args.base_config)

    # ------------------------------------------------------------------
    # Run all three ablations
    # ------------------------------------------------------------------
    all_results = {}

    # Ablation 1: Group size
    abl1 = run_ablation_group_size(args, base_cfg)
    abl1["summary"] = compute_summary(abl1)
    abl1["theory_trends"] = compute_theory_trends(abl1)
    all_results["ablation_group_size"] = abl1

    # Ablation 2: KL coefficient
    abl2 = run_ablation_kl_coef(args, base_cfg)
    abl2["summary"] = compute_summary(abl2)
    abl2["theory_trends"] = compute_theory_trends(abl2)
    all_results["ablation_kl_coef"] = abl2

    # Ablation 3: Interaction
    abl3 = run_ablation_interaction(args, base_cfg)
    abl3["summary"] = compute_summary(abl3)
    abl3["theory_trends"] = compute_theory_trends(abl3)
    all_results["ablation_interaction"] = abl3

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    total_runs = sum(
        r["summary"]["n_runs"]
        for r in [abl1, abl2, abl3]
    )
    total_matches = sum(
        r["summary"]["n_matches"]
        for r in [abl1, abl2, abl3]
    )
    overall_accuracy = total_matches / max(total_runs, 1)

    # Theory trend validation:
    # - Ablation 1: larger G should yield lower GSR and wider stability region
    # - Ablation 2: larger lambda_KL should yield higher rho_max
    g_trends = abl1["theory_trends"].get("trends", [])
    g_gsr_monotone = all(
        g_trends[i]["GSR"] >= g_trends[i + 1]["GSR"]
        for i in range(len(g_trends) - 1)
    ) if len(g_trends) > 1 else True
    g_width_monotone = all(
        g_trends[i]["stability_width"] <= g_trends[i + 1]["stability_width"]
        for i in range(len(g_trends) - 1)
    ) if len(g_trends) > 1 else True

    kl_trends = abl2["theory_trends"].get("trends", [])
    kl_rhomax_monotone = all(
        kl_trends[i]["rho_max"] <= kl_trends[i + 1]["rho_max"]
        for i in range(len(kl_trends) - 1)
    ) if len(kl_trends) > 1 else True

    overall_summary = {
        "total_runs": total_runs,
        "total_matches": total_matches,
        "overall_prediction_accuracy": overall_accuracy,
        "per_ablation_accuracy": {
            "group_size": abl1["summary"]["accuracy"],
            "kl_coef": abl2["summary"]["accuracy"],
            "interaction": abl3["summary"]["accuracy"],
        },
        "theory_validation": {
            "larger_G_lowers_GSR": g_gsr_monotone,
            "larger_G_widens_stability": g_width_monotone,
            "larger_kl_raises_rho_max": kl_rhomax_monotone,
        },
        "conclusion": (
            "rho stability analysis remains predictive across confounders"
            if overall_accuracy >= 0.7
            else "mixed results -- confounders may interact non-trivially with rho"
        ),
    }
    all_results["overall_summary"] = overall_summary

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = os.path.join(args.output_dir, "confounder_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    summary_path = os.path.join(args.output_dir, "confounder_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)

    # ------------------------------------------------------------------
    # Log final summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("CONFOUNDER ABLATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total runs:        %d", total_runs)
    logger.info("Prediction matches: %d / %d (%.1f%%)",
                total_matches, total_runs, 100 * overall_accuracy)
    logger.info("")
    logger.info("Per-ablation accuracy:")
    logger.info("  Group size G:     %.1f%%", 100 * abl1["summary"]["accuracy"])
    logger.info("  KL coef lambda:   %.1f%%", 100 * abl2["summary"]["accuracy"])
    logger.info("  Interaction:      %.1f%%", 100 * abl3["summary"]["accuracy"])
    logger.info("")
    logger.info("Theory trend validation:")
    logger.info("  Larger G -> lower GSR:          %s", g_gsr_monotone)
    logger.info("  Larger G -> wider stability:     %s", g_width_monotone)
    logger.info("  Larger lambda_KL -> higher rho_max: %s", kl_rhomax_monotone)
    logger.info("")
    logger.info("Conclusion: %s", overall_summary["conclusion"])
    logger.info("Results: %s", out_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
