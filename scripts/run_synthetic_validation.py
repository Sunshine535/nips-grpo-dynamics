#!/usr/bin/env python3
"""Synthetic validation of GRPO stability theory against ground truth.

Generates groups of binary rewards from Bernoulli(p) where the theoretical
assumptions hold exactly, then compares predicted variance components and
regime classifications against Monte Carlo empirical measurements.

Outputs:
    synthetic_calibration.json  -- predicted vs observed for each (rho, p, G)
    regime_accuracy.json        -- classification accuracy breakdown
    calibration_r2.json         -- R^2, RMSE per (p, G) combination
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stability_analysis import (
    compute_advantage_variance_components,
    group_starvation_rate,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("synthetic_validation")


# ---------------------------------------------------------------------------
# Empirical measurement helpers
# ---------------------------------------------------------------------------

def simulate_groups(p: float, G: int, n_groups: int, rng: np.random.Generator):
    """Draw n_groups of G Bernoulli(p) rewards and return (n_groups, G) array."""
    return rng.binomial(1, p, size=(n_groups, G)).astype(np.float64)


def measure_empirical_variance(
    rewards_grouped: np.ndarray, rho: float,
    emp_Vp: float, emp_Vm: float, emp_C: float,
) -> float:
    """Compute empirical gradient variance matching the theoretical definition.

    The theory defines:
        Var(grad) = rho^2 * V+ + V- + 2*rho*C
    where V+, V-, C are per-group sums averaged over the group distribution.
    We apply the same quadratic form to the empirically measured components.
    """
    return rho ** 2 * emp_Vp + emp_Vm + 2 * rho * emp_C


def measure_empirical_variance_direct(
    rewards_grouped: np.ndarray, rho: float,
) -> float:
    """Direct Monte Carlo: compute per-group rho-weighted quadratic form, average.

    For each group, the effective gradient magnitude squared is:
        Q_g = (rho * sum_pos(a_i) + sum_neg(a_j))^2  (divided by G^2)
    This computes the per-group squared gradient norm and averages across groups.
    """
    n_groups, G = rewards_grouped.shape
    m = rewards_grouped.sum(axis=1, keepdims=True)
    group_mean = m / G
    sigma = np.sqrt(m * (G - m)) / G
    delta = 1.0 / G
    sigma = np.maximum(sigma, delta)

    advantages = (rewards_grouped - group_mean) / sigma

    pos_mask = (rewards_grouped > 0).astype(np.float64)
    neg_mask = (rewards_grouped <= 0).astype(np.float64)

    degenerate = (m.squeeze(axis=1) == 0) | (m.squeeze(axis=1) == G)

    # Per-group quadratic form: rho^2 * sum(a_pos^2) + sum(a_neg^2) + 2*rho*cross
    # Equivalently, the sum of squared rho-weighted advantages per group
    rho_weighted = advantages * (pos_mask * rho + neg_mask * 1.0)
    rho_weighted[degenerate] = 0.0
    per_group_ssq = np.sum(rho_weighted ** 2, axis=1)  # sum_i (rho_a_i)^2

    return float(np.mean(per_group_ssq))


def measure_empirical_components(rewards_grouped: np.ndarray):
    """Decompose empirical variance into V+, V-, C components.

    Matches the theoretical definition in compute_advantage_variance_components:
        V+ = E_m[ n_pos * a_pos^2 ]     (averaged over ALL groups incl. degenerate)
        V- = E_m[ n_neg * a_neg^2 ]
        C  = E_m[ (n_pos * a_pos)(n_neg * a_neg) / G ]

    For degenerate groups (m=0 or m=G), the advantage is zero, contributing 0.
    Averaging over all groups (including degenerate) gives the correct expectation
    since the theory sums Binom(m;G,p) over m=0..G with m=0,G contributing zero.
    """
    n_groups, G = rewards_grouped.shape
    m = rewards_grouped.sum(axis=1, keepdims=True)
    group_mean = m / G
    sigma = np.sqrt(m * (G - m)) / G
    delta = 1.0 / G
    sigma = np.maximum(sigma, delta)

    advantages = (rewards_grouped - group_mean) / sigma

    degenerate = (m.squeeze(axis=1) == 0) | (m.squeeze(axis=1) == G)
    # Zero out degenerate groups so they contribute 0 to the expectation
    advantages[degenerate] = 0.0

    pos_mask = (rewards_grouped > 0).astype(np.float64)
    neg_mask = (rewards_grouped <= 0).astype(np.float64)

    # Per-group sums of squared advantages (matches n_pos * a_pos^2 since all
    # positives in a group have the same advantage value)
    v_plus_per_group = np.sum(advantages ** 2 * pos_mask, axis=1)
    v_minus_per_group = np.sum(advantages ** 2 * neg_mask, axis=1)

    # Cross term: (sum of positive adv) * (sum of negative adv) / G
    sum_pos = np.sum(advantages * pos_mask, axis=1)
    sum_neg = np.sum(advantages * neg_mask, axis=1)
    cross_per_group = sum_pos * sum_neg / G

    # Average over ALL groups (including degenerate ones which contribute 0)
    V_plus = float(np.mean(v_plus_per_group))
    V_minus = float(np.mean(v_minus_per_group))
    C_pG = float(np.mean(cross_per_group))

    return V_plus, V_minus, C_pG


# ---------------------------------------------------------------------------
# Regime classification via empirical variance decomposition
# ---------------------------------------------------------------------------

def classify_empirical_regime_from_variance(
    rho: float, p: float, G: int, n_groups: int, rng: np.random.Generator,
) -> str:
    """Classify regime by directly testing variance behavior.

    The theoretical regime boundaries are:
      - Below rho_min: negative gradients dominate (Var(rho) > Var(0))
      - Above rho_max: Var(rho) / Var(rho*) exceeds instability threshold
      - Between: convergent (Var is manageable)

    We measure the empirical variance at the test rho and at rho* to check
    whether the point is in the convergent, starved, or unstable zone.
    """
    rewards_grouped = rng.binomial(1, p, size=(n_groups, G)).astype(np.float64)
    emp_Vp, emp_Vm, emp_C = measure_empirical_components(rewards_grouped)

    # Empirical rho_min and rho_star from the observed components
    if emp_C < -1e-12:
        emp_rho_min = emp_Vm / (2.0 * abs(emp_C))
    else:
        emp_rho_min = 0.0

    if emp_Vp > 1e-12:
        emp_rho_star = -emp_C / emp_Vp
    else:
        emp_rho_star = 1.0
    emp_rho_star = max(emp_rho_star, emp_rho_min)

    # Variance at test point and at rho_star
    var_at_rho = rho ** 2 * emp_Vp + emp_Vm + 2 * rho * emp_C
    var_at_star = emp_rho_star ** 2 * emp_Vp + emp_Vm + 2 * emp_rho_star * emp_C

    # Starved: rho is below the empirical rho_min
    if rho < emp_rho_min:
        return "gradient_starved"

    # Unstable: variance at rho is much larger than at rho_star
    if var_at_star > 1e-10:
        ratio = var_at_rho / var_at_star
        if ratio > 10.0:
            return "unstable"

    return "convergent"


# ---------------------------------------------------------------------------
# Theoretical predictions
# ---------------------------------------------------------------------------

def predict_regime(rho: float, p: float, G: int) -> str:
    """Predict regime from theory using variance decomposition.

    Uses the core theoretical result: the gradient variance quadratic
        Var(rho) = rho^2 V+ + V- + 2 rho C
    The regime depends on where rho sits relative to rho_min and rho_max.

    Both boundaries are derived from the variance quadratic, ensuring that the
    theoretical prediction uses the same criterion as the empirical test:
      - Starved: rho < rho_min (below the lower bound from Theorem 3)
      - Unstable: Var(rho) / Var(rho*) > 10 (variance ratio threshold)
      - Convergent: otherwise
    """
    V_plus, V_minus, C_pG = compute_advantage_variance_components(p, G)

    # rho_min: below this, negative gradients dominate (starvation)
    if C_pG < 0:
        rho_min = V_minus / (2.0 * abs(C_pG))
    else:
        rho_min = 0.0

    # rho_star: minimum variance point
    if V_plus > 1e-12:
        rho_star = -C_pG / V_plus
    else:
        rho_star = 1.0
    rho_star = max(rho_star, rho_min)

    # Variance at rho_star (minimum) and at test rho
    var_at_star = rho_star ** 2 * V_plus + V_minus + 2 * rho_star * C_pG
    var_at_rho = rho ** 2 * V_plus + V_minus + 2 * rho * C_pG

    # Starvation: below rho_min
    if rho < rho_min:
        return "gradient_starved"

    # Unstable: variance ratio exceeds threshold (same 10x used empirically)
    if var_at_star > 1e-10:
        ratio = var_at_rho / var_at_star
        if ratio > 10.0:
            return "unstable"

    return "convergent"


# ---------------------------------------------------------------------------
# Calibration study
# ---------------------------------------------------------------------------

def run_calibration(
    p_values: list,
    G_values: list,
    rho_values: np.ndarray,
    n_groups: int,
    seed: int,
) -> list:
    """Run variance calibration across all (rho, p, G) points.

    Returns list of dicts with predicted and observed variance.
    """
    rng = np.random.default_rng(seed)
    results = []
    total = len(p_values) * len(G_values) * len(rho_values)
    count = 0

    for G in G_values:
        for p in p_values:
            # Generate one large batch of groups
            rewards_grouped = simulate_groups(p, G, n_groups, rng)

            # Measure empirical V+, V-, C once per (p, G)
            emp_Vp, emp_Vm, emp_C = measure_empirical_components(rewards_grouped)

            # Theoretical components
            theo_Vp, theo_Vm, theo_C = compute_advantage_variance_components(p, G)

            gsr_theoretical = group_starvation_rate(p, G)
            m_per_group = rewards_grouped.sum(axis=1)
            gsr_empirical = float(
                np.mean((m_per_group == 0) | (m_per_group == G))
            )

            for rho in rho_values:
                # Theoretical variance: rho^2 * V+ + V- + 2*rho*C
                var_predicted = rho ** 2 * theo_Vp + theo_Vm + 2 * rho * theo_C

                # Empirical variance via quadratic form of empirical components
                var_observed = measure_empirical_variance(
                    rewards_grouped, rho, emp_Vp, emp_Vm, emp_C
                )

                # Direct Monte Carlo: per-group sum of squared rho-weighted advs
                var_direct = measure_empirical_variance_direct(
                    rewards_grouped, rho
                )

                # Variance from empirical components (same as var_observed by
                # construction, included for completeness)
                var_from_emp_components = (
                    rho ** 2 * emp_Vp + emp_Vm + 2 * rho * emp_C
                )

                results.append({
                    "p": float(p),
                    "G": int(G),
                    "rho": float(rho),
                    "var_predicted": float(var_predicted),
                    "var_observed": float(var_direct),
                    "var_from_emp_components": float(var_from_emp_components),
                    "V_plus_predicted": float(theo_Vp),
                    "V_plus_observed": float(emp_Vp),
                    "V_minus_predicted": float(theo_Vm),
                    "V_minus_observed": float(emp_Vm),
                    "C_predicted": float(theo_C),
                    "C_observed": float(emp_C),
                    "GSR_predicted": float(gsr_theoretical),
                    "GSR_observed": float(gsr_empirical),
                })

                count += 1
                if count % 100 == 0:
                    logger.info(
                        "Calibration progress: %d/%d (%.1f%%)",
                        count, total, 100 * count / total,
                    )

    return results


def compute_r2_rmse(predicted: np.ndarray, observed: np.ndarray):
    """Compute R-squared and RMSE between predicted and observed arrays."""
    ss_res = np.sum((predicted - observed) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    return float(r2), float(rmse)


def aggregate_calibration(calibration_results: list) -> dict:
    """Compute per-(p, G) R^2 and RMSE, plus overall metrics."""
    # Group by (p, G)
    groups = {}
    for entry in calibration_results:
        key = (entry["p"], entry["G"])
        if key not in groups:
            groups[key] = {"predicted": [], "observed": []}
        groups[key]["predicted"].append(entry["var_predicted"])
        groups[key]["observed"].append(entry["var_observed"])

    per_pG = {}
    all_pred = []
    all_obs = []

    for (p, G), data in sorted(groups.items()):
        pred = np.array(data["predicted"])
        obs = np.array(data["observed"])
        r2, rmse = compute_r2_rmse(pred, obs)
        per_pG[f"p={p:.2f}_G={G}"] = {
            "p": p,
            "G": G,
            "R2": r2,
            "RMSE": rmse,
            "n_points": len(pred),
            "mean_predicted": float(np.mean(pred)),
            "mean_observed": float(np.mean(obs)),
        }
        all_pred.extend(data["predicted"])
        all_obs.extend(data["observed"])

    all_pred = np.array(all_pred)
    all_obs = np.array(all_obs)
    overall_r2, overall_rmse = compute_r2_rmse(all_pred, all_obs)

    # Also compute component-level R^2
    comp_metrics = {}
    for comp_name, pred_key, obs_key in [
        ("V_plus", "V_plus_predicted", "V_plus_observed"),
        ("V_minus", "V_minus_predicted", "V_minus_observed"),
        ("C", "C_predicted", "C_observed"),
        ("GSR", "GSR_predicted", "GSR_observed"),
    ]:
        # Deduplicate: components are per (p, G), not per rho
        seen = set()
        pred_vals, obs_vals = [], []
        for entry in calibration_results:
            key = (entry["p"], entry["G"])
            if key not in seen:
                seen.add(key)
                pred_vals.append(entry[pred_key])
                obs_vals.append(entry[obs_key])
        if len(pred_vals) > 1:
            r2, rmse = compute_r2_rmse(np.array(pred_vals), np.array(obs_vals))
            comp_metrics[comp_name] = {"R2": r2, "RMSE": rmse}

    return {
        "overall_R2": overall_r2,
        "overall_RMSE": overall_rmse,
        "n_total_points": len(all_pred),
        "per_pG": per_pG,
        "component_calibration": comp_metrics,
    }


# ---------------------------------------------------------------------------
# Regime classification accuracy
# ---------------------------------------------------------------------------

def run_regime_classification(
    p_values: list,
    G_values: list,
    rho_values: np.ndarray,
    n_groups: int,
    seed: int,
) -> dict:
    """Classify predicted vs observed regimes for all (rho, p, G) points.

    Theoretical prediction uses the closed-form variance decomposition.
    Empirical measurement draws fresh Monte Carlo samples and computes
    rho_min and the variance ratio from the observed components.
    """
    rng = np.random.default_rng(seed)

    predictions = []
    total = len(p_values) * len(G_values) * len(rho_values)
    count = 0

    confusion = {
        "convergent": {"convergent": 0, "gradient_starved": 0, "unstable": 0},
        "gradient_starved": {"convergent": 0, "gradient_starved": 0, "unstable": 0},
        "unstable": {"convergent": 0, "gradient_starved": 0, "unstable": 0},
    }

    for G in G_values:
        for p in p_values:
            for rho in rho_values:
                predicted = predict_regime(rho, p, G)

                observed = classify_empirical_regime_from_variance(
                    rho, p, G, n_groups, rng,
                )

                match = predicted == observed
                predictions.append({
                    "p": float(p),
                    "G": int(G),
                    "rho": float(rho),
                    "predicted_regime": predicted,
                    "observed_regime": observed,
                    "match": match,
                })

                if predicted in confusion and observed in confusion[predicted]:
                    confusion[predicted][observed] += 1

                count += 1
                if count % 50 == 0:
                    logger.info(
                        "Regime classification progress: %d/%d (%.1f%%)",
                        count, total, 100 * count / total,
                    )

    n_correct = sum(1 for r in predictions if r["match"])
    n_total = len(predictions)
    accuracy = n_correct / max(n_total, 1)

    # Per-regime accuracy
    per_regime = {}
    for regime in ["convergent", "gradient_starved", "unstable"]:
        subset = [r for r in predictions if r["predicted_regime"] == regime]
        if subset:
            n_right = sum(1 for r in subset if r["match"])
            per_regime[regime] = {
                "n_predicted": len(subset),
                "n_correct": n_right,
                "accuracy": n_right / len(subset),
            }

    # Per-G accuracy
    per_G = {}
    for G in G_values:
        subset = [r for r in predictions if r["G"] == G]
        if subset:
            n_right = sum(1 for r in subset if r["match"])
            per_G[f"G={G}"] = {
                "n_points": len(subset),
                "n_correct": n_right,
                "accuracy": n_right / len(subset),
            }

    # Per-p accuracy
    per_p = {}
    for p in p_values:
        subset = [r for r in predictions if abs(r["p"] - p) < 1e-6]
        if subset:
            n_right = sum(1 for r in subset if r["match"])
            per_p[f"p={p:.2f}"] = {
                "n_points": len(subset),
                "n_correct": n_right,
                "accuracy": n_right / len(subset),
            }

    return {
        "overall_accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "confusion_matrix": confusion,
        "per_regime_accuracy": per_regime,
        "per_G_accuracy": per_G,
        "per_p_accuracy": per_p,
        "detailed_predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic validation of GRPO stability theory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/synthetic_validation",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--n_groups",
        type=int,
        default=10000,
        help="Number of Monte Carlo groups per (p, G) combination",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parameter grid
    p_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    G_values = [2, 4, 8, 16]
    rho_values = np.logspace(-1, 1, 20)  # 0.1 to 10

    logger.info("=" * 70)
    logger.info("Synthetic Validation of GRPO Stability Theory")
    logger.info("=" * 70)
    logger.info("Parameters:")
    logger.info("  p values: %s", p_values)
    logger.info("  G values: %s", G_values)
    logger.info("  rho range: [%.2f, %.2f], %d points", rho_values[0], rho_values[-1], len(rho_values))
    logger.info("  n_groups: %d", args.n_groups)
    logger.info("  seed: %d", args.seed)
    logger.info(
        "  Total calibration points: %d",
        len(p_values) * len(G_values) * len(rho_values),
    )

    # --- Phase 1: Variance calibration ---
    logger.info("")
    logger.info("-" * 70)
    logger.info("Phase 1: Variance Calibration")
    logger.info("-" * 70)

    calibration_results = run_calibration(
        p_values, G_values, rho_values, args.n_groups, args.seed
    )

    cal_path = os.path.join(args.output_dir, "synthetic_calibration.json")
    with open(cal_path, "w") as f:
        json.dump(calibration_results, f, indent=2)
    logger.info("Calibration data saved to %s (%d points)", cal_path, len(calibration_results))

    # --- Phase 2: R^2 and RMSE ---
    logger.info("")
    logger.info("-" * 70)
    logger.info("Phase 2: Calibration Metrics (R^2, RMSE)")
    logger.info("-" * 70)

    r2_results = aggregate_calibration(calibration_results)

    r2_path = os.path.join(args.output_dir, "calibration_r2.json")
    with open(r2_path, "w") as f:
        json.dump(r2_results, f, indent=2)

    logger.info("Overall R^2 = %.6f, RMSE = %.6f", r2_results["overall_R2"], r2_results["overall_RMSE"])
    logger.info("Component calibration:")
    for comp, metrics in r2_results.get("component_calibration", {}).items():
        logger.info("  %s: R^2 = %.6f, RMSE = %.6f", comp, metrics["R2"], metrics["RMSE"])

    logger.info("Per (p, G) breakdown:")
    for key, entry in r2_results["per_pG"].items():
        logger.info(
            "  %s: R^2 = %.6f, RMSE = %.6f",
            key, entry["R2"], entry["RMSE"],
        )

    # --- Phase 3: Regime classification ---
    logger.info("")
    logger.info("-" * 70)
    logger.info("Phase 3: Regime Classification Accuracy")
    logger.info("-" * 70)

    regime_results = run_regime_classification(
        p_values, G_values, rho_values, args.n_groups, args.seed,
    )

    regime_output = {
        "overall_accuracy": regime_results["overall_accuracy"],
        "n_correct": regime_results["n_correct"],
        "n_total": regime_results["n_total"],
        "confusion_matrix": regime_results["confusion_matrix"],
        "per_regime_accuracy": regime_results["per_regime_accuracy"],
        "per_G_accuracy": regime_results["per_G_accuracy"],
        "per_p_accuracy": regime_results["per_p_accuracy"],
        "detailed_predictions": regime_results["detailed_predictions"],
    }

    regime_path = os.path.join(args.output_dir, "regime_accuracy.json")
    with open(regime_path, "w") as f:
        json.dump(regime_output, f, indent=2)

    logger.info(
        "Overall regime classification accuracy: %.1f%% (%d/%d)",
        100 * regime_results["overall_accuracy"],
        regime_results["n_correct"],
        regime_results["n_total"],
    )
    logger.info("Confusion matrix (rows=predicted, cols=observed):")
    header = f"{'':>20s}  {'convergent':>12s}  {'starved':>12s}  {'unstable':>12s}"
    logger.info(header)
    for pred_regime in ["convergent", "gradient_starved", "unstable"]:
        row = regime_results["confusion_matrix"].get(pred_regime, {})
        logger.info(
            "%20s  %12d  %12d  %12d",
            pred_regime,
            row.get("convergent", 0),
            row.get("gradient_starved", 0),
            row.get("unstable", 0),
        )

    logger.info("Per-regime accuracy:")
    for regime, metrics in regime_results["per_regime_accuracy"].items():
        logger.info(
            "  %s: %.1f%% (%d/%d)",
            regime, 100 * metrics["accuracy"],
            metrics["n_correct"], metrics["n_predicted"],
        )

    logger.info("Per-G accuracy:")
    for g_key, metrics in regime_results["per_G_accuracy"].items():
        logger.info(
            "  %s: %.1f%% (%d/%d)",
            g_key, 100 * metrics["accuracy"],
            metrics["n_correct"], metrics["n_points"],
        )

    # --- Summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info("Variance calibration:  R^2 = %.6f", r2_results["overall_R2"])
    logger.info("Regime classification: %.1f%% accuracy", 100 * regime_results["overall_accuracy"])
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Files written:")
    logger.info("  %s", cal_path)
    logger.info("  %s", r2_path)
    logger.info("  %s", regime_path)


if __name__ == "__main__":
    main()
