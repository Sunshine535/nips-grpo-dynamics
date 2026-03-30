import numpy as np
from scipy.stats import binom
from dataclasses import dataclass
from typing import Optional


@dataclass
class StabilityBounds:
    rho_min: float
    rho_max: float
    rho_star: float
    V_plus: float
    V_minus: float
    C_pG: float
    GSR: float
    p_0: float
    regime: str


def binomial_pmf(m: int, G: int, p: float) -> float:
    return binom.pmf(m, G, p)


def group_starvation_rate(p: float, G: int) -> float:
    return (1 - p) ** G + p ** G


def compute_advantage_variance_components(
    p: float,
    G: int,
    grad_var_plus: float = 1.0,
    grad_var_minus: float = 1.0,
) -> tuple[float, float, float]:
    V_plus = 0.0
    V_minus = 0.0
    C_pG = 0.0

    for m in range(1, G):
        pmf = binomial_pmf(m, G, p)
        if pmf < 1e-15:
            continue

        mu_m = m / G
        sigma_m = np.sqrt(m * (G - m)) / G
        if sigma_m < 1e-12:
            continue

        a_pos = (1.0 - mu_m) / sigma_m
        a_neg = (0.0 - mu_m) / sigma_m

        n_pos = m
        n_neg = G - m

        V_plus += pmf * n_pos * a_pos ** 2 * grad_var_plus
        V_minus += pmf * n_neg * a_neg ** 2 * grad_var_minus

        pos_total = n_pos * a_pos
        neg_total = n_neg * a_neg
        C_pG += pmf * pos_total * neg_total / G

    return V_plus, V_minus, C_pG


def compute_rho_min(V_minus: float, C_pG: float) -> float:
    if C_pG >= 0:
        return 0.0
    return V_minus / (2.0 * abs(C_pG))


def compute_rho_star(V_plus: float, C_pG: float) -> float:
    if V_plus < 1e-12:
        return 1.0
    return -C_pG / V_plus


def compute_rho_max(
    p: float,
    G: int,
    kl_coef: float = 0.05,
    clip_range: float = 0.2,
    grad_pos_norm: float = 1.0,
) -> float:
    sigma_bar = np.sqrt(p * (1 - p)) if 0 < p < 1 else 1.0

    if grad_pos_norm < 1e-12:
        return float('inf')

    rho_max = (1.0 / clip_range) * (kl_coef / grad_pos_norm) * sigma_bar
    return max(rho_max, 5.0)


def analyze_stability(
    p: float,
    G: int,
    kl_coef: float = 0.05,
    clip_range: float = 0.2,
    grad_var_plus: float = 1.0,
    grad_var_minus: float = 1.0,
    grad_pos_norm: float = 1.0,
) -> StabilityBounds:
    V_plus, V_minus, C_pG = compute_advantage_variance_components(
        p, G, grad_var_plus, grad_var_minus
    )

    gsr = group_starvation_rate(p, G)
    rho_min = compute_rho_min(V_minus, C_pG)
    rho_star = compute_rho_star(V_plus, C_pG)
    rho_max = compute_rho_max(p, G, kl_coef, clip_range, grad_pos_norm)

    rho_star = np.clip(rho_star, rho_min, rho_max)

    bounds = StabilityBounds(
        rho_min=rho_min,
        rho_max=rho_max,
        rho_star=rho_star,
        V_plus=V_plus,
        V_minus=V_minus,
        C_pG=C_pG,
        GSR=gsr,
        p_0=gsr,
        regime="convergent",
    )
    bounds.regime = classify_regime(rho_star, bounds)
    return bounds


def classify_regime(
    rho: float,
    bounds: StabilityBounds,
    p_0_threshold: float = 0.8,
    kl_ratio: float = 1.0,
    kl_threshold: float = 2.0,
    entropy_drop: float = 0.0,
    entropy_threshold: float = 0.5,
    reward_stagnation_steps: int = 0,
    stagnation_window: int = 100,
) -> str:
    n_conditions_met = 0

    if bounds.GSR > p_0_threshold and kl_ratio > kl_threshold:
        n_conditions_met += 1

    if kl_ratio > 3.0:
        n_conditions_met += 1

    if reward_stagnation_steps > stagnation_window and entropy_drop > entropy_threshold:
        n_conditions_met += 1

    if n_conditions_met >= 2:
        return "gradient_starved" if rho < bounds.rho_star else "unstable"

    if rho < bounds.rho_min:
        return "gradient_starved"

    if rho > bounds.rho_max:
        if kl_ratio > kl_threshold:
            return "unstable"
        return "unstable_mild"

    if bounds.GSR > p_0_threshold:
        return "at_risk"

    return "convergent"


def compute_gradient_variance(rho: float, bounds: StabilityBounds) -> float:
    return (rho ** 2) * bounds.V_plus + bounds.V_minus + 2 * rho * bounds.C_pG


def build_stability_map(
    rho_range: np.ndarray,
    p_range: np.ndarray,
    G: int,
    kl_coef: float = 0.05,
    clip_range: float = 0.2,
) -> dict:
    n_rho = len(rho_range)
    n_p = len(p_range)

    regime_map = np.zeros((n_p, n_rho), dtype=object)
    rho_min_curve = np.zeros(n_p)
    rho_max_curve = np.zeros(n_p)
    rho_star_curve = np.zeros(n_p)
    gsr_curve = np.zeros(n_p)
    variance_map = np.zeros((n_p, n_rho))

    regime_to_int = {
        "convergent": 0, "gradient_starved": 1, "unstable": 2,
        "unstable_mild": 3, "at_risk": 4,
    }

    regime_int_map = np.zeros((n_p, n_rho), dtype=int)

    for i, p in enumerate(p_range):
        bounds = analyze_stability(p, G, kl_coef, clip_range)
        rho_min_curve[i] = bounds.rho_min
        rho_max_curve[i] = bounds.rho_max
        rho_star_curve[i] = bounds.rho_star
        gsr_curve[i] = bounds.GSR

        for j, rho in enumerate(rho_range):
            regime = classify_regime(rho, bounds)
            regime_map[i, j] = regime
            regime_int_map[i, j] = regime_to_int.get(regime, -1)
            variance_map[i, j] = compute_gradient_variance(rho, bounds)

    return {
        "rho_range": rho_range,
        "p_range": p_range,
        "regime_map": regime_map,
        "regime_int_map": regime_int_map,
        "rho_min_curve": rho_min_curve,
        "rho_max_curve": rho_max_curve,
        "rho_star_curve": rho_star_curve,
        "gsr_curve": gsr_curve,
        "variance_map": variance_map,
    }


def estimate_from_telemetry(
    success_counts: np.ndarray,
    group_size: int,
    grad_pos_norms: Optional[np.ndarray] = None,
    grad_neg_norms: Optional[np.ndarray] = None,
) -> dict:
    p_hat = success_counts.mean() / group_size

    total_groups = len(success_counts)
    degenerate = np.sum((success_counts == 0) | (success_counts == group_size))
    p_0_hat = degenerate / total_groups

    gsr_hat = group_starvation_rate(p_hat, group_size)

    grad_var_plus = float(np.var(grad_pos_norms)) if grad_pos_norms is not None else 1.0
    grad_var_minus = float(np.var(grad_neg_norms)) if grad_neg_norms is not None else 1.0

    grad_pos_norm = float(np.mean(grad_pos_norms)) if grad_pos_norms is not None else 1.0

    bounds = analyze_stability(
        p_hat, group_size,
        grad_var_plus=grad_var_plus,
        grad_var_minus=grad_var_minus,
        grad_pos_norm=grad_pos_norm,
    )

    return {
        "p_hat": p_hat,
        "p_0_hat": p_0_hat,
        "gsr_hat": gsr_hat,
        "bounds": bounds,
    }
