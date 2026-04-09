"""
CSD (Contrastive Self-Distillation) logging for GRPO training.

Tracks the CSD components predicted by Theorem 1:
  ∇L_GRPO = √(p(1-p)) · [∇KL(τ⁻‖π) - ρ·∇KL(τ⁺‖π)]

Logs per-step:
  - p (group success rate), n⁺, n⁻
  - CSD signal strength: √(p(1-p))
  - Q_CSD collapse predictor: diversity(τ⁺) · (n⁺/G) · advantage_alignment
  - Collapse label (for post-hoc AUROC analysis)
"""

import math
import logging
from collections import deque

import numpy as np
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class CSDLoggingCallback(TrainerCallback):
    """Track CSD-specific metrics during GRPO training."""

    def __init__(self, group_size: int = 4, collapse_threshold: float = 0.1,
                 window: int = 20):
        self.group_size = group_size
        self.collapse_threshold = collapse_threshold
        self.window = window
        self.csd_logs = []
        self.reward_history = deque(maxlen=200)
        self._trainer_ref = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Get trainer reference
        trainer = self._trainer_ref
        if trainer is None:
            return

        step = state.global_step
        G = self.group_size

        # Extract from trainer's step stats
        step_stats = getattr(trainer, '_rho_step_stats', [])
        if not step_stats:
            return

        latest = step_stats[-1]
        n_pos = latest.get("n_positive", 0)
        n_neg = latest.get("n_negative", 0)
        n_deg = latest.get("n_degenerate", 0)
        n_total = n_pos + n_neg + n_deg
        rho = latest.get("rho", 1.0)

        if n_total == 0:
            return

        # --- CSD Components ---
        # Per-batch success rate
        p = n_pos / max(n_pos + n_neg, 1)

        # CSD signal strength (Theorem 1): √(p(1-p))
        csd_signal = math.sqrt(max(p * (1 - p), 0))

        # Advantage statistics
        mean_pos_adv = latest.get("mean_pos_adv", 0.0)
        mean_neg_adv = latest.get("mean_neg_adv", 0.0)

        # Distillation/anti-distillation balance
        # |ρ·A⁺| vs |A⁻| — measures the relative strength
        distill_strength = abs(rho * mean_pos_adv) if n_pos > 0 else 0.0
        anti_distill_strength = abs(mean_neg_adv) if n_neg > 0 else 0.0
        balance_ratio = (distill_strength / max(anti_distill_strength, 1e-8)
                         if anti_distill_strength > 0 else float('inf'))

        # --- Q_CSD Collapse Predictor (Proposition 1) ---
        # Approximation without gradient cosine:
        # Q_CSD ≈ diversity(τ⁺) · availability(τ⁺) · signal_quality
        #
        # diversity: use std of per-group success counts as proxy
        # availability: n⁺/G (fraction of correct responses)
        # signal: csd_signal itself (√(p(1-p)))

        # Compute per-group stats if we have enough data
        n_groups = n_total // G if G > 0 else 1
        availability = n_pos / max(n_total, 1)

        # Diversity proxy: how varied are success rates across groups?
        # Higher diversity = better τ⁺ (more diverse correct responses)
        # We approximate using p * (1-p) / max(n_groups, 1) which is the variance
        # of a binomial. More precise computation would need per-group data.
        diversity_proxy = min(1.0, 4 * p * (1 - p))  # Normalized to [0,1], max at p=0.5

        # Q_CSD: product of three terms
        q_csd = diversity_proxy * availability * csd_signal

        # --- Collapse Detection ---
        reward_mean = logs.get("reward/mean", logs.get("reward_mean", 0))
        self.reward_history.append(reward_mean)

        # Detect collapse: reward drops below threshold after initial period
        is_collapsed = False
        if step > 50 and len(self.reward_history) >= self.window:
            recent = list(self.reward_history)[-self.window:]
            recent_mean = np.mean(recent)
            is_collapsed = recent_mean < self.collapse_threshold

        # --- Gradient variance (for comparison with Q_CSD) ---
        grad_norm = logs.get("grad_norm", 0)

        record = {
            "step": step,
            "rho": rho,
            # CSD Theorem 1 components
            "p": round(p, 4),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_degenerate": n_deg,
            "csd_signal_strength": round(csd_signal, 6),
            "mean_pos_advantage": round(mean_pos_adv, 6),
            "mean_neg_advantage": round(mean_neg_adv, 6),
            "distill_strength": round(distill_strength, 6),
            "anti_distill_strength": round(anti_distill_strength, 6),
            "balance_ratio": round(min(balance_ratio, 100), 4),
            # CSD Proposition 1
            "q_csd": round(q_csd, 6),
            "diversity_proxy": round(diversity_proxy, 4),
            "availability": round(availability, 4),
            # Collapse
            "is_collapsed": is_collapsed,
            "reward_mean": round(reward_mean, 4) if isinstance(reward_mean, float) else reward_mean,
            "grad_norm": round(grad_norm, 6) if isinstance(grad_norm, float) else grad_norm,
        }

        self.csd_logs.append(record)

        # Log to wandb/console
        logs["csd/signal_strength"] = csd_signal
        logs["csd/q_csd"] = q_csd
        logs["csd/p"] = p
        logs["csd/balance_ratio"] = min(balance_ratio, 100)
        logs["csd/is_collapsed"] = int(is_collapsed)


def compute_step0_qcsd(rewards: np.ndarray, group_size: int) -> float:
    """Compute Q_CSD from step-0 rollout rewards for collapse prediction."""
    n_total = len(rewards)
    n_pos = int((rewards > 0).sum())
    p = n_pos / max(n_total, 1)
    csd_signal = math.sqrt(max(p * (1 - p), 0))
    availability = n_pos / max(n_total, 1)
    diversity_proxy = min(1.0, 4 * p * (1 - p))
    return diversity_proxy * availability * csd_signal
