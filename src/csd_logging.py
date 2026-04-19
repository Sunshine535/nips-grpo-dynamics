"""
CSD (Contrastive Self-Distillation) logging for GRPO training.

Tracks the CSD components predicted by Theorem 1:
  ∇L_GRPO = √(p(1-p)) · [∇KL(τ⁻‖π) - ρ·∇KL(τ⁺‖π)]

Logs per-step:
  - p (group success rate), n⁺, n⁻
  - CSD signal strength: √(p(1-p))
  - Q_CSD canonical: H_norm(τ⁺) · (n⁺/G) — computed inside the trainer
    where completion_ids are available; this callback reads it from
    trainer._rho_step_stats[-1]["q_csd"].
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

        # --- Q_CSD Collapse Predictor (canonical) ---
        # Q_CSD := H_norm(τ⁺) · (n⁺/G), per FINAL_PROPOSAL.md §"Empirical Hypothesis 1".
        # The trainer computes this inside _apply_rho_weighting where completion_ids
        # are available; we read it here.
        q_csd = float(latest.get("q_csd", 0.0))
        h_norm_pos = float(latest.get("h_norm_pos", 0.0))
        availability = float(latest.get("availability", n_pos / max(G, 1)))

        # --- Collapse Detection ---
        reward_mean = logs.get("reward/mean", logs.get("reward_mean", 0))
        self.reward_history.append(reward_mean)

        # Detect collapse: reward drops below threshold after initial period
        is_collapsed = False
        if step > 50 and len(self.reward_history) >= self.window:
            recent = list(self.reward_history)[-self.window:]
            recent_mean = float(np.mean(recent))  # cast to Python float
            is_collapsed = bool(recent_mean < self.collapse_threshold)  # numpy.bool_ → bool

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
            # Q_CSD (canonical): H_norm(τ⁺) · (n⁺/G)
            "q_csd": round(q_csd, 6),
            "h_norm_pos": round(h_norm_pos, 4),
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


def compute_step0_qcsd(
    rewards: np.ndarray,
    group_size: int,
    completion_ids: "np.ndarray | None" = None,
) -> float:
    """Compute canonical Q_CSD = H_norm(τ⁺) · (n⁺/G) from step-0 rollouts.

    τ⁺ is a *per-group* object. For a batch of B·G rewards (B groups of
    G consecutive rollouts) we compute Q_CSD per group and average, so the
    returned value is bounded in [0, 1].

    If ``completion_ids`` is supplied, H_norm(τ⁺) is computed from the
    entropy of the empirical correct-response distribution; otherwise we
    return the optimistic upper bound (H_norm = 1 ⇒ Q_CSD = n⁺/G).
    """
    rewards = np.asarray(rewards)
    G = max(group_size, 1)
    n_total = len(rewards)
    n_groups = n_total // G
    if n_groups == 0:
        return 0.0
    per_group = []
    for b in range(n_groups):
        sl = slice(b * G, (b + 1) * G)
        r_b = rewards[sl]
        n_pos_b = int((r_b > 0).sum())
        avail_b = n_pos_b / G  # ∈ [0, 1]
        if n_pos_b < 2:
            per_group.append(0.0)
            continue
        if completion_ids is None:
            per_group.append(avail_b)  # upper bound (H_norm = 1)
            continue
        pos_rows = np.asarray(completion_ids)[sl][r_b > 0]
        hashes = [hash(tuple(row.tolist())) for row in pos_rows]
        _, counts = np.unique(hashes, return_counts=True)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        h_norm = entropy / float(np.log(n_pos_b))
        per_group.append(h_norm * avail_b)
    return float(np.mean(per_group))
