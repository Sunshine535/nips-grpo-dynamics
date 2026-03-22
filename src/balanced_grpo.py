"""
Balanced GRPO: Custom GRPO loss with configurable positive/negative signal weighting.

The core idea: in standard GRPO, positive (correct) and negative (incorrect) completions
contribute equally to the policy gradient. This module allows independent control over
their relative influence via `positive_ratio` (alpha) and `negative_weight` (beta).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from transformers import TrainerCallback


@dataclass
class BalancedGRPOConfig:
    positive_ratio: float = 0.5
    negative_weight: float = 1.0
    clip_range: float = 0.2
    kl_coef: float = 0.05


def compute_balanced_grpo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    config: BalancedGRPOConfig,
) -> dict:
    """
    Compute GRPO loss with asymmetric weighting of positive/negative signals.

    Args:
        logprobs: Log-probs under current policy, shape (B, G, L)
        ref_logprobs: Log-probs under reference policy, shape (B, G, L)
        advantages: Per-generation advantages, shape (B, G)
        mask: Token-level mask, shape (B, G, L)
        config: Balancing configuration

    Returns:
        Dictionary with 'loss', 'pg_loss', 'kl_loss', and diagnostics.
    """
    ratio = torch.exp(logprobs - ref_logprobs)

    positive_mask = (advantages > 0).float()
    negative_mask = (advantages <= 0).float()

    alpha = config.positive_ratio
    beta = config.negative_weight

    # Reweight advantages: scale positive by alpha, negative by beta * (1 - alpha)
    weights = positive_mask * alpha + negative_mask * (1.0 - alpha) * beta
    weighted_advantages = advantages * weights

    # Clipped surrogate objective
    ratio_clamped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)

    # Token-level policy gradient
    # advantages is (B, G), ratio is (B, G, L) -> broadcast
    weighted_adv = weighted_advantages.unsqueeze(-1)  # (B, G, 1)
    pg1 = ratio * weighted_adv
    pg2 = ratio_clamped * weighted_adv

    pg_loss_token = -torch.min(pg1, pg2)
    pg_loss = (pg_loss_token * mask).sum() / mask.sum().clamp(min=1)

    # KL penalty
    kl = logprobs - ref_logprobs
    kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

    loss = pg_loss + config.kl_coef * kl_loss

    # Diagnostics
    n_pos = positive_mask.sum().item()
    n_neg = negative_mask.sum().item()
    frac_positive = n_pos / max(n_pos + n_neg, 1)
    mean_pos_adv = (advantages * positive_mask).sum().item() / max(n_pos, 1)
    mean_neg_adv = (advantages * negative_mask).sum().item() / max(n_neg, 1)

    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "frac_positive": frac_positive,
        "mean_pos_advantage": mean_pos_adv,
        "mean_neg_advantage": mean_neg_adv,
        "effective_weight_pos": alpha,
        "effective_weight_neg": (1.0 - alpha) * beta,
    }


def build_grpo_reward_fn(positive_ratio: float, negative_weight: float):
    """
    Build a reward-shaping wrapper that re-weights rewards for TRL's GRPOTrainer.

    This adjusts the reward signal before advantage computation so that the trainer
    naturally produces the desired positive/negative balance.
    """
    alpha = positive_ratio
    beta = negative_weight

    def shaped_reward(rewards: list[float], **kwargs) -> list[float]:
        shaped = []
        for r in rewards:
            if r > 0:
                shaped.append(r * alpha)
            else:
                shaped.append(r * (1.0 - alpha) * beta)
        return shaped

    return shaped_reward


class BalancedGRPOCallback(TrainerCallback):
    """Callback to log balanced GRPO diagnostics during training."""

    def __init__(self, positive_ratio: float, negative_weight: float):
        self.positive_ratio = positive_ratio
        self.negative_weight = negative_weight
        self.step_metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs["balanced_grpo/positive_ratio"] = self.positive_ratio
        logs["balanced_grpo/negative_weight"] = self.negative_weight
        self.step_metrics.append({
            "step": state.global_step,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        })
