"""
Zero-score gradient handling strategies for GRPO training.

When GRPO generates completions that receive a reward of 0 (incorrect),
the standard policy gradient gives zero learning signal. This module
implements four strategies to extract useful signal from these samples.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F


class ZeroScoreStrategy(str, Enum):
    CLIP = "clip"
    TEMPERATURE = "temperature"
    CURRICULUM = "curriculum"
    RELABEL = "relabel"


@dataclass
class ZeroScoreConfig:
    strategy: ZeroScoreStrategy = ZeroScoreStrategy.CURRICULUM
    clip_factor: float = 0.1
    temperature_boost: float = 1.5
    curriculum_warmup_steps: int = 500
    relabel_epsilon: float = 0.01


class ZeroScoreHandler:
    """Handles zero-score samples in GRPO training with configurable strategies."""

    def __init__(self, config: ZeroScoreConfig):
        self.config = config
        self.strategy = ZeroScoreStrategy(config.strategy)

    def reweight_advantages(
        self,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        global_step: int = 0,
    ) -> torch.Tensor:
        """
        Reweight advantage estimates to handle zero-score samples.

        Args:
            advantages: [batch_size] advantage estimates from GRPO
            rewards: [batch_size] raw reward signals (0 or 1 for binary correctness)
            global_step: current training step (for curriculum)

        Returns:
            Modified advantages tensor
        """
        zero_mask = rewards == 0.0
        nonzero_mask = ~zero_mask

        if not zero_mask.any():
            return advantages

        if self.strategy == ZeroScoreStrategy.CLIP:
            return self._apply_clip(advantages, zero_mask, nonzero_mask)
        elif self.strategy == ZeroScoreStrategy.TEMPERATURE:
            return self._apply_temperature(advantages, zero_mask, nonzero_mask)
        elif self.strategy == ZeroScoreStrategy.CURRICULUM:
            return self._apply_curriculum(advantages, zero_mask, nonzero_mask, global_step)
        elif self.strategy == ZeroScoreStrategy.RELABEL:
            return self._apply_relabel(advantages, rewards, zero_mask)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _apply_clip(
        self,
        advantages: torch.Tensor,
        zero_mask: torch.Tensor,
        nonzero_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scale down gradients for 0-score samples by clip_factor."""
        modified = advantages.clone()
        if nonzero_mask.any():
            nonzero_mean_abs = advantages[nonzero_mask].abs().mean()
        else:
            nonzero_mean_abs = advantages.abs().mean()
        modified[zero_mask] = (
            torch.sign(advantages[zero_mask])
            * nonzero_mean_abs
            * self.config.clip_factor
        )
        return modified

    def _apply_temperature(
        self,
        advantages: torch.Tensor,
        zero_mask: torch.Tensor,
        nonzero_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Boost exploration for 0-score samples via temperature scaling."""
        modified = advantages.clone()
        modified[zero_mask] = advantages[zero_mask] / self.config.temperature_boost
        return modified

    def _apply_curriculum(
        self,
        advantages: torch.Tensor,
        zero_mask: torch.Tensor,
        nonzero_mask: torch.Tensor,
        global_step: int,
    ) -> torch.Tensor:
        """Gradually include zero-score samples over training."""
        modified = advantages.clone()
        if global_step >= self.config.curriculum_warmup_steps:
            inclusion_ratio = 1.0
        else:
            inclusion_ratio = global_step / self.config.curriculum_warmup_steps
        scale = 0.5 * (1.0 - math.cos(math.pi * inclusion_ratio))
        modified[zero_mask] = advantages[zero_mask] * scale
        return modified

    def _apply_relabel(
        self,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        zero_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Relabel 0-score with small positive reward to provide gradient signal."""
        modified_rewards = rewards.clone()
        modified_rewards[zero_mask] = self.config.relabel_epsilon
        mean_r = modified_rewards.mean()
        std_r = modified_rewards.std().clamp(min=1e-8)
        return (modified_rewards - mean_r) / std_r


def compute_gradient_stats(
    model: torch.nn.Module,
    zero_mask: torch.Tensor,
    per_sample_grads: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute diagnostic gradient statistics for zero vs non-zero score samples.

    Returns dict with gradient norms and cosine similarity between
    zero-score and non-zero-score gradient directions.
    """
    stats = {}

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    stats["total_grad_norm"] = total_norm ** 0.5

    if per_sample_grads is not None and zero_mask.any() and (~zero_mask).any():
        zero_grads = per_sample_grads[zero_mask].mean(dim=0)
        nonzero_grads = per_sample_grads[~zero_mask].mean(dim=0)

        stats["zero_grad_norm"] = zero_grads.norm(2).item()
        stats["nonzero_grad_norm"] = nonzero_grads.norm(2).item()

        cos_sim = F.cosine_similarity(
            zero_grads.unsqueeze(0), nonzero_grads.unsqueeze(0)
        )
        stats["grad_cosine_similarity"] = cos_sim.item()
        stats["grad_norm_ratio"] = (
            stats["zero_grad_norm"] / max(stats["nonzero_grad_norm"], 1e-8)
        )

    return stats
