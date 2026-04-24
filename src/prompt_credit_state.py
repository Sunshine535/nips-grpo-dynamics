"""Prompt-conditioned credit state with Beta-posterior for TRACE-GRPO.

Replaces naive EMA-only PromptStatsStore with posterior evidence tracking.
Each prompt maintains success/fail counts, frontier score, uncertainty,
and replay exposure — enabling trust-calibrated replay decisions.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import math


@dataclass
class PromptCredit:
    alpha: float = 1.0
    beta: float = 1.0
    baseline_ema: float = 0.0
    replay_exposure: int = 0
    last_seen_step: int = -1

    @property
    def n_obs(self) -> int:
        return int(self.alpha + self.beta - 2)

    @property
    def p_hat(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def frontier(self, n_min: int = 5) -> float:
        """High for evidence-backed uncertain prompts, low otherwise.
        F = 4 * p_hat * (1-p_hat) * min(1, n_obs/n_min)
        """
        evidence_conf = min(1.0, self.n_obs / max(n_min, 1))
        p = self.p_hat
        return 4.0 * p * (1.0 - p) * evidence_conf

    def saturation(self, max_exposure: int = 10) -> float:
        return min(1.0, self.replay_exposure / max(max_exposure, 1))


class PromptCreditStore:
    """Maintains per-prompt Beta-posterior credit states for TRACE-GRPO."""

    def __init__(self, alpha_baseline: float = 0.1, n_min: int = 5,
                 max_exposure: int = 10):
        self.alpha_baseline = alpha_baseline
        self.n_min = n_min
        self.max_exposure = max_exposure
        self.credits: dict = defaultdict(PromptCredit)

    def update(self, prompt_id: int, mean_reward: float, step: int) -> None:
        c = self.credits[prompt_id]
        c.alpha += mean_reward
        c.beta += (1.0 - mean_reward)
        c.baseline_ema = (1 - self.alpha_baseline) * c.baseline_ema + self.alpha_baseline * mean_reward
        c.last_seen_step = step

    def get_baseline(self, prompt_id: int) -> float:
        return self.credits[prompt_id].baseline_ema

    def get_frontier(self, prompt_id: int) -> float:
        return self.credits[prompt_id].frontier(self.n_min)

    def get_p_hat(self, prompt_id: int) -> float:
        return self.credits[prompt_id].p_hat

    def get_uncertainty(self, prompt_id: int) -> float:
        return self.credits[prompt_id].uncertainty

    def get_saturation(self, prompt_id: int) -> float:
        return self.credits[prompt_id].saturation(self.max_exposure)

    def record_replay(self, prompt_id: int) -> None:
        self.credits[prompt_id].replay_exposure += 1

    def dump(self) -> dict:
        return {
            pid: {
                "alpha": c.alpha, "beta": c.beta, "p_hat": round(c.p_hat, 4),
                "frontier": round(c.frontier(self.n_min), 4),
                "uncertainty": round(c.uncertainty, 4),
                "n_obs": c.n_obs, "baseline_ema": round(c.baseline_ema, 4),
                "replay_exposure": c.replay_exposure,
                "last_seen_step": c.last_seen_step,
            }
            for pid, c in self.credits.items()
        }
