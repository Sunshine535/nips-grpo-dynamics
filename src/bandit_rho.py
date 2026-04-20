"""
UCB1 bandit over ρ values, targeting training reward directly.

Rationale (from the Round-4 Codex critique + our Wave 2 data):
  - The CSD Theorem-2 ρ* minimizes *gradient variance*, not test accuracy.
  - Our empirical sweep shows ρ=0.7 (55.0%) beats ρ=1.0 (45.7%) by ~9 pp,
    but ADQ converges to ρ ≈ 0.85 because it targets the wrong objective.
  - A bandit over discrete ρ choices that uses training reward as the
    feedback signal avoids both layers of approximation error
    (wrong objective + bad proxy estimator for Cov/Var).

This controller is deliberately simple:
  - Arms: a user-supplied list of ρ values.
  - Reward signal: mean rollout reward of the most recent K training batches.
  - Decision rule: UCB1, choose argmax_{ρ} (μ̂_ρ + c · sqrt(ln N / N_ρ)).
  - Update cadence: every `update_every` calls to `update()`.

Compared to ADQ, the bandit is:
  + Directly optimizing what we care about (training reward, correlated with test acc).
  - Non-differentiable (discrete ρ grid), slower to converge than a good
    continuous proxy would be.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class BanditRhoConfig:
    rho_grid: List[float] = field(default_factory=lambda: [0.3, 0.7, 1.0, 2.0, 3.0])
    exploration_c: float = 1.0      # UCB exploration strength
    warmup_steps: int = 5            # always-explore phase (one visit to each arm)
    update_every: int = 1            # pick a new arm every N update() calls
    reward_window: int = 10           # moving average window over recent per-arm rewards


class UCBBanditRho:
    """UCB1 bandit over a discrete ρ grid with rolling reward window per arm."""

    def __init__(self, config: BanditRhoConfig):
        self.config = config
        self.n_arms = len(config.rho_grid)
        self._visits = np.zeros(self.n_arms, dtype=np.int64)   # N_arm
        self._reward_queues: List[list] = [[] for _ in range(self.n_arms)]
        self._last_pick: int = 0                               # index into rho_grid
        self._update_calls = 0
        self._telemetry: list = []

    # ----- public API (used by the trainer) ----------------------------
    def initial_rho(self) -> float:
        """Pick the first ρ — uniform random over the grid."""
        idx = int(np.random.choice(self.n_arms))
        self._last_pick = idx
        return float(self.config.rho_grid[idx])

    def update(self, observed_reward: float) -> float:
        """Observe the batch-mean training reward for the most recently-used ρ;
        return the next ρ to use. If not time to switch yet, return the same ρ."""
        self._update_calls += 1
        idx = self._last_pick
        self._visits[idx] += 1
        q = self._reward_queues[idx]
        q.append(float(observed_reward))
        if len(q) > self.config.reward_window:
            q.pop(0)

        if self._update_calls % self.config.update_every != 0:
            return float(self.config.rho_grid[idx])

        next_idx = self._select_next()
        self._last_pick = next_idx
        rho = float(self.config.rho_grid[next_idx])
        self._telemetry.append({
            "call": self._update_calls,
            "picked_idx": int(next_idx),
            "picked_rho": rho,
            "visits": self._visits.tolist(),
            "arm_means": [float(np.mean(q)) if q else 0.0 for q in self._reward_queues],
        })
        return rho

    def _select_next(self) -> int:
        # Force visiting unseen arms during warm-up.
        total = int(self._visits.sum())
        if total < max(self.config.warmup_steps, self.n_arms):
            unseen = np.where(self._visits == 0)[0]
            if len(unseen) > 0:
                return int(unseen[0])

        # UCB1 score per arm.
        scores = np.zeros(self.n_arms)
        for k in range(self.n_arms):
            n_k = self._visits[k]
            if n_k == 0:
                scores[k] = float("inf")
                continue
            mu = float(np.mean(self._reward_queues[k])) if self._reward_queues[k] else 0.0
            bonus = self.config.exploration_c * math.sqrt(math.log(max(total, 1)) / n_k)
            scores[k] = mu + bonus
        return int(np.argmax(scores))

    def get_telemetry(self) -> dict:
        return {
            "rho_grid": list(self.config.rho_grid),
            "visits": self._visits.tolist(),
            "arm_means": [float(np.mean(q)) if q else 0.0 for q in self._reward_queues],
            "n_updates": int(self._update_calls),
            "n_telemetry_rows": len(self._telemetry),
        }

    def dump(self) -> dict:
        return {"telemetry": self._telemetry, "summary": self.get_telemetry()}
