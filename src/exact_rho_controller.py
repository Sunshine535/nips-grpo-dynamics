"""
Exact ρ* controller — computes Theorem-2 Cov(g⁺,g⁻)/Var(g⁺) from per-group
gradient samples via two additional autograd.grad calls per group.

This refutes / rescues the "ADQ uses a wrong proxy" critique by directly
estimating the *theoretical* quantity Theorem 2 prescribes, instead of
the binomial-variance approximation in src/adabalance.py.

Cost: 2 × B extra backward-pass traversals per aux update step
(retain_graph=True on the original forward graph). With B=2 groups per
batch and K=20 update interval, this is roughly +1-2 main-step worth of
compute per aux update — well under doubling overall wall-clock.

Estimator:
  Sample B group-level (g⁺_b, g⁻_b) pairs from the current training batch
  (one pair per prompt-group). Then:
    Var_s(g⁺) := (1/B) Σ_b ‖g⁺_b − ḡ⁺‖²
    Cov_s(g⁺, g⁻) := (1/B) Σ_b ⟨g⁺_b − ḡ⁺, g⁻_b − ḡ⁻⟩
  ρ̂* = Cov_s(g⁺,g⁻) / max(Var_s(g⁺), eps)
EMA-smoothed and clipped to [ρ_min, ρ_max].

When a group is degenerate (n⁺=0 or n⁻=0) we skip it for that update.
If too few non-degenerate groups remain, we keep the previous ρ.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class ExactRhoConfig:
    update_every: int = 20         # call exact_estimator() every N main steps
    rho_init: float = 1.0           # starting ρ
    rho_min: float = 0.1
    rho_max: float = 10.0
    ema_alpha: float = 0.7          # ρ_new = α·ρ_prev + (1-α)·ρ_estimate
    min_groups_for_update: int = 2  # skip update if fewer non-degenerate groups
    eps: float = 1e-6


class ExactRhoController:
    def __init__(self, config: ExactRhoConfig):
        self.config = config
        self._rho = float(config.rho_init)
        self._n_updates = 0
        self._n_skipped = 0
        self._telemetry: list = []

    @property
    def rho(self) -> float:
        return self._rho

    @torch.no_grad()
    def _flat_var_cov(self, g_plus: List[torch.Tensor], g_minus: List[torch.Tensor]
                     ) -> Tuple[float, float]:
        """g_plus[b], g_minus[b] are 1-D flat gradient tensors per group b. Returns (Var_s, Cov_s)."""
        B = len(g_plus)
        if B < 2:
            return 0.0, 0.0
        gp = torch.stack(g_plus, dim=0)   # (B, D)
        gn = torch.stack(g_minus, dim=0)
        gp_mean = gp.mean(dim=0)
        gn_mean = gn.mean(dim=0)
        dgp = gp - gp_mean
        dgn = gn - gn_mean
        var_p = float((dgp * dgp).sum(dim=1).mean().item())
        cov_pn = float((dgp * dgn).sum(dim=1).mean().item())
        return var_p, cov_pn

    def update(self, g_plus: List[torch.Tensor], g_minus: List[torch.Tensor],
               step: Optional[int] = None) -> float:
        """Receive per-group flat gradient samples; return new ρ (EMA-smoothed).

        Both lists must have the same length B = number of non-degenerate
        groups. If B < min_groups_for_update, ρ is left unchanged.
        """
        self._n_updates += 1
        B = len(g_plus)
        if B < self.config.min_groups_for_update or B != len(g_minus):
            self._n_skipped += 1
            self._telemetry.append({
                "step": step, "rho": self._rho, "n_groups_used": B,
                "skipped": True,
            })
            return self._rho

        var_p, cov_pn = self._flat_var_cov(g_plus, g_minus)
        rho_hat = cov_pn / max(var_p, self.config.eps)
        rho_hat = float(max(self.config.rho_min, min(self.config.rho_max, rho_hat)))
        # EMA
        self._rho = self.config.ema_alpha * self._rho + (1.0 - self.config.ema_alpha) * rho_hat
        self._telemetry.append({
            "step": step, "rho": self._rho, "n_groups_used": B,
            "var_p": var_p, "cov_pn": cov_pn, "rho_hat_raw": rho_hat,
            "skipped": False,
        })
        return self._rho

    def get_telemetry(self) -> dict:
        return {
            "n_updates": self._n_updates,
            "n_skipped": self._n_skipped,
            "current_rho": self._rho,
        }

    def dump(self) -> dict:
        return {"telemetry": self._telemetry, "summary": self.get_telemetry()}
