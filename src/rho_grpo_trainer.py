"""
Custom GRPOTrainers that intercept TRL's advantage tensor in compute_loss.

Three trainers for three parameterizations of the same idea:
  - BalancedGRPOTrainer: (alpha, beta) weighting on positive/negative advantages
  - RhoGRPOTrainer: rho weighting (positive * rho, negative * 1.0)
  - AdaBalanceGRPOTrainer: adaptive rho via online controller

All rely on TRL >= 0.15 storing pre-computed advantages in the batch dict.
"""

import logging
from typing import Optional

import numpy as np
import torch
from trl import GRPOTrainer
from transformers import TrainerCallback

from .rho_grpo import RhoGRPOConfig, compute_group_statistics
from .stability_analysis import (
    analyze_stability,
    classify_regime,
    compute_gradient_variance,
    group_starvation_rate,
)

logger = logging.getLogger(__name__)


class BalancedGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with independent (alpha, beta) weighting on advantages.

    positive advantages are scaled by alpha;
    negative advantages are scaled by (1 - alpha) * beta.
    This is NOT equivalent to a single rho when beta=0.
    """

    def __init__(self, *args, alpha: float = 0.5, beta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._balanced_step_stats = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "advantages" in inputs and isinstance(inputs.get("advantages"), torch.Tensor):
            advantages = inputs["advantages"]

            pos_mask = advantages > 0
            neg_mask = advantages < 0
            zero_mask = advantages == 0

            raw_pos_w = self._alpha
            raw_neg_w = (1.0 - self._alpha) * self._beta

            norm_sum = raw_pos_w + raw_neg_w
            if norm_sum > 0:
                norm_pos_w = raw_pos_w * 2.0 / norm_sum
                norm_neg_w = raw_neg_w * 2.0 / norm_sum
            else:
                norm_pos_w = 1.0
                norm_neg_w = 1.0

            weighted = advantages.clone()
            weighted[pos_mask] = advantages[pos_mask] * norm_pos_w
            weighted[neg_mask] = advantages[neg_mask] * norm_neg_w

            inputs = dict(inputs)
            inputs["advantages"] = weighted

            n_pos = int(pos_mask.sum().item())
            n_neg = int(neg_mask.sum().item())
            n_deg = int(zero_mask.sum().item())
            step = self.state.global_step if self.state else 0
            self._balanced_step_stats.append({
                "step": step,
                "alpha": self._alpha,
                "beta": self._beta,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_degenerate": n_deg,
                "raw_pos_weight": raw_pos_w,
                "raw_neg_weight": raw_neg_w,
                "normalized_pos_weight": norm_pos_w,
                "normalized_neg_weight": norm_neg_w,
                "pos_neg_ratio_raw": raw_pos_w / max(raw_neg_w, 1e-8),
                "pos_neg_ratio_normalized": norm_pos_w / max(norm_neg_w, 1e-8),
            })

        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


class RhoGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with rho-asymmetric advantage weighting.

    rho > 1 → upweight positive signal → exploitative
    rho < 1 → downweight positive signal → exploratory
    rho = 1 → standard symmetric GRPO
    """

    def __init__(self, *args, rho: float = 1.0, degenerate_floor: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._rho = rho
        self._degenerate_floor = degenerate_floor
        self._rho_step_stats = []

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value: float):
        self._rho = value

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "advantages" in inputs and isinstance(inputs.get("advantages"), torch.Tensor):
            advantages = inputs["advantages"]

            pos_mask = advantages > 0
            neg_mask = advantages < 0
            zero_mask = advantages == 0

            raw_pos_w = self._rho
            raw_neg_w = 1.0
            norm_factor = 2.0 / (raw_pos_w + raw_neg_w)
            norm_pos_w = raw_pos_w * norm_factor
            norm_neg_w = raw_neg_w * norm_factor

            weighted = advantages.clone()
            weighted[pos_mask] = advantages[pos_mask] * norm_pos_w
            weighted[neg_mask] = advantages[neg_mask] * norm_neg_w
            if self._degenerate_floor != 0.0:
                weighted[zero_mask] = self._degenerate_floor

            inputs = dict(inputs)
            inputs["advantages"] = weighted

            n_pos = int(pos_mask.sum().item())
            n_neg = int(neg_mask.sum().item())
            n_deg = int(zero_mask.sum().item())
            step = self.state.global_step if self.state else 0
            self._rho_step_stats.append({
                "step": step,
                "rho": self._rho,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_degenerate": n_deg,
                "mean_pos_adv": float(advantages[pos_mask].mean()) if n_pos > 0 else 0.0,
                "mean_neg_adv": float(advantages[neg_mask].mean()) if n_neg > 0 else 0.0,
                "degenerate_ratio": n_deg / max(n_pos + n_neg + n_deg, 1),
                "raw_pos_weight": raw_pos_w,
                "raw_neg_weight": raw_neg_w,
                "normalized_pos_weight": norm_pos_w,
                "normalized_neg_weight": norm_neg_w,
            })

        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


class AdaBalanceGRPOTrainer(RhoGRPOTrainer):
    """
    RhoGRPOTrainer with adaptive rho from AdaBalanceController.

    On each compute_loss call, estimates group success counts from the
    advantages tensor and feeds them to the controller. The controller
    updates rho based on its internal stability analysis.
    """

    def __init__(self, *args, controller=None, group_size: int = 4,
                 kl_coef: float = 0.05, clip_range: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ada_controller = controller
        self._ada_group_size = group_size
        self._ada_kl_coef = kl_coef
        self._ada_clip_range = clip_range
        self._ada_telemetry = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (
            self.ada_controller is not None
            and "advantages" in inputs
            and isinstance(inputs.get("advantages"), torch.Tensor)
        ):
            advantages = inputs["advantages"]
            G = self._ada_group_size
            n_samples = advantages.shape[0]
            n_groups = n_samples // G

            if n_groups > 0:
                adv_groups = advantages[:n_groups * G].reshape(n_groups, G)

                if "rewards" in inputs and isinstance(inputs.get("rewards"), torch.Tensor):
                    reward_groups = inputs["rewards"][:n_groups * G].reshape(n_groups, G)
                    success_counts = reward_groups.sum(dim=1).cpu().numpy().astype(float)
                else:
                    pos_counts = (adv_groups > 0).sum(dim=1)
                    is_degenerate = (adv_groups == 0).all(dim=1)
                    non_degen_mask = ~is_degenerate
                    p_batch = (
                        pos_counts[non_degen_mask].float().mean().item() / G
                        if non_degen_mask.any() else 0.5
                    )
                    success_counts = pos_counts.float()
                    # AUDIT NOTE: the block below fabricates synthetic
                    # success-count labels for degenerate groups from the
                    # binomial `p_batch**G` / `(1-p_batch)**G` mixture. This
                    # injects *non-observed* signal into `ada_controller.update`
                    # and is a known confounder for any AdaBalance run that
                    # encountered degenerate groups. V14 (`rho_grpo_trainer_v14.py`)
                    # does NOT do this — it only uses real rewards. If you are
                    # pushing paper-grade claims, use V14; see RETRACTIONS.md §3.
                    if is_degenerate.any():
                        all_success_prob = p_batch ** G
                        all_fail_prob = (1 - p_batch) ** G
                        denom = max(all_success_prob + all_fail_prob, 1e-8)
                        frac_success = all_success_prob / denom
                        degen_idx = is_degenerate.nonzero(as_tuple=True)[0]
                        for idx in degen_idx:
                            success_counts[idx] = float(G) if torch.rand(1).item() < frac_success else 0.0
                    success_counts = success_counts.cpu().numpy()

                pos_mask = advantages > 0
                neg_mask = advantages < 0
                grad_pos_proxy = advantages[pos_mask].abs().cpu().numpy() if pos_mask.any() else None
                grad_neg_proxy = advantages[neg_mask].abs().cpu().numpy() if neg_mask.any() else None

                new_rho = self.ada_controller.update(
                    success_counts, G,
                    grad_pos_norms=grad_pos_proxy,
                    grad_neg_norms=grad_neg_proxy,
                    kl_coef=self._ada_kl_coef,
                    clip_range=self._ada_clip_range,
                )
                self._rho = new_rho

                step = self.state.global_step if self.state else 0
                n_pos = int(pos_mask.sum().item())
                n_neg = int(neg_mask.sum().item())
                n_degen = int((adv_groups == 0).all(dim=1).sum().item())
                self._ada_telemetry.append({
                    "step": step,
                    "rho": new_rho,
                    "p_hat": float(success_counts.mean() / G),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n_groups": n_groups,
                    "n_degenerate_groups": n_degen,
                    "success_count_std": float(success_counts.std()),
                    **self.ada_controller.get_telemetry(),
                })

        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)


class RhoStabilityCallback(TrainerCallback):
    """Log stability diagnostics from the rho-GRPO trainer at each step."""

    def __init__(self, trainer, group_size: int,
                 kl_coef: float = 0.05, clip_range: float = 0.2):
        self.trainer = trainer
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.initial_kl = None
        self.telemetry = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        reward_mean = logs.get("reward/mean", logs.get("reward_mean"))
        if reward_mean is None:
            return

        p_est = max(0.01, min(0.99, reward_mean))
        gsr = group_starvation_rate(p_est, self.group_size)

        rho_val = getattr(self.trainer, '_rho', 1.0)
        bounds = analyze_stability(p_est, self.group_size, self.kl_coef, self.clip_range)
        grad_var = compute_gradient_variance(rho_val, bounds)

        kl_val = logs.get("kl", 0)
        if self.initial_kl is None and kl_val > 0:
            self.initial_kl = kl_val
        kl_ratio = kl_val / self.initial_kl if self.initial_kl and self.initial_kl > 0 else 1.0

        regime = classify_regime(rho_val, bounds, kl_ratio=kl_ratio)

        record = {
            "step": state.global_step,
            "rho": rho_val,
            "p_hat": p_est,
            "GSR": gsr,
            "rho_min": bounds.rho_min,
            "rho_max": bounds.rho_max,
            "rho_star": bounds.rho_star,
            "V_plus": bounds.V_plus,
            "V_minus": bounds.V_minus,
            "C_pG": bounds.C_pG,
            "gradient_variance": grad_var,
            "regime": regime,
            "kl_ratio": kl_ratio,
        }
        self.telemetry.append(record)

        logs["rho_grpo/rho"] = rho_val
        logs["rho_grpo/regime"] = regime
        logs["rho_grpo/GSR"] = gsr
        logs["rho_grpo/gradient_variance"] = grad_var
