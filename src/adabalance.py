import json
import logging
import os

import numpy as np
from collections import deque
from dataclasses import dataclass
from transformers import TrainerCallback
from typing import Optional

from .stability_analysis import (
    compute_advantage_variance_components,
    compute_rho_star,
    compute_rho_min,
    compute_rho_max,
    group_starvation_rate,
)


@dataclass
class AdaBalanceConfig:
    K: int = 50
    tau: float = 0.1
    rho_init: float = 1.0
    rho_min_floor: float = 0.1
    rho_max_ceil: float = 10.0
    warmup_steps: int = 50
    history_window: int = 200


class AdaBalanceController:
    def __init__(self, config: AdaBalanceConfig):
        self.config = config
        self.rho = config.rho_init
        self.rho_ema = config.rho_init
        self.step = 0

        self.p_hat_ema = 0.5
        self.gsr_ema = 0.0

        self.V_plus_ema = 1.0
        self.V_minus_ema = 1.0
        self.C_pG_ema = 0.0

        self.success_history = deque(maxlen=config.history_window)
        self.grad_pos_history = deque(maxlen=config.history_window)
        self.grad_neg_history = deque(maxlen=config.history_window)

        self.rho_history = []
        self.p_hat_history = []
        self.gsr_history = []
        self.bounds_history = []

    def update(
        self,
        success_counts: np.ndarray,
        group_size: int,
        grad_pos_norms: Optional[np.ndarray] = None,
        grad_neg_norms: Optional[np.ndarray] = None,
        kl_coef: float = 0.05,
        clip_range: float = 0.2,
    ) -> float:
        self.step += 1

        p_batch = success_counts.mean() / group_size
        gsr_batch = group_starvation_rate(p_batch, group_size)

        tau = self.config.tau
        self.p_hat_ema = (1 - tau) * self.p_hat_ema + tau * p_batch
        self.gsr_ema = (1 - tau) * self.gsr_ema + tau * gsr_batch

        self.success_history.extend(success_counts.tolist())

        grad_var_plus = 1.0
        grad_var_minus = 1.0
        grad_pos_norm = 1.0

        if grad_pos_norms is not None:
            self.grad_pos_history.extend(grad_pos_norms.tolist())
            grad_var_plus = float(np.var(list(self.grad_pos_history))) if len(self.grad_pos_history) > 1 else 1.0
            grad_pos_norm = float(np.mean(list(self.grad_pos_history)))

        if grad_neg_norms is not None:
            self.grad_neg_history.extend(grad_neg_norms.tolist())
            grad_var_minus = float(np.var(list(self.grad_neg_history))) if len(self.grad_neg_history) > 1 else 1.0

        if self.step < self.config.warmup_steps:
            self._record(p_batch, gsr_batch)
            return self.rho

        if self.step % self.config.K != 0:
            return self.rho

        V_plus, V_minus, C_pG = compute_advantage_variance_components(
            self.p_hat_ema, group_size, grad_var_plus, grad_var_minus
        )

        self.V_plus_ema = (1 - tau) * self.V_plus_ema + tau * V_plus
        self.V_minus_ema = (1 - tau) * self.V_minus_ema + tau * V_minus
        self.C_pG_ema = (1 - tau) * self.C_pG_ema + tau * C_pG

        rho_star = compute_rho_star(self.V_plus_ema, self.C_pG_ema)
        rho_min = compute_rho_min(self.V_minus_ema, self.C_pG_ema)
        rho_max = compute_rho_max(
            self.p_hat_ema, group_size, kl_coef, clip_range, grad_pos_norm
        )

        rho_target = np.clip(rho_star, rho_min, rho_max)
        rho_target = np.clip(rho_target, self.config.rho_min_floor, self.config.rho_max_ceil)

        self.rho_ema = (1 - tau) * self.rho_ema + tau * rho_target
        self.rho = float(self.rho_ema)

        self._record(p_batch, gsr_batch, rho_min, rho_max, rho_star)

        return self.rho

    def _record(self, p_batch, gsr_batch, rho_min=None, rho_max=None, rho_star=None):
        self.rho_history.append(self.rho)
        self.p_hat_history.append(self.p_hat_ema)
        self.gsr_history.append(self.gsr_ema)
        self.bounds_history.append({
            "step": self.step,
            "rho": self.rho,
            "p_hat": self.p_hat_ema,
            "gsr": self.gsr_ema,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "rho_star": rho_star,
            "V_plus": self.V_plus_ema,
            "V_minus": self.V_minus_ema,
            "C_pG": self.C_pG_ema,
        })

    def get_telemetry(self) -> dict:
        return {
            "rho": self.rho,
            "rho_ema": self.rho_ema,
            "p_hat_ema": self.p_hat_ema,
            "gsr_ema": self.gsr_ema,
            "V_plus_ema": self.V_plus_ema,
            "V_minus_ema": self.V_minus_ema,
            "C_pG_ema": self.C_pG_ema,
            "step": self.step,
        }

    def state_dict(self) -> dict:
        return {
            "rho": self.rho,
            "rho_ema": self.rho_ema,
            "step": self.step,
            "p_hat_ema": self.p_hat_ema,
            "gsr_ema": self.gsr_ema,
            "V_plus_ema": self.V_plus_ema,
            "V_minus_ema": self.V_minus_ema,
            "C_pG_ema": self.C_pG_ema,
            "success_history": list(self.success_history),
            "grad_pos_history": list(self.grad_pos_history),
            "grad_neg_history": list(self.grad_neg_history),
            "rho_history": self.rho_history,
            "p_hat_history": self.p_hat_history,
            "gsr_history": self.gsr_history,
            "bounds_history": self.bounds_history,
        }

    def load_state_dict(self, state: dict):
        self.rho = state["rho"]
        self.rho_ema = state["rho_ema"]
        self.step = state["step"]
        self.p_hat_ema = state["p_hat_ema"]
        self.gsr_ema = state["gsr_ema"]
        self.V_plus_ema = state["V_plus_ema"]
        self.V_minus_ema = state["V_minus_ema"]
        self.C_pG_ema = state["C_pG_ema"]
        self.success_history = deque(state.get("success_history", []),
                                     maxlen=self.config.history_window)
        self.grad_pos_history = deque(state.get("grad_pos_history", []),
                                      maxlen=self.config.history_window)
        self.grad_neg_history = deque(state.get("grad_neg_history", []),
                                      maxlen=self.config.history_window)
        self.rho_history = state.get("rho_history", [])
        self.p_hat_history = state.get("p_hat_history", [])
        self.gsr_history = state.get("gsr_history", [])
        self.bounds_history = state.get("bounds_history", [])


class AdaBalanceCallback(TrainerCallback):
    """
    Callback that feeds real training statistics to the AdaBalanceController
    and updates the trainer's rho on-the-fly.

    Requires the trainer to be a RhoGRPOTrainer (or subclass) so that
    the rho attribute exists and is respected in loss computation.
    """

    def __init__(
        self,
        controller: AdaBalanceController,
        group_size: int = 4,
        kl_coef: float = 0.05,
        clip_range: float = 0.2,
    ):
        self.controller = controller
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.step_metrics = []
        self._trainer_ref = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return

        reward_mean = logs.get("reward/mean", logs.get("reward_mean", None))
        if reward_mean is not None and self._trainer_ref is not None:
            trainer = self._trainer_ref
            last_stats = getattr(trainer, '_rho_step_stats', [])
            if last_stats:
                latest = last_stats[-1]
                n_pos = latest.get("n_positive", 0)
                n_neg = latest.get("n_negative", 0)
                n_total = n_pos + n_neg
                if n_total > 0:
                    n_groups = max(1, n_total // self.group_size)
                    p_estimate = max(0.0, min(1.0, n_pos / n_total))
                    success_counts = np.array([
                        round(p_estimate * self.group_size)
                    ] * n_groups)

                    new_rho = self.controller.update(
                        success_counts, self.group_size,
                        kl_coef=self.kl_coef, clip_range=self.clip_range,
                    )

                    if hasattr(trainer, 'rho'):
                        trainer.rho = new_rho

                    logs["adabalance/rho"] = new_rho
            else:
                p_estimate = max(0.0, min(1.0, reward_mean))
                n_groups = max(1, args.per_device_train_batch_size)
                success_counts = np.array([
                    round(p_estimate * self.group_size)
                ] * n_groups)

                new_rho = self.controller.update(
                    success_counts, self.group_size,
                    kl_coef=self.kl_coef, clip_range=self.clip_range,
                )

                if hasattr(self._trainer_ref, 'rho'):
                    self._trainer_ref.rho = new_rho

                logs["adabalance/rho"] = new_rho

        telemetry = self.controller.get_telemetry()
        for key, val in telemetry.items():
            if isinstance(val, (int, float)):
                logs[f"adabalance/{key}"] = val

        self.step_metrics.append({
            "step": state.global_step,
            **telemetry,
        })

    def on_save(self, args, state, control, **kwargs):
        """Persist controller state inside each checkpoint for correct resume."""
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            path = os.path.join(ckpt_dir, "adabalance_state.json")
            with open(path, "w") as f:
                json.dump(self.controller.state_dict(), f, indent=2)
            logging.getLogger(__name__).info(
                "Saved AdaBalance state to %s (step %d, rho=%.4f)",
                path, state.global_step, self.controller.rho,
            )
