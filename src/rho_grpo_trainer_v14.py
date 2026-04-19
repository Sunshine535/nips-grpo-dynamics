"""
RhoGRPOTrainer rewritten for TRL 0.14 where compute_loss internally
computes advantages (not exposed via inputs dict).

Replicates TRL 0.14's compute_loss exactly + applies ρ-asymmetric weighting
to advantages + collects per-step stats for AdaBalanceController.
"""
import logging
from typing import Optional, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import is_conversational
from trl.data_utils import maybe_apply_chat_template, apply_chat_template
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class RhoGRPOTrainerV14(GRPOTrainer):
    """TRL 0.14-compatible RhoGRPO + AdaBalance.

    Copies TRL 0.14's compute_loss body but injects:
      - ρ-asymmetric advantage weighting after advantages are computed
      - Per-step stats collection for AdaBalance controller
      - Online ρ updates from AdaBalance controller
    """

    def __init__(
        self, *args,
        rho: float = 1.0,
        degenerate_floor: float = 0.0,
        controller=None,
        group_size_for_ada: int = 4,
        ada_kl_coef: float = 0.05,
        ada_clip_range: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rho = rho
        self._degenerate_floor = degenerate_floor
        self.ada_controller = controller
        self._ada_group_size = group_size_for_ada
        self._ada_kl_coef = ada_kl_coef
        self._ada_clip_range = ada_clip_range
        self._rho_step_stats = []
        self._ada_telemetry = []

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, v):
        self._rho = v

    def _apply_rho_weighting(
        self,
        advantages: torch.Tensor,
        completion_ids: Optional[torch.Tensor] = None,
        completion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply ρ-asymmetric weighting to pre-computed advantages.

        When ``completion_ids`` is provided (normal training path), we also
        compute the canonical per-group Q_CSD := H_norm(τ⁺) · (n⁺/G) collapse
        predictor, where H_norm is the entropy of the empirical correct-response
        distribution divided by log(n⁺), and aggregate it as the batch mean.

        Without ``completion_ids`` we cannot distinguish distinct correct
        responses from duplicates, so ``h_norm_pos`` is set to 0 and ``q_csd``
        becomes 0 for that step. Callers that want the optimistic upper bound
        (``H_norm = 1 ⇒ q_csd = n⁺/(B·G)``) should use
        :func:`src.csd_logging.compute_step0_qcsd` without completions instead.
        """
        pos_mask = advantages > 0
        neg_mask = advantages < 0

        raw_pos_w = self._rho
        raw_neg_w = 1.0
        norm_factor = 2.0 / (raw_pos_w + raw_neg_w)
        norm_pos_w = raw_pos_w * norm_factor
        norm_neg_w = raw_neg_w * norm_factor

        weighted = advantages.clone()
        weighted[pos_mask] = advantages[pos_mask] * norm_pos_w
        weighted[neg_mask] = advantages[neg_mask] * norm_neg_w

        step = self.state.global_step if self.state else 0
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        n_deg = int((advantages == 0).sum().item())

        # Canonical Q_CSD = H_norm(τ⁺) · (n⁺/G) per FINAL_PROPOSAL.md §"Empirical Hypothesis 1".
        # τ⁺ is a *per-group* object (group = G consecutive rows). For a batch of
        # B groups we compute Q_CSD_b per group and average, so batch Q_CSD ∈ [0, 1].
        G = max(self._ada_group_size, 1)
        n_total = advantages.shape[0]
        n_groups = n_total // G
        q_csd_per_group = []
        h_norm_per_group = []
        avail_per_group = []
        for b in range(n_groups):
            sl = slice(b * G, (b + 1) * G)
            grp_adv = advantages[sl]
            grp_pos_mask = grp_adv > 0
            n_pos_b = int(grp_pos_mask.sum().item())
            avail_b = n_pos_b / G  # ∈ [0, 1] by construction
            if n_pos_b >= 2 and completion_ids is not None:
                grp_completions = completion_ids[sl][grp_pos_mask]
                if completion_mask is not None:
                    lengths = completion_mask[sl][grp_pos_mask].sum(dim=1).tolist()
                else:
                    lengths = [grp_completions.size(1)] * grp_completions.size(0)
                hashes = [
                    hash(tuple(row[:int(L)].tolist()))
                    for row, L in zip(grp_completions, lengths)
                ]
                _, counts = np.unique(hashes, return_counts=True)
                probs = counts / counts.sum()
                entropy = float(-(probs * np.log(probs)).sum())
                h_norm_b = entropy / float(np.log(n_pos_b))
            else:
                h_norm_b = 0.0
            q_csd_per_group.append(h_norm_b * avail_b)
            h_norm_per_group.append(h_norm_b)
            avail_per_group.append(avail_b)

        if n_groups > 0:
            h_norm_pos = float(np.mean(h_norm_per_group))
            availability = float(np.mean(avail_per_group))
            q_csd = float(np.mean(q_csd_per_group))
        else:
            h_norm_pos = 0.0
            availability = 0.0
            q_csd = 0.0

        self._rho_step_stats.append({
            "step": step,
            "rho": self._rho,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_degenerate": n_deg,
            "n_groups": n_groups,
            "mean_pos_adv": float(advantages[pos_mask].mean()) if n_pos > 0 else 0.0,
            "mean_neg_adv": float(advantages[neg_mask].mean()) if n_neg > 0 else 0.0,
            "normalized_pos_weight": norm_pos_w,
            "normalized_neg_weight": norm_neg_w,
            "h_norm_pos": h_norm_pos,
            "availability": availability,
            "q_csd": q_csd,
            "q_csd_per_group": q_csd_per_group,
        })

        return weighted

    def _update_adabalance(self, rewards: torch.Tensor, advantages: torch.Tensor):
        """Feed AdaBalance controller with per-group stats, update ρ."""
        if self.ada_controller is None:
            return

        G = self._ada_group_size
        n = rewards.shape[0]
        n_groups = n // G
        if n_groups == 0:
            return

        reward_groups = rewards[:n_groups * G].reshape(n_groups, G)
        success_counts = reward_groups.sum(dim=1).cpu().numpy().astype(float)

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
        self._ada_telemetry.append({
            "step": step,
            "rho": new_rho,
            "p_hat": float(success_counts.mean() / G),
            "n_groups": n_groups,
            "success_count_std": float(success_counts.std()),
            **self.ada_controller.get_telemetry(),
        })

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Replicate TRL 0.14 GRPOTrainer.compute_loss body, injecting ρ weighting
        # and AdaBalance stats collection.
        if return_outputs:
            raise ValueError("RhoGRPOTrainerV14 does not support return_outputs")

        device = self.accelerator.device

        # ─── Prepare prompts ───
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        # Manually move to accelerator device. The parent
        # GRPOTrainer._prepare_inputs is the inline-rollout-and-advantage path
        # which we are explicitly replacing here, so we must not call it.
        prompt_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in prompt_inputs.items()
        }
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        # ─── Generate completions ───
        from accelerate.utils import gather_object
        with self.accelerator.unwrap_model(model).disable_adapter() if hasattr(
            self.accelerator.unwrap_model(model), "disable_adapter"
        ) else _NullCtx():
            pass

        unwrapped_model = self.accelerator.unwrap_model(model)
        prompt_completion_ids = unwrapped_model.generate(
            **prompt_inputs, generation_config=self.generation_config
        )
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after first EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # ─── Compute logps ───
        def get_per_token_logps(mdl, input_ids, num_logits_to_keep):
            logits = mdl(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits
            logits = logits[:, :-1, :]
            input_ids = input_ids[:, -num_logits_to_keep:]
            logits = logits[:, -num_logits_to_keep:]
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        num_logits_to_keep = completion_ids.size(1)
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, num_logits_to_keep)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # ─── Decode completions ───
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": c}] for c in completions]

        # ─── Compute rewards ───
        # Shape convention: completion_ids has B*G rows (B prompts × G generations).
        # rewards_per_func must therefore have (B*G, num_reward_funcs) — NOT (B, num_reward_funcs).
        n_total = len(completions)  # = B * G
        assert n_total % self.num_generations == 0, \
            f"completions={n_total} not divisible by num_generations={self.num_generations}"

        rewards_per_func = torch.zeros(n_total, len(self.reward_funcs), device=device)
        prompts_for_reward = []
        for example in inputs:
            prompts_for_reward.extend([example["prompt"]] * self.num_generations)
        for i, (reward_func, reward_proc) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                raise NotImplementedError("Reward model not supported in V14 trainer")
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts_for_reward, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)  # shape: (B*G,)

        # ─── Compute advantages ───
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # ─── INJECT: ρ-asymmetric weighting + AdaBalance update ───
        self._update_adabalance(rewards, advantages)
        advantages = self._apply_rho_weighting(advantages, completion_ids, completion_mask)

        # ─── Loss ───
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # ─── Metrics (same as TRL 0.14) ───
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                name = reward_func.config._name_or_path.split("/")[-1]
            else:
                name = reward_func.__name__
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass
