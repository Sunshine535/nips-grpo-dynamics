"""
ASERTrainerV14 — Adaptive Support Expansion with Replay (MVP).

Forked from RhoGRPOTrainerV14 with the four surgeries in `RETRACTIONS.md`-style
audit clarity:

  A. Drop ρ / ADQ / bandit / exact-ρ* controllers entirely — this is not a
     ρ-controller paper.
  B. Replace advantage computation:
       dr_grpo:   A_ij = r_ij − r̄_i           (no std normalisation)
       spo:       A_ij = r_ij − b(prompt_id)   (persistent EMA baseline)
  C. Drop per-row completion-length normalisation — use sum-then-batch-mean
     aggregation instead (this is what Dr. GRPO actually prescribes).
  D. Add a small replay-CE loss from the verified-success replay bank.

This trainer expects each example in `inputs` to have a `prompt_id` field (int).
The data pipeline (see `scripts/run_aser_mvp.py`) adds it via `dataset.map(..., with_indices=True)`.
"""
import logging
from typing import Any, Optional, List

import numpy as np
import torch
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import is_conversational
from trl.data_utils import maybe_apply_chat_template
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


class ASERTrainerV14(GRPOTrainer):
    """SPO / Dr. GRPO backbone + per-prompt baseline + verified replay CE."""

    def __init__(
        self, *args,
        backbone_mode: str = "spo",                # "spo" | "dr_grpo"
        prompt_stats=None,
        replay_bank=None,
        lambda_rep: float = 0.05,
        replay_batch_size: int = 4,
        replay_warmup_steps: int = 50,
        success_threshold: float = 0.5,            # reward >= this → push to replay bank
        pg_weight: float = 1.0,                    # 0.0 → pure RFT (bank-only, no GRPO gradient)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert backbone_mode in ("spo", "dr_grpo"), backbone_mode
        self.backbone_mode = backbone_mode
        self.prompt_stats = prompt_stats
        self.replay_bank = replay_bank
        self.lambda_rep = float(lambda_rep)
        self.replay_batch_size = int(replay_batch_size)
        self.replay_warmup_steps = int(replay_warmup_steps)
        self.success_threshold = float(success_threshold)
        self.pg_weight = float(pg_weight)

        self._aser_step_stats: list = []

    # ------------------------------------------------------------------
    def _compute_replay_loss(self, model, device) -> torch.Tensor:
        """Compute a supervised-fine-tuning-style cross-entropy loss on a small
        batch of verified-success completions drawn from the replay bank."""
        if self.replay_bank is None or self.lambda_rep <= 0:
            return torch.tensor(0.0, device=device)
        step = int(self.state.global_step) if self.state else 0
        if step < self.replay_warmup_steps:
            return torch.tensor(0.0, device=device)
        items = self.replay_bank.sample(self.replay_batch_size)
        if not items:
            return torch.tensor(0.0, device=device)

        tok = self.processing_class
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id
        # Each stored item has the raw prompt text (template-applied) and the completion token ids.
        prompts = [it["prompt"] for it in items]
        prompt_enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                         add_special_tokens=False)

        input_ids_rows = []
        labels_rows = []
        for p_ids, item in zip(prompt_enc["input_ids"], items):
            p = [int(t) for t in p_ids.tolist() if int(t) != pad_id]
            c = [int(t) for t in item["token_ids"]]
            input_ids_rows.append(p + c)
            labels_rows.append([-100] * len(p) + c)

        max_len = max(len(r) for r in input_ids_rows)
        for i in range(len(input_ids_rows)):
            pad_len = max_len - len(input_ids_rows[i])
            input_ids_rows[i] = input_ids_rows[i] + [pad_id] * pad_len
            labels_rows[i] = labels_rows[i] + [-100] * pad_len

        input_ids = torch.tensor(input_ids_rows, device=device, dtype=torch.long)
        labels = torch.tensor(labels_rows, device=device, dtype=torch.long)
        attn = (input_ids != pad_id).long()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        return out.loss

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("ASERTrainerV14 does not support return_outputs")
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
        prompt_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in prompt_inputs.items()
        }
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        # ─── Generate completions ───
        unwrapped_model = self.accelerator.unwrap_model(model)
        prompt_completion_ids = unwrapped_model.generate(
            **prompt_inputs, generation_config=self.generation_config
        )
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after first EOS.
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # ─── Compute logps (current + reference) ───
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

        # ─── Rewards ───
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": c}] for c in completions]
        n_total = len(completions)
        assert n_total % self.num_generations == 0, \
            f"completions={n_total} not divisible by num_generations={self.num_generations}"

        rewards_per_func = torch.zeros(n_total, len(self.reward_funcs), device=device)
        prompts_for_reward: list = []
        for example in inputs:
            prompts_for_reward.extend([example["prompt"]] * self.num_generations)
        for i, (reward_func, _rp) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                raise NotImplementedError("Reward model not supported in ASER trainer")
            # NB: drop prompt_id from reward kwargs so it doesn't pollute the reward fn.
            reward_kwargs = {k: [] for k in inputs[0].keys() if k not in ("prompt", "completion", "prompt_id")}
            for k in reward_kwargs:
                for example in inputs:
                    reward_kwargs[k].extend([example[k]] * self.num_generations)
            out = reward_func(prompts=prompts_for_reward, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)  # (B*G,)

        # ─── Advantages (SPO / Dr. GRPO — NO std normalisation) ───
        group_rewards = rewards.view(-1, self.num_generations)                      # (B, G)
        group_mean_rewards = group_rewards.mean(dim=1)                              # (B,)
        group_prompt_ids = [int(x.get("prompt_id", -1)) for x in inputs]            # (B,)

        if self.backbone_mode == "dr_grpo":
            advantages = rewards - group_mean_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.backbone_mode == "spo":
            assert self.prompt_stats is not None, "SPO backbone requires prompt_stats"
            baseline_vals = torch.tensor(
                [self.prompt_stats.get_baseline(pid) for pid in group_prompt_ids],
                device=device, dtype=rewards.dtype,
            )
            advantages = rewards - baseline_vals.repeat_interleave(self.num_generations, dim=0)
        else:
            raise ValueError(self.backbone_mode)

        # ─── Update per-prompt stats AFTER advantages are computed (use-then-update) ───
        if self.prompt_stats is not None:
            with torch.no_grad():
                for pid, mr in zip(group_prompt_ids, group_mean_rewards.detach().cpu().tolist()):
                    if pid >= 0:
                        self.prompt_stats.update(pid, float(mr))

        # ─── Push successes into replay bank ───
        if self.replay_bank is not None:
            with torch.no_grad():
                step = int(self.state.global_step) if self.state else 0
                valid_lens = completion_mask.sum(dim=1).tolist()
                reward_list = rewards.detach().cpu().tolist()
                # expand prompt_id / prompt_text per rollout (G per prompt)
                expanded_pids: list = []
                expanded_prompts: list = []
                for ex, ptxt in zip(inputs, prompts_text):
                    expanded_pids.extend([int(ex.get("prompt_id", -1))] * self.num_generations)
                    expanded_prompts.extend([ptxt] * self.num_generations)
                for pid, ptxt, comp_ids_row, L, r in zip(
                    expanded_pids, expanded_prompts, completion_ids, valid_lens, reward_list
                ):
                    if pid < 0 or r < self.success_threshold:
                        continue
                    toks = comp_ids_row[: int(L)].detach().cpu().tolist()
                    if not toks:
                        continue
                    txt = self.processing_class.decode(toks, skip_special_tokens=True)
                    self.replay_bank.add_success(
                        prompt_id=pid, prompt_text=ptxt, token_ids=toks, text=txt, step=step,
                    )

        # ─── Policy-gradient loss — NO per-row length normalisation ───
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # sum over tokens per row, then mean over rows (faithful Dr. GRPO / SPO aggregation).
        loss_pg = (per_token_loss * completion_mask).sum(dim=1).mean()

        # ─── Replay loss ───
        if self.lambda_rep > 0:
            loss_rep = self._compute_replay_loss(model, device)
        else:
            loss_rep = torch.tensor(0.0, device=device)
        loss = self.pg_weight * loss_pg + self.lambda_rep * loss_rep

        # ─── Metrics (TRL-compatible) ───
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
        # replace the usual "reward_std" metric with "hardness_ema_mean" for ASER telemetry
        if self.prompt_stats is not None:
            hard_vals = [self.prompt_stats.get_hardness(pid) for pid in group_prompt_ids if pid >= 0]
            if hard_vals:
                self._metrics.setdefault("aser/hardness_ema", []).append(float(np.mean(hard_vals)))
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # ─── step stats ───
        step = int(self.state.global_step) if self.state else 0
        # Count duplicate prompt_ids in batch (adaptive-dup telemetry)
        pid_counts = {}
        for pid in group_prompt_ids:
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
        batch_n_dup = sum(c - 1 for c in pid_counts.values() if c > 1)
        self._aser_step_stats.append({
            "step": step,
            "backbone": self.backbone_mode,
            "mean_reward": float(rewards.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "replay_bank_size": int(self.replay_bank.size()) if self.replay_bank is not None else 0,
            "replay_bank_prompts": int(self.replay_bank.n_prompts()) if self.replay_bank is not None else 0,
            "loss_pg": float(loss_pg.detach().item()),
            "loss_rep": float(loss_rep.detach().item()) if isinstance(loss_rep, torch.Tensor) else 0.0,
            "pg_weight": self.pg_weight,
            "batch_n_dup": batch_n_dup,
        })
        return loss
