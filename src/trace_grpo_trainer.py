"""
TRACE-GRPO Trainer — Trust-Calibrated Replay and Prompt-Conditioned Credit
Assignment for Sparse Binary GRPO.

Minimal fork of ASERTrainerV14. Key difference: replaces fixed lambda_rep
uniform replay with adaptive lambda_eff controlled by PromptCreditState
and TrustGatedReplayBank.

Three modes (for A/B/C ablation):
  --trace-mode full           : adaptive lambda_eff (C = full TRACE)
  --trace-mode constant_gate  : lambda_eff = lambda_max always (B = no mechanism)
  --trace-mode no_replay      : lambda_eff = 0 always (A-like, SPO-only)
"""
import logging
import math
from typing import Optional

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


class TraceGRPOTrainer(GRPOTrainer):
    """TRACE-GRPO: trust-calibrated replay for sparse binary GRPO."""

    def __init__(
        self, *args,
        backbone_mode: str = "spo",
        prompt_credit_store=None,
        trust_replay_bank=None,
        lambda_max: float = 0.05,
        replay_batch_size: int = 2,
        replay_warmup_steps: int = 50,
        success_threshold: float = 0.5,
        trace_mode: str = "full",
        drift_budget_cap: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert backbone_mode in ("spo", "dr_grpo", "tasa"), backbone_mode
        assert trace_mode in ("full", "constant_gate", "no_replay"), trace_mode
        self.backbone_mode = backbone_mode
        self.credit_store = prompt_credit_store
        self.trust_bank = trust_replay_bank
        self.lambda_max = float(lambda_max)
        self.replay_batch_size = int(replay_batch_size)
        self.replay_warmup_steps = int(replay_warmup_steps)
        self.success_threshold = float(success_threshold)
        self.trace_mode = trace_mode
        self.drift_budget_cap = float(drift_budget_cap)
        self._trace_step_stats: list = []
        self._replay_token_total = 0
        self._pg_token_total = 0

    def _compute_trace_replay_loss(self, model, device) -> tuple:
        """Trust-gated replay CE with adaptive lambda_eff."""
        if self.trust_bank is None or self.trace_mode == "no_replay":
            return torch.tensor(0.0, device=device), 0.0
        step = int(self.state.global_step) if self.state else 0
        if step < self.replay_warmup_steps:
            return torch.tensor(0.0, device=device), 0.0

        if self.trace_mode == "full":
            items = self.trust_bank.weighted_sample(
                self.replay_batch_size, step, credit_store=self.credit_store)
        else:
            items = self.trust_bank.weighted_sample(
                self.replay_batch_size, step, credit_store=None)
        if not items:
            return torch.tensor(0.0, device=device), 0.0

        if self.trace_mode == "full":
            mean_trust = np.mean([it.get("trust_weight", 1.0) for it in items])
            drift_ratio = self._replay_token_total / max(self._pg_token_total, 1)
            drift_budget = max(0.0, 1.0 - drift_ratio / self.drift_budget_cap)
            lambda_eff = self.lambda_max * min(mean_trust, 1.0) * drift_budget
        else:
            lambda_eff = self.lambda_max

        if lambda_eff < 1e-8:
            return torch.tensor(0.0, device=device), 0.0

        tok = self.processing_class
        pad_id = tok.pad_token_id or tok.eos_token_id
        prompts = [it["prompt"] for it in items]
        prompt_enc = tok(prompts, return_tensors="pt", padding=True,
                         padding_side="left", add_special_tokens=False)
        input_ids_rows, labels_rows = [], []
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

        replay_tokens = sum(len(it["token_ids"]) for it in items)
        self._replay_token_total += replay_tokens

        return out.loss * lambda_eff, float(lambda_eff)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("TraceGRPOTrainer does not support return_outputs")
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
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

        unwrapped = self.accelerator.unwrap_model(model)
        prompt_completion_ids = unwrapped.generate(
            **prompt_inputs, generation_config=self.generation_config)
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        # GPT-5.5 P0: pass full attention_mask to logprob computation (prompt + completion up to EOS)
        prompt_attention_mask = prompt_inputs["attention_mask"].to(dtype=completion_mask.dtype)
        if prompt_attention_mask.size(0) != prompt_completion_ids.size(0):
            factor = prompt_completion_ids.size(0) // prompt_attention_mask.size(0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(factor, dim=0)
        full_attention_mask = torch.cat([prompt_attention_mask, completion_mask], dim=1)

        def get_per_token_logps(mdl, ids, attention_mask, n_keep):
            logits = mdl(ids, attention_mask=attention_mask,
                         num_logits_to_keep=n_keep + 1).logits
            logits = logits[:, :-1, :]
            target = ids[:, -n_keep:]
            logits = logits[:, -n_keep:]
            per_token = []
            for lr, tr in zip(logits, target):
                lp = lr.log_softmax(dim=-1)
                per_token.append(torch.gather(lp, 1, tr.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token)

        n_keep = completion_ids.size(1)
        per_token_logps = get_per_token_logps(
            model, prompt_completion_ids, full_attention_mask, n_keep)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_logps = get_per_token_logps(
                    self.ref_model, prompt_completion_ids, full_attention_mask, n_keep)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_logps = get_per_token_logps(
                        model, prompt_completion_ids, full_attention_mask, n_keep)
        per_token_kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1

        # GPT-5.5 P1: decode only up to EOS (completion_mask), not raw completion_ids
        valid_lens = completion_mask.sum(dim=1).tolist()
        completions = []
        for row, vl in zip(completion_ids, valid_lens):
            text = self.processing_class.decode(row[:int(vl)], skip_special_tokens=True)
            completions.append(text)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": c}] for c in completions]

        n_total = len(completions)
        rewards_per_func = torch.zeros(n_total, len(self.reward_funcs), device=device)
        prompts_for_reward = []
        for ex in inputs:
            prompts_for_reward.extend([ex["prompt"]] * self.num_generations)
        for i, (rf, _rp) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(rf, PreTrainedModel):
                raise NotImplementedError
            rk = {k: [] for k in inputs[0].keys() if k not in ("prompt", "completion", "prompt_id")}
            for k in rk:
                for ex in inputs:
                    rk[k].extend([ex[k]] * self.num_generations)
            out = rf(prompts=prompts_for_reward, completions=completions, **rk)
            rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)
        rewards = rewards_per_func.sum(dim=1)

        group_rewards = rewards.view(-1, self.num_generations)
        group_mean = group_rewards.mean(dim=1)
        group_pids = [int(x.get("prompt_id", -1)) for x in inputs]

        if self.backbone_mode == "dr_grpo":
            advantages = rewards - group_mean.repeat_interleave(self.num_generations, dim=0)
        elif self.backbone_mode == "spo":
            assert self.credit_store is not None
            baselines = torch.tensor(
                [self.credit_store.get_baseline(pid) for pid in group_pids],
                device=device, dtype=rewards.dtype)
            advantages = rewards - baselines.repeat_interleave(self.num_generations, dim=0)
        elif self.backbone_mode == "tasa":
            from src.aser_trainer_v14 import ASERTrainerV14
            advantages = ASERTrainerV14._compute_tasa_advantages(group_rewards, 0.5, device)
        else:
            raise ValueError(self.backbone_mode)

        # Update prompt credit state
        if self.credit_store is not None:
            step = int(self.state.global_step) if self.state else 0
            with torch.no_grad():
                for pid, mr in zip(group_pids, group_mean.detach().cpu().tolist()):
                    if pid >= 0:
                        self.credit_store.update(pid, float(mr), step)

        # Push successes into trust replay bank
        if self.trust_bank is not None:
            with torch.no_grad():
                step = int(self.state.global_step) if self.state else 0
                reward_list = rewards.detach().cpu().tolist()
                exp_pids, exp_prompts = [], []
                for ex, ptxt in zip(inputs, prompts_text):
                    exp_pids.extend([int(ex.get("prompt_id", -1))] * self.num_generations)
                    exp_prompts.extend([ptxt] * self.num_generations)
                for pid, ptxt, cids, vl, r in zip(
                    exp_pids, exp_prompts, completion_ids, valid_lens, reward_list
                ):
                    if pid < 0 or r < self.success_threshold:
                        continue
                    toks = cids[:int(vl)].detach().cpu().tolist()
                    if toks:
                        self.trust_bank.add_success(
                            pid, ptxt, toks,
                            self.processing_class.decode(toks, skip_special_tokens=True),
                            r, step)

        # PG loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss_pg = (per_token_loss * completion_mask).sum(dim=1).mean()

        pg_tokens = int(completion_mask.sum().item())
        self._pg_token_total += pg_tokens

        # Replay loss (trust-gated)
        loss_rep, lambda_eff = self._compute_trace_replay_loss(model, device)
        loss = loss_pg + loss_rep

        # Metrics
        comp_len = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(comp_len)
        rpf = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, rf in enumerate(self.reward_funcs):
            name = rf.__name__ if hasattr(rf, '__name__') else rf.config._name_or_path.split("/")[-1]
            self._metrics[f"rewards/{name}"].append(rpf[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # TRACE-specific metrics
        step = int(self.state.global_step) if self.state else 0
        frontiers = [self.credit_store.get_frontier(pid) for pid in group_pids if pid >= 0] if self.credit_store else []
        all_fail = int((group_rewards.max(dim=1).values == 0).sum().item())
        all_pass = int((group_rewards.min(dim=1).values > 0).sum().item())

        self._trace_step_stats.append({
            "step": step,
            "trace_mode": self.trace_mode,
            "mean_reward": float(rewards.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "loss_pg": float(loss_pg.detach().item()),
            "loss_rep": float(loss_rep.detach().item()) if isinstance(loss_rep, torch.Tensor) else 0.0,
            "lambda_eff": lambda_eff,
            "bank_size": self.trust_bank.size() if self.trust_bank else 0,
            "bank_prompts": self.trust_bank.n_prompts() if self.trust_bank else 0,
            "mean_frontier": float(np.mean(frontiers)) if frontiers else 0.0,
            "all_fail_groups": all_fail,
            "all_pass_groups": all_pass,
            "replay_token_ratio": self._replay_token_total / max(self._pg_token_total, 1),
        })
        return loss
