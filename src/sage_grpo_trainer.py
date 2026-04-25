"""SAGE-GRPO Trainer — Signed Advantage-Guided Evidence Replay.

Combines threshold-signed on-policy credit (TASA) with prompt-local
contrastive evidence replay over verified successes and failures.

Modes (for A/B/C/D ablation):
  tasa_only         : signed PG only, no replay (B)
  positive_ce_only  : signed PG + positive-only CE replay (D)
  pair_only         : signed PG + pairwise contrastive replay only
  full              : signed PG + pairwise contrastive + optional positive CE (C)
"""
import logging
import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import is_conversational
from trl.data_utils import maybe_apply_chat_template
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class SageGRPOTrainer(GRPOTrainer):

    def __init__(
        self, *args,
        evidence_bank=None,
        prompt_credit_store=None,
        lambda_pair: float = 0.05,
        lambda_pos: float = 0.0,
        pair_batch_size: int = 2,
        replay_warmup_steps: int = 50,
        success_threshold: float = 0.5,
        failure_threshold: float = 0.0,
        tasa_c: float = 0.5,
        ref_coef: float = 1.0,
        sage_mode: str = "full",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        valid = ("tasa_only", "positive_ce_only", "pair_only", "full")
        assert sage_mode in valid, sage_mode
        self.evidence_bank = evidence_bank
        self.credit_store = prompt_credit_store
        self.lambda_pair = float(lambda_pair)
        self.lambda_pos = float(lambda_pos)
        self.pair_batch_size = int(pair_batch_size)
        self.replay_warmup_steps = int(replay_warmup_steps)
        self.success_threshold = float(success_threshold)
        self.failure_threshold = float(failure_threshold)
        self.tasa_c = float(tasa_c)
        self.ref_coef = float(ref_coef)
        self.sage_mode = sage_mode
        self._sage_step_stats: list = []

    @staticmethod
    def _compute_tasa_advantages(group_rewards, c, device):
        from src.aser_trainer_v14 import ASERTrainerV14
        return ASERTrainerV14._compute_tasa_advantages(group_rewards, c, device)

    def _get_per_token_logps(self, mdl, input_ids, attention_mask, n_keep):
        logits = mdl(input_ids, attention_mask=attention_mask,
                     num_logits_to_keep=n_keep + 1).logits
        logits = logits[:, :-1, :]
        target = input_ids[:, -n_keep:]
        logits = logits[:, -n_keep:]
        per_token = []
        for lr, tr in zip(logits, target):
            lp = lr.log_softmax(dim=-1)
            per_token.append(torch.gather(lp, 1, tr.unsqueeze(1)).squeeze(1))
        return torch.stack(per_token)

    def _compute_pair_loss(self, model, device) -> dict:
        """Pairwise contrastive replay: log sigma(logp_pos - logp_neg - ref_delta)."""
        info = {"loss_pair": 0.0, "n_pairs": 0, "pair_reward_gap_mean": 0.0,
                "pair_frontier_mean": 0.0, "pair_age_mean": 0.0, "mechanism_active": False}
        if self.evidence_bank is None or self.sage_mode == "tasa_only":
            return info
        step = int(self.state.global_step) if self.state else 0
        if step < self.replay_warmup_steps:
            return info

        pairs = self.evidence_bank.sample_pairs(
            self.pair_batch_size, step, credit_store=self.credit_store)
        if not pairs:
            return info

        tok = self.processing_class
        pad_id = tok.pad_token_id or tok.eos_token_id

        def _build_ids_and_comp_mask(prompt_text, comp_ids):
            p_enc = tok(prompt_text, add_special_tokens=False)["input_ids"]
            c = [int(t) for t in comp_ids]
            full = p_enc + c
            comp_mask = [0] * len(p_enc) + [1] * len(c)
            return full, comp_mask

        all_pos_ids, all_neg_ids = [], []
        all_pos_cmask, all_neg_cmask = [], []
        for pair in pairs:
            pi, pm = _build_ids_and_comp_mask(pair["prompt"], pair["pos_token_ids"])
            ni, nm = _build_ids_and_comp_mask(pair["prompt"], pair["neg_token_ids"])
            all_pos_ids.append(pi); all_pos_cmask.append(pm)
            all_neg_ids.append(ni); all_neg_cmask.append(nm)

        def _pad_and_mask(id_lists, cmask_lists):
            max_len = max(len(ids) for ids in id_lists)
            padded, attn_masks, comp_masks = [], [], []
            for ids, cm in zip(id_lists, cmask_lists):
                pad_len = max_len - len(ids)
                padded.append(ids + [pad_id] * pad_len)
                attn_masks.append([1] * len(ids) + [0] * pad_len)
                comp_masks.append(cm + [0] * pad_len)
            return (torch.tensor(padded, device=device, dtype=torch.long),
                    torch.tensor(attn_masks, device=device, dtype=torch.long),
                    torch.tensor(comp_masks, device=device, dtype=torch.long))

        pos_ids, pos_attn, pos_cmask = _pad_and_mask(all_pos_ids, all_pos_cmask)
        neg_ids, neg_attn, neg_cmask = _pad_and_mask(all_neg_ids, all_neg_cmask)

        def _comp_logp(mdl, ids, attn_mask, comp_mask):
            """Completion-only sequence log-prob (GPT-5.5 Task 3)."""
            logits = mdl(ids, attention_mask=attn_mask).logits[:, :-1]
            targets = ids[:, 1:]
            lp = logits.log_softmax(dim=-1)
            token_lp = torch.gather(lp, 2, targets.unsqueeze(2)).squeeze(2)
            return (token_lp * comp_mask[:, 1:]).sum(dim=1)

        logp_pos = _comp_logp(model, pos_ids, pos_attn, pos_cmask)
        logp_neg = _comp_logp(model, neg_ids, neg_attn, neg_cmask)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_logp_pos = _comp_logp(self.ref_model, pos_ids, pos_attn, pos_cmask)
                ref_logp_neg = _comp_logp(self.ref_model, neg_ids, neg_attn, neg_cmask)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_logp_pos = _comp_logp(model, pos_ids, pos_attn, pos_cmask)
                    ref_logp_neg = _comp_logp(model, neg_ids, neg_attn, neg_cmask)

        margin = (logp_pos - logp_neg) - self.ref_coef * (ref_logp_pos - ref_logp_neg)

        # GPT-5.5 Task 3: weight pairs by reward_gap * frontier * age_decay
        pair_weights = torch.tensor(
            [max(0.05, min(1.0, p["reward_gap"] * p["frontier"] * p["age_decay"]))
             for p in pairs], device=device, dtype=margin.dtype)
        loss_pair = (pair_weights * -F.logsigmoid(margin)).sum() / pair_weights.sum().clamp(min=1e-8)

        info["loss_pair"] = float(loss_pair.detach().item())
        info["n_pairs"] = len(pairs)
        info["pair_reward_gap_mean"] = float(np.mean([p["reward_gap"] for p in pairs]))
        info["pair_frontier_mean"] = float(np.mean([p["frontier"] for p in pairs]))
        info["pair_age_mean"] = float(np.mean([p["age_decay"] for p in pairs]))
        info["pair_weight_mean"] = float(pair_weights.mean().item())
        info["pair_weight_min"] = float(pair_weights.min().item())
        info["pair_weight_max"] = float(pair_weights.max().item())
        info["completion_only_pair_logp"] = True
        info["mechanism_active"] = True
        return info, loss_pair

    def _compute_positive_ce_loss(self, model, device) -> tuple:
        """Optional positive-only CE replay (variant D / full with lambda_pos > 0)."""
        if self.evidence_bank is None or self.lambda_pos <= 0:
            return torch.tensor(0.0, device=device), 0
        step = int(self.state.global_step) if self.state else 0
        if step < self.replay_warmup_steps:
            return torch.tensor(0.0, device=device), 0

        eligible = [pid for pid in self.evidence_bank.pos if self.evidence_bank.pos[pid]]
        if not eligible:
            return torch.tensor(0.0, device=device), 0

        items = []
        for _ in range(self.pair_batch_size):
            pid = random.choice(eligible)
            items.append(random.choice(self.evidence_bank.pos[pid]))

        tok = self.processing_class
        pad_id = tok.pad_token_id or tok.eos_token_id
        input_rows, label_rows = [], []
        for item in items:
            p_enc = tok(item.prompt, add_special_tokens=False)["input_ids"]
            c = item.token_ids
            input_rows.append(p_enc + c)
            label_rows.append([-100] * len(p_enc) + c)

        max_len = max(len(r) for r in input_rows)
        for i in range(len(input_rows)):
            pad_len = max_len - len(input_rows[i])
            input_rows[i] = input_rows[i] + [pad_id] * pad_len
            label_rows[i] = label_rows[i] + [-100] * pad_len

        ids = torch.tensor(input_rows, device=device, dtype=torch.long)
        labels = torch.tensor(label_rows, device=device, dtype=torch.long)
        attn = (ids != pad_id).long()
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        return out.loss * self.lambda_pos, len(items)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("SageGRPOTrainer does not support return_outputs")
        device = self.accelerator.device
        import random

        prompts_text = [
            maybe_apply_chat_template(ex, self.processing_class)["prompt"]
            for ex in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False)
        prompt_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in prompt_inputs.items()}
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

        prompt_attn = prompt_inputs["attention_mask"].to(dtype=completion_mask.dtype)
        if prompt_attn.size(0) != prompt_completion_ids.size(0):
            factor = prompt_completion_ids.size(0) // prompt_attn.size(0)
            prompt_attn = prompt_attn.repeat_interleave(factor, dim=0)
        full_attn = torch.cat([prompt_attn, completion_mask], dim=1)

        n_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, full_attn, n_keep)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, full_attn, n_keep)
            else:
                with unwrapped.disable_adapter():
                    ref_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, full_attn, n_keep)
        per_token_kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1

        valid_lens = completion_mask.sum(dim=1).tolist()
        completions = [self.processing_class.decode(row[:int(vl)], skip_special_tokens=True)
                       for row, vl in zip(completion_ids, valid_lens)]
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

        advantages = self._compute_tasa_advantages(group_rewards, self.tasa_c, device)

        # Update credit state
        if self.credit_store is not None:
            step = int(self.state.global_step) if self.state else 0
            with torch.no_grad():
                for pid, mr in zip(group_pids, group_mean.detach().cpu().tolist()):
                    if pid >= 0:
                        self.credit_store.update(pid, float(mr), step)

        # Push to evidence bank (both successes AND failures)
        if self.evidence_bank is not None:
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
                    if pid < 0:
                        continue
                    toks = cids[:int(vl)].detach().cpu().tolist()
                    if not toks:
                        continue
                    is_success = r >= self.success_threshold
                    is_failure = r <= self.failure_threshold
                    if is_success or is_failure:
                        self.evidence_bank.add(
                            pid, ptxt, toks,
                            self.processing_class.decode(toks, skip_special_tokens=True),
                            r, is_success, step)

        # PG loss (TASA signed advantage)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss_pg = (per_token_loss * completion_mask).sum(dim=1).mean()

        # Pair loss
        pair_info = {"loss_pair": 0.0, "n_pairs": 0, "mechanism_active": False}
        loss_pair = torch.tensor(0.0, device=device)
        if self.sage_mode in ("pair_only", "full"):
            result = self._compute_pair_loss(model, device)
            if isinstance(result, tuple):
                pair_info, loss_pair = result

        # Positive CE loss
        loss_pos_ce = torch.tensor(0.0, device=device)
        n_pos_ce = 0
        if self.sage_mode in ("positive_ce_only", "full") and self.lambda_pos > 0:
            loss_pos_ce, n_pos_ce = self._compute_positive_ce_loss(model, device)

        loss = loss_pg + self.lambda_pair * loss_pair + loss_pos_ce

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

        step = int(self.state.global_step) if self.state else 0
        all_fail = int((group_rewards.max(dim=1).values == 0).sum().item())
        all_pass = int((group_rewards.min(dim=1).values > 0).sum().item())
        bank_summary = self.evidence_bank.summary() if self.evidence_bank else {}

        self._sage_step_stats.append({
            "step": step,
            "sage_mode": self.sage_mode,
            "mechanism_active": pair_info.get("mechanism_active", False),
            "mean_reward": float(rewards.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "loss_pg": float(loss_pg.detach().item()),
            "loss_pair": pair_info.get("loss_pair", 0.0),
            "loss_pos_ce": float(loss_pos_ce.detach().item()) if isinstance(loss_pos_ce, torch.Tensor) else 0.0,
            "lambda_pair_eff": self.lambda_pair if pair_info.get("mechanism_active") else 0.0,
            "lambda_pos_eff": self.lambda_pos if n_pos_ce > 0 else 0.0,
            "n_pairs_sampled": pair_info.get("n_pairs", 0),
            "bank_pos_total": bank_summary.get("n_pos", 0),
            "bank_neg_total": bank_summary.get("n_neg", 0),
            "bank_prompt_count": bank_summary.get("n_prompts_with_pairs", 0),
            "pair_reward_gap_mean": pair_info.get("pair_reward_gap_mean", 0.0),
            "pair_frontier_mean": pair_info.get("pair_frontier_mean", 0.0),
            "pair_age_mean": pair_info.get("pair_age_mean", 0.0),
            "all_fail_groups": all_fail,
            "all_pass_groups": all_pass,
            "completion_length": comp_len,
        })
        return loss
