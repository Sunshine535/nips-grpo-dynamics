import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import TrainerCallback


@dataclass
class RhoGRPOConfig:
    rho: float = 1.0
    clip_range: float = 0.2
    kl_coef: float = 0.05
    group_size: int = 4
    degenerate_floor: float = 0.0


def compute_group_statistics(rewards: torch.Tensor, group_size: int):
    B = rewards.shape[0]
    n_groups = B // group_size
    rewards_grouped = rewards[:n_groups * group_size].view(n_groups, group_size)

    m = rewards_grouped.sum(dim=1)
    p_hat = m / group_size

    sigma = torch.sqrt(m * (group_size - m)) / group_size
    delta = 1.0 / group_size
    sigma_safe = torch.clamp(sigma, min=delta)

    degenerate_mask = (m == 0) | (m == group_size)
    p_0 = degenerate_mask.float().mean()
    gsr = p_0

    return {
        "m": m,
        "p_hat": p_hat,
        "sigma": sigma_safe,
        "degenerate_mask": degenerate_mask,
        "p_0": p_0.item(),
        "gsr": gsr.item(),
        "n_groups": n_groups,
    }


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    rho: float = 1.0,
    degenerate_floor: float = 0.0,
):
    B = rewards.shape[0]
    n_groups = B // group_size
    rewards_grouped = rewards[:n_groups * group_size].view(n_groups, group_size)

    m = rewards_grouped.sum(dim=1, keepdim=True)
    group_mean = m / group_size
    sigma = torch.sqrt(m * (group_size - m)) / group_size
    delta = 1.0 / group_size
    sigma = torch.clamp(sigma, min=delta)

    advantages = (rewards_grouped - group_mean) / sigma

    positive_mask = (rewards_grouped > 0).float()
    negative_mask = (rewards_grouped <= 0).float()

    rho_weighted = advantages * (positive_mask * rho + negative_mask * 1.0)

    degenerate = (m.squeeze(-1) == 0) | (m.squeeze(-1) == group_size)
    rho_weighted[degenerate] = degenerate_floor

    return rho_weighted.view(-1), advantages.view(-1)


def compute_rho_grpo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rho_advantages: torch.Tensor,
    mask: torch.Tensor,
    config: RhoGRPOConfig,
) -> dict:
    ratio = torch.exp(logprobs - ref_logprobs)
    ratio_clamped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)

    adv = rho_advantages.unsqueeze(-1)
    pg1 = ratio * adv
    pg2 = ratio_clamped * adv
    pg_loss_token = -torch.min(pg1, pg2)
    pg_loss = (pg_loss_token * mask).sum() / mask.sum().clamp(min=1)

    kl = logprobs - ref_logprobs
    kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

    loss = pg_loss + config.kl_coef * kl_loss

    positive_mask = (rho_advantages > 0).float()
    negative_mask = (rho_advantages < 0).float()
    n_pos = positive_mask.sum().item()
    n_neg = negative_mask.sum().item()
    n_zero = (rho_advantages == 0).sum().item()

    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_degenerate": n_zero,
        "mean_pos_advantage": (rho_advantages * positive_mask).sum().item() / max(n_pos, 1),
        "mean_neg_advantage": (rho_advantages * negative_mask).sum().item() / max(n_neg, 1),
    }


def build_gsm8k_binary_reward_function():
    import re
    # Strip Qwen3/3.5 thinking blocks: <think>...</think>
    _think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    # Match "#### <number>" with optional text between #### and the number
    _pattern = re.compile(r"####\s*(?:.*?)(-?\d[\d,]*\.?\d*)")
    # Fallback: last number in the text (after stripping think blocks)
    _fallback = re.compile(r"(-?\d[\d,]*\.?\d*)")

    def reward_fn(completions, answer, **kwargs):
        rewards = []
        for completion, gold in zip(completions, answer):
            if isinstance(completion, list):
                text = completion[0].get("content", "") if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", str(completion))
            else:
                text = str(completion)

            # Strip thinking blocks for answer extraction
            text_clean = _think_pattern.sub("", text).strip()

            # Try #### pattern first (on both original and cleaned text)
            match = _pattern.search(text) or _pattern.search(text_clean)
            if match:
                pred = match.group(1).replace(",", "")
            else:
                # Fallback: last number in cleaned text (skip think block numbers)
                nums = _fallback.findall(text_clean)
                pred = nums[-1].replace(",", "") if nums else ""

            gold_clean = str(gold).strip()
            rewards.append(1.0 if pred == gold_clean else 0.0)
        return rewards

    return reward_fn


build_rho_reward_function = build_gsm8k_binary_reward_function


class RhoGRPOCallback(TrainerCallback):
    def __init__(self, rho: float, group_size: int = 4):
        self.rho = rho
        self.group_size = group_size
        self.step_metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs["rho_grpo/rho"] = self.rho
        logs["rho_grpo/group_size"] = self.group_size
        self.step_metrics.append({
            "step": state.global_step,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        })
