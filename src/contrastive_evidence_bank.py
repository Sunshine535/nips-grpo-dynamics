"""Contrastive evidence bank for SAGE-GRPO.

Stores both verified successes AND verified failures per prompt, enabling
prompt-local (positive, negative) pair sampling for contrastive replay.
"""
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


def _token_hash(token_ids) -> str:
    m = hashlib.sha1()
    m.update(str(tuple(int(t) for t in token_ids)).encode("utf-8"))
    return m.hexdigest()


@dataclass
class EvidenceItem:
    prompt_id: int
    prompt: str
    token_ids: list
    text: str
    reward: float
    is_success: bool
    source_step: int
    length: int
    hash: str
    replay_count: int = 0


class ContrastiveEvidenceBank:
    def __init__(self, max_pos_per_prompt: int = 2, max_neg_per_prompt: int = 2,
                 age_tau: float = 200.0):
        self.max_pos = max_pos_per_prompt
        self.max_neg = max_neg_per_prompt
        self.age_tau = age_tau
        self.pos: dict = defaultdict(list)
        self.neg: dict = defaultdict(list)
        self.hashes: dict = defaultdict(set)

    def add(self, prompt_id: int, prompt_text: str, token_ids,
            text: str, reward: float, is_success: bool, step: int) -> bool:
        h = _token_hash(token_ids)
        if h in self.hashes[prompt_id]:
            return False
        self.hashes[prompt_id].add(h)
        item = EvidenceItem(
            prompt_id=prompt_id, prompt=prompt_text,
            token_ids=[int(t) for t in token_ids], text=text,
            reward=reward, is_success=is_success,
            source_step=step, length=len(token_ids), hash=h,
        )
        bank = self.pos if is_success else self.neg
        cap = self.max_pos if is_success else self.max_neg
        bank[prompt_id].append(item)
        bank[prompt_id] = sorted(
            bank[prompt_id], key=lambda x: x.source_step, reverse=True)[:cap]
        return True

    def sample_pairs(self, n: int, current_step: int,
                     credit_store=None) -> list:
        """Sample prompt-local (positive, negative) pairs.
        Only samples from prompts that have BOTH a positive and a negative."""
        eligible = [pid for pid in self.pos if pid in self.neg
                    and self.pos[pid] and self.neg[pid]]
        if not eligible:
            return []

        pairs = []
        for _ in range(n):
            pid = random.choice(eligible)
            pos_item = random.choice(self.pos[pid])
            neg_item = random.choice(self.neg[pid])
            pos_item.replay_count += 1
            neg_item.replay_count += 1

            frontier = 1.0
            if credit_store is not None:
                frontier = credit_store.get_frontier(pid)

            age_pos = max(0, current_step - pos_item.source_step)
            age_neg = max(0, current_step - neg_item.source_step)
            age_decay = math.exp(-(age_pos + age_neg) / (2 * self.age_tau))

            pairs.append({
                "prompt_id": pid,
                "prompt": pos_item.prompt,
                "pos_token_ids": pos_item.token_ids,
                "neg_token_ids": neg_item.token_ids,
                "pos_reward": pos_item.reward,
                "neg_reward": neg_item.reward,
                "reward_gap": pos_item.reward - neg_item.reward,
                "frontier": frontier,
                "age_decay": age_decay,
            })
        return pairs

    def n_pos(self) -> int:
        return sum(len(v) for v in self.pos.values())

    def n_neg(self) -> int:
        return sum(len(v) for v in self.neg.values())

    def n_prompts_with_pairs(self) -> int:
        return sum(1 for pid in self.pos if pid in self.neg
                   and self.pos[pid] and self.neg[pid])

    def summary(self) -> dict:
        return {
            "n_pos": self.n_pos(),
            "n_neg": self.n_neg(),
            "n_prompts_with_pairs": self.n_prompts_with_pairs(),
            "n_prompts_pos_only": sum(1 for pid in self.pos
                                      if pid not in self.neg or not self.neg[pid]),
            "n_prompts_neg_only": sum(1 for pid in self.neg
                                      if pid not in self.pos or not self.pos[pid]),
        }
