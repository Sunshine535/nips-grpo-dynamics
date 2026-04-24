"""Trust-gated replay bank for TRACE-GRPO.

Replaces uniform sampling with weighted sampling based on:
- prompt frontier score (from PromptCreditStore)
- item trust: age decay, diversity, length guard
- replay exposure penalty
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
class ReplayItem:
    prompt_id: int
    prompt: str
    token_ids: list
    text: str
    reward: float
    source_step: int
    length: int
    hash: str
    replay_count: int = 0
    last_replayed_step: int = -1


class TrustGatedReplayBank:
    def __init__(self, max_per_prompt: int = 2, age_tau: float = 200.0,
                 max_length: int = 512, diversity_bonus: float = 0.5):
        self.max_per_prompt = max_per_prompt
        self.age_tau = age_tau
        self.max_length = max_length
        self.diversity_bonus = diversity_bonus
        self.bank: dict = defaultdict(list)
        self.hashes: dict = defaultdict(set)

    def add_success(self, prompt_id: int, prompt_text: str,
                    token_ids, text: str, reward: float, step: int) -> bool:
        h = _token_hash(token_ids)
        if h in self.hashes[prompt_id]:
            return False
        self.hashes[prompt_id].add(h)
        length = len(token_ids)
        item = ReplayItem(
            prompt_id=prompt_id, prompt=prompt_text,
            token_ids=[int(t) for t in token_ids], text=text,
            reward=reward, source_step=step, length=length, hash=h,
        )
        self.bank[prompt_id].append(item)
        self.bank[prompt_id] = sorted(
            self.bank[prompt_id], key=lambda x: x.source_step, reverse=True
        )[:self.max_per_prompt]
        return True

    def compute_item_trust(self, item: ReplayItem, current_step: int,
                           frontier: float, saturation: float) -> float:
        age = max(0, current_step - item.source_step)
        age_decay = math.exp(-age / self.age_tau)
        length_guard = 1.0 if item.length <= self.max_length else 0.5
        n_unique = len(self.hashes[item.prompt_id])
        diversity = min(1.0, n_unique * self.diversity_bonus)
        sat_penalty = max(0.0, 1.0 - saturation)
        trust = age_decay * length_guard * diversity * sat_penalty
        return frontier * trust

    def weighted_sample(self, n: int, current_step: int,
                        credit_store=None) -> list:
        all_items = [item for items in self.bank.values() for item in items]
        if not all_items:
            return []

        weights = []
        for item in all_items:
            frontier = 1.0
            saturation = 0.0
            if credit_store is not None:
                frontier = credit_store.get_frontier(item.prompt_id)
                saturation = credit_store.get_saturation(item.prompt_id)
            w = self.compute_item_trust(item, current_step, frontier, saturation)
            weights.append(max(w, 1e-8))

        total = sum(weights)
        probs = [w / total for w in weights]

        n = min(n, len(all_items))
        chosen_indices = []
        for _ in range(n):
            r = random.random()
            cumsum = 0.0
            for j, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    chosen_indices.append(j)
                    break
            else:
                chosen_indices.append(len(all_items) - 1)

        result = []
        for idx in chosen_indices:
            item = all_items[idx]
            item.replay_count += 1
            item.last_replayed_step = current_step
            if credit_store is not None:
                credit_store.record_replay(item.prompt_id)
            result.append({
                "prompt_id": item.prompt_id,
                "prompt": item.prompt,
                "token_ids": item.token_ids,
                "text": item.text,
                "trust_weight": weights[idx],
            })
        return result

    def size(self) -> int:
        return sum(len(items) for items in self.bank.values())

    def n_prompts(self) -> int:
        return len(self.bank)
