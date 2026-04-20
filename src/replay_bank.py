"""Verified replay bank — small per-prompt store of hash-deduped success trajectories."""
from collections import defaultdict
import hashlib
import random


def _token_hash(token_ids) -> str:
    m = hashlib.sha1()
    # tuple -> bytes via repr; avoids the bytes() signed-int issue for large vocab IDs
    m.update(str(tuple(int(t) for t in token_ids)).encode("utf-8"))
    return m.hexdigest()


class VerifiedReplayBank:
    def __init__(self, max_per_prompt: int = 2):
        self.max_per_prompt = max_per_prompt
        self.bank: dict = defaultdict(list)
        self.hashes: dict = defaultdict(set)

    def add_success(self, prompt_id: int, prompt_text: str,
                    token_ids, text: str, step: int) -> bool:
        h = _token_hash(token_ids)
        if h in self.hashes[prompt_id]:
            return False
        self.hashes[prompt_id].add(h)
        self.bank[prompt_id].append({
            "prompt_id": int(prompt_id),
            "prompt": prompt_text,
            "token_ids": [int(t) for t in token_ids],
            "text": text,
            "step": int(step),
        })
        # Keep only the latest max_per_prompt (by step)
        self.bank[prompt_id] = sorted(
            self.bank[prompt_id], key=lambda x: x["step"], reverse=True
        )[: self.max_per_prompt]
        return True

    def sample(self, n: int):
        all_items = [x for xs in self.bank.values() for x in xs]
        if not all_items:
            return []
        n = min(n, len(all_items))
        return random.sample(all_items, n)

    def size(self) -> int:
        return sum(len(x) for x in self.bank.values())

    def n_prompts(self) -> int:
        return len(self.bank)
