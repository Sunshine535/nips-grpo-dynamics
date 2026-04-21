"""Difficulty-adaptive duplication batch sampler.

Given a fixed batch size B, each batch contains:
  - `(1-dup_frac)*B` uniformly sampled prompts
  - `dup_frac*B` prompts sampled proportionally to (hardness + eps)^tau

Because `num_generations` G stays fixed, the same prompt appearing twice
in a batch effectively gets 2*G rollouts this step — a token-budget-neutral
way to give more exploration to hard prompts without touching TRL's group-
size assumptions.
"""
import random

import numpy as np
from torch.utils.data import Sampler


class AdaptiveDupBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, stats_store,
                 dup_frac: float = 0.25, hardness_temp: float = 2.0,
                 warmup_steps: int = 100, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.stats_store = stats_store
        self.dup_frac = float(dup_frac)
        self.hardness_temp = float(hardness_temp)
        self.warmup_steps = int(warmup_steps)
        self.rng = random.Random(seed)
        self.step = 0
        self.prompt_ids = [int(dataset[i]["prompt_id"]) for i in range(len(dataset))]

    def _current_dup_frac(self) -> float:
        if self.step < self.warmup_steps:
            return 0.0
        # Linear warmup over 2x warmup_steps
        progress = min(1.0, (self.step - self.warmup_steps) / max(1, self.warmup_steps))
        return self.dup_frac * progress

    def __iter__(self):
        n = len(self.dataset)
        while True:
            frac = self._current_dup_frac()
            # Semantic fix (Round 2 review): the earlier code "replaced slot[i] with a
            # hardness-weighted sample", which is biased sampling, NOT duplication.
            # For batch_size=2, even with n_dup=1 chosen, the resulting batch was two
            # distinct prompts (1 uniform + 1 hardness-biased), so batch_n_dup=0 every
            # step. The paper's claim is that hard prompts get G→2G effective rollouts,
            # which requires the SAME prompt_id to appear MULTIPLE times in a batch.
            #
            # New semantics: with probability `frac` per step, pick one hardness-weighted
            # prompt and fill ALL `batch_size` slots with it — that is a G-wide replication
            # of one hard prompt. Otherwise, do `batch_size` uniform samples.
            # For batch_size=2, this gives batch_n_dup=1 on `frac` fraction of steps.
            do_dup = frac > 0 and self.rng.random() < frac
            if do_dup:
                hard = np.array([
                    max(self.stats_store.get_hardness(pid), 1e-3) ** self.hardness_temp
                    for pid in self.prompt_ids
                ], dtype=np.float64)
                hard = hard / hard.sum()
                hard_idx = int(np.random.choice(np.arange(n), p=hard))
                batch = [hard_idx] * self.batch_size
                self.last_n_dup = self.batch_size - 1
            else:
                batch = [self.rng.randrange(n) for _ in range(self.batch_size)]
                self.last_n_dup = 0
            self.step += 1
            yield batch

    def __len__(self) -> int:
        return 10 ** 12  # effectively unbounded; train loop stops on max_steps
