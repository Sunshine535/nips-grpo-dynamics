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
            # Probabilistic rounding — unbiased, so small batches still get duplicates.
            # n_dup_expected = batch_size * frac; floor + Bernoulli on fractional part.
            expected = self.batch_size * frac
            base = int(expected)
            residual = expected - base
            n_dup = base + (1 if (residual > 0 and self.rng.random() < residual) else 0)
            n_dup = min(n_dup, self.batch_size)
            # Telemetry: record realised n_dup so downstream analysis can verify it fired.
            self.last_n_dup = n_dup
            batch = [self.rng.randrange(n) for _ in range(self.batch_size)]
            if n_dup > 0:
                hard = np.array([
                    max(self.stats_store.get_hardness(pid), 1e-3) ** self.hardness_temp
                    for pid in self.prompt_ids
                ], dtype=np.float64)
                hard = hard / hard.sum()
                dup_idxs = np.random.choice(np.arange(n), size=n_dup, replace=True, p=hard)
                batch[:n_dup] = dup_idxs.tolist()
            self.step += 1
            yield batch

    def __len__(self) -> int:
        return 10 ** 12  # effectively unbounded; train loop stops on max_steps
