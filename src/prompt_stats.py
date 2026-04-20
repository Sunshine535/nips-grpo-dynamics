"""Per-prompt persistent statistics for SPO-style baselines + difficulty tracking."""
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class PromptStat:
    baseline: float = 0.0
    success_ema: float = 0.0
    seen: int = 0


class PromptStatsStore:
    """EMA-smoothed per-prompt reward statistics.

    - `baseline` : persistent SPO baseline b(x), used in
                   advantage = r - b(x) when trainer is in SPO mode.
    - `success_ema` : exponentially smoothed success rate, used by the
                      adaptive-duplication sampler to score hardness.
    """

    def __init__(self, alpha_baseline: float = 0.1, alpha_success: float = 0.1):
        self.alpha_baseline = alpha_baseline
        self.alpha_success = alpha_success
        self.stats: dict = defaultdict(PromptStat)

    def get_baseline(self, prompt_id: int) -> float:
        return self.stats[prompt_id].baseline

    def get_success_ema(self, prompt_id: int) -> float:
        return self.stats[prompt_id].success_ema

    def get_hardness(self, prompt_id: int) -> float:
        """Return 1 - EMA success rate. Returns 1.0 for unseen prompts (max hard)."""
        s = self.stats[prompt_id]
        if s.seen == 0:
            return 1.0
        return 1.0 - s.success_ema

    def update(self, prompt_id: int, mean_reward: float) -> None:
        st = self.stats[prompt_id]
        st.baseline = (1 - self.alpha_baseline) * st.baseline + self.alpha_baseline * mean_reward
        st.success_ema = (1 - self.alpha_success) * st.success_ema + self.alpha_success * mean_reward
        st.seen += 1

    def dump(self) -> dict:
        return {pid: {"baseline": st.baseline, "success_ema": st.success_ema, "seen": st.seen}
                for pid, st in self.stats.items()}
