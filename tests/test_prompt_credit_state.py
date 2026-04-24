"""Tests for PromptCreditState (GPT-5.5 Task 5).

Verifies the Beta-posterior frontier behaviour:
- Unseen prompts have frontier 0.
- Evidence-backed uncertain prompts have HIGH frontier.
- Easy or hopeless prompts (saturated) have LOW frontier.
- Saturation reduces effective trust.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prompt_credit_state import PromptCreditStore


def test_unseen_frontier_is_zero():
    store = PromptCreditStore(n_min=5)
    assert store.get_frontier(0) == 0.0


def test_evidence_backed_uncertain_has_highest_frontier():
    store = PromptCreditStore(n_min=5)
    # 50%-success prompt with 10 observations
    for i in range(10):
        store.update(1, 0.5, step=i)
    f_uncertain = store.get_frontier(1)
    # easy prompt (always 1.0)
    for i in range(10):
        store.update(2, 1.0, step=i)
    f_easy = store.get_frontier(2)
    # hopeless prompt (always 0.0)
    for i in range(10):
        store.update(3, 0.0, step=i)
    f_hopeless = store.get_frontier(3)
    assert f_uncertain > f_easy, \
        f"uncertain frontier {f_uncertain} must exceed easy {f_easy}"
    assert f_uncertain > f_hopeless, \
        f"uncertain frontier {f_uncertain} must exceed hopeless {f_hopeless}"
    assert f_uncertain >= 0.8, f"uncertain should be ~1.0, got {f_uncertain}"


def test_low_evidence_frontier_lower_than_full_evidence():
    store = PromptCreditStore(n_min=10)
    store.update(1, 1.0, step=0)
    store.update(1, 0.0, step=1)
    f_low = store.get_frontier(1)
    for i in range(10):
        store.update(2, 0.5, step=i)
    f_full = store.get_frontier(2)
    assert f_low < f_full, f"low evidence {f_low} must be < full evidence {f_full}"


def test_saturation_bounded():
    store = PromptCreditStore(max_exposure=10)
    for _ in range(5):
        store.record_replay(0)
    assert store.get_saturation(0) == 0.5
    for _ in range(20):
        store.record_replay(0)
    assert store.get_saturation(0) == 1.0


def test_baseline_ema_tracks_mean_reward():
    store = PromptCreditStore(alpha_baseline=0.5)
    for _ in range(5):
        store.update(0, 1.0, step=0)
    assert store.get_baseline(0) > 0.5


def test_dump_contains_required_fields():
    store = PromptCreditStore()
    store.update(0, 0.7, step=5)
    d = store.dump()
    assert 0 in d
    entry = d[0]
    for k in ("alpha", "beta", "p_hat", "frontier",
             "uncertainty", "n_obs", "baseline_ema",
             "replay_exposure", "last_seen_step"):
        assert k in entry, f"missing key {k}"


if __name__ == "__main__":
    test_unseen_frontier_is_zero()
    test_evidence_backed_uncertain_has_highest_frontier()
    test_low_evidence_frontier_lower_than_full_evidence()
    test_saturation_bounded()
    test_baseline_ema_tracks_mean_reward()
    test_dump_contains_required_fields()
    print("ALL PromptCreditState TESTS PASSED")
