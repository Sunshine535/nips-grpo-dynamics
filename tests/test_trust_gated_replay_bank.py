"""Tests for TrustGatedReplayBank (GPT-5.5 Task 6).

Verifies:
- dedup by token hash
- frontier-weighted sampling prefers high-frontier prompts
- age decay reduces trust for stale items
- saturation penalty (via credit_store) reduces trust
- sampling increments replay_count
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prompt_credit_state import PromptCreditStore
from src.trust_gated_replay_bank import TrustGatedReplayBank


def test_dedup_by_token_hash():
    bank = TrustGatedReplayBank(max_per_prompt=5)
    added1 = bank.add_success(0, "p", [1, 2, 3], "ans", 1.0, step=0)
    added2 = bank.add_success(0, "p", [1, 2, 3], "ans", 1.0, step=1)
    added3 = bank.add_success(0, "p", [4, 5, 6], "ans2", 1.0, step=2)
    assert added1 is True
    assert added2 is False, "duplicate tokens should not be added again"
    assert added3 is True


def test_max_per_prompt_retains_most_recent():
    bank = TrustGatedReplayBank(max_per_prompt=2)
    for i in range(5):
        bank.add_success(0, "p", list(range(i, i+3)), f"a{i}", 1.0, step=i)
    assert len(bank.bank[0]) == 2, "bank must enforce max_per_prompt"
    source_steps = sorted(item.source_step for item in bank.bank[0])
    assert source_steps == [3, 4], f"should retain most recent, got {source_steps}"


def test_frontier_prefers_uncertain_prompt():
    credit = PromptCreditStore(n_min=5)
    bank = TrustGatedReplayBank(max_per_prompt=2, age_tau=100.0)
    for i in range(10):
        credit.update(0, 0.5, step=i)
        credit.update(1, 1.0, step=i)
    bank.add_success(0, "p0", [1, 2, 3], "a0", 1.0, step=10)
    bank.add_success(1, "p1", [4, 5, 6], "a1", 1.0, step=10)
    t_uncertain = bank.compute_item_trust(
        bank.bank[0][0], 15,
        credit.get_frontier(0), credit.get_saturation(0))
    t_easy = bank.compute_item_trust(
        bank.bank[1][0], 15,
        credit.get_frontier(1), credit.get_saturation(1))
    assert t_uncertain > t_easy, \
        f"uncertain {t_uncertain} must exceed easy {t_easy}"


def test_age_decay_reduces_trust():
    bank = TrustGatedReplayBank(age_tau=50.0)
    bank.add_success(0, "p", [1, 2], "old", 1.0, step=0)
    bank.add_success(1, "p", [3, 4], "new", 1.0, step=90)
    t_old = bank.compute_item_trust(bank.bank[0][0], 100, 1.0, 0.0)
    t_new = bank.compute_item_trust(bank.bank[1][0], 100, 1.0, 0.0)
    assert t_new > t_old


def test_saturation_reduces_trust():
    bank = TrustGatedReplayBank()
    bank.add_success(0, "p", [1, 2, 3], "a", 1.0, step=5)
    t_fresh = bank.compute_item_trust(bank.bank[0][0], 10, 1.0, 0.0)
    t_saturated = bank.compute_item_trust(bank.bank[0][0], 10, 1.0, 0.8)
    assert t_saturated < t_fresh


def test_sampling_updates_replay_count():
    credit = PromptCreditStore(n_min=5)
    bank = TrustGatedReplayBank(age_tau=100.0)
    for i in range(10):
        credit.update(0, 0.5, step=i)
    bank.add_success(0, "p", [1, 2, 3], "a", 1.0, step=10)
    count_before = bank.bank[0][0].replay_count
    items = bank.weighted_sample(1, current_step=15, credit_store=credit)
    assert len(items) == 1
    assert bank.bank[0][0].replay_count == count_before + 1


def test_empty_bank_returns_empty_list():
    bank = TrustGatedReplayBank()
    items = bank.weighted_sample(5, current_step=10)
    assert items == []


if __name__ == "__main__":
    test_dedup_by_token_hash()
    test_max_per_prompt_retains_most_recent()
    test_frontier_prefers_uncertain_prompt()
    test_age_decay_reduces_trust()
    test_saturation_reduces_trust()
    test_sampling_updates_replay_count()
    test_empty_bank_returns_empty_list()
    print("ALL TrustGatedReplayBank TESTS PASSED")
