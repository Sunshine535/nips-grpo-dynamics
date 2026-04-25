"""Tests for ContrastiveEvidenceBank (SAGE-GRPO Task 2)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.contrastive_evidence_bank import ContrastiveEvidenceBank


def test_no_pairs_without_both():
    bank = ContrastiveEvidenceBank()
    bank.add(0, "p0", [1,2,3], "good", 1.0, True, 0)
    pairs = bank.sample_pairs(5, current_step=10)
    assert pairs == [], "No pairs when only positives exist"


def test_pairs_after_both():
    bank = ContrastiveEvidenceBank()
    bank.add(0, "p0", [1,2,3], "good", 1.0, True, 0)
    bank.add(0, "p0", [4,5,6], "bad", 0.0, False, 1)
    pairs = bank.sample_pairs(1, current_step=5)
    assert len(pairs) == 1
    assert pairs[0]["pos_reward"] > pairs[0]["neg_reward"]
    assert pairs[0]["prompt_id"] == 0


def test_dedup():
    bank = ContrastiveEvidenceBank()
    assert bank.add(0, "p", [1,2], "a", 1.0, True, 0) == True
    assert bank.add(0, "p", [1,2], "a", 1.0, True, 1) == False


def test_max_per_prompt():
    bank = ContrastiveEvidenceBank(max_pos_per_prompt=2, max_neg_per_prompt=2)
    for i in range(10):
        bank.add(0, "p", list(range(i, i+3)), f"s{i}", 1.0, True, i)
    assert len(bank.pos[0]) == 2


def test_cross_prompt_isolation():
    bank = ContrastiveEvidenceBank()
    bank.add(0, "p0", [1,2], "good0", 1.0, True, 0)
    bank.add(0, "p0", [3,4], "bad0", 0.0, False, 0)
    bank.add(1, "p1", [5,6], "good1", 1.0, True, 0)
    # prompt 1 has no negative — should never appear in pairs
    for _ in range(20):
        pairs = bank.sample_pairs(1, current_step=5)
        assert all(p["prompt_id"] == 0 for p in pairs)


def test_summary():
    bank = ContrastiveEvidenceBank()
    bank.add(0, "p", [1,2], "g", 1.0, True, 0)
    bank.add(0, "p", [3,4], "b", 0.0, False, 0)
    bank.add(1, "p1", [5,6], "g1", 1.0, True, 0)
    s = bank.summary()
    assert s["n_pos"] == 2
    assert s["n_neg"] == 1
    assert s["n_prompts_with_pairs"] == 1
    assert s["n_prompts_pos_only"] == 1


def test_reward_gap_positive():
    bank = ContrastiveEvidenceBank()
    bank.add(0, "p", [1,2], "g", 1.0, True, 0)
    bank.add(0, "p", [3,4], "b", 0.0, False, 0)
    pairs = bank.sample_pairs(1, current_step=5)
    assert pairs[0]["reward_gap"] == 1.0


if __name__ == "__main__":
    test_no_pairs_without_both()
    test_pairs_after_both()
    test_dedup()
    test_max_per_prompt()
    test_cross_prompt_isolation()
    test_summary()
    test_reward_gap_positive()
    print("ALL ContrastiveEvidenceBank TESTS PASSED")
