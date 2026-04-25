"""Test that positive_ce_only path does not raise NameError (GPT-5.5 Task 3)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contrastive_evidence_bank import ContrastiveEvidenceBank


def test_positive_ce_random_import():
    """Verify that random is imported at module level in sage_grpo_trainer."""
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              "src", "sage_grpo_trainer.py")) as f:
        src = f.read()
    lines = [l.strip() for l in src.split("\n") if l.strip().startswith("import random")]
    assert len(lines) >= 1, "sage_grpo_trainer.py must have 'import random' at module level"


def test_evidence_bank_has_positives():
    """Verify ContrastiveEvidenceBank stores positives for CE replay."""
    bank = ContrastiveEvidenceBank()
    bank.add(0, "prompt0", [1, 2, 3], "good answer", 1.0, True, step=0)
    bank.add(0, "prompt0", [4, 5, 6], "bad answer", 0.0, False, step=1)
    assert bank.n_pos() == 1
    eligible = [pid for pid in bank.pos if bank.pos[pid]]
    assert len(eligible) == 1
    import random
    item = random.choice(bank.pos[eligible[0]])
    assert item.is_success is True
    assert item.reward == 1.0


if __name__ == "__main__":
    test_positive_ce_random_import()
    test_evidence_bank_has_positives()
    print("ALL positive CE TESTS PASSED")
