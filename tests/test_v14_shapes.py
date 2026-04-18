"""
Shape-logic smoke test for RhoGRPOTrainerV14.

This test exercises the dimensional bug fix flagged by Round 1 review:
rewards_per_func must have shape (B*G, num_reward_funcs), and
_apply_rho_weighting must accept advantages of shape (B*G,) alongside
completion_ids of shape (B*G, max_len).

Runs without requiring a real model or TRL trainer setup. Instead, we
bind _apply_rho_weighting to a stand-in object that carries only the
attributes the method reads.

Execute locally:
    python -m pytest tests/test_v14_shapes.py -v
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_trainer_class():
    """Import RhoGRPOTrainerV14 with TRL / transformers stubbed so the
    test can run in a CPU-only env without the training stack."""
    for mod in [
        "trl",
        "trl.trainer",
        "trl.trainer.grpo_trainer",
        "trl.data_utils",
        "transformers",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()
    sys.modules["trl"].GRPOTrainer = type("_StubTrainer", (), {})
    sys.modules["trl.trainer.grpo_trainer"].is_conversational = lambda *a, **k: False
    sys.modules["trl.data_utils"].maybe_apply_chat_template = lambda *a, **k: a[0]
    sys.modules["trl.data_utils"].apply_chat_template = lambda *a, **k: a[0]
    sys.modules["transformers"].PreTrainedModel = type("_PM", (), {})
    from src.rho_grpo_trainer_v14 import RhoGRPOTrainerV14
    return RhoGRPOTrainerV14


def _make_stub(rho=1.0, group_size=4):
    """Construct a minimal stand-in object with the attributes the
    trainer method touches, without instantiating GRPOTrainer."""
    trainer_cls = _load_trainer_class()
    stub = types.SimpleNamespace()
    stub._rho = rho
    stub._ada_group_size = group_size
    stub._rho_step_stats = []
    stub.state = types.SimpleNamespace(global_step=0)
    stub._apply_rho_weighting = trainer_cls._apply_rho_weighting.__get__(
        stub, type(stub)
    )
    return stub


class TestV14Shapes:

    def test_apply_rho_weighting_accepts_bg_shape(self):
        """advantages of shape (B*G,) should be accepted."""
        B, G = 2, 4
        stub = _make_stub(rho=1.5, group_size=G)
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
        completion_ids = torch.randint(0, 100, (B * G, 5))
        completion_mask = torch.ones(B * G, 5, dtype=torch.long)

        weighted = stub._apply_rho_weighting(advantages, completion_ids, completion_mask)

        assert weighted.shape == advantages.shape
        # positive entries scale by norm_pos_w = 2·1.5/2.5 = 1.2
        assert torch.allclose(weighted[advantages > 0], advantages[advantages > 0] * 1.2)
        # negative entries scale by norm_neg_w = 2·1.0/2.5 = 0.8
        assert torch.allclose(weighted[advantages < 0], advantages[advantages < 0] * 0.8)

    def test_q_csd_matches_canonical_definition(self):
        """Q_CSD = H_norm(τ⁺) · (n⁺/G)."""
        G = 4
        stub = _make_stub(group_size=G)
        advantages = torch.tensor([1.0, 1.0, 1.0, -1.0])
        # All three positive responses distinct → H_norm = 1
        completion_ids = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        completion_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        stub._apply_rho_weighting(advantages, completion_ids, completion_mask)
        stats = stub._rho_step_stats[-1]
        assert stats["n_positive"] == 3
        assert stats["h_norm_pos"] == pytest.approx(1.0, abs=1e-6)
        assert stats["availability"] == pytest.approx(3 / G)
        assert stats["q_csd"] == pytest.approx(3 / G)

    def test_q_csd_degenerate_all_duplicate(self):
        """Three identical correct responses → H_norm = 0 → Q_CSD = 0."""
        G = 4
        stub = _make_stub(group_size=G)
        advantages = torch.tensor([1.0, 1.0, 1.0, -1.0])
        same_row = [9, 9, 9, 0, 0]
        completion_ids = torch.tensor([same_row, same_row, same_row, [0] * 5])
        completion_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        stub._apply_rho_weighting(advantages, completion_ids, completion_mask)
        stats = stub._rho_step_stats[-1]
        assert stats["h_norm_pos"] == pytest.approx(0.0, abs=1e-6)
        assert stats["q_csd"] == pytest.approx(0.0)

    def test_q_csd_single_correct_is_zero(self):
        """n⁺ = 1 is degenerate → H_norm := 0 by convention."""
        G = 4
        stub = _make_stub(group_size=G)
        advantages = torch.tensor([1.0, -1.0, -1.0, -1.0])
        completion_ids = torch.randint(0, 100, (G, 5))
        completion_mask = torch.ones(G, 5, dtype=torch.long)
        stub._apply_rho_weighting(advantages, completion_ids, completion_mask)
        stats = stub._rho_step_stats[-1]
        assert stats["n_positive"] == 1
        assert stats["h_norm_pos"] == 0.0
        assert stats["q_csd"] == 0.0

    def test_rewards_per_func_uses_bg_rows(self):
        """Regression guard: rewards_per_func allocation must use B*G rows.

        Reviewer Round 1 pointed at the old bug where rewards_per_func was
        allocated per-prompt (B) but filled per-generation (B*G). We check
        the current compute_loss source to ensure the fixed shape is used.
        """
        src = Path(__file__).resolve().parent.parent / "src" / "rho_grpo_trainer_v14.py"
        text = src.read_text()
        # The fix must use n_total = len(completions), not len(inputs).
        assert "n_total = len(completions)" in text, \
            "regression: rewards_per_func no longer uses B*G rows"
        assert "rewards_per_func = torch.zeros(n_total," in text, \
            "regression: rewards_per_func shape regressed"


class TestV14BatchInvariants:
    """Round 3 reviewer: Q_CSD must satisfy [0, 1] in the batch setting (B > 1)."""

    def test_q_csd_bounded_by_one_multi_group(self):
        """With B=2 groups of G=4 (8 rows), all correct → Q_CSD must not exceed 1."""
        G = 4
        stub = _make_stub(group_size=G)
        advantages = torch.ones(8)  # every row positive
        completion_ids = torch.arange(1, 41).reshape(8, 5)  # all distinct
        completion_mask = torch.ones(8, 5, dtype=torch.long)
        stub._apply_rho_weighting(advantages, completion_ids, completion_mask)
        stats = stub._rho_step_stats[-1]
        # Per-group: avail=1.0, H_norm=1.0, Q_CSD_group=1.0 for each of 2 groups
        assert stats["n_groups"] == 2
        assert stats["availability"] == pytest.approx(1.0)
        assert stats["h_norm_pos"] == pytest.approx(1.0)
        assert stats["q_csd"] == pytest.approx(1.0)
        assert stats["q_csd"] <= 1.0 + 1e-9
        for q in stats["q_csd_per_group"]:
            assert 0.0 <= q <= 1.0 + 1e-9

    def test_q_csd_mixed_groups_averages(self):
        """Group 1: 3 distinct correct (Q=0.75). Group 2: 1 correct → Q=0. Batch Q=0.375."""
        G = 4
        stub = _make_stub(group_size=G)
        advantages = torch.tensor([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
        completion_ids = torch.tensor([
            [1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0], [0, 0, 0, 0, 0],
            [10, 11, 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        ])
        completion_mask = torch.ones(8, 5, dtype=torch.long)
        stub._apply_rho_weighting(advantages, completion_ids, completion_mask)
        stats = stub._rho_step_stats[-1]
        # Group 0: n⁺=3 distinct → H_norm=1, avail=3/4 → Q=0.75
        # Group 1: n⁺=1 → H_norm=0 → Q=0
        assert stats["q_csd_per_group"][0] == pytest.approx(0.75)
        assert stats["q_csd_per_group"][1] == pytest.approx(0.0)
        assert stats["q_csd"] == pytest.approx(0.375)

    def test_q_csd_invariant_under_random_batches(self):
        """Fuzz: 100 random batches, Q_CSD ∈ [0, 1] must hold."""
        G = 4
        rng = np.random.default_rng(0)
        stub = _make_stub(group_size=G)
        for _ in range(100):
            B = int(rng.integers(1, 5))
            n = B * G
            advantages = torch.tensor(
                rng.choice([-1.0, 1.0], size=n).astype(np.float32)
            )
            completion_ids = torch.randint(0, 50, (n, 6))
            completion_mask = torch.ones(n, 6, dtype=torch.long)
            stub._apply_rho_weighting(advantages, completion_ids, completion_mask)
            stats = stub._rho_step_stats[-1]
            assert 0.0 <= stats["q_csd"] <= 1.0 + 1e-9, stats

    def test_trainer_and_step0_qcsd_agree(self):
        """Trainer-side Q_CSD and compute_step0_qcsd must agree on the same data."""
        from src.csd_logging import compute_step0_qcsd
        G = 4
        stub = _make_stub(group_size=G)
        advantages = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
        completion_ids = torch.tensor([
            [1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0],
            [0, 0, 0, 0], [10, 11, 12, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        ])
        completion_mask = torch.ones(8, 4, dtype=torch.long)
        stub._apply_rho_weighting(advantages, completion_ids, completion_mask)
        trainer_q = stub._rho_step_stats[-1]["q_csd"]

        rewards_np = np.where(advantages.numpy() > 0, 1.0, 0.0)
        step0_q = compute_step0_qcsd(rewards_np, G, completion_ids.numpy())
        assert trainer_q == pytest.approx(step0_q, abs=1e-6), \
            f"trainer Q_CSD={trainer_q} vs. step-0 Q_CSD={step0_q}"

    def test_step0_qcsd_bounded(self):
        """compute_step0_qcsd must also respect [0, 1] under B > 1."""
        from src.csd_logging import compute_step0_qcsd
        G = 4
        rewards = np.ones(12)  # 3 groups × 4 responses, all correct
        completion_ids = np.arange(12 * 5).reshape(12, 5)  # all distinct
        q = compute_step0_qcsd(rewards, G, completion_ids)
        assert 0.0 <= q <= 1.0 + 1e-9
        assert q == pytest.approx(1.0)

        # No completion_ids → H_norm=1 upper bound; with all correct it should be 1.0 exactly
        q_ub = compute_step0_qcsd(rewards, G, None)
        assert q_ub == pytest.approx(1.0)
