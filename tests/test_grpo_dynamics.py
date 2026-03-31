"""
Basic tests for nips-grpo-dynamics core modules.
Run: python -m pytest tests/test_grpo_dynamics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.balanced_grpo import BalancedGRPOConfig, compute_balanced_grpo_loss
from src.rho_grpo import RhoGRPOConfig, compute_group_statistics, compute_grpo_advantages
from src.stability_analysis import (
    StabilityBounds,
    analyze_stability,
    classify_regime,
    compute_gradient_variance,
    group_starvation_rate,
)


# ---------------------------------------------------------------------------
# stability_analysis tests
# ---------------------------------------------------------------------------
class TestStabilityAnalysis:

    def test_gsr_extreme_p(self):
        assert group_starvation_rate(0.0, 4) == pytest.approx(1.0)
        assert group_starvation_rate(1.0, 4) == pytest.approx(1.0)

    def test_gsr_balanced_p(self):
        gsr = group_starvation_rate(0.5, 4)
        assert 0.0 < gsr < 1.0
        assert gsr == pytest.approx(0.5**4 + 0.5**4)

    def test_analyze_returns_bounds(self):
        bounds = analyze_stability(0.5, 4)
        assert isinstance(bounds, StabilityBounds)
        assert bounds.rho_min >= 0
        assert bounds.rho_max >= bounds.rho_min
        assert bounds.rho_star >= bounds.rho_min
        assert bounds.rho_star <= bounds.rho_max

    def test_classify_regime_convergent(self):
        bounds = analyze_stability(0.5, 4)
        regime = classify_regime(bounds.rho_star, bounds)
        assert regime == "convergent"

    def test_gradient_variance_nonneg(self):
        bounds = analyze_stability(0.5, 4)
        for rho in [0.1, 0.5, 1.0, 2.0, 5.0]:
            var = compute_gradient_variance(rho, bounds)
            assert var >= -1e-9, f"Negative variance at rho={rho}: {var}"


# ---------------------------------------------------------------------------
# rho_grpo tests
# ---------------------------------------------------------------------------
class TestRhoGRPO:

    def test_compute_group_statistics_basic(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        stats = compute_group_statistics(rewards, group_size=4)
        assert stats["n_groups"] == 2
        assert stats["gsr"] >= 0.0

    def test_compute_group_statistics_all_success(self):
        rewards = torch.ones(8)
        stats = compute_group_statistics(rewards, group_size=4)
        assert stats["p_0"] > 0  # all-success groups are degenerate

    def test_grpo_advantages_shape(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        weighted, raw = compute_grpo_advantages(rewards, group_size=4, rho=1.5)
        assert weighted.shape == (8,)
        assert raw.shape == (8,)

    def test_grpo_advantages_rho1_matches_raw(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        weighted, raw = compute_grpo_advantages(rewards, group_size=4, rho=1.0)
        pos_mask = rewards > 0
        torch.testing.assert_close(weighted[pos_mask], raw[pos_mask])


# ---------------------------------------------------------------------------
# balanced_grpo weight normalization tests
# ---------------------------------------------------------------------------
class TestBalancedGRPONormalization:

    def _make_inputs(self, B=8, G=4, L=16):
        logprobs = torch.randn(B, G, L) * 0.1
        ref_logprobs = torch.randn(B, G, L) * 0.1
        advantages = torch.randn(B, G)
        mask = torch.ones(B, G, L)
        return logprobs, ref_logprobs, advantages, mask

    def test_symmetric_config_gives_equal_weights(self):
        cfg = BalancedGRPOConfig(positive_ratio=0.5, negative_weight=1.0)
        lp, rlp, adv, mask = self._make_inputs()
        result = compute_balanced_grpo_loss(lp, rlp, adv, mask, cfg)
        assert result["normalized_weight_pos"] == pytest.approx(1.0)
        assert result["normalized_weight_neg"] == pytest.approx(1.0)

    def test_normalization_preserves_ratio(self):
        cfg = BalancedGRPOConfig(positive_ratio=0.8, negative_weight=2.0)
        lp, rlp, adv, mask = self._make_inputs()
        result = compute_balanced_grpo_loss(lp, rlp, adv, mask, cfg)
        raw_ratio = result["pos_neg_ratio_raw"]
        norm_ratio = result["pos_neg_ratio_normalized"]
        assert raw_ratio == pytest.approx(norm_ratio, rel=1e-5)

    def test_normalized_weights_sum_to_two(self):
        for alpha, beta in [(0.3, 0.5), (0.7, 2.0), (0.1, 3.0)]:
            cfg = BalancedGRPOConfig(positive_ratio=alpha, negative_weight=beta)
            lp, rlp, adv, mask = self._make_inputs()
            result = compute_balanced_grpo_loss(lp, rlp, adv, mask, cfg)
            total = result["normalized_weight_pos"] + result["normalized_weight_neg"]
            assert total == pytest.approx(2.0, abs=1e-6)

    def test_loss_is_finite(self):
        cfg = BalancedGRPOConfig(positive_ratio=0.5, negative_weight=1.0)
        lp, rlp, adv, mask = self._make_inputs()
        result = compute_balanced_grpo_loss(lp, rlp, adv, mask, cfg)
        assert torch.isfinite(result["loss"]).all()


# ---------------------------------------------------------------------------
# eval_phase_point: answer extraction
# ---------------------------------------------------------------------------
class TestAnswerExtraction:

    def test_extract_boxed_simple(self):
        from scripts.eval_phase_point import _extract_boxed
        assert _extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_extract_boxed_nested(self):
        from scripts.eval_phase_point import _extract_boxed
        assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_extract_numeric_gsm8k(self):
        from scripts.eval_phase_point import extract_numeric_answer
        assert extract_numeric_answer("Therefore #### 42") == "42"

    def test_extract_numeric_fallback(self):
        from scripts.eval_phase_point import extract_numeric_answer
        assert extract_numeric_answer("The answer is 7.") == "7"
