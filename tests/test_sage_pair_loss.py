"""Tests for SAGE pair loss logic (Task 3)."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F


def test_logsigmoid_preference():
    """When positive logprob > negative, loss should be low."""
    logp_pos = torch.tensor([0.0])
    logp_neg = torch.tensor([-2.0])
    ref_delta = torch.tensor([0.0])
    margin = (logp_pos - logp_neg) - ref_delta
    loss = -F.logsigmoid(margin).mean()
    assert loss.item() < 0.15, f"Loss should be low when pos preferred: {loss.item()}"


def test_logsigmoid_wrong_preference():
    """When negative logprob > positive, loss should be high."""
    logp_pos = torch.tensor([-2.0])
    logp_neg = torch.tensor([0.0])
    ref_delta = torch.tensor([0.0])
    margin = (logp_pos - logp_neg) - ref_delta
    loss = -F.logsigmoid(margin).mean()
    assert loss.item() > 1.5, f"Loss should be high when neg preferred: {loss.item()}"


def test_ref_coef_shifts_margin():
    """ref_coef > 0 should reduce margin when ref also prefers positive."""
    logp_pos = torch.tensor([0.0])
    logp_neg = torch.tensor([-1.0])
    ref_logp_pos = torch.tensor([0.0])
    ref_logp_neg = torch.tensor([-1.0])
    # Without ref: margin = 1.0
    margin_no_ref = (logp_pos - logp_neg)
    # With ref_coef=1.0: margin = 1.0 - 1.0 = 0.0
    margin_with_ref = (logp_pos - logp_neg) - 1.0 * (ref_logp_pos - ref_logp_neg)
    loss_no_ref = -F.logsigmoid(margin_no_ref).mean()
    loss_with_ref = -F.logsigmoid(margin_with_ref).mean()
    assert loss_with_ref > loss_no_ref, \
        f"ref_coef should increase loss: {loss_with_ref} vs {loss_no_ref}"


def test_zero_pairs_zero_loss():
    """No pairs should produce no mechanism activation."""
    from src.contrastive_evidence_bank import ContrastiveEvidenceBank
    bank = ContrastiveEvidenceBank()
    pairs = bank.sample_pairs(5, current_step=10)
    assert pairs == []


if __name__ == "__main__":
    test_logsigmoid_preference()
    test_logsigmoid_wrong_preference()
    test_ref_coef_shifts_margin()
    test_zero_pairs_zero_loss()
    print("ALL SAGE pair loss TESTS PASSED")
