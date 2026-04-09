---
type: idea
node_id: idea:csd-equivalence
title: "RLVR is Contrastive Self-Distillation"
stage: refined
outcome: pending
based_on: [paper:srpo2026, paper:clipo2026, paper:rlvr_limit2025, paper:sdpo2026]
target_gaps: [G1, G2, G3, G4, G5, G6]
created_at: 2026-04-09
---

# GRPO gradient under binary rewards = contrastive self-distillation

## Hypothesis

∇L_GRPO = √(p(1-p)) · [∇KL(τ⁻‖π) - ρ·∇KL(τ⁺‖π)]

GRPO simultaneously self-distills from correct responses and anti-distills from incorrect.

## Key Predictions
1. Capacity bound: accuracy ≤ pass@G·T_eff (Theorem 2)
2. Optimal ρ* = Cov(g⁺,g⁻)/Var(g⁺) (Theorem 3)
3. Q_CSD predicts collapse better than variance (Proposition 1)

## Method: CSDPO
- EA: Experience-augmented τ⁺ (fixes zero-success)
- QW: Quality-weighted distillation
- ADQ: Adaptive ρ from CSD variance minimization
- GCR: Gradient consistency regularization

## Status
- Novelty: 8/10 confirmed
- Nightmare review: 5→8/10 (after addressing weaknesses)
- Experiments: PENDING (490 GPU-hours planned)
