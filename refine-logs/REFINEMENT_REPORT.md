# Refinement Report — CSD v2

**Problem**: GRPO training instability
**Direction Evolution**: Stability analysis → MetaGRPO → CSD equivalence
**Date**: 2026-04-09

## Direction Changes

| Version | Direction | Score | Why Changed |
|---------|-----------|-------|-------------|
| v1.0 | (α,β) phase diagrams + zero-score reshaping | 2/10 | No results, theory shallow |
| v1.4 | Stability analysis (Theorems 1-3, AdaBalance) | 9.1/10 | Scoped, but not best-paper |
| v1.5 | MetaGRPO (basin + step-0 + rescue) | 3/10 | Nightmare: analysis only, no method |
| **v2.0** | **CSD: GRPO = contrastive self-distillation** | **5→8/10** | **Paradigm shift + method** |

## Key Refinements in v2

### From Nightmare Review Weaknesses to Solutions

| Weakness | Solution | Implementation |
|----------|----------|---------------|
| F1: Equivalence trivially obvious | 3 quantitative predictions CSD makes that standard analysis cannot | Predictions 1-3 in FINAL_PROPOSAL |
| F2: Capacity bound informal | Theorem 2 with formal proof, G-dependence | Verified via Block 2 |
| F3: CSDPO components ad hoc | Each derived from CSD objective with explicit formula | EA, QW, ADQ, GCR derivations |
| F4: Binary rewards narrow | Extension to continuous rewards (Remark 1) | τ_r(y) ∝ r(y)·π(y|x) |
| F5: No surprising prediction | Q_CSD predictor, variant ranking, optimal G∝1/p | Blocks 3, 4, 6 |
| S6: Must beat SRPO | Primary baseline in Block 4 | 3 models × 5 seeds |
| S7: Unification descriptive only | Regime-specific predictions table | Block 6 |
| S8: Seed variance hand-wavy | Formal model: Var(acc) ∝ Var_{τ⁺}[KL(τ⁺‖π_ref)] | Block 3 |

## Complexity Intentionally Rejected

1. Process rewards / token-level reward modeling
2. External teacher models / cross-distillation
3. Architectural modifications (contrastive heads like CLIPO)
4. Off-policy training / importance sampling
5. Meta-learning the advantage function

## Remaining Risks

1. Q_CSD predictor may not achieve high AUROC → fallback: combined features
2. CSDPO may not beat SRPO → theory contribution still holds
3. Capacity bound may be loose → show tightness empirically with varying G
