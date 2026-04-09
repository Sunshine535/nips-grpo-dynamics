# Review Summary — CSD Direction

**Problem**: GRPO training instability → RLVR is Contrastive Self-Distillation
**Date**: 2026-04-09

## Review History

### v1: Stability Analysis (2026-03-29 → 2026-04-04)
- 4 rounds GPT-5.4 refinement: 4.5 → 6.5 → 8.1 → 9.1/10
- Result: READY for submission as scoped stability paper

### v1 Nightmare Review (2026-04-07): 3/10
- 5 FATAL: Theory obvious, stats thin, Kramers metaphor, models all Qwen, not proven metastable
- Pivoted to MetaGRPO (basin analysis + step-0 prediction + transient rescue)
- MetaGRPO estimated ceiling: 7-8/10 — not enough for best paper

### v2: CSD Direction (2026-04-09)
- Full re-ideation: 12 literature searches, 60+ papers, mathematical derivation
- Core theorem: ∇L_GRPO = √(p(1-p))·[∇KL(τ⁻‖π) - ρ·∇KL(τ⁺‖π)]
- Novelty check: 8/10 (no prior proof of CSD equivalence)

### v2 Self-Nightmare Review: 5/10
- F1: Equivalence may be trivially obvious → Added 3 quantitative predictions
- F2: Capacity bound needs formal proof → Theorem 2 with G-dependence
- F3: CSDPO components ad hoc → Formal derivation from CSD for each component
- F4: Binary-only scope → Extension to continuous rewards (Remark 1)
- F5: Need surprising prediction → Q_CSD collapse predictor, variant performance ranking
- S6: Must beat SRPO → Included as primary baseline
- S7: Unification must be predictive → Regime-specific predictions table
- S8: Seed variance model → Formal stochastic τ⁺ model

### v2 Refined (post-nightmare): estimated 7-8/10
- If all experiments pass → 8-9/10 (theory + method + predictions)
- If CSDPO doesn't beat SRPO but theory holds → 7/10 (strong theory paper)
