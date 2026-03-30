# Review Summary

**Problem**: GRPO signal balance stability under binary rewards
**Initial Approach**: Phase diagrams of (α,β) space + zero-score gradient reshaping
**Date**: 2026-03-29
**Rounds**: 4 / 5
**Final Score**: 9.1 / 10
**Final Verdict**: READY

## Problem Anchor
GRPO is the dominant RL post-training algorithm for LLM reasoning, yet practitioners lack principled guidance for the positive/negative signal balance. Training has hidden stability regimes. No theory maps or navigates these.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|-------------------------|------------------------------------------|---------|----------------|
| 1 | (α,β) may collapse to 1D; theory needs m/G; "phase transition" too strong; V_pos/V_zero not operational; AdaBalance second headline; budget | Reparameterized to ρ; theory on m/G; "stability map"; trainer telemetry; AdaBalance demoted; two-stage protocol | yes | Theory specifics still vague |
| 2 | P(m) insufficient for Var; upper bound needs λ_KL/ε; collapse p_0>0.8 alone too weak; E[∇L] preservation not proved | Explicit assumptions A1-A3; split Theorem/Proposition; joint collapse conditions; narrowed to binary rewards | yes | E[∇L] claim; overclaiming; i.i.d. robustness |
| 3 | ρ* claim wrong for original GRPO; "stability law" overclaimed; no i.i.d. violation test | Corrected to L_ρ family; "stability analysis"; added robustness Exp 3; K/τ acknowledged as hyperparams | yes | Non-blocking cosmetics |
| 4 | None blocking | — | — | Prop 1 labeling; 27B framing; GSM8K subset definition |

## Overall Evolution
- **Method became more concrete**: From vague (α,β) theory to exact modified GRPO objective with ρ placement
- **Dominant contribution became more focused**: From "phase diagram + theory + navigator" to "stability analysis (theorems) + derived controller (corollary)"
- **Unnecessary complexity removed**: 2D → 1D parameterization, gradient measurement → trainer telemetry, two headlines → one theorem + corollary
- **Modern leverage**: Appropriately uses current GRPO setting; no forced trendy components
- **Drift avoided**: Stayed on signal balance understanding throughout

## Final Status
- Anchor status: preserved
- Focus status: tight
- Modernity status: appropriately frontier-aware
- Strongest parts: Clean theory under explicit assumptions, practical controller as corollary, honest scope
- Remaining weaknesses: Proposition 1 is approximate; theory limited to binary rewards
