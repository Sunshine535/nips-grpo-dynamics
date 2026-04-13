# Proof Audit — CSD Theorems (Nightmare Difficulty)

**Date**: 2026-04-14
**Reviewer**: Claude Opus 4.6 (self-audit, nightmare)
**Target**: refine-logs/FINAL_PROPOSAL.md — Theorems 1-3, Proposition 1

---

## Issue Summary

| ID | Category | Status | Impact | Severity | Location |
|----|----------|--------|--------|----------|----------|
| I1 | NORMALIZATION_MISMATCH | UNDERSTATED | LOCAL | MINOR | Theorem 1 |
| I2 | SCOPE_OVERCLAIM | OVERSTATED | GLOBAL | **CRITICAL** | Theorem 1 |
| I3 | UNJUSTIFIED_ASSERTION | UNJUSTIFIED | GLOBAL | **FATAL** | Theorem 2 |
| I4 | HIDDEN_ASSUMPTION | UNDERSTATED | LOCAL | MAJOR | Theorem 3 |
| I5 | UNJUSTIFIED_ASSERTION | UNJUSTIFIED | LOCAL | **CRITICAL** | Proposition 1 |
| I6 | SCOPE_OVERCLAIM | OVERSTATED | LOCAL | MAJOR | Remark 1 |
| I7 | CASE_INCOMPLETE | UNDERSTATED | LOCAL | MAJOR | Theorem 1 |
| I8 | QUANTIFIER_ERROR | UNCLEAR | LOCAL | MINOR | Theorem 3 |
| I9 | HIDDEN_ASSUMPTION | UNDERSTATED | LOCAL | MAJOR | Theorem 1 |

---

## Detailed Issue Reports

### I1: Missing normalization factor (MINOR)

**Location**: Theorem 1 statement
**Status**: UNDERSTATED
**Impact**: LOCAL

**Statement**: Theorem claims ∇L = √(p(1-p))·[∇KL(τ⁻‖π) - ρ·∇KL(τ⁺‖π)]

**Problem**: The actual ρ-weighted GRPO (from rho_grpo_trainer.py) normalizes:
  pos_weight = 2ρ/(ρ+1), neg_weight = 2/(ρ+1)

So the correct formula includes a factor c(ρ) = 2/(ρ+1):
  ∇L_ρ = c(ρ) · √(p(1-p)) · [∇KL(τ⁻‖π) - ρ·∇KL(τ⁺‖π)]

**Counterexample**: NO — the gradient DIRECTION is identical; only magnitude differs.

**Fix**: State c(ρ) explicitly. For standard GRPO (ρ=1), c=1 and the formula is exact.

**Downstream**: Does not affect CSD interpretation, optimal ρ direction, or predictions.

---

### I2: Theorem 1 overclaims "GRPO = CSD" but is really "ρ-GRPO = CSD" (CRITICAL)

**Location**: Theorem 1 framing
**Status**: OVERSTATED
**Impact**: GLOBAL

**Problem**: Standard GRPO (TRL, DeepSeek-R1) does NOT have ρ weighting. It uses symmetric advantages. The CSD decomposition with ρ=1 gives:
  ∇L = √(p(1-p)) · [∇KL(τ⁻‖π) - ∇KL(τ⁺‖π)]

This IS a valid CSD decomposition, but the ρ parameter is OUR EXTENSION, not part of standard GRPO. The paper narrative ("GRPO is not RL, it's self-distillation") is correct for ALL ρ values including ρ=1.

**Fix**: Present Theorem 1 in two parts:
  (a) Standard GRPO (ρ=1): exact CSD decomposition
  (b) ρ-weighted extension: adds control over distillation/anti-distillation balance

**Downstream**: The CSD interpretation holds for standard GRPO (ρ=1). The ρ extension is our methodological contribution.

---

### I3: Theorem 2 capacity bound is NOT PROVEN (FATAL)

**Location**: Theorem 2 proof sketch
**Status**: UNJUSTIFIED
**Impact**: GLOBAL

**Why FATAL**: The "proof" cites the NeurIPS 2025 BPR paper (an EMPIRICAL finding) as a proof step:
  "these are already in π₀'s support (RL doesn't add new capabilities, by NeurIPS 2025 BPR)"

This is NOT a formal proof. The NeurIPS paper showed empirically that pass@k of base ≥ pass@1 of RLVR-trained at large k. This does not constitute a mathematical theorem.

**Deeper issue**: The claim "supp(π_T) ⊆ supp(π₀)" is TRIVIALLY TRUE for autoregressive LMs with softmax (both have full support over all token sequences). The meaningful claim is about PROBABILITY MASS, not support.

**What the actual bound should say**: GRPO concentrates probability mass on responses that received reward 1 during training. The maximum achievable accuracy is bounded by the fraction of prompts for which the base model can generate at least one correct response in G·T_eff samples. This is pass@(G·T_eff) by definition.

But even this restated bound has problems:
- π_t changes over training, so later rollouts sample from a DIFFERENT distribution than π₀
- The effective exploration is NOT simply G·T independent samples from π₀
- Distillation from τ⁺_t may find NEW correct paths that π₀ would not find via independent sampling (through policy-guided search)

**Counterexample candidate**: If GRPO-trained π_T consistently finds correct responses via a different reasoning path than π₀, the "bounded by pass@k(π₀)" claim could be violated.

**Fix options**:
  (a) WEAKEN_CLAIM: Downgrade to "Informal Prediction" or "Empirical Conjecture"
  (b) STRENGTHEN_ASSUMPTION: Add formal assumptions about distribution stability
  (c) ADD_DERIVATION: Prove a weaker but rigorous bound

**RECOMMENDED**: Option (a) — call it an "empirical prediction motivated by CSD" not a "theorem."

---

### I4: Theorem 3 missing assumptions (MAJOR)

**Location**: Theorem 3 statement and proof
**Status**: UNDERSTATED
**Impact**: LOCAL

**Missing assumptions**:
1. Var(g⁺) > 0 (otherwise ρ* undefined)
2. g⁺ and g⁻ treated as random vectors — over WHAT probability space? (Different groups? Different prompts? Different steps?)
3. The formula assumes ρ does not affect g⁺ and g⁻ within the same step (true for current-step computation, false across steps)
4. The variance formula Var(a·X + b·Y) = a²Var(X) + b²Var(Y) + 2ab·Cov(X,Y) assumes SCALAR ρ weighting. But g⁺ and g⁻ are VECTORS. The variance should be trace(Cov matrix) or expected squared norm.

**Fix**: State assumptions explicitly. Clarify that Var(g⁺) means E[‖g⁺‖² ] - ‖E[g⁺]‖² and similarly for Cov. The scalar formula is an approximation when g⁺ and g⁻ have aligned principal components.

---

### I5: Proposition 1 is empirical conjecture, not a mathematical statement (CRITICAL)

**Location**: Proposition 1
**Status**: UNJUSTIFIED
**Impact**: LOCAL

**Problem**: "P(collapse | step t) is monotonically decreasing in Q_CSD(t)" is stated without ANY justification. There is no:
- Definition of "collapse" (binary event? continuous measure?)
- Model of how collapse happens
- Proof of monotonicity
- Even an informal argument

The claim is that a product of three heuristic metrics predicts a poorly-defined event. This is an empirical hypothesis, not a proposition.

**Fix**: WEAKEN_CLAIM — rename to "Empirical Hypothesis" or "Heuristic" and validate experimentally. Do not call it a "Proposition" unless proven.

---

### I6: Remark 1 (continuous reward extension) is overstated (MAJOR)

**Location**: Remark 1
**Status**: OVERSTATED
**Impact**: LOCAL

**Problem**: Claims "CSD structure persists for any bounded reward." This is FALSE for the exact CSD decomposition.

For continuous r_i ∈ [0,1], the advantage A_i = (r_i - μ)/σ assigns a DIFFERENT weight to each response. There is no clean binary partition into τ⁺/τ⁻. The GRPO gradient becomes:
  ∇L = (1/Gσ) Σ_i (r_i - μ) ∇log π(y_i)

This is a WEIGHTED score function estimator, which can be interpreted as distillation from a reward-weighted distribution τ_r(y) ∝ r(y)·δ_yi, but NOT as KL(τ⁻‖π) - KL(τ⁺‖π) with discrete distributions.

The claim τ_r(y) ∝ r(y)·π_θ(y|x) is also wrong — τ_r should be proportional to r(y) over the SAMPLE, not over the policy.

**Fix**: State that the binary decomposition MOTIVATES a continuous analogue, but the exact KL form only holds for binary rewards. Remove "CSD structure persists."

---

### I7: Degenerate case p ∈ {0, 1} not handled (MAJOR)

**Location**: Theorem 1
**Status**: UNDERSTATED
**Impact**: LOCAL

**Problem**: When p=0 (all incorrect) or p=1 (all correct):
- σ = √(p(1-p)) = 0 → division by zero in advantage computation
- Standard GRPO handles this by setting advantages to 0 (degenerate group)
- CSD formula gives √(0) · [...] = 0, which is consistent

But the proof steps assume p ∈ (0,1). The theorem should explicitly exclude p ∈ {0,1} or state that degenerate groups contribute zero gradient (which is the CSD "zero-success trap" corollary).

**Fix**: Add "where 0 < p < 1" to the theorem statement. Note that p ∈ {0,1} gives zero gradient, consistent with the CSD "zero-success trap."

---

### I8: Var/Cov in Theorem 3 — over what? (MINOR)

**Location**: Theorem 3
**Status**: UNCLEAR
**Impact**: LOCAL

**Problem**: Var(g⁺) and Cov(g⁺,g⁻) are not specified over what random variable. Options:
- Over different random groups for the same prompt (within-prompt variance)
- Over different prompts in a batch (cross-prompt variance)
- Over training steps (temporal variance)

The practical implementation uses EMA over recent batches (temporal), but the theorem statement doesn't specify.

**Fix**: Specify "Var and Cov are taken over random groups sampled from π_θ for a fixed prompt x."

---

### I9: τ⁺ = Uniform assumption is implicit (MAJOR)

**Location**: Theorem 1
**Status**: UNDERSTATED
**Impact**: LOCAL

**Problem**: The proof defines τ⁺ = Uniform({y_i : r_i=1}). This means all correct responses in the group are weighted equally. But this is only true when the GRPO advantages are computed with standard normalization (A_i depends only on r_i, not on y_i).

If token-level loss normalization is used (as in some TRL versions), the effective weight per response depends on sequence length, not just reward. This would break the uniform τ⁺ assumption.

**Fix**: State explicitly: "τ⁺ is uniform over correct responses when using sequence-level (not token-level) advantage weighting."

---

## Acceptance Gate Check

1. Zero FATAL or CRITICAL issues? **NO** — 1 FATAL (I3), 2 CRITICAL (I2, I5)
2. Every theorem has explicit hypotheses + full proof? **NO** — T2 lacks proof
3. All big-O statements with declared dependence? **N/A** (no asymptotic claims)
4. Counterexample pass? **PARTIAL** — candidate for T2, none found for T1/T3

**VERDICT: DOES NOT PASS** — needs fixes for I2, I3, I5 before publication.

---

## Salvage Assessment

### What IS proven (solid):
- **Theorem 1 (CSD Equivalence)** for standard GRPO (ρ=1): CORRECT, needs minor fixes (I1, I7, I9)
- **Theorem 3 (Optimal ρ)**: CORRECT under stated assumptions, needs explicit conditions (I4, I8)

### What is NOT proven:
- **Theorem 2 (Capacity Bound)**: Empirical conjecture, not a theorem
- **Proposition 1 (Q_CSD)**: Empirical hypothesis, not a proposition
- **Remark 1 (Continuous)**: Overstated, needs weakening

### Recommended Fix Priority:
1. **I3 (FATAL)**: Downgrade Theorem 2 to "Prediction" or prove rigorously
2. **I2 (CRITICAL)**: Clarify ρ=1 is standard GRPO, ρ≠1 is our extension
3. **I5 (CRITICAL)**: Downgrade Proposition 1 to "Empirical Hypothesis"
4. **I6 (MAJOR)**: Weaken Remark 1
5. **I4, I7, I8, I9 (MAJOR)**: Add missing assumptions and edge cases
