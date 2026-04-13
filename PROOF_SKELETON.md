# Proof Skeleton — CSD Theorems

## Dependency DAG

```
A1 (binary rewards) ──┐
A2 (on-policy)  ──────┤
A3 (group structure) ──┤
                       ▼
               Theorem 1 (CSD Equivalence)
                  │         │
                  ▼         ▼
          Theorem 2     Theorem 3
        (Capacity)    (Optimal ρ)
                          │
                          ▼
                    Proposition 1
                     (Q_CSD)
```

## Assumption Ledger

| Assumption | Statement | Used by | Verified? |
|------------|-----------|---------|-----------|
| A1 | r_i ∈ {0,1} (binary verifiable) | T1, T2, T3, P1 | STIPULATED |
| A2 | Responses sampled on-policy from π_θ | T1, T2 | STIPULATED |
| A3 | G responses per prompt, evaluated independently | T1 | STIPULATED |
| (implicit) | π_θ(y\|x) > 0 for all y in supp(τ⁺) | T1 (KL well-defined) | UNVERIFIED — true for softmax LMs |
| (implicit) | Var(g⁺) > 0 | T3 (division by Var) | UNVERIFIED |
| (implicit) | ρ does not affect π_θ within a step | T3 (∂/∂ρ) | UNVERIFIED |
| (implicit) | supp(π_T) ⊆ supp(π₀) | T2 | CITED (NeurIPS 2025 BPR, empirical) |

## Typed Symbol Table

| Symbol | Type | Depends on | Notes |
|--------|------|-----------|-------|
| p | scalar ∈ (0,1) | prompt x, group sample | empirical success rate n⁺/G |
| τ⁺ | distribution on {y_i : r_i=1} | group sample | FIXED w.r.t. θ at gradient time ✓ |
| τ⁻ | distribution on {y_j : r_j=0} | group sample | FIXED w.r.t. θ ✓ |
| g⁺ | vector ∈ ℝ^d | θ, τ⁺ | = ∇_θ KL(τ⁺‖π_θ) |
| g⁻ | vector ∈ ℝ^d | θ, τ⁻ | = ∇_θ KL(τ⁻‖π_θ) |
| ρ | scalar ∈ (0,∞) | user-set | positive signal weight |
| A⁺ | scalar | p | = √((1-p)/p) |
| A⁻ | scalar | p | = -√(p/(1-p)) |

## Canonical Quantified Statements

**Theorem 1:** ∀ prompt x, ∀ group {y_1,...,y_G} ~ π_θ(·|x), let p = n⁺/G where 0 < p < 1:
  ∇_θ L_GRPO(x) = c(ρ) · √(p(1-p)) · [∇_θ KL(τ⁻ ‖ π_θ) − ρ · ∇_θ KL(τ⁺ ‖ π_θ)]
  where c(ρ) = 2/(ρ+1) is a normalization factor.
  
  NOTE: Undefined when p ∈ {0, 1} (degenerate groups). These contribute zero gradient.

**Theorem 2:** (INFORMAL) 𝔼[acc(π_T)] ≤ pass@G_eff(π₀) — lacks formal proof.

**Theorem 3:** ∀θ such that Var(g⁺) > 0:
  ρ* = argmin_ρ Var(∇L_CSD) = Cov(g⁺, g⁻) / Var(g⁺)

**Proposition 1:** (EMPIRICAL CONJECTURE) P(collapse) decreasing in Q_CSD — no formal proof.
