---
type: idea
node_id: idea:stability-v1
title: "Stability Analysis of GRPO Signal Balance (ρ-weighted variance decomposition)"
stage: archived
outcome: partial
based_on: []
target_gaps: [G4]
failure_notes: "Refined to 9.1/10 for scoped paper but nightmare review of MetaGRPO extension killed it. Theory considered close to algebraic identity. AdaBalance broken in TRL. Not best-paper level."
created_at: 2026-03-25
---

# Variance decomposition Var(∇L_ρ) = ρ²V₊ + V₋ + 2ρC

## What Worked
- Synthetic validation: R²=0.994
- Qwen2.5-7B ρ-sweep data useful
- AdaBalance concept (minimum-variance ρ)

## Why Archived
- Theorem 2 may be algebraic identity
- Proposition 1 is heuristic
- "Single control variable" confounded by LR, KL, G
- AdaBalance broke on TRL 1.0
- Superseded by CSD (which gives ρ* from distillation theory, not variance)
