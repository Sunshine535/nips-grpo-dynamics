# TASA-GRPO: Threshold-Anchored Signed Advantage for Continuous-Reward Group Policy Optimization

## Problem
Standard GRPO computes advantage as A_i = r_i - mean(r_group). With continuous/partial-credit rewards, below-threshold completions (wrong answers) can receive positive advantage when they are "less wrong" than the group mean. This wrong-sign gradient reinforces incorrect reasoning.

## Method (one equation)

For G completions with continuous rewards r_1...r_G and correctness threshold c (default c=0.5):

```
A_i = (r_i - c)_+ / Z+  -  (c - r_i)_+ / Z-

where Z+ = sum_j (r_j - c)_+,  Z- = sum_j (c - r_j)_+
```

Degenerate cases:
- Z+ = 0 (all wrong): A_i = -(c - r_i) / Z-  (pure avoidance, weighted by severity)
- Z- = 0 (all correct): A_i = (r_i - c) / Z+  (pure reinforcement, weighted by quality)

Loss: L = -sum_i A_i * log pi(y_i|x) + beta * KL(pi || pi_ref)

Implementation: stop-gradient on Z+, Z-. Standard TRL clipping. KL to LoRA-disabled reference.

## Properties (invariants)
- P1 (Sign correctness): r_i > c => A_i > 0; r_i <= c => A_i <= 0
- P2 (Within-side monotonicity): higher reward => higher advantage within each side
- P3 (Zero-sum): sum A_i = 0 when both S+ and S- nonempty
- P4 (Binary directional equivalence): with binary {0,1} rewards, TASA is directionally equivalent to Dr. GRPO. Specifically: TASA_i = G/(n+*n-) * A_Dr.GRPO_i

## Contribution
The minimal threshold-anchored advantage transform for continuous-reward GRPO. Changes only the advantage computation. Zero new parameters, zero new modules.

## Validation Plan
1. Main table: TASA vs {GRPO, Dr. GRPO, CoRPO} on MATH (partial credit), 3 seeds, full test set
2. Core diagnostic (Figure 1): wrong-sign advantage frequency — % of below-threshold completions receiving positive A_i under each method
3. Sanity: TASA on binary GSM8K matches Dr. GRPO direction
4. Ablation: c sensitivity (0.3, 0.5, 0.7)
5. Optional appendix: (alpha, beta) sensitivity analysis

## Reward Design (engineering, not method)
- r = 1.0: exact match (symbolic/numeric via sympy)
- r = 0.7: final answer within 1% relative tolerance
- r = 0.0: otherwise
- Correctness threshold c = 0.5

## Training Setup
- Qwen3.5-9B + LoRA r=64, G=8, batch=1, grad_accum=4
- lr=2e-5, warmup 5%, cosine decay
- 200 steps, KL beta=0.04
- Full MATH test evaluation (n=5000)

## Compute: ~50 GPU-hours (8h on 8xA800)
## Timeline: Implementation 3 days, experiments 2 days, analysis 2 days
