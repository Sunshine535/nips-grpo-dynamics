# ARIS Automated Review — GRPO Stability Analysis

## Paper Title
"Why GRPO Variants Work: A Unified Stability Map Under Binary Verifiable Rewards"

## Research Question
Why does GRPO training sometimes collapse and sometimes succeed? Can we predict and prevent it?

## Core Contribution
A unified stability framework that:
1. Maps GRPO training dynamics onto two coordinates: (ρ_eff, GSR_eff)
2. Derives computable stability boundaries (ρ_min, ρ_max)
3. Explains WHY existing variants (DAPO, GSPO, GTPO) work through the same framework
4. Predicts when variants help vs. when they're redundant

## Theoretical Results
- Theorem 1: Degenerate group starvation (m=0 or m=G → zero gradient)
- Theorem 2: Gradient variance decomposition Var = ρ²V+ + V- + 2ρC
- Theorem 3: Sharp lower bound ρ_min = V_/(2|C|)
- Proposition 1: Approximate upper bound ρ_max
- Corollary 1: Optimal ρ* = -C/V+

## Experimental Evidence

### Synthetic Validation
- R² = 0.994 (predicted vs observed gradient variance)
- 99.8% regime classification accuracy (559/560)

### Qwen2.5-7B-Instruct (complete, 6 seeds at critical points)
- Standard GRPO (ρ=1.0): 45.5% accuracy, 50% failure rate
- Optimal ρ=3.0: 87.2% accuracy, 0% failure rate
- +41.7% absolute improvement

### Qwen3.5-9B (preliminary, 3 seeds — NEEDS MORE REPETITIONS)
- ρ=1.0: 95.0% (0/3 fail) — different from Qwen2.5!
- ρ=3.0: 95.0% (0/3 fail) — consistent
- Interpretation: model-dependent stability boundaries (theory predicts this)

### GRPO Variant Unification
| Variant | Mechanism in Our Framework |
|---------|--------------------------|
| DAPO | ↑ρ_eff via clip asymmetry + ↓GSR via dynamic sampling |
| GSPO | ↓ρ_min via sequence-level variance reduction |
| GTPO | ↓GSR→0 via degenerate group masking |

### Confounder Ablation (8 runs on Qwen2.5)
- ρ stability robust across G=2/4/8
- ρ=3.0 robust even under high λ_KL=0.20 (where ρ=0.7 collapses)

## Limitations
1. Qwen3.5-9B results need ≥5 seeds per condition for statistical reliability
2. AdaBalance controller broken (TRL 1.0 compatibility) — removed from claims
3. Proposition 1 (upper bound) is approximate, not sharp
4. Theory restricted to binary verifiable rewards
5. Single dataset (GSM8K)

## What Remains To Run
See `configs/full_experiment_plan.yaml` — 180 runs, ~322 GPU-hours
- 65 core sweep runs (10 seeds at critical ρ)
- 40 variant comparison runs (5 seeds each)
- 15 long-horizon runs (600 steps)
- 60 confounder ablation runs (5 seeds each)

## External Review History (7 rounds via GPT-5.4)
- Round 1: 2/10 (zero experiments)
- Round 2: 5/10 (coarse sweep done)
- Round 3: 6/10 (GSM8K eval done)
- Round 4: 6/10 (scoped as stability paper)
- Round 5: 6/10 (confounder added)
- Round 6: 3/10 best paper (6/10 accept)
- Round 7: 5.5→8/10 projected (variant unification framework)
