# Experiment Plan: CSD Equivalence + CSDPO

**Source**: `refine-logs/FINAL_PROPOSAL.md` (CSD v2)
**Date**: 2026-04-09
**Budget**: ~490 GPU-hours
**Models**: Qwen2.5-7B, Qwen3-8B, Qwen3.5-9B

## Claims to Prove

| # | Claim | Type | Priority |
|---|-------|------|----------|
| C1 | GRPO gradient = CSD gradient (Theorem 1) | Theory + Empirical | MUST |
| C2 | CSD capacity bound holds (Theorem 2) | Theory + Empirical | MUST |
| C3 | Closed-form ρ* improves stability (Theorem 3) | Empirical | MUST |
| C4 | Q_CSD predicts collapse better than variance | Empirical | MUST |
| C5 | CSDPO eliminates collapse and beats SRPO | Empirical | MUST |
| C6 | CSD predicts variant relative performance | Empirical | SHOULD |
| C7 | CSDPO components are each necessary | Empirical | SHOULD |

## Block 1: CSD Equivalence Verification (C1) — 54 GPU-hrs

- 3 models × 4 ρ × 3 seeds × 200 steps
- Log per-step: ∇KL(τ⁺‖π), ∇KL(τ⁻‖π), ∇L_GRPO
- Metric: R² > 0.95, cosine > 0.99
- Gate: R² < 0.8 → investigate A2 assumption

## Block 2: Capacity Bound (C2) — 84 GPU-hrs

- Base pass@k: k ∈ {1,4,8,16,32,64,128,256}, 3 models
- G sweep: G ∈ {4,8,16,32}, 3 models × 2 seeds
- Metric: accuracy(CSDPO,G) ≤ pass@(G·T_eff)(π₀)

## Block 3: Optimal ρ + Collapse Predictor (C3+C4) — 120 GPU-hrs

- Qwen2.5-7B, ρ ∈ {0.3,0.5,0.7,1.0,1.5,2.0,3.0,5.0}, 10 seeds, 300 steps
- Log: H(τ⁺), n⁺/G, cos(g⁺,g⁻), collapse label
- C3 metric: ADQ 0% collapse at ρ=1.0 (vs 50%)
- C4 metric: Q_CSD AUROC > 0.85

## Block 4: CSDPO vs Baselines (C5) — 112 GPU-hrs

- 3 models × 5 methods (GRPO, DAPO, SRPO, GRPO-λ, CSDPO) × 5 seeds × 300 steps
- Eval: GSM8K + MATH-500
- Metrics: accuracy, collapse rate, convergence speed

## Block 5: Ablation (C7) — 45 GPU-hrs

- Qwen2.5-7B × 6 variants (full, -EA, -QW, -ADQ, -GCR, GRPO) × 5 seeds

## Block 6: Variant Prediction (C6) — 75 GPU-hrs

- Qwen2.5-7B × 5 methods × 5 difficulty groups × 3 seeds

## Run Order

| Week | Blocks | Wall time (8 GPU) | Gate |
|------|--------|-------------------|------|
| 1 | 1 + 3 (parallel) | ~15h | R² > 0.8, AUROC > 0.7 |
| 2 | 2 + 4 (parallel) | ~14h | CSDPO > GRPO +3% |
| 3 | 5 + 6 (parallel) | ~15h | — |

## First 3 Runs

1. `python train_csd_verification.py --model Qwen2.5-7B --rho 1.0 --seed 42 --log_csd`
2. `python train_grpo_sweep.py --model Qwen2.5-7B --rho 1.0 --seeds 10 --steps 300`
3. `python eval_passk.py --model Qwen2.5-7B --k 1,4,8,16,32,64,128,256`
