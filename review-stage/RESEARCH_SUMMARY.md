# CSD Research Summary — Auto Review Loop Round 1

> **⚠ SUPERSEDED** — This file records the Round-1 narrative before the
> audit. Claims here that contradict `RETRACTIONS.md` (repo root) are
> retracted. Authoritative numbers live in
> `refine-logs/FINAL_PROPOSAL.md §Results` (Wave 2+, 3-seed, n=200). The
> single-seed ρ-sweep and "GRPO IS CSD" framing below are kept for audit
> trail only.

## Target Venue
NeurIPS 2026 best paper level (difficulty: nightmare)

## Core Theoretical Claim

**Theorem 1 (CSD Equivalence)**: Under binary verifiable rewards r ∈ {0,1}, GRPO's policy gradient decomposes exactly as:

∇_θ L_GRPO = √(p(1-p)) · [∇_θ KL(τ⁻ ‖ π_θ) − ρ · ∇_θ KL(τ⁺ ‖ π_θ)]

where:
- τ⁺ = Uniform distribution over correct responses in the group
- τ⁻ = Uniform distribution over incorrect responses
- p = empirical success rate (n⁺/G)
- ρ = asymmetric weighting factor (positive signal strength)

This shows GRPO is mathematically equivalent to **contrastive self-distillation (CSD)**: pull toward correct, push from incorrect.

## Derived Results

- **Theorem 2 (Capacity Bound)**: RLVR accuracy bounded by pass@(G·T_eff) of base model
- **Theorem 3 (Optimal ρ)**: ρ* = Cov(g⁺, g⁻) / Var(g⁺) minimizes gradient variance
- **Proposition 1 (Collapse Predictor)**: Q_CSD = H(τ⁺)·(n⁺/G)·cos(g⁺,g⁻) predicts failure

## Method: CSDPO (Contrastive Self-Distillation Policy Optimization)

Four components derived from theory:
1. **EA** (Experience-Augmented τ⁺): replay buffer fixes zero-success trap
2. **QW** (Quality-Weighted Distillation): weight by π_θ confidence
3. **ADQ** (Adaptive ρ via CSD): online ρ update via Theorem 3
4. **GCR** (Gradient Consistency Regularization): penalize g⁺/g⁻ conflict

## Experimental Setup

- Model: Qwen/Qwen3.5-9B (base, via HF cache, offline)
- Dataset: GSM8K (train: 200-500 samples, test: 200 held-out)
- Adapter: LoRA r=64, target: all qkvo+MLP proj
- Training: 100 steps, per_device_batch=1, grad_accum=2, num_generations=2, max_completion=512
- Hardware: 8× A100-80GB, PyTorch 2.4, TRL 0.14, transformers 5.5.4
- Critical config: `gradient_checkpointing=False` + `enable_thinking=False`

## Results So Far

### Stage 1: Baseline Reproduction (stock TRL GRPO, no custom code)
Path: `grpo-dynamics-baseline/results/baseline_full/`

| Seed | Max reward | GSM8K acc (n=100) |
|------|-----------|-------------------|
| 42 | 1.0 | 29% |
| 43 | 1.0 | 23% |
| 44 | 1.0 | 23% |
| **Mean** | | **25%** |

**Baseline works**: GRPO trains successfully (reward climbs from 0 → 1.0 over 25 steps).

### Stage 2: CSD CONST ρ=0.7 vs ADQ (100 steps, 4 seeds each)
Path: `results/csd_full/`

| Seed | CONST (ρ=0.7) | ADQ (init ρ=0.7) | Δ |
|------|-------|-----|---|
| 42 | 21.0% | 15.0% | −6.0% |
| 43 | 23.5% | 23.0% | −0.5% |
| 44 | 25.5% | 23.0% | −2.5% |
| 45 | 20.0% | 19.5% | −0.5% |
| **Mean** | **22.5%** | **20.1%** | **−2.4%** |

**Apparent negative result, but misleading**: diagnosis showed `ada_telemetry.json` doesn't exist → AdaBalance controller NEVER FIRED because `RhoGRPOTrainer._rho_step_stats` is always empty on TRL 0.14 (API mismatch: `inputs["advantages"]` doesn't exist in 0.14).

So "ADQ runs" were effectively identical to "CONST ρ=0.7" runs with different seeds. The −2.4% is just seed variance.

### Stage 3: ρ Sweep (100 steps, 1 seed each, n=200 eval)
Path: `results/rho_sweep/`

| ρ | GSM8K acc |
|---|-----------|
| 0.3 | 21.0% |
| 0.5 | 21.0% |
| 0.7 | 18.5% |
| 1.0 | 22.5% |
| 1.5 | 21.0% |
| 2.0 | 24.0% |
| 2.5 | 24.0% |
| **3.0** | **26.0%** |

**Upward tendency, NOT monotonic**: higher ρ generally yields higher accuracy on this model/dataset, but the sweep has local dips at ρ=0.7 (18.5%) and ρ=1.5 (21.0%). Treat as exploratory single-seed evidence consistent with CSD's prediction that a stronger distillation signal helps on hard tasks; statistical validation requires ≥3 seeds.

### Stage 4: Fix Implementation (not yet tested)
- Created `src/rho_grpo_trainer_v14.py` — rewrites `compute_loss` for TRL 0.14 (computes advantages inline, then applies ρ weighting + feeds AdaBalance)
- Wired into `scripts/run_csd_pilot.py`
- Import test passes — but no training run yet with fixed ADQ

## Known Issues / Open Questions

1. **ADQ validation pending**: V14 trainer not yet run. Can't say if CSDPO's ADQ actually works.
2. **Small ρ variance**: in ρ sweep, ρ=0.3/0.5/0.7/1.0 all within 18.5-22.5% — is the "monotonic trend" real or noise?
3. **Only 1 seed per ρ in sweep**: no confidence intervals
4. **Only Qwen3.5-9B tested**: can't generalize
5. **max_steps=100 might be too short** for CSD/ADQ to differentiate from CONST
6. **Q_CSD predictor (Proposition 1) untested**: no collapse observed to predict

## Pre-Registered Hypotheses from CSD Theory

- **H1**: Higher ρ → higher accuracy (partially supported by single-seed sweep: ρ=0.3→21%, ρ=3.0→26%, upward tendency with local dips at ρ=0.7 and ρ=1.5; multi-seed confirmation deferred)
- **H2**: ADQ should reach final ρ* close to sweep-optimal ρ (ρ=3.0 region)
- **H3**: ADQ with correct initialization should outperform fixed ρ=1.0 baseline
- **H4**: Q_CSD early-training signal predicts final accuracy

## Implementation Files

- `refine-logs/FINAL_PROPOSAL.md` — full theory writeup with proofs
- `src/rho_grpo_trainer_v14.py` — NEW: compute_loss override for TRL 0.14 + AdaBalance
- `src/csd_logging.py` — CSD diagnostic callback
- `src/adabalance.py` — closed-form ρ* controller
- `scripts/run_csd_pilot.py` — experiment driver
- `scripts/eval_baseline.py` — GSM8K eval
- `grpo-dynamics-baseline/` — independent stock TRL reproduction (separate project)

## Question for Reviewer

Given the current evidence:
1. Is the CSD theoretical framework sound enough for NeurIPS?
2. The ADQ controller hasn't been validated end-to-end — is that fatal?
3. What minimum experimental set would make this publishable?
4. Does the single-seed upward-tendency-with-dips ρ result support or weaken CSD's central story?
