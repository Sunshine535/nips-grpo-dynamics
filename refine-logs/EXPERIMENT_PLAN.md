# Experiment Plan: Stability Analysis of GRPO Signal Balance

**Source**: `refine-logs/FINAL_PROPOSAL.md` (Round 4, Score 9.1/10, READY)
**Date**: 2026-03-29
**Budget**: ~45 GPU-hours on 8×H100

---

## Core Claims

| ID | Claim | Strength | Key Evidence |
|----|-------|----------|--------------|
| C1 | The ρ-weighted GRPO stability analysis (Theorems 1-3, Proposition 1) predicts high-risk regimes with >85% accuracy | Strong | Exp 1: coarse+fine sweep, regime classification |
| C2 | AdaBalance achieves accuracy within 1% of oracle best-static ρ without hyperparameter search | Strong | Exp 2: 5 full runs, GSM8K+MATH accuracy |
| C3 | Stability boundaries degrade gracefully (<10% accuracy drop) under moderate i.i.d. violation | Moderate | Exp 3: GSM8K step-count binning |
| C4 | Stability boundaries transfer predictably across model scale (9B→27B) | Sanity | Exp 4: 3 representative ρ values on 27B |

---

## Assumptions (Must Validate)

- **(A1)** Binary verifiable rewards: r_i ∈ {0,1} — enforced by reward function design
- **(A2)** Within-group i.i.d. Bernoulli(p(x)) — approximately holds for random sampling; tested in Exp 3
- **(A3)** Per-prompt p(x) is sufficient statistic — by construction under A1-A2

---

## Experiment 1: Stability Prediction Accuracy

### Objective
Validate that theoretical stability boundaries (Theorems 1-3, Proposition 1) predict empirical training outcomes.

### Protocol

**Stage 1a — Coarse Sweep (5.4 GPU-hours)**
- Grid: ρ ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0} × seed ∈ {42, 43, 44} × short training (200 steps)
- 27 coarse runs (some ρ values doubled for boundary regions)
- Total: 54 short runs × 0.1 GPU-hr = 5.4 GPU-hours
- Collect: per-step reward mean, reward std, KL divergence, gradient norms, p_0 (zero-group rate), accuracy at step 200
- Classify each run as: convergent / gradient-starved / unstable using joint collapse conditions (Definition 4 in paper Section 4.6)

**Stage 1b — Fine Sweep (15 GPU-hours)**
- Select 10 critical points near predicted boundaries (ρ_min, ρ_max)
- Full training (2 epochs, ~600 steps) per point
- 10 fine runs × 1.5 GPU-hr = 15 GPU-hours
- Collect same metrics + final GSM8K accuracy

### Metrics
- **Regime classification accuracy**: fraction of runs where predicted regime matches empirical outcome (target: >85%)
- **Rank correlation (Spearman)**: between predicted stability score and empirical accuracy (target: >0.8)
- **Boundary precision**: predicted ρ_min/ρ_max vs empirical boundary location (within 15%)

### Ablation
- ρ-only prediction (ignore p_0) — expect accuracy drop ~15%
- p_0-only prediction (ignore ρ) — expect accuracy drop ~25%
- Combined (ρ, p_0) prediction — full model

### Controls
- Standard GRPO baseline (ρ=1.0) always included
- Consistent data ordering across all runs (fixed data seed=0)

---

## Experiment 2: AdaBalance Competitiveness

### Objective
Show AdaBalance matches oracle best-static ρ without grid search.

### Protocol (7.5 GPU-hours)
5 full training runs, each 2 epochs on GSM8K:

| Run | Method | ρ | Details |
|-----|--------|---|---------|
| 1 | Vanilla GRPO | 1.0 | Standard TRL GRPOTrainer |
| 2 | Best-static ρ (oracle) | ρ* from Exp 1 | Best fixed ρ from coarse+fine sweep |
| 3 | AdaBalance | adaptive | K=50, τ=0.1; ρ updated online per Corollary 1 |
| 4 | Linear ρ scheduler | 0.5→2.0 | Linear warmup over training |
| 5 | GTPO-style | — | Skip zero-score groups entirely |

Each run: 3 seeds × 1.5 GPU-hr = 4.5; plus analysis overhead → 7.5 GPU-hours total

### Metrics
- GSM8K accuracy (test set, 500 samples, greedy decoding)
- MATH accuracy (test set, 200 samples, greedy decoding)
- p_0 trajectory over training
- ρ trajectory (for AdaBalance)
- KL divergence trajectory
- Gradient norm trajectory

### Success Criteria
- AdaBalance GSM8K accuracy within 1% of oracle best-static ρ
- AdaBalance > vanilla GRPO by ≥2%
- AdaBalance converges to stable ρ trajectory (no oscillation >20% after step 100)

### Ablation: AdaBalance Hyperparameters
| K (update interval) | τ (EMA smoothing) | Expected Behavior |
|---------------------|--------------------|--------------------|
| 10 | 0.1 | Noisy ρ updates, may oscillate |
| 50 | 0.1 | Default: stable, responsive |
| 100 | 0.1 | Smoother but slower adaptation |
| 50 | 0.05 | Faster response, more noise |
| 50 | 0.2 | Slower response, more stable |

---

## Experiment 3: Robustness Under i.i.d. Violation

### Objective
Test stability prediction accuracy when Assumption A2 (group i.i.d.) is violated.

### Protocol (2.4 GPU-hours)

**Data Construction:**
- Bin GSM8K training problems by number of reasoning steps (1-2, 3-4, 5-6, 7+)
- Within each bin, problems have correlated difficulty → violates i.i.d. within groups
- Form groups by sampling from same bin → within-group reward correlation > 0

**Runs:**
- 4 bins × 3 ρ values (near ρ_min, ρ*_optimal, near ρ_max) × 2 seeds = 24 short runs
- Each: 200 steps × 0.1 GPU-hr = 2.4 GPU-hours

### Metrics
- Stability prediction accuracy per bin
- Within-group reward correlation (measured from training data)
- Accuracy degradation vs correlation strength (scatter plot)

### Success Criteria
- <10% accuracy drop for moderate within-group correlation (r < 0.3)
- Graceful degradation curve (monotonically decreasing, no cliff)
- Clear correlation between violation severity and prediction error

---

## Experiment 4: Scale Transfer (27B Sanity Check)

### Objective
Verify stability boundaries shift predictably from 9B to 27B.

### Protocol (9 GPU-hours)
- 3 representative ρ values: one near ρ_min (expect starved), one at ρ* (expect convergent), one near ρ_max (expect unstable)
- Qwen3.5-27B, GSM8K, 2 epochs each
- 3 runs × 3 GPU-hr = 9 GPU-hours

### Metrics
- Same regime classification as Exp 1
- Boundary shift direction and magnitude
- Correlation of p_0 trajectories between 9B and 27B

### Success Criteria
- Predicted regime matches for at least 2/3 runs
- Boundaries shift in predicted direction (larger models tolerate wider ρ range)
- NOT claimed as general scaling law — sanity check only

---

## Threshold Sensitivity Analysis

### Objective
Show collapse classification is robust to threshold perturbation.

### Protocol (included in Exp 1 analysis, no extra GPU)
- Re-classify all Exp 1 runs with varied thresholds:
  - p_0 threshold ∈ {0.7, 0.8, 0.9} (default: 0.8)
  - KL multiplier ∈ {1.5, 2.0, 3.0} (default: 2.0)
- 3×3 = 9 threshold combinations

### Metrics
- Classification agreement with default thresholds
- Fleiss' κ across threshold combinations

### Success Criteria
- ≥80% agreement for ±20% threshold variation
- κ > 0.7 (substantial agreement)

---

## Run Order & Dependencies

```
Phase 0: Model download (Qwen3.5-9B, 27B)
    ↓
Phase 1: Exp 1a — Coarse ρ sweep (54 short runs)         [5.4 GPU-hr]
    ↓
Phase 2: Stability analysis — compute theoretical boundaries from Exp 1a telemetry
    ↓
Phase 3: Exp 1b — Fine sweep near boundaries (10 full runs) [15 GPU-hr]
    ↓
Phase 4: Compute oracle best-static ρ from Exp 1
    ↓
Phase 5: Exp 2 — AdaBalance comparison (5 methods × 3 seeds) [7.5 GPU-hr]
    ↓  (can parallelize with Phase 6)
Phase 6: Exp 3 — i.i.d. robustness (24 short runs)          [2.4 GPU-hr]
    ↓
Phase 7: Exp 4 — 27B transfer (3 runs)                       [9 GPU-hr]
    ↓
Phase 8: Analysis — all figures, tables, threshold sensitivity
    ↓
Phase 9: Paper writing
```

**Total GPU budget**: 5.4 + 15 + 7.5 + 2.4 + 9 + 5 = **44.3 GPU-hours** (within 45 budget)

---

## Output Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Coarse sweep results | `results/sweep_coarse/` | Per-run metrics JSON + telemetry |
| Fine sweep results | `results/sweep_fine/` | Full training metrics |
| Stability analysis | `results/stability_analysis/` | Theoretical boundaries, regime maps |
| AdaBalance comparison | `results/adabalance/` | 5 method × 3 seed results |
| Robustness test | `results/robustness/` | Per-bin prediction accuracy |
| 27B validation | `results/validation_27b/` | 3-run transfer check |
| Threshold sensitivity | `results/threshold_sensitivity/` | 9-combination agreement matrix |
| Figures | `results/figures/` | All paper-ready figures |
| Paper | `paper/` | LaTeX source + compiled PDF |

---

## Diagnostic Telemetry (Logged Every K Steps)

All training runs log the following to `step_metrics.json`:

```json
{
  "step": 100,
  "rho": 1.0,
  "p_0": 0.35,
  "reward_mean": 0.42,
  "reward_std": 0.49,
  "kl_divergence": 0.15,
  "grad_norm": 1.23,
  "grad_pos_norm": 0.89,
  "grad_neg_norm": 0.67,
  "V_plus": 0.012,
  "V_minus": 0.008,
  "C_pG": -0.003,
  "rho_min": 0.4,
  "rho_max": 3.2,
  "rho_star": 1.1,
  "GSR": 0.15,
  "entropy": 2.34,
  "accuracy_running": 0.38
}
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Proposition 1 upper bound too loose | Report calibration factor; honest about "approximate" |
| AdaBalance oscillates | EMA smoothing (τ); ablate K values |
| p_0 measurement noisy for small groups | Use running average (window=50); report confidence intervals |
| 27B too expensive if GPU time runs low | Drop 27B (marked as "sanity", not central claim) |
| i.i.d. violation too strong in some bins | Report correlation strength; exclude bins with r > 0.5 |

---

## Paper Figure List (Planned)

| Figure | Content | Source |
|--------|---------|--------|
| Fig 1 | Stability Map: (ρ, p_0) with three regime annotations | Exp 1 |
| Fig 2 | Regime classification confusion matrix | Exp 1 |
| Fig 3 | Training dynamics (reward, KL, p_0) for representative ρ values | Exp 1 |
| Fig 4 | AdaBalance vs baselines: accuracy curves | Exp 2 |
| Fig 5 | AdaBalance ρ trajectory over training | Exp 2 |
| Fig 6 | Robustness: prediction accuracy vs within-group correlation | Exp 3 |
| Table 1 | Main results: GSM8K/MATH accuracy for all methods | Exp 2 |
| Table 2 | Regime classification accuracy with ablations | Exp 1 |
| Table 3 | AdaBalance hyperparameter ablation | Exp 2 |
| Table 4 | Threshold sensitivity (Fleiss' κ) | Threshold analysis |
| Table 5 | 27B transfer results | Exp 4 |
