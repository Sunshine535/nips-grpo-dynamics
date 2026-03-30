# Stability Analysis of GRPO Signal Balance Under Binary Verifiable Rewards

## Problem Anchor
- **Bottom-line problem**: GRPO (Group Relative Policy Optimization) is the dominant RL post-training algorithm for LLM reasoning (DeepSeek-R1, Qwen3), yet practitioners lack principled guidance for configuring the positive/negative signal balance. The training landscape contains hidden stability regimes between healthy convergence and catastrophic collapse, but no theoretical or empirical framework exists to map and navigate these regimes.
- **Must-solve bottleneck**: Current GRPO tuning is trial-and-error. Existing fixes (GTPO, DaGRPO, TR-GRPO) each address one symptom without understanding the underlying geometry of the training landscape.
- **Non-goals**: (1) Not building a new RL algorithm. (2) Not multi-objective. (3) Not token-level credit assignment. (4) Not continuous rewards.
- **Constraints**: 8×H100 GPUs, ~300 GPU-hours budget, Qwen3.5-9B primary, 27B validation. GSM8K + MATH.
- **Success condition**: (1) A theory-backed stability analysis that predicts high-risk regimes from first principles. (2) An adaptive controller derived from the analysis. (3) Distinct from GTPO/DaGRPO/TR-GRPO.

## Technical Gap
No existing work characterizes the stability landscape of GRPO as a function of the positive/negative signal balance. Mroueh (2025) derives GRPO dynamics and success amplification but does not address signal balance. Zhou et al. (2026) show GRPO's gradient is a U-statistic and derive optimal group size but say nothing about the signal-balance dimension. GTPO/DaGRPO/TR-GRPO provide ad-hoc fixes (skip negatives, mask pairs, token reweighting) without a unifying stability characterization.

## Method Thesis
Under binary verifiable rewards and group-internal i.i.d. sampling (Assumptions A1-A3), we derive a reduced-model stability analysis of the GRPO policy gradient in terms of the effective balance ratio ρ and the zero-group rate p_0. This analysis predicts high-risk regimes (gradient starvation, instability) and yields a principled adaptive controller with two hyperparameters. The stability analysis is a reduced-model approximation that predicts high-risk regimes, not a complete description of all GRPO failure modes.

## Contribution Focus
- **Dominant contribution**: Stability analysis (Theorems 1-3, Proposition 1) for GRPO under binary rewards, characterizing convergent/starved/unstable regimes as f(ρ, p_0, G, λ_KL, ε) under explicit assumptions (A1-A3).
- **Supporting contribution (corollary)**: AdaBalance — minimum-variance controller for the ρ-weighted GRPO objective family, with two hyperparameters (K=50, τ=0.1) with principled defaults.
- **Non-contributions**: No new RL algorithm, no token-level mechanism, no off-policy augmentation, no continuous-reward generalization.

## Proposed Method

### Formal Setup

**Setting**: GRPO with group size G, binary reward r_i ∈ {0,1}, KL coefficient λ_KL, clip range ε.

**Assumptions**:
- (A1) Rewards are binary verifiable: r_i ∈ {0,1}
- (A2) Within a group for prompt x, rewards are i.i.d. Bernoulli(p(x))
- (A3) The per-prompt success probability p(x) is the sufficient statistic for reward distribution

Under (A1-A3), the group success count m ~ Binomial(G, p(x)).

**Standard GRPO advantage** for sample i in group with outcome m:
  a_i(m) = (r_i - m/G) / max(σ(m), δ)
  where σ(m) = √(m(G-m))/G, δ = 1/G (floor for degenerate groups)

**ρ-weighted advantage**:
  ã_i(m) = ρ · a_i(m)  if r_i = 1
  ã_i(m) = a_i(m)      if r_i = 0
  where ρ > 0 is the effective balance ratio (ρ=1 recovers standard GRPO)

**Modified GRPO objective** (ρ enters AFTER group-level normalization, BEFORE clipped surrogate):
  L_ρ(θ) = E_x E_{m~Bin(G,p(x))} [ Σ_{i=1}^G  min(r_θ · ã_i, clip(r_θ, 1-ε, 1+ε) · ã_i) ] + λ_KL · KL(π_θ || π_ref)

### Theoretical Results

**Theorem 1 (Degenerate Group Starvation)**: For m=0 or m=G, a_i(m) = 0 for all i. These groups contribute zero gradient regardless of ρ.

**Theorem 2 (Gradient Variance Decomposition)**: Under (A1-A3), the variance of the ρ-weighted GRPO gradient estimator for a single prompt x with success probability p = p(x) is:
  Var(∇̂L_ρ | x) = ρ² · V_+(p, G) + V_-(p, G) + 2ρ · C(p, G)
where V_+(p,G), V_-(p,G), C(p,G) are functions of the binomial distribution Bin(G,p) and per-sample gradient variance, computable from trainer telemetry.

**Theorem 3 (Lower Stability Bound — Gradient Starvation)**: Training suffers gradient starvation when the zero-group rate GSR(p,G) = (1-p)^G + p^G exceeds threshold τ_star. The minimum ρ to maintain positive signal:
  ρ_min(p, G) = V_-(p, G) / (2|C(p, G)|) when C < 0
Depends only on p, G, and gradient statistics — sharp result, no extra hyperparameters.

**Proposition 1 (Upper Instability Bound — approximate)**: Training becomes unstable when ρ exceeds:
  ρ_max(p, G, λ_KL, ε) ≈ (1/ε) · (λ_KL / ||∇_+||) · σ(m̄)
This upper bound additionally depends on λ_KL and ε. It is an empirically-calibrated guideline, not a sharp theorem. (Labeled approximate everywhere.)

**Corollary 1 (AdaBalance — Minimum-Variance ρ*)**: The ρ that minimizes Var(∇̂L_ρ) within the {L_ρ : ρ > 0} objective family:
  ρ* = -C(p, G) / V_+(p, G)
Estimable online from the success-count histogram at cost O(G) per group. NOTE: ρ* minimizes variance of the MODIFIED objective L_ρ, not the original GRPO objective. Interpretation: selecting the most efficient member of the {L_ρ} family, analogous to choosing the best learning rate.

### Stability Map
2D visualization of (ρ, p_0) annotated with three regimes:
- **Convergent**: ρ ∈ [ρ_min, ρ_max], moderate p_0
- **Gradient-starved**: below ρ_min or p_0 too high
- **Unstable**: above ρ_max
Theoretical boundaries from Theorem 3 and Proposition 1 (approximate), validated by sweep.

### AdaBalance Controller
Callback in TRL GRPOTrainer. Reads group success counts. Computes p_0 EMA. Updates ρ every K=50 steps via Corollary 1 formula with EMA smoothing (τ=0.1). Two hyperparameters with principled defaults derivable from the stability analysis.

### Collapse Definitions (Mechanistic, with cross-checks)
Collapse requires JOINT conditions:
1. **Gradient starvation**: p_0 > 0.8 AND ΔKL > 2× baseline in last 50 steps
2. **KL divergence blow-up**: KL > 3× initial AND reward not improving
3. **Entropy + reward**: stagnation >100 steps AND entropy drop >50%
Pure high p_0 without KL/entropy change = hard task, not collapse.
Threshold sensitivity: report with ±20% variation.

## Experiments

### Exp 1: Stability analysis predicts high-risk regimes
- Two-stage sweep on Qwen3.5-9B/GSM8K: 54 coarse (short) + 10 fine (full) runs
- Metric: Regime classification accuracy (>85%), rank correlation (>0.8)
- Ablation: p_0-only prediction, ρ-only prediction

### Exp 2: AdaBalance competitive without search
- 5 full runs: AdaBalance, oracle best-static ρ, vanilla GRPO (ρ=1), linear scheduler, GTPO
- Metric: GSM8K accuracy (within 1% oracle), MATH accuracy, p_0 trajectory
- Ablation: K ∈ {10, 50, 100}, τ ∈ {0.05, 0.1, 0.2}

### Exp 3: Robustness under i.i.d. violation
- GSM8K subsets grouped by reasoning-step count (within-group reward correlations violate Bernoulli i.i.d.)
- Construction: bin problems by number of solution steps (1-2, 3-4, 5-6, 7+), form groups from same bin
- Compare predicted stability boundaries to empirical outcomes
- Report prediction accuracy degradation vs within-group correlation
- Expected: graceful degradation (<10% accuracy drop for moderate correlation)

### Exp 4: Scaling transfer check (27B, not central)
- 3 representative ρ values on Qwen3.5-27B
- Verify boundaries shift predictably with model scale
- Presented as transfer sanity check, not scaling evidence

### Threshold sensitivity analysis
- Collapse classification with p_0 ∈ {0.7, 0.8, 0.9}, KL multiplier ∈ {1.5, 2.0, 3.0}
- Show robustness to ±20% threshold variation

## Compute & Timeline
- Coarse sweep: 54 runs × 0.1 GPU-hr = 5.4 GPU-hours
- Fine sweep: 10 runs × 1.5 GPU-hr = 15 GPU-hours
- AdaBalance + baselines: 5 runs × 1.5 = 7.5 GPU-hours
- Robustness test: 4 subsets × 3 ρ × 2 seeds = 24 short runs × 0.1 = 2.4 GPU-hours
- 27B: 3 runs × 3 GPU-hr = 9 GPU-hours
- Analysis + diagnostics: 5 GPU-hours
- **Total: ~45 GPU-hours** (well within 300 GPU-hour budget)
- **Timeline**: Sweep (3d) → Analysis + theory validation (2d) → AdaBalance (2d) → Robustness + 27B (2d) → Paper (5d) = 14 days

## Novelty and Elegance Argument
First stability characterization of GRPO signal balance. Distinguished from all prior work by:
1. Identifying ρ as the effective control variable (vs ad-hoc α/β, GTPO's binary skip, DaGRPO's pair masking, TR-GRPO's token reweighting)
2. Conditioning on group outcome m/G under explicit assumptions (A1-A3)
3. Deriving both sharp lower bound (Theorem 3) and approximate upper guideline (Proposition 1)
4. Yielding a practical 2-hyperparameter controller as a direct corollary

The contribution is a geometry-of-optimization result, not a new trick.
