# Round 3 Refinement

## Problem Anchor
(Verbatim — unchanged.)

## Anchor Check
- **Original bottleneck**: Understanding GRPO signal balance stability under binary rewards.
- **Still on target?**: YES — we correct overclaims and add robustness, not new components.
- **Rejected as drift**: NONE.

## Simplicity Check
- **Dominant contribution**: A reduced-model stability analysis for GRPO under binary rewards.
- **Removed/merged**: Corrected E[∇L] claim; softened "stability law" to "stability analysis."
- **Rejected as complexity**: NONE.

## Changes Made

### 1. Corrected E[∇L] preservation claim
- **Reviewer said**: ρ* minimizes Var of modified L_ρ, not original L. Overclaimed.
- **Action**: Corrected. ρ* minimizes Var(∇̂L_ρ) for the ρ-weighted objective, not the original GRPO objective. The correct interpretation: choosing ρ defines a FAMILY of GRPO objectives; ρ* is the member of this family with minimum gradient variance. This is analogous to choosing the learning rate — you're selecting the best member of a family, not reducing variance of a fixed target. We do NOT claim variance reduction for the original GRPO objective.
- **Impact**: Honest and correct. The practical benefit is clear: ρ* is the family member that learns most efficiently.

### 2. Softened "stability law" to "stability analysis"
- **Reviewer said**: Overclaimed for a reduced model.
- **Action**: Changed throughout: "stability law" → "stability analysis" or "stability characterization." The theorems are valid under the stated assumptions (A1-A3) as a reduced model. We explicitly state: "The stability analysis provides a reduced-model approximation that predicts high-risk regimes, not a complete description of all GRPO failure modes."
- **Impact**: Defensible claims.

### 3. Added robustness test for i.i.d. violation
- **Reviewer said**: Need one test when group-i.i.d. is imperfect.
- **Action**: Added Experiment 3: "Robustness under i.i.d. violation." Use a dataset with prompt-conditional difficulty variation (e.g., GSM8K subsets grouped by number of reasoning steps). Within each group, reward correlations violate pure Bernoulli i.i.d. Compare predicted stability boundaries to empirical outcomes. Report degradation of prediction quality as a function of within-group correlation.
- **Impact**: Addresses the theory-to-practice bridge concern.

### 4. Corrected "zero parameters" to "two controller hyperparameters"
- **Reviewer said**: K and τ are hyperparameters.
- **Action**: Changed to "two controller hyperparameters (K=50, τ=0.1) with principled defaults derivable from the stability analysis." These are not learned; they are analogous to learning rate warmup steps.
- **Impact**: Honest.

### 5. Added σ(m) floor handling and threshold sensitivity
- **Reviewer said**: Define floor for degenerate groups; sensitivity analysis for collapse thresholds.
- **Action**: σ(m) floor δ = 1/G (prevents division by zero). Collapse threshold sensitivity: report results with p_0 thresholds {0.7, 0.8, 0.9} and KL multipliers {1.5×, 2×, 3×}. Show classification accuracy is robust to ±20% threshold variation.
- **Impact**: Rigorous.

## Final Revised Proposal

# Stability Analysis of GRPO Signal Balance Under Binary Verifiable Rewards

## Problem Anchor
(Verbatim.)

## Technical Gap
No existing work characterizes the stability landscape of GRPO as a function of the positive/negative signal balance. Mroueh (2025) and Zhou et al. (2026) analyze GRPO dynamics without addressing signal balance. GTPO/DaGRPO/TR-GRPO provide ad-hoc fixes without a unifying stability characterization.

## Method Thesis
Under binary verifiable rewards and group-internal i.i.d. sampling (Assumptions A1-A3), we derive a reduced-model stability analysis of the GRPO policy gradient in terms of the effective balance ratio ρ and the zero-group rate p_0. This analysis predicts high-risk regimes (gradient starvation, instability) and yields a principled adaptive controller with two hyperparameters.

## Contribution Focus
- **Dominant**: Stability analysis (Theorems 1-3, Proposition 1) characterizing convergent/starved/unstable regimes as f(ρ, p_0, G, λ_KL, ε) under explicit assumptions. A reduced-model approximation that predicts high-risk regimes, not a complete description of all GRPO failure modes.
- **Supporting (corollary)**: AdaBalance — minimum-variance controller for the ρ-weighted GRPO objective family, with two hyperparameters (K, τ).
- **Non-contributions**: No new RL algorithm, no token-level, no continuous rewards. Generalization beyond binary rewards is stated as future work.

## Method

### Setup
GRPO, G samples/group, binary r ∈ {0,1}, KL coeff λ_KL, clip ε.
Assumptions: (A1) binary rewards, (A2) group i.i.d. Bernoulli(p(x)), (A3) p(x) sufficient.
m ~ Bin(G, p(x)). σ(m) = √(m(G-m))/G, floored at δ = 1/G.

ρ-weighted advantage: ã_i = ρ·a_i if r_i=1, else a_i (ρ=1 = standard GRPO).
ρ enters AFTER group normalization, BEFORE clipped surrogate in TRL GRPOTrainer.

### Theoretical Results
- **Thm 1**: m=0 or m=G → zero gradient (degenerate starvation).
- **Thm 2**: Var(∇̂L_ρ|x) = ρ²V_+ + V_- + 2ρC, functions of Bin(G,p) and per-sample gradient variance.
- **Thm 3**: Lower stability bound ρ_min(p,G) from gradient-starvation condition. Sharp, depends only on p, G.
- **Prop 1**: Upper instability bound ρ_max(p,G,λ_KL,ε). Approximate, depends on KL/clip.
- **Cor 1**: ρ* = -C/V_+ minimizes Var(∇̂L_ρ) within the ρ-weighted objective family. NOT claimed as variance reduction for original GRPO objective. Interpretation: ρ* selects the most efficient member of the {L_ρ : ρ > 0} family.

### AdaBalance Controller
Two hyperparameters: K=50 (update interval), τ=0.1 (EMA rate). Defaults derivable from stability analysis.
Reads group success counts from trainer. Computes p_0 EMA. Updates ρ via Cor 1 formula. 3-line hook in TRL.

### Collapse Definitions
Joint conditions:
1. Gradient starvation: p_0 > 0.8 AND ΔKL > 2× in 50 steps
2. KL blow-up: KL > 3× initial AND no reward improvement
3. Entropy+reward: stagnation >100 steps AND entropy drop >50%
Sensitivity analysis: report with thresholds ±20%.

### Stability Map
2D plot of (ρ, p_0) with three zones and theoretical boundaries overlaid. Reduced-model predictions vs empirical outcomes.

## Experiments

### Exp 1: Stability analysis predicts high-risk regimes
- Two-stage sweep: 54 coarse + 10 fine runs on Qwen3.5-9B/GSM8K
- Metric: Regime classification accuracy (>85%), rank correlation (>0.8) between predicted instability score and actual training divergence
- Ablation: p_0-only prediction (no ρ), ρ-only prediction (no p_0)

### Exp 2: AdaBalance competitive without search
- 5 full runs: AdaBalance, oracle best-static, vanilla GRPO, linear scheduler, GTPO
- Metric: GSM8K accuracy (within 1% of oracle), MATH accuracy, p_0 trajectory
- Ablation: AdaBalance with K ∈ {10, 50, 100}, τ ∈ {0.05, 0.1, 0.2}

### Exp 3: Robustness under i.i.d. violation
- GSM8K subsets grouped by reasoning-step count (violates pure Bernoulli i.i.d.)
- Compare predicted boundaries to empirical outcomes
- Report prediction accuracy degradation as function of within-group reward correlation
- Expected: predictions degrade gracefully (accuracy drops <10% for moderate correlation)

### Exp 4: Scaling check (27B)
- 3 ρ values on Qwen3.5-27B
- Verify boundaries shift predictably with scale
- Not central — a transfer sanity check

### Threshold sensitivity analysis
- Report collapse classification with p_0 ∈ {0.7, 0.8, 0.9}, KL multiplier ∈ {1.5, 2.0, 3.0}
- Show robustness to ±20% variation

## Compute: ~45 GPU-hours. Timeline: 14 days total.

## Novelty Argument
First stability characterization of GRPO signal balance. Distinguished from all prior work by: (1) identifying ρ as the effective control variable, (2) conditioning on group outcome m/G under explicit assumptions, (3) deriving both lower bound (theorem) and upper guideline (proposition), (4) yielding a practical controller as a corollary.
