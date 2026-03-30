# Research Proposal: Phase Diagrams of GRPO Training Dynamics — Theory, Geometry, and Adaptive Navigation

## Problem Anchor
- **Bottom-line problem**: GRPO (Group Relative Policy Optimization) is the dominant RL post-training algorithm for LLM reasoning (DeepSeek-R1, Qwen3), yet practitioners lack principled guidance for configuring the positive/negative signal balance. The training landscape contains hidden phase transitions between healthy convergence and catastrophic collapse, but no theoretical or empirical framework exists to map and navigate these transitions.
- **Must-solve bottleneck**: Current GRPO tuning is trial-and-error. When the positive-signal ratio α and negative-signal weight β are misconfigured, training silently enters collapse zones (entropy collapse, reward hacking, or gradient starvation). Existing fixes (GTPO, DaGRPO, TR-GRPO) each address one symptom without understanding the underlying geometry of the training landscape.
- **Non-goals**: (1) We do NOT aim to build a new RL algorithm from scratch; GRPO is our object of study. (2) We do NOT aim to solve multi-objective reward hacking (MO-GRPO). (3) We do NOT aim to address token-level credit assignment (GRPO-λ).
- **Constraints**: 8×H100 GPUs, 2-week training budget (~300 GPU-hours), Qwen3.5-9B as primary model, Qwen3.5-27B for scaling validation. GSM8K primary + MATH/ARC-Challenge secondary.
- **Success condition**: (1) A theoretically grounded phase diagram that predicts collapse boundaries from first principles (validated by experiments). (2) An adaptive navigation algorithm that matches or beats the best static (α,β) without grid search. (3) Clear, focused contribution distinct from GTPO/DaGRPO/TR-GRPO.

## Technical Gap

### Why current methods fail
Existing GRPO stabilization methods each target a single failure mode without understanding the global geometry:
- **GTPO** (Wen et al., 2025): Skips negative updates and filters high-entropy completions. Ad-hoc — does not explain *when* negative signals become destructive.
- **DaGRPO** (2025): Masks low-distinctiveness pairs and augments with off-policy data. Addresses gradient conflict but not the structural cause of zero-signal collapse.
- **TR-GRPO** (2026): Token-level probability-correlated weighting. Fixes gradient over-amplification at token level but ignores the sequence-level positive/negative balance.
- **Mroueh (2025)**: Proves GRPO amplifies PoS and derives dynamics. Beautiful theory but does not address the stability landscape — which (α,β) configurations converge and which collapse.
- **Zhou et al. (2026)**: Shows GRPO gradient is a U-statistic. Provides optimal group size but says nothing about the signal balance between positive and negative samples.

### The missing piece
No existing work provides a **geometrical understanding of the GRPO training landscape** as a function of the positive/negative signal balance. We need:
1. A formal connection between (α,β) and the **gradient variance** of the GRPO policy gradient
2. Conditions under which the U-statistic estimator's variance explodes (phase boundaries)
3. An **adaptive** algorithm that reads the local gradient geometry and automatically navigates away from collapse zones

### Why naive fixes are insufficient
Simply searching (α,β) by grid sweep is combinatorially expensive and doesn't transfer across models or datasets. Ad-hoc clipping/filtering (GTPO) can accidentally suppress useful negative signal. We need a principled, geometry-aware approach.

## Method Thesis
- **One-sentence thesis**: The GRPO training landscape admits a low-dimensional phase diagram in (α,β) space where collapse boundaries are theoretically predictable from the gradient variance of the U-statistic policy gradient, enabling an adaptive navigator that avoids collapse without hyperparameter search.
- **Why this is the smallest adequate intervention**: We modify only the signal weighting mechanism in GRPO (2 scalar parameters α,β) and their scheduling. No new modules, no architecture changes, no off-policy augmentation.
- **Why this route is timely in the foundation-model era**: As GRPO becomes the default post-training recipe for reasoning LLMs, understanding its training dynamics at a phase-diagram level is immediately actionable for every team training reasoning models.

## Contribution Focus
- **Dominant contribution**: A theoretical and empirical phase diagram of GRPO training dynamics, showing that the positive/negative signal balance (α,β) controls a variance-bias tradeoff in the U-statistic policy gradient, with provable phase transition boundaries between convergence and collapse.
- **Supporting contribution**: An adaptive navigation algorithm (AdaBalance) that uses online gradient statistics to automatically adjust (α,β) during training, matching or exceeding the best static configuration found by exhaustive search.
- **Explicit non-contributions**: We do not propose a new RL algorithm, do not address token-level credit assignment, do not handle multi-objective rewards.

## Proposed Method

### Complexity Budget
- **Frozen / reused backbone**: Qwen3.5-9B/27B, TRL GRPOTrainer, LoRA fine-tuning
- **New trainable components**: ZERO new parameters. Only an adaptive scheduler for (α,β) that reads gradient statistics.
- **Tempting additions intentionally not used**: Token-level reweighting (TR-GRPO), off-policy augmentation (DaGRPO), entropy filtering (GTPO). These are baselines to compare against, not components to add.

### System Overview
```
Input: Prompt batch from GSM8K/MATH
  → GRPO generates G completions per prompt
  → Binary reward r_i ∈ {0,1} (correctness check)
  → Compute advantages a_i = (r_i - μ_r) / σ_r  (standard GRPO)
  → Apply signal balance: ã_i = α·a_i·𝟙[r_i>0] + (1-α)·β·a_i·𝟙[r_i=0]
  → Policy gradient: ∇L = E[ã_i · ∇log π(y_i|x)]
  → AdaBalance reads gradient variance ratio V(zero)/V(nonzero) → updates (α,β)
  → Next training step
```

### Core Mechanism: Phase Diagram Theory
**Input**: GRPO training configuration (α, β, group size G, model π)
**Output**: Predicted phase (convergent / transitional / collapsed) + gradient variance bound

#### Theoretical Foundation
Building on Mroueh (2025) and Zhou et al. (2026):

1. **GRPO gradient as weighted U-statistic**: The policy gradient under (α,β) weighting is:
   ∇J(α,β) = E_x [ U_G^{α,β}(x) ] where U_G is a U-statistic of order 2 over G samples.

2. **Variance decomposition**: 
   Var(U_G^{α,β}) = α²·Var_+(x) + (1-α)²·β²·Var_0(x) + cross-terms
   where Var_+ is the gradient variance from positive-reward samples and Var_0 from zero-score samples.

3. **Phase transition condition**: Collapse occurs when the zero-score variance term dominates:
   (1-α)²·β²·Var_0(x) >> α²·Var_+(x)
   This gives a theoretical boundary curve in (α,β) space.

4. **Gradient starvation condition**: When zero-score ratio p_0 → 1, the positive-signal term α²·(1-p_0)·Var_+ → 0, leading to gradient starvation regardless of β.

5. **Optimal balance**: The minimum-variance (α*,β*) satisfies:
   α* = √(Var_0) / (√(Var_+) + √(Var_0)), β* = √(Var_+)/√(Var_0)

### AdaBalance: Adaptive (α,β) Navigator

Instead of static (α,β), AdaBalance updates them every K steps using online estimates of the gradient variance decomposition:

```python
# Every K steps:
V_pos = running_var(grad_norms[positive_samples])
V_zero = running_var(grad_norms[zero_samples])
p_zero = running_mean(zero_score_ratio)

alpha_new = sqrt(V_zero) / (sqrt(V_pos) + sqrt(V_zero) + eps)
beta_new = sqrt(V_pos) / (sqrt(V_zero) + eps)

# Exponential moving average for stability
alpha = (1-tau) * alpha + tau * alpha_new
beta = (1-tau) * beta + tau * beta_new
```

**Why this is the main novelty**: Unlike GTPO (which skips negatives entirely) or DaGRPO (which masks pairs), AdaBalance continuously adjusts the signal balance based on the local geometry of the gradient landscape. It's derived from the minimum-variance condition of the U-statistic estimator, not from heuristics.

### Modern Primitive Usage
- **Which LLM/RL primitive**: TRL's GRPOTrainer (standard GRPO implementation)
- **Exact role**: The backbone RL training loop. We modify only the advantage reweighting and scheduler, not the core algorithm.
- **Why more natural than old-school alternative**: GRPO is the current standard for reasoning RL. Our contribution is understanding its dynamics, not replacing it.

### Integration into Base Generator / Downstream Pipeline
- AdaBalance is a **callback** attached to GRPOTrainer. It reads logged gradient statistics every K steps and updates (α,β) in the reward shaping function.
- No changes to model architecture, tokenizer, or data pipeline.
- At inference time, only the final trained model is used — AdaBalance is purely a training-time algorithm.

### Training Plan
1. **Phase Diagram Construction** (empirical validation of theory):
   - Grid sweep: α ∈ {0.1, 0.2, ..., 0.9}, β ∈ {0.0, 0.25, 0.5, 1.0, 2.0} — 45 points × 3 seeds = 135 runs
   - Each run: 2 epochs on GSM8K train, LoRA r=64, batch=2, grad_accum=4
   - Collect: accuracy, gradient norms, zero-score ratio, KL divergence, reward dynamics
   - Construct phase diagram heatmap + boundary detection (Sobel filter on accuracy surface)
   - Compare predicted boundaries (from theory) with empirical boundaries

2. **AdaBalance Training**:
   - Run AdaBalance with K=50 (update interval), τ=0.1 (EMA rate)
   - Compare against: best static (α,β), GTPO, DaGRPO, standard GRPO, curriculum schedules
   - Evaluate on GSM8K test + MATH + ARC-Challenge

3. **Scaling Validation**:
   - Repeat key experiments with Qwen3.5-27B
   - Verify that phase boundaries shift predictably with model scale

### Failure Modes and Diagnostics
- **Failure mode**: Gradient variance estimates are noisy with small batch sizes → AdaBalance oscillates
  - **Detection**: Monitor α/β trajectory for high-frequency oscillation
  - **Mitigation**: Increase EMA window τ, increase K
- **Failure mode**: Phase boundaries are too diffuse to detect clearly
  - **Detection**: Sobel filter gradient magnitude is uniformly low
  - **Mitigation**: Increase grid resolution, use finer β steps near boundaries
- **Failure mode**: AdaBalance converges to a suboptimal fixed point
  - **Detection**: Final accuracy below best static point
  - **Mitigation**: Add exploration noise to (α,β) updates

### Novelty and Elegance Argument
**Closest work and differences:**
| Method | What it does | Key difference from ours |
|--------|-------------|--------------------------|
| Mroueh (2025) | Derives GRPO PoS dynamics | No phase diagram, no (α,β) analysis |
| Zhou et al. (2026) | U-statistic theory for group size | No signal balance analysis |
| GTPO (2025) | Skip negatives + entropy filter | Heuristic; doesn't map the landscape |
| DaGRPO (2025) | Mask low-distinctiveness + augment | Off-policy augmentation; no theory |
| TR-GRPO (2026) | Token-level reweighting | Token-level; misses sequence dynamics |

**Our contribution is unique**: We are the first to (1) construct a phase diagram of GRPO training dynamics, (2) theoretically predict phase boundaries from gradient variance analysis, and (3) derive an adaptive navigator from the minimum-variance principle. This is a geometry-of-optimization contribution, not a new trick.

## Claim-Driven Validation Sketch

### Claim 1: Phase transitions exist in the (α,β) space and are predictable
- **Minimal experiment**: Full grid sweep (45 points × 3 seeds). Measure accuracy + gradient statistics at each point. Apply Sobel boundary detection. Compare empirical boundaries with theoretical predictions from the variance decomposition formula.
- **Baselines / ablations**: Random (α,β) vs grid-predicted optimal vs theoretical optimal
- **Metric**: (1) Boundary detection F1 between predicted and empirical phase boundaries. (2) Correlation between predicted variance and empirical gradient norm.
- **Expected evidence**: Empirical boundaries align with theoretical predictions (F1 > 0.7); high correlation (r > 0.8) between predicted variance and empirical gradient norms.

### Claim 2: AdaBalance matches or exceeds the best static (α,β) without search
- **Minimal experiment**: AdaBalance vs best-static-from-sweep vs GRPO-baseline vs GTPO vs DaGRPO on GSM8K and MATH.
- **Baselines / ablations**: (1) Standard GRPO (α=0.5, β=1.0). (2) Best static (α,β) from sweep. (3) GTPO. (4) DaGRPO. (5) Curriculum schedules (cosine, linear anneal). (6) AdaBalance without EMA (ablation). (7) AdaBalance with fixed K values (ablation).
- **Metric**: GSM8K accuracy, MATH accuracy, zero-score ratio trajectory, training stability (reward variance)
- **Expected evidence**: AdaBalance achieves ≥98% of best-static accuracy with zero hyperparameter search; outperforms GTPO/DaGRPO on at least one benchmark.

## Experiment Handoff Inputs
- **Must-prove claims**: (1) Phase transitions are real and predictable. (2) AdaBalance is competitive.
- **Must-run ablations**: (1) AdaBalance without EMA. (2) AdaBalance with different K values. (3) Phase diagram with/without LoRA.
- **Critical datasets/metrics**: GSM8K accuracy (primary), MATH accuracy (secondary), gradient norm ratio (diagnostic)
- **Highest-risk assumptions**: (1) Variance decomposition formula holds empirically with finite samples. (2) Online gradient statistics are accurate enough for AdaBalance to navigate.

## Compute & Timeline Estimate
- **Phase diagram sweep**: 135 runs × ~0.5 GPU-hours = ~68 GPU-hours
- **AdaBalance + baselines**: 10 runs × ~1 GPU-hour = ~10 GPU-hours
- **27B validation**: 5 runs × ~4 GPU-hours = ~20 GPU-hours
- **Gradient analysis + diagnostics**: ~10 GPU-hours
- **Total**: ~108 GPU-hours (well within 300 GPU-hour budget)
- **Timeline**: Phase diagram (5 days) → Analysis + AdaBalance (3 days) → 27B validation (2 days) → Paper (4 days)
