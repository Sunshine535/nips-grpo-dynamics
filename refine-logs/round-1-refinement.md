# Round 1 Refinement

## Problem Anchor
- **Bottom-line problem**: GRPO is the dominant RL post-training algorithm for LLM reasoning, yet practitioners lack principled guidance for configuring the positive/negative signal balance. The training landscape contains hidden stability regimes between healthy convergence and catastrophic collapse, but no theoretical or empirical framework exists to map and navigate these regimes.
- **Must-solve bottleneck**: Current GRPO tuning is trial-and-error. Existing fixes (GTPO, DaGRPO, TR-GRPO) each address one symptom without understanding the underlying geometry of the training landscape.
- **Non-goals**: (1) Not building a new RL algorithm. (2) Not solving multi-objective reward hacking. (3) Not addressing token-level credit assignment.
- **Constraints**: 8×H100, ~300 GPU-hours, Qwen3.5-9B primary, 27B validation. GSM8K + MATH.
- **Success condition**: (1) A theoretically grounded stability map that predicts collapse from first principles. (2) An adaptive controller derived from the theory. (3) Distinct from GTPO/DaGRPO/TR-GRPO.

## Anchor Check
- **Original bottleneck**: Understanding the stability landscape of GRPO signal balance to avoid collapse without brute-force search.
- **Does the revised method still solve it?**: YES — we sharpen the theoretical framework and simplify the control variable.
- **Reviewer suggestions rejected as drift**: NONE — all reviewer suggestions stay on-anchor. They ask us to simplify and sharpen, not to add components.

## Simplicity Check
- **Dominant contribution after revision**: A stability law for GRPO under binary rewards, parameterized by the effective balance ratio and conditioned on group success count m/G.
- **Components removed or merged**: (1) Collapsed (α,β) to one effective balance ratio ρ. (2) Removed "phase diagram" language → "stability map". (3) AdaBalance demoted from headline to corollary. (4) Removed expensive gradient variance estimation → use cheap trainer telemetry.
- **Reviewer suggestions rejected as unnecessary complexity**: NONE — all simplification suggestions accepted.
- **Why the remaining mechanism is still the smallest adequate route**: One scalar balance parameter + one stability theorem + one derived controller. Cannot be simpler while still being useful.

## Changes Made

### 1. Reparameterized from (α,β) to effective balance ratio ρ
- **Reviewer said**: w+ = α and w0 = (1-α)β may collapse to one effective ratio. Prove why both dimensions matter or collapse them.
- **Action**: Analyzed the exact GRPO loss with clipping and KL penalty. The effective positive weight is α and effective negative weight is (1-α)β. Under global rescaling invariance of advantages (GRPO normalizes by std), only the ratio ρ = α / ((1-α)β) matters. We collapse to ρ as the single control variable.
- **Reasoning**: The reviewer is right — the absolute scale is absorbed by GRPO's advantage normalization. The meaningful control is the relative weight of positive vs negative signal. ρ > 1 upweights positives; ρ < 1 upweights negatives; ρ = 1 is standard GRPO.
- **Impact on core method**: Simpler theory, one-dimensional stability map instead of 2D phase diagram. More elegant.

### 2. Built theory around group success count m/G
- **Reviewer said**: For binary rewards, the key discrete state is m out of G. Degenerate groups (m=0, m=G) are the dominant pathologies.
- **Action**: Rewrote the variance decomposition conditioned on the group success count m. For a group of G samples with m correct:
  - Mean reward: μ = m/G, Std: σ = √(m(G-m)/G²)
  - When m=0: all advantages are 0, gradient is ZERO (gradient starvation)
  - When m=G: all advantages are 0, gradient is also ZERO (saturation)
  - Critical regime: m ∈ {1, 2} or m ∈ {G-1, G-2} — extreme advantage values
  - The balance ratio ρ controls how these degenerate cases contribute to the total gradient
- **Reasoning**: This gives a discrete, provable stability law instead of a vague continuous approximation.
- **Impact on core method**: Theory is now grounded in the actual GRPO computation. Stability conditions are stated in terms of observable quantities (the m/G distribution).

### 3. Renamed "phase diagram" → "stability map"
- **Reviewer said**: "Phase transition" is too strong unless sharp threshold proved.
- **Action**: Renamed throughout. The main object is a "stability map" showing regimes (convergent / transitional / collapsed) as a function of ρ and the zero-score ratio p_0 = P(m=0). We reserve "phase transition" only if the theorem yields a provably sharp boundary.
- **Reasoning**: Honest terminology. We can always upgrade to "phase transition" if the theory delivers.
- **Impact**: More defensible claim, same intellectual content.

### 4. AdaBalance demoted to derived controller
- **Reviewer said**: AdaBalance should be a corollary, not a second headline contribution.
- **Action**: The paper's dominant contribution is the stability law. AdaBalance is presented as "the natural controller derived from the theorem" — one formula, no new parameters, no separate method section.
- **Reasoning**: Cleaner paper structure. The theorem does the heavy lifting; the controller is the engineering payoff.
- **Impact**: One focused contribution instead of two.

### 5. Replaced gradient variance estimation with trainer telemetry
- **Reviewer said**: V_pos and V_zero are not operationally defined. Expensive gradient computation.
- **Action**: The stability law is now stated in terms of directly observable trainer telemetry: (1) zero-score ratio p_0 (fraction of groups with m=0), (2) success-count histogram P(m), (3) clip fraction, (4) entropy/KL to reference. The AdaBalance controller reads p_0 and the m-distribution, not raw gradient norms.
- **Reasoning**: Every TRL GRPOTrainer already logs these quantities. Zero additional overhead.
- **Impact**: Practical and implementable.

### 6. Simplified experimental plan
- **Reviewer said**: Budget may not fit; use two-stage protocol. Collapse definitions should be mechanistic.
- **Action**: (1) Two-stage sweep: coarse (9 ρ values × 3 p_0 regimes × 2 seeds = 54 short runs), then fine near boundaries. (2) Collapse defined as: sustained all-zero-group rate > 80% OR KL divergence exceeding 3× initial OR reward stagnation for >100 steps. (3) 27B reduced to 3 representative ρ values. (4) Baselines: vanilla GRPO, oracle best-static, one heuristic scheduler. GTPO/DaGRPO/TR-GRPO secondary.
- **Reasoning**: Two-stage is more efficient and scientifically sound.
- **Impact**: Fits budget comfortably.

## Revised Proposal

# Research Proposal: Stability Map of GRPO Signal Balance — A Group-Outcome Theory with Adaptive Control

## Problem Anchor
(Same as above — verbatim.)

## Technical Gap
No existing work provides a stability law for GRPO training that (1) identifies the effective control variable governing the positive/negative signal balance, (2) characterizes collapse conditions in terms of group success-count statistics, or (3) derives an adaptive controller from the stability condition.

Mroueh (2025) and Zhou et al. (2026) analyze GRPO dynamics and the U-statistic structure but do not address the signal-balance dimension. GTPO/DaGRPO/TR-GRPO offer ad-hoc fixes without a unifying stability condition.

## Method Thesis
- **One-sentence thesis**: Under binary rewards, the GRPO policy gradient's variance is governed by the effective balance ratio ρ = w+/w0 and the group success-count distribution P(m); a stability law in these two variables predicts collapse regimes and yields a zero-overhead adaptive controller.
- **Why smallest adequate intervention**: One scalar parameter ρ, one stability theorem, one derived controller formula. No new modules, no architecture changes.
- **Why timely**: GRPO is the default post-training recipe; a stability law is immediately actionable.

## Contribution Focus
- **Dominant contribution**: A stability law for GRPO under binary rewards, characterizing convergent / transitional / collapsed regimes as a function of the effective balance ratio ρ and the zero-score group rate p_0 = P(m=0). The law is stated in terms of directly observable trainer telemetry.
- **Supporting contribution (corollary)**: AdaBalance, a zero-parameter adaptive controller that adjusts ρ online using the stability condition and p_0 monitoring.
- **Explicit non-contributions**: No new RL algorithm, no token-level mechanism, no off-policy augmentation.

## Proposed Method

### Complexity Budget
- Frozen/reused: Qwen3.5-9B/27B, TRL GRPOTrainer, LoRA
- New trainable components: ZERO
- New hyperparameters: K (update interval, default 50), τ (EMA, default 0.1) — both derivable from the theorem
- Intentionally excluded: Token-level reweighting, off-policy augmentation, entropy filtering, learned controllers

### System Overview
```
Prompt batch → GRPO generates G completions per prompt
  → Binary reward r_i ∈ {0,1}
  → Compute group success count m per group
  → Standard advantage: a_i = (r_i - m/G) / σ_m
  → Apply balance ratio: ã_i = ρ·a_i if r_i=1, else a_i  (ρ ≥ 0)
  → Policy gradient step
  → Monitor p_0 = fraction of all-zero groups
  → AdaBalance: ρ_{t+1} = f(p_0, P(m))  [derived from stability condition]
```

### Core Mechanism: Group-Outcome Stability Law

#### Setup
Consider GRPO with group size G and binary reward r ∈ {0,1}. For a given prompt x, let m = Σr_i be the group success count. The advantage for sample i in a group with outcome m is:

a_i(m) = (r_i - m/G) / σ(m), where σ(m) = √(m(G-m)) / G

With balance ratio ρ, the effective advantage is:
ã_i(m) = ρ · a_i(m) if r_i = 1, else a_i(m)

#### Key Observations (to be proved)

**Theorem 1 (Degenerate Group Starvation)**: When m = 0 or m = G, all advantages are identically 0 regardless of ρ. These groups contribute zero gradient signal.

**Theorem 2 (Variance Decomposition)**: The variance of the weighted GRPO gradient estimator decomposes as:
Var(∇̂J_ρ) = ρ² · V_+(P(m)) + V_0(P(m)) + 2ρ · Cov(P(m))

where V_+, V_0, Cov are functions of the success-count distribution P(m) computable from trainer telemetry.

**Theorem 3 (Stability Condition)**: Training is stable when:
ρ ∈ [ρ_min(p_0), ρ_max(p_0)]

where ρ_min prevents gradient starvation (too little positive signal when p_0 is high) and ρ_max prevents negative-signal dominance (too much positive bias causing reward hacking). Both bounds are explicit functions of p_0 and G.

**Corollary (AdaBalance Controller)**: The minimum-variance ρ* satisfies:
ρ* = -Cov(P(m)) / V_+(P(m))

which can be estimated online from the success-count histogram at cost O(G) per group.

#### Stability Map
The stability map is a 2D plot of (ρ, p_0) annotated with three regimes:
- **Convergent**: ρ ∈ [ρ_min, ρ_max], p_0 moderate → healthy gradient, stable KL
- **Gradient-starved**: ρ too low OR p_0 too high → insufficient positive signal
- **Reward-hacking**: ρ too high AND p_0 low → model over-fits correct samples, ignores errors

The boundaries are derived from Theorem 3 and validated empirically.

### AdaBalance Controller (Corollary)
```python
class AdaBalanceCallback(TrainerCallback):
    def __init__(self, G, K=50, tau=0.1):
        self.G = G
        self.K = K
        self.tau = tau
        self.rho = 1.0  # start at standard GRPO
        self.p0_ema = 0.0
        self.success_counts = []

    def on_step_end(self, args, state, control, **kwargs):
        # Read success count m from latest batch (already available in trainer)
        m = self._get_group_success_count(kwargs)
        self.success_counts.append(m)
        
        if state.global_step % self.K == 0 and len(self.success_counts) >= self.K:
            recent = self.success_counts[-self.K:]
            p0 = sum(1 for x in recent if x == 0) / len(recent)
            self.p0_ema = (1 - self.tau) * self.p0_ema + self.tau * p0
            
            # From Theorem 3 corollary:
            rho_star = self._compute_optimal_rho(recent, self.G)
            self.rho = (1 - self.tau) * self.rho + self.tau * rho_star
```

### Training Plan
1. **Stability Map Construction** (empirical validation):
   - Stage 1 (coarse): 9 values of ρ ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0} × 3 difficulty regimes (easy/medium/hard subsets controlling p_0) × 2 seeds = 54 short runs (50 steps each)
   - Stage 2 (fine): Refine near predicted boundaries with 5 additional ρ values × 2 seeds = 10 full runs (2 epochs)
   - Collect: accuracy, p_0 trajectory, m-histogram, KL, entropy, clip fraction
   - Overlay theoretical boundaries from Theorem 3 on empirical stability map

2. **AdaBalance Validation**:
   - 3 full runs: AdaBalance, oracle best-static ρ, vanilla GRPO (ρ=1)
   - Secondary: 1 heuristic linear scheduler, 1 GTPO comparison
   - Evaluate on GSM8K test + MATH

3. **Scaling Check (27B)**:
   - 3 representative ρ values on Qwen3.5-27B
   - Verify stability boundaries shift predictably

### Collapse Definitions (Mechanistic)
A run is classified as "collapsed" if any of:
1. **Gradient starvation**: p_0 > 0.8 sustained for >100 steps
2. **KL divergence blow-up**: KL > 3× initial value
3. **Reward stagnation**: reward mean changes < 0.01 for >100 steps
4. **Entropy collapse**: generation entropy drops below 0.1× initial

### Failure Modes and Diagnostics
- **ρ* estimate noisy with few groups**: Increase K, use EMA
- **Stability boundaries too diffuse**: Use finer ρ grid near boundaries
- **27B doesn't transfer**: Report as limitation; the stability law may need model-size correction terms

### Novelty and Elegance Argument
| Method | What | Key difference |
|--------|------|----------------|
| Mroueh (2025) | GRPO PoS dynamics | No balance ratio analysis |
| Zhou et al. (2026) | U-statistic, optimal G | No signal balance, no stability map |
| GTPO (2025) | Skip negatives | Heuristic, not principled |
| DaGRPO (2025) | Mask + augment | Off-policy, no theory |
| **Ours** | Stability law in (ρ, p_0) | First theory-backed stability map + derived controller |

## Claim-Driven Validation Sketch

### Claim 1: The stability law predicts collapse regimes
- **Experiment**: Two-stage sweep. Plot empirical stability map (ρ vs p_0) colored by run outcome (convergent/collapsed). Overlay theoretical boundaries.
- **Metric**: (1) Classification accuracy of theorem-predicted regime vs actual outcome. (2) Rank correlation between predicted instability score and actual training divergence.
- **Expected**: Classification accuracy > 85%, rank correlation > 0.8.

### Claim 2: AdaBalance matches best-static without search
- **Experiment**: AdaBalance vs oracle best-static ρ vs vanilla GRPO on GSM8K + MATH.
- **Metric**: GSM8K accuracy (±std), MATH accuracy, p_0 trajectory.
- **Expected**: AdaBalance within 1% of oracle best-static; >3% above vanilla GRPO.

## Experiment Handoff Inputs
- **Must-prove claims**: (1) Stability law is predictive. (2) AdaBalance is competitive.
- **Must-run ablations**: (1) AdaBalance with different K/τ. (2) Stability map with different G values.
- **Critical datasets/metrics**: GSM8K accuracy (primary), MATH accuracy (secondary), p_0 (diagnostic).
- **Highest-risk assumptions**: (1) Variance decomposition holds empirically with finite G. (2) p_0 is sufficient statistic for stability.

## Compute & Timeline Estimate
- Coarse sweep: 54 short runs × 0.1 GPU-hr = 5.4 GPU-hours
- Fine sweep: 10 full runs × 1.5 GPU-hr = 15 GPU-hours
- AdaBalance + baselines: 5 full runs × 1.5 GPU-hr = 7.5 GPU-hours
- 27B validation: 3 runs × 3 GPU-hr = 9 GPU-hours
- Analysis + diagnostics: 5 GPU-hours
- **Total: ~42 GPU-hours** (well within 300 GPU-hour budget)
- **Timeline**: Sweep (3 days) → Analysis + AdaBalance (2 days) → 27B (1 day) → Paper (4 days)
