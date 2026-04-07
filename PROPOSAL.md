# Metastable Training Dynamics in GRPO: Seed-Resolved Basin Analysis and Transient Rescue

## One-Sentence Summary

We demonstrate that GRPO training collapse reflects structured seed variance with basin-like dynamics — detectable at step 0 from gradient subspace geometry, characterizable via Binder cumulant order parameters, and rescuable via transient hyperparameter pulses that produce permanent escape — validated across Qwen2.5-7B, Qwen3-8B, and Qwen3.5-9B.

## Problem

GRPO exhibits extreme seed variance at certain hyperparameter settings: under identical ρ=1.0 on Qwen2.5-7B, three seeds yield 0.5%, 0.0%, and 89.0% accuracy. This is not Gaussian noise — it is a structured, multi-modal distribution suggesting competing basins in optimization space.

40+ GRPO papers (2025-2026) propose fixes (DAPO, GSPO, GTPO, etc.) without asking: **why do some seeds collapse while others converge under identical settings?** And critically: **can we predict and rescue the failing seeds?**

## Approach

### Contribution 1: Seed-Resolved Basin Analysis

We apply statistical mechanics diagnostics to dense ρ-sweeps (≥20 seeds per condition):
- **Binder cumulant** U₄ = 1 - ⟨R⁴⟩/(3⟨R²⟩²) identifies the ρ range where outcomes become multi-modal (U₄ dip below Gaussian reference 2/3)
- **Susceptibility** χ = N(⟨R²⟩ - ⟨R⟩²) peaks at the boundary between unimodal and multi-modal regimes
- **Residence time analysis**: for seeds near the boundary, measure how long rewards stay in each basin before committing — evidence of genuine metastability vs early divergence

Key: We do NOT claim phase transitions or universality. We use these tools as sensitive diagnostics for structured variance.

### Contribution 2: Step-0 Trainability Prediction

After the initial rollout (step 0), compute:
1. **Gradient subspace alignment** S₀ = cos(g⁺, g⁻) between mean positive-reward and negative-reward gradient directions
2. **Initial success diversity** D₀ = entropy of per-group success counts
3. **Gradient magnitude ratio** M₀ = ‖g⁺‖/‖g⁻‖

These features are available from a single rollout before any parameter update. We train a binary classifier (converge vs collapse) on Qwen2.5-7B (≥180 labeled runs), validate on Qwen3-8B and Qwen3.5-9B.

Must beat trivial baselines: (a) ρ alone, (b) initial reward mean, (c) initial entropy.

### Contribution 3: Transient Rescue

The killer experiment (per reviewer guidance): take collapsing seeds at near-critical ρ, apply a **transient** ρ or λ_KL pulse for K steps (K ∈ {5, 10, 20}), then **revert to original hyperparameters**. If the seed permanently escapes to the convergent basin after the pulse, this is genuine basin escape, not mere retuning.

Measure:
- **Rescue probability** P_rescue(t, K, Δρ): probability of permanent escape as function of intervention time t, duration K, and magnitude Δρ
- **Irreversibility point** t_irrev: the step after which no transient pulse rescues
- **Minimum effective pulse**: smallest Δρ × K product that achieves escape

## Experiments

| Experiment | Model | Runs | Seeds | GPU-hours |
|------------|-------|------|-------|-----------|
| Dense ρ-sweep (9 ρ values) | Qwen2.5-7B | 180 | 20 × 9ρ | ~360 |
| Dense ρ-sweep (9 ρ values) | Qwen3-8B | 180 | 20 × 9ρ | ~360 |
| Dense ρ-sweep (9 ρ values) | Qwen3.5-9B | 90 | 10 × 9ρ | ~180 |
| Step-0 probes | All models | 450 | from sweeps | ~20 |
| Transient rescue | Qwen2.5-7B | 120 | 3 collapsed × 8t × 5pulse | ~60 |
| Transient rescue | Qwen3-8B | 120 | 3 collapsed × 8t × 5pulse | ~60 |
| **Total** | | | | **~1040** |

## Benchmarks

- GSM8K (1319 test) — primary
- MATH (5000 test) — secondary

## Expected Contributions

1. Dense seed-resolved sweep (20 seeds) revealing structured multi-modal outcome distributions at near-critical ρ, characterized by Binder cumulant diagnostics
2. Step-0 trainability predictor from gradient geometry that beats trivial baselines — practical tool for avoiding wasted compute
3. Transient rescue protocol demonstrating genuine basin escape (not retuning) with measured irreversibility points
4. Cross-model comparison showing how basin structure shifts between Qwen2.5-7B, Qwen3-8B, and Qwen3.5-9B

## Key Differentiators

| Prior work | What they do | What we add |
|------------|-------------|-------------|
| Mroueh 2503.06639 | Aggregate dynamics, fixed points | Seed-resolved basin analysis, individual fate prediction |
| Grokking FSS (2603.24746) | Binder cumulant for grokking | Different phenomenon, we use U₄ as diagnostic not as universality claim |
| Kramers CL (2604.04154) | Kramers for continual learning | We do transient rescue (pulse + revert), not permanent barrier modification |
| GAC (2603.01501) | Gradient cosine during training | We predict at step 0, before any updates |
| All variant papers | Fix collapse by changing algorithm | We predict and rescue within vanilla GRPO |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Metastability is just noise | 20 seeds gives statistical power; residence time analysis distinguishes |
| Step-0 predictor is trivial proxy | Must beat ρ + initial_reward + entropy baselines |
| Transient rescue = retuning | Revert to original params after pulse; permanent escape is the test |
| Only Qwen family | Include Qwen3 (different architecture/training) as minimal cross-family |
| 1040 GPU-hours too much | Qwen3.5 reduced to 10 seeds; prioritize transient rescue on 2 models |
