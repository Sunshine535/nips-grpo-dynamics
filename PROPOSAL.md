# GRPO Dynamics: Phase Diagrams and Zero-Score Gradient Reshaping for Stable RL Post-Training

## One-Sentence Summary

We provide the first systematic characterization of GRPO training stability through phase diagrams over positive/negative signal balance, revealing a zero-score collapse zone and proposing four gradient reshaping strategies that recover stable training.

## Problem

Group Relative Policy Optimization (GRPO) is the dominant RL algorithm for LLM post-training (DeepSeek-R1, Qwen). However, practitioners observe unstable training: reward collapses, KL divergence spikes, and inconsistent final performance depending on hyperparameters. Two specific pathologies are poorly understood:

1. **Signal imbalance**: The relative weighting of positive (correct) vs negative (incorrect) completions in the policy gradient is never explicitly controlled, leading to gradient domination by one signal type.
2. **Zero-score collapse**: When all completions in a GRPO group score 0, the normalized advantage is zero everywhere, producing zero gradient signal—wasting compute and creating dead zones in training.

## Approach

### Track 1: Phase Diagram Analysis

We introduce two hyperparameters: α (positive signal ratio) and β (negative signal weight), and sweep a grid of 45+ configurations × 3 seeds on GSM8K/MATH with Qwen3.5-9B. For each (α, β) point, we:
- Train for 2 epochs with the modified GRPO loss
- Evaluate on GSM8K and MATH test sets
- Record per-step reward, KL, loss, and gradient norms

From the sweep, we construct:
- **Accuracy heatmap**: identify optimal (α, β) regions
- **Phase boundaries**: gradient magnitude analysis to detect transition zones
- **Stability map**: cross-seed variance to identify reliable vs chaotic regions
- **Pareto frontier**: accuracy vs effective negative pressure tradeoff

### Track 2: Zero-Score Gradient Reshaping

We implement four strategies to handle zero-score groups:
1. **Clip**: Scale zero-score gradients by a small factor (0.1×) to maintain signal
2. **Temperature**: Divide zero-score advantages by a boost factor to encourage exploration
3. **Curriculum**: Gradually include zero-score samples over warmup steps
4. **Relabel**: Assign small positive reward (ε=0.01) to zero-score samples

Each strategy is evaluated across 3 hyperparameter settings × 2 seeds, with gradient analysis diagnostics comparing zero-score vs non-zero-score gradient directions.

### Integration

We show that zero-score collapse corresponds to a specific region in the phase diagram (low α, high β), and that the reshaping strategies effectively expand the stable training region.

## Experiments

| Experiment | Details |
|------------|---------|
| Phase diagram sweep | 9α × 5β × 3 seeds = 135 training runs |
| Zero-score strategies | 4 strategies × 3 configs × 2 seeds = 24 runs |
| Curriculum comparison | 4 schedules (anneal-pos, anneal-neg, cosine, static) |
| Gradient diagnostics | Per-step gradient norms, cosine similarity, norm ratios |
| 27B validation | Best configurations validated on Qwen3.5-27B |

## Benchmarks

- **GSM8K** (grade school math, 1319 test problems)
- **MATH** (competition math, 5000 test problems)

## Expected Contributions

1. First phase diagram characterization of GRPO signal balance
2. Identification of zero-score collapse as a distinct failure mode
3. Four practical gradient reshaping strategies with theoretical motivation
4. Curriculum strategies for dynamic α/β scheduling
5. Practical guidelines for stable GRPO training

## NeurIPS Justification

Combines rigorous empirical analysis (phase diagrams) with practical algorithmic contributions (reshaping strategies). Directly relevant to the growing community working on RL post-training for reasoning LLMs.
