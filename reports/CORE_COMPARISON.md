# Core Comparison (A / B / C) — FINAL

**Date**: 2026-04-25 04:32 UTC. All 6 evaluations complete.

## Setup

- Model: Qwen/Qwen3.5-9B + LoRA (r=64)
- Dataset: GSM8K train (training); GSM8K test n=1319 full (evaluation)
- Group size G = 4
- Batch size 1, grad-accum 4
- Learning rate 2e-5
- Max steps 200
- Reward: binary {0, 1} (`build_gsm8k_binary_reward_function`)
- 2 seeds per variant: 42, 43

## Variants

| Variant | Trainer | Mechanism |
|---------|---------|-----------|
| A_legacy | `run_aser_mvp.py --backbone spo` | Legacy ASERTrainerV14 + `prompt_stats.py` (EMA baseline) + `replay_bank.py` (uniform sampling), λ_rep=0.05 fixed |
| B_constant | `run_trace_grpo.py --trace-mode constant_gate` | New `TraceGRPOTrainer` + `PromptCreditState` (Beta posterior) + `TrustGatedReplayBank` (weighted sampling), but λ_eff = λ_max=0.05 always |
| C_full | `run_trace_grpo.py --trace-mode full` | Same as B + adaptive λ_eff = λ_max × mean_trust × drift_budget |

## Results table

| Variant | seed42 | seed43 | mean | std | min | max |
|---------|:------:|:------:|:----:|:---:|:---:|:---:|
| A_legacy | **83.55%** | 29.26% | 56.41% | 38.38 pp | 29.26 | 83.55 |
| B_constant | 30.86% | 29.72% | 30.29% | 0.81 pp | 29.72 | 30.86 |
| C_full | 33.66% | 34.19% | **33.93%** | 0.37 pp | 33.66 | 34.19 |

For reference:
- Base Qwen3.5-9B: ~25.5%
- TASA G=4 (4 seeds): 62.57 ± 9.52% (different backbone, different trainer)
- Dr.GRPO G=4 (4 seeds): 27.77 ± 0.59%
- SFT-gold (n=200, not full): 84.6%

## Pairwise deltas

| Comparison | Delta | Interpretation |
|------------|:-----:|----------------|
| C − A | **−22.48 pp** | Full TRACE BELOW legacy SPO+replay mean |
| C − B | **+3.64 pp** | Trust gate adds marginal benefit over constant gate |
| B − A | **−26.12 pp** | TRACE infra (without trust gate) BELOW legacy mean |

## Verdict per GPT-5.5 decision tree

> **C < A: new method hurts. Check implementation or diagnosis.**

This is the protocol-prescribed verdict. However, the result is more nuanced
than a clean rejection — see "Variance analysis" below.

## Variance analysis

A_legacy is **highly bimodal**:
- seed 42 → 83.55% (essentially SFT-gold level)
- seed 43 → 29.26% (essentially base level)
- std = 38.38 pp (n=2)

B_constant and C_full are **highly stable but at mediocre level**:
- B_constant std = 0.81 pp; mean 30.29%
- C_full std = 0.37 pp; mean 33.93%
- ~50× variance reduction vs A_legacy

**Interpretation**: TRACE infrastructure eliminates the seed-dependent collapse that
GPT-5.5's diagnosis warned about (the "500-step collapse"), AND the seed-dependent
upside that A_seed42 exhibited. It produces a stable but conservative outcome ~5-8 pp
above base.

## Mechanism activation evidence (from trace_step_stats.json)

### B_constant (constant gate)
| Seed | λ_eff mean | λ_eff max | nonzero_steps | bank_size | replay_token_ratio |
|:----:|:----------:|:---------:|:-------------:|:---------:|:------------------:|
| 42 | 0.0375 | 0.0500 | 600/800 | 647 | 0.348 |
| 43 | 0.0375 | 0.0500 | 600/800 | 643 | 0.343 |

`λ_eff` is at the configured 0.05 max; behavior is correct for the "constant gate" mode.

### C_full (adaptive gate)
| Seed | λ_eff mean | λ_eff max | nonzero_steps | bank_size | replay_token_ratio |
|:----:|:----------:|:---------:|:-------------:|:---------:|:------------------:|
| 42 | 0.00108 | 0.00822 | 526/800 | 738 | 0.300 |
| 43 | 0.00108 | 0.00875 | 520/800 | 712 | 0.300 |

The trust gate **dramatically suppresses** λ_eff to ~1/35 of the configured maximum.
The drift budget hits the 0.30 cap, further suppressing replay. **Mean frontier is only
~0.18-0.20**, indicating most prompts have low frontier scores under the current
PromptCreditState formula.

## Why is the trust gate so conservative?

The frontier formula is `F = 4 * p_hat * (1-p_hat) * min(1, n_obs/n_min)` with
`n_min=5`. With 200 training steps × 1 prompt-batch × 4 generations = 800 group
observations spread across ~7,473 unique GSM8K prompts, most prompts see only 0-2
observations during training. So `min(1, n_obs/5) ≤ 0.4`, capping max frontier at
`4 * 0.5 * 0.5 * 0.4 = 0.4` even for ideal 50%-success prompts.

This means the trust gate is structurally too conservative for short-horizon
training. With only 200 steps, the credit state never accumulates enough evidence
per prompt to license aggressive replay.

## Why does B (constant gate) underperform A?

B and A use:
- Same model, same dataset, same hyperparameters
- Same reward function
- BOTH attention_mask + EOS-decode bug fixes applied

The differences are:
1. Trainer class: legacy `ASERTrainerV14` (A) vs new `TraceGRPOTrainer` (B/C)
2. Baseline storage: `PromptStatsStore` (A) vs `PromptCreditStore` (B/C)
3. Replay sampling: uniform `ReplayBank` (A) vs `TrustGatedReplayBank` (B/C)

Even with constant gate (λ_eff=0.05), B underperforms A by 26 pp in mean. This
suggests the TraceGRPOTrainer fork has a behavioural difference vs the legacy
trainer that is not just about the trust gate. Candidate causes:

- `PromptCreditState.get_baseline` returns 0 for unseen prompts (~7,000 prompts
  in GSM8K train); legacy `PromptStatsStore` may have different default behaviour.
- Reward function or completion construction may differ subtly.
- The TraceGRPOTrainer reward path constructs `prompts_for_reward` with
  `[ex["prompt"]] * num_generations`, which is correct, but the legacy trainer
  may use a different format.

Need to debug TraceGRPOTrainer vs ASERTrainerV14 with constant_gate fixed before
attributing remaining gap to trust mechanism.

## Decision per GPT-5.5 protocol

| Possible decision | Triggered? | Action |
|------------------|:----------:|--------|
| CONTINUE: minimal evidence supports the new mechanism | No | — |
| **DEBUG MORE**: code or logging is still unreliable | **Yes** | Investigate B vs A trainer-level discrepancy |
| RUN MORE SEEDS: signal exists but variance is too high | Yes | A_legacy bimodal pattern needs n≥5 |
| RUN FULL BASELINES | Premature | Until trainer parity established |
| REVISE METHOD: new mechanism does not help | Partial | Trust gate over-conservative for short horizon |
| STOP / RETURN TO GPT-5.5 | Recommended | Diagnosis partially correct (variance ↓) but mean ↓ unexpected |

**Overall**: protocol verdict is `C < A`, but the variance reduction (50× reduction)
provides genuine evidence that the TRACE diagnosis is partially correct. The trust
gate however is over-conservative for 200-step training. Recommend:

1. **Report honestly**: TRACE in current form trades upside for stability.
2. **Do not claim** TRACE wins.
3. **Investigate** the unexpected B<A gap (TraceGRPOTrainer vs ASERTrainerV14
   under matched config).
4. **Return to GPT-5.5** with these findings and ask whether the trust gate
   formula should be relaxed (e.g., n_min=2 instead of 5; or remove drift budget
   for short-horizon training).
