# Current Result Audit

**Date**: 2026-04-24

## Result Table

| Result | File | Dataset | Config | Seed | Metric | Value | Compared Against | Supports GPT-5.5? | Notes |
|--------|------|---------|--------|------|--------|-------|------------------|--------------------|-------|
| SPO+Replay n=200 | results/stratified_eval/spo_full_seed*.json | GSM8K first-200 | aser_mvp G=2 | 42-51 (9) | acc | 69.4+/-10.4% | fixed-rho 54.2% | Yes (positive signal) | High variance, prefix only |
| SPO-only n=200 | results/stratified_eval/spo_only_seed*.json | GSM8K first-200 | aser no replay | 42-44 (3) | acc | 52.2+/-11.0% | SPO+Replay | Yes (SPO alone not enough) | Only 3 seeds |
| RFT-only n=200 | results/stratified_eval_wave11/ | GSM8K first-200 | pg_weight=0 | mixed | acc | ~36% | SPO+Replay | Yes (CE alone weak) | Provenance issue |
| SFT-gold n=200 | results/stratified_eval_wave13/sft_gold_*.json | GSM8K first-200 | SFT on gold | 42-45 (4) | acc | 84.6+/-0.6% | SPO+Replay | Yes (strong baseline) | Upper bound |
| 500-step full-set | results/wave14_500step/evals/*.json | GSM8K full-1319 | aser_mvp 500step | 42-44 (3) | acc | 44.6% mean | n=200 positive | Yes (collapse) | Strong negative |
| Phase diagram | results/wave14_phase_diagram/evals/*.json | GSM8K full-1319 | alpha/beta grid | 42 | acc | 25-27% | base 25.5% | Yes (null effect) | Near baseline |
| Base model | results/stratified_eval/base.json | GSM8K first-200 | none | - | acc | 25.5% | all | Reference | |
| Dr. GRPO n=200 | results/stratified_eval/drgrpo_seed*.json | GSM8K first-200 | dr_grpo G=2 | 42-44 | acc | ~39% | SPO+Replay | Yes (weak backbone) | |
| true-dup n=200 | results/stratified_eval_wave13/truedup_*.json | GSM8K first-200 | true-dup | 42-45 | acc | 68.9+/-11.5% | SPO+Replay | Yes (dup no effect) | High variance |
| TASA G=4 (running) | results/tasa_g4_safe/ | GSM8K | G=4 TASA | 42-45 | acc | PENDING | Dr.GRPO G=4 | Unknown | Currently training on server |

## Variant Existence Check

A. **Existing Best Positive Fragment Only**: YES — fixed SPO+Replay, 9 seeds, n=200 only
B. **New MAIN METHOD Without New Mechanism**: NO — TRACE with constant gate not yet implemented
C. **Full New MAIN METHOD (TRACE-GRPO)**: NO — not yet implemented

**Missing**: Variants B and C. Must implement before any new claims.

## Result-Based Execution Decision

**PROCEED** — with the following priority ordering:
1. Fix eval protocol (first-N prefix issue contaminates all existing positive results)
2. Fix trainer bugs (attention_mask, EOS decode)
3. Implement TRACE-GRPO (the missing mechanism)
4. Run A/B/C comparison on full-set n=1319

Reason: Diagnosis is well-supported by current evidence (positive, negative, unstable). No contradiction found. The currently running TASA experiment is orthogonal (advantage computation vs replay mechanism) and does not conflict with TRACE implementation.
