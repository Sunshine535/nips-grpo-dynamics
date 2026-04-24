# GPT-5.5 Pro Report Extraction

## Diagnosis File Used
`GPT55_DIAGNOSIS.md` — repository root, 1089 lines, dated 2026-04-24.

## Recommended MAIN METHOD PATH
**TRACE-GRPO**: Trust-Calibrated Replay and Prompt-Conditioned Credit Assignment for Sparse Binary GRPO.

Replace fixed lambda_rep verified replay CE with prompt-conditioned posterior / trust gate / drift control.

## Missing Mechanism
**Prompt-conditioned trust-calibrated replay credit with drift control.**

Current method treats verified successes as equally trustworthy, equally relevant, and infinitely reusable. It has no mechanism for:
- Which prompt's success is still on the learning frontier
- Which replay item is stale or over-replayed
- How strongly to replay based on evidence confidence
- When replay is causing distribution drift

## Evidence From Positive Results
- SPO+Replay n=200 first-200 prefix: 69.4 +/- 10.4% (9 seeds) vs fixed-rho 54.2%
- Shows "per-prompt baseline + positive memory" signal exists (P1)

## Evidence From Negative Results
- 500-step full-set collapse: 44.6% mean (P2) — fixed replay drifts
- Phase diagram (alpha,beta) near base 25% (P4) — global scalar control is not the mechanism
- RFT-only: 35-38% (P5) — CE alone insufficient

## Evidence From Unstable Results
- lambda=0.02: two seeds split (75% vs 39%) (P7)
- true-dup: 54-82% across seeds (P8)
- SPO+Replay std=10.4 (P1)

## Evidence From Failed Ablations
- SPO-only: 52.2% — baseline alone not enough (P6)
- Adaptive duplication: no effect (P8)
- Phase alpha/beta: no effect (P4)

## Why Existing Best Positive Fragment Is Insufficient
Fixed SPO+Replay cannot explain 500-step collapse, phase null, lambda instability, or SFT-gold gap. It is a local trick that works on n=200 prefix short horizon only.

## Files to Inspect
- src/aser_trainer_v14.py (attention_mask bug, EOS decode)
- src/prompt_stats.py (lacks posterior/trust)
- src/replay_bank.py (uniform sampling, no trust)
- scripts/eval_stratified.py (first-N prefix)
- scripts/run_aser_mvp.py (provenance)

## Files to Edit
- scripts/eval_stratified.py (fix selection mode)
- src/aser_trainer_v14.py (fix mask/EOS)

## New Files to Create
- src/prompt_credit_state.py
- src/trust_gated_replay_bank.py
- src/trace_grpo_trainer.py
- scripts/run_trace_grpo.py
- configs/trace_grpo_minimal.yaml
- src/provenance.py

## Files to Archive
- PROPOSAL_SPO_REPLAY.md (superseded)
- paper/main.tex (stale CSD/rho)
- src/adabalance.py, src/csd_logging.py, src/bandit_rho.py, src/exact_rho_controller.py, src/stability_analysis.py, src/rho_grpo_trainer.py (retracted routes)

## Files to Keep
- RETRACTIONS.md (freeze)
- review-stage/AUTO_REVIEW.md (freeze)
- src/qwen35_compat.py, src/torch_compat.py (infrastructure)
- All raw result JSONs (historical evidence)

## Files to Keep Only as Baseline
- src/aser_trainer_v14.py (variant A)
- src/prompt_stats.py (legacy)
- src/replay_bank.py (legacy)
- configs/aser_mvp.yaml (variant A config)
- SFT-gold results

## Files to Keep Only as Ablation
- Adaptive duplication results
- True-dup results
- Phase diagram results
- Zero-score handler results

## Suspected Bugs
1. P0: attention_mask not passed in get_per_token_logps (trainer line ~155)
2. P1: completions decoded for reward include post-EOS tokens
3. P2: RFT-only label/count inconsistency in analysis

## Required Logging
- lambda_eff distribution per step
- frontier histogram
- replay item age/exposure
- replay token ratio vs PG tokens
- all-pass/all-fail group rates
- KL, length, reward per step
- prompt posterior state snapshots
- run manifest with git hash, config, seed, eval ids

## Required Minimal Experiments
1. Smoke test (2 steps)
2. Data/metric/checkpoint sanity
3. Reproduce fixed SPO+Replay positive (n=200 + full)
4. A/B/C comparison on full GSM8K n=1319, seeds 42/43/44:
   A. Existing fixed SPO+Replay
   B. TRACE with constant gate (lambda_eff = lambda_max always)
   C. Full TRACE-GRPO

## Required Core Comparison
C vs A vs B on full-set mean accuracy with matched seeds, config, eval.

## Required Baselines
Base, stock GRPO, Dr.GRPO, SPO-only, RFT-only, SFT-gold, fixed-rho.
Official: RePO, DAPO if compute allows.

## Stop / Continue / Pivot Criteria
- CONTINUE: C > A and C > B on full-set mean, variance not worse, no 500-step collapse
- STOP: C only improves first-200 not full-set
- PIVOT: B approximately equals C (infrastructure, not mechanism, explains gain)
- HARD STOP: Official baselines dominate and TRACE has no differentiating signal
