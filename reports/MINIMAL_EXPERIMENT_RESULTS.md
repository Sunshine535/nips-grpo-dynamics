# Minimal Experiment Results

**Date**: 2026-04-25 (TASA vs DrGRPO evals COMPLETE; TRACE A/B/C running)

## Experiment ID: TASA-G4-vs-DrGRPO-G4 (not TRACE A/B/C)

This is a **parallel baseline** experiment that was running when the GPT-5.5 Pro
diagnosis was executed. It does NOT test TRACE-GRPO's trust gate — that test is
launch_trace_abc.sh and pending GPU availability. But it does measure whether
the TASA advantage formulation improves on Dr.GRPO under matched setup.

### Setup
| Field | Value |
|-------|-------|
| Model | Qwen/Qwen3.5-9B + LoRA (r=64) |
| Dataset | GSM8K train (`openai/gsm8k`) |
| Reward | binary {0, 1} via `build_gsm8k_binary_reward_function` |
| Group size G | 4 |
| Batch size | 1 per device × grad-accum 4 |
| Learning rate | 2e-5 |
| Max steps | 200 |
| Eval | `scripts/eval_stratified.py --n 1319` (full GSM8K test) |
| Seeds | 42, 43, 44, 45 |
| GPUs | 8× H100-80GB |

### Results Table (partial; updating as evals complete)

| Experiment | Config | Dataset | Seed | Metric | Result | Expected | Pass/Fail | Interpretation |
|------------|--------|---------|------|--------|--------|----------|-----------|----------------|
| TASA G4 | aser_g4_safe (tasa backbone) | GSM8K full 1319 | 42 | acc | 0.5610 | >base 0.255 | PASS | +30.6 pp over base |
| TASA G4 | aser_g4_safe (tasa backbone) | GSM8K full 1319 | 43 | acc | 0.7362 | >base 0.255 | PASS | +48.1 pp over base |
| TASA G4 | aser_g4_safe (tasa backbone) | GSM8K full 1319 | 44 | acc | 0.6725 | >base 0.255 | PASS | +41.7 pp over base |
| TASA G4 | aser_g4_safe (tasa backbone) | GSM8K full 1319 | 45 | acc | 0.5330 | >base 0.255 | PASS | +27.8 pp over base |
| Dr.GRPO G4 | aser_g4_safe (dr_grpo backbone) | GSM8K full 1319 | 42 | acc | 0.2851 | >base 0.255 | MARGINAL | +3.0 pp over base |
| Dr.GRPO G4 | aser_g4_safe (dr_grpo backbone) | GSM8K full 1319 | 43 | acc | 0.2782 | >base 0.255 | MARGINAL | +2.3 pp over base |
| Dr.GRPO G4 | aser_g4_safe (dr_grpo backbone) | GSM8K full 1319 | 44 | acc | 0.2707 | >base 0.255 | MARGINAL | +1.6 pp over base |
| Dr.GRPO G4 | aser_g4_safe (dr_grpo backbone) | GSM8K full 1319 | 45 | acc | 0.2767 | >base 0.255 | MARGINAL | +2.2 pp over base |

### Aggregate (n=4 seeds per variant)

| Method | Mean accuracy | Std | Delta vs base (25.5%) |
|--------|:-------------:|:---:|:---------------------:|
| **TASA G=4** | **62.57%** | 9.52 pp | **+37.1 pp** |
| **Dr.GRPO G=4** | **27.77%** | 0.59 pp | **+2.3 pp** |
| Base Qwen3.5-9B | ~25.5% | — | — |

**TASA improves GSM8K full-set accuracy by +37.1 pp over base and +34.8 pp over Dr.GRPO.**
Dr.GRPO under G=4 + binary reward barely moves from baseline — its mean-centered
advantage collapses when only {0, 1} rewards are available with a small group.

**Training reward (log_history, last step):**

| Backbone | seed42 | seed43 | seed44 | seed45 | mean |
|----------|:------:|:------:|:------:|:------:|:----:|
| TASA | 0.813 | 1.000 | 0.750 | 0.500 | 0.766 |
| Dr.GRPO | 0.625 | 0.313 | 0.500 | 0.438 | 0.469 |

Training reward gap: TASA mean 0.766 vs Dr.GRPO mean 0.469 (+0.297 absolute, ~63% relative).

### Observations so far

1. TASA full-set accuracy is far above base (25.5%) and far above the Wave-14 500-step
   SPO+Replay full-set collapse point (~44.6%).
2. TASA shows high seed variance (0.56–0.74, std ~7.2 pp) — consistent with small-G RLVR
   instability observed in prior work (RePO, DAPO).
3. Final Dr.GRPO comparison pending.
4. Dr.GRPO training reward is already substantially lower (0.469 vs 0.766) — suggests
   Dr.GRPO's mean-centered advantage is less effective than TASA's threshold-anchored
   signed form under binary reward + G=4.

### What this does NOT show
- Does NOT test TRACE-GRPO's trust-gated replay mechanism.
- Does NOT separate "TRACE mechanism" from "TASA mechanism" — both are orthogonal
  changes and currently the running experiment only tests TASA.
- Does NOT compare to SFT-gold (84.6% per Wave 13). TASA is still behind SFT-gold.
- Does NOT include variance analysis beyond point estimates (4 seeds is n=4).

## Experiment ID: TRACE A/B/C (PENDING)

**Will launch**: `bash launch_trace_abc.sh` after current evals free GPUs.

| Variant | Command |
|---------|---------|
| A. Existing best fragment | `run_aser_mvp.py --backbone spo --lambda-rep 0.05 --config configs/aser_g4_safe.yaml` |
| B. TRACE infra, no trust gate | `run_trace_grpo.py --trace-mode constant_gate` |
| C. Full TRACE-GRPO | `run_trace_grpo.py --trace-mode full` |

Expected 2 seeds × 3 variants × 200 steps = 6 runs. Evaluate on GSM8K full n=1319
with --selection full.

### Smoke Test (PENDING)
`python3 scripts/run_trace_grpo.py --seed 42 --max-steps 2 --trace-mode full`

Needs GPU; deferred until current evals complete.

### Unit Tests (COMPLETED)
| Test | Command | Status |
|------|---------|--------|
| PromptCreditState frontier | `python3 tests/test_prompt_credit_state.py` | PASS (6/6) |
| TrustGatedReplayBank trust weights | `python3 tests/test_trust_gated_replay_bank.py` | PASS (7/7) |
| Provenance manifest | `python3 tests/test_provenance.py` | PASS (3/3) |
| Eval selection modes | `python3 tests/test_eval_selection.py` | PASS (3/3) |
