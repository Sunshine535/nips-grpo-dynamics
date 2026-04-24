# Core Comparison (A / B / C)

**Status**: INCOMPLETE — TASA vs Dr.GRPO evals in progress; TRACE A/B/C launch pending GPU availability.

## Structure per GPT-5.5 Pro protocol

| Variant | Meaning | Config |
|---------|---------|--------|
| A | Existing Best Positive Fragment Only | `run_aser_mvp.py --backbone spo --lambda-rep 0.05` (legacy ASER with SPO + fixed-λ verified replay CE) |
| B | New MAIN METHOD Without New Mechanism | `run_trace_grpo.py --trace-mode constant_gate` (TRACE infra but `lambda_eff = lambda_max` always) |
| C | Full New MAIN METHOD | `run_trace_grpo.py --trace-mode full` (adaptive lambda_eff with prompt credit, trust, drift budget) |

## Planned comparison table

| Variant | Config | Dataset | Seeds | Metric Mean | Std | Compared To | Result | Interpretation |
|---------|--------|---------|-------|-------------|-----|-------------|--------|----------------|
| A | aser_g4_safe + lambda_rep=0.05 | GSM8K full 1319 | 42, 43 | PENDING | PENDING | baseline control | PENDING | "Is the existing best fragment still positive on full-set?" |
| B | trace_grpo_minimal --trace-mode constant_gate | GSM8K full 1319 | 42, 43 | PENDING | PENDING | A | PENDING | "Does TRACE infra alone (without trust gate) beat A?" |
| C | trace_grpo_minimal --trace-mode full | GSM8K full 1319 | 42, 43 | PENDING | PENDING | A and B | PENDING | "Does the trust gate ITSELF cause the gain?" |

## Interpretation decision tree (GPT-5.5 protocol)

Once results populate, apply:

1. If C > A and C > B consistently → TRACE trust gate likely adds real value. Proceed to broader baselines (SFT-gold, Dr.GRPO, RePO).
2. If C ≈ A → TRACE may be reusing old positive fragment. Do NOT claim new mechanism. Diagnose why the mechanism is inactive.
3. If C ≈ B → trust gate inactive or irrelevant; check `lambda_eff` / `mean_frontier` in `trace_step_stats.json`.
4. If C < A → new method hurts. Inspect implementation OR diagnosis may be wrong.
5. If results are unstable across seeds → do NOT claim success; add stability analysis (more seeds or narrower variance).
6. If C only wins on one subset → treat as narrow signal; document limits.

## Intermediate signal (TASA vs Dr.GRPO training reward, not TRACE)

| Backbone | seed42 | seed43 | seed44 | seed45 | mean |
|----------|:------:|:------:|:------:|:------:|:----:|
| TASA | 0.813 | 1.000 | 0.750 | 0.500 | 0.766 |
| Dr.GRPO | 0.625 | 0.313 | 0.500 | 0.438 | 0.469 |

TASA training reward ~63% higher than Dr.GRPO. Full-set test accuracy pending.
**This is tangential to TRACE; TASA changes the advantage, TRACE changes replay trust.**
Both mechanisms are orthogonal and can be combined.

## Files produced after full comparison

When TRACE A/B/C completes:
- `results/trace_abc/A_legacy/evals/eval_A_legacy_spo_replay_seed*.json`
- `results/trace_abc/B_constant/evals/eval_B_trace_constant_gate_seed*.json`
- `results/trace_abc/C_full/evals/eval_C_trace_full_seed*.json`
- `results/trace_abc/*/trace_step_stats.json` (for B, C) containing `lambda_eff` trajectory
- `results/trace_abc/*/prompt_credit_dump.json` (for B, C) with final posterior state
