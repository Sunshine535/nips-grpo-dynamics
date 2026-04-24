# Claude Execution Plan

**Date**: 2026-04-24
**Source**: GPT55_DIAGNOSIS.md (repository root, 1089 lines)

## 1. Diagnosis File Location
`/home/tarkoy/nips/nips-grpo-dynamics/GPT55_DIAGNOSIS.md`

## 2. MAIN METHOD PATH
**TRACE-GRPO**: Trust-Calibrated Replay and Prompt-Conditioned Credit Assignment for Sparse Binary GRPO.

Core change: Replace fixed lambda_rep uniform replay CE with adaptive lambda_eff controlled by prompt posterior, trust gate, frontier score, age decay, and drift budget.

## 3. Missing Mechanism to Implement
**Prompt-conditioned trust-calibrated replay credit with drift control.**

Three new modules:
1. `PromptCreditState`: Beta-posterior per prompt (success/fail counts, frontier, uncertainty)
2. `TrustGatedReplayBank`: Weighted sampling by trust/age/diversity/length
3. `TraceGRPOTrainer`: Adaptive lambda_eff = lambda_max * frontier * trust * drift_budget

## 4. Current Evidence Supporting the Diagnosis

| Evidence | Supports? |
|----------|-----------|
| SPO+Replay n=200 positive signal (69.4%) | Yes — short-horizon verified replay has value |
| 500-step full-set collapse (44.6%) | Yes — fixed replay drifts |
| Phase diagram null (25-27%) | Yes — global scalar not the mechanism |
| lambda=0.02 instability (75% vs 39%) | Yes — fixed lambda is fragile |
| RFT-only weak (36%) | Yes — CE alone insufficient |
| SFT-gold strong (84.6%) | Yes — need honest baseline discipline |

## 5. Current Evidence Contradicting or Weakening the Diagnosis

| Evidence | Weakens? |
|----------|----------|
| TASA-GRPO (just implemented, running on server) | Partially — TASA changes advantage computation, not replay. GPT-5.5 focuses on replay trust instead. These are orthogonal; both could contribute. |
| No full-set 200-step eval yet | Unknown — if full-set 200-step shows no gain over baseline, the positive signal itself is questionable |
| Confidence rated "medium-low" by GPT-5.5 | Acknowledged — mechanism is hypothesis, not proven |

**Note on TASA vs TRACE**: GPT-5.5 Pro was not aware of the TASA-GRPO work (implemented after the diagnosis). TASA changes the advantage computation; TRACE changes the replay mechanism. These are complementary, not conflicting. TRACE can be implemented on top of either SPO or TASA backbone.

## 6. Files to Inspect
- `src/aser_trainer_v14.py` (attention_mask bug lines ~155, EOS decode ~175)
- `src/prompt_stats.py` (lacks posterior)
- `src/replay_bank.py` (uniform sampling)
- `scripts/eval_stratified.py` (first-N prefix line ~60)
- `scripts/run_aser_mvp.py` (provenance gaps)

## 7. Files to Edit
- `scripts/eval_stratified.py` — add --selection mode, save eval ids
- `src/aser_trainer_v14.py` — fix attention_mask and EOS decode

## 8. Files to Create
- `src/prompt_credit_state.py`
- `src/trust_gated_replay_bank.py`
- `src/trace_grpo_trainer.py`
- `src/provenance.py`
- `scripts/run_trace_grpo.py`
- `configs/trace_grpo_minimal.yaml`
- `tests/test_prompt_credit_state.py`
- `tests/test_trust_gated_replay_bank.py`
- `tests/test_eval_selection.py`

## 9. Files to Archive
- `PROPOSAL_SPO_REPLAY.md` → note as superseded
- `paper/main.tex` → note as stale

## 10. Files NOT to Touch
- `RETRACTIONS.md` (frozen)
- `review-stage/AUTO_REVIEW.md` (frozen for this iteration)
- All raw result JSONs in `results/`
- `src/qwen35_compat.py`, `src/torch_compat.py`, `src/rho_grpo.py` (infrastructure/baseline)

## 11. Tests to Run Before and After Changes
**Before**: `python -m compileall src scripts` (verify no syntax errors)
**After each task**:
- Task 3 (eval fix): verify --selection flag works, eval ids saved
- Task 4 (mask fix): verify logprobs change with padding
- Task 5 (PromptCreditState): unit test frontier behavior
- Task 6 (TrustGatedReplayBank): unit test weighted sampling
- Task 7 (TraceTrainer): 2-step smoke test

## 12. Rollback Conditions
- If attention_mask fix breaks TRL API → revert and document
- If PromptCreditState causes training crash → use old PromptStatsStore as fallback
- If TrustGatedReplayBank produces empty samples → fall back to uniform with warning
- If TRACE trainer fails smoke test → debug before any benchmark run

## Execution Order
1. Write LOCAL_REPO_SCAN.md ✅
2. Write GPT55_REPORT_EXTRACTION.md ✅
3. Write CLAUDE_EXECUTION_PLAN.md ✅ (this file)
4. Write CURRENT_RESULT_AUDIT.md
5. Write KEEP_REWRITE_ARCHIVE_PLAN.md
6. Fix eval protocol (Task 3)
7. Fix trainer mask/EOS (Task 4)
8. Implement PromptCreditState (Task 5)
9. Implement TrustGatedReplayBank (Task 6)
10. Implement TraceGRPOTrainer (Task 7)
11. Add configs and launcher (Task 8)
12. Add sanity tests (Task 9)
13. Run minimal verification (Task 10)
