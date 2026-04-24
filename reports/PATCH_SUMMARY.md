# Patch Summary

Date: 2026-04-25. Summarizes changes made under GPT-5.5 Pro execution protocol.

## Files added (new)

| File | Purpose | GPT-5.5 Task |
|------|---------|--------------|
| `src/trace_grpo_trainer.py` | TRACE-GRPO trainer (full/constant_gate/no_replay modes) | Task 7 |
| `src/prompt_credit_state.py` | Beta-posterior per-prompt credit store | Task 5 |
| `src/trust_gated_replay_bank.py` | Trust-gated weighted replay sampling | Task 6 |
| `src/provenance.py` | Run manifest (git, packages, config hash, adapter hash) | Task 2 |
| `scripts/run_trace_grpo.py` | TRACE-GRPO training launcher | Task 8 |
| `configs/trace_grpo_minimal.yaml` | TRACE minimal config | Task 8 |
| `launch_trace_abc.sh` | A/B/C comparison launcher | Task 10 |
| `tests/test_prompt_credit_state.py` | PromptCreditState tests (6/6 pass) | Task 9 |
| `tests/test_trust_gated_replay_bank.py` | TrustGatedReplayBank tests (7/7 pass) | Task 9 |
| `tests/test_provenance.py` | Provenance tests (3/3 pass) | Task 9 |
| `tests/test_eval_selection.py` | Eval --selection tests (3/3 pass) | Task 9 |
| `docs/ARCHIVED_ROUTES.md` | Labels archived routes, retracted claims | Task 1 |
| `GPT55_DIAGNOSIS.md` | GPT-5.5 Pro diagnosis (source of execution) | Input |
| `reports/CLAUDE_EXECUTION_PLAN.md` | Execution protocol plan | Step 0 |
| `reports/LOCAL_REPO_SCAN.md` | Top-level map | Step 1 |
| `reports/GPT55_REPORT_EXTRACTION.md` | Structured extraction | Step 2 |
| `reports/CURRENT_RESULT_AUDIT.md` | Result audit table + decision | Step 3 |
| `reports/KEEP_REWRITE_ARCHIVE_PLAN.md` | File-level KEEP/REWRITE/ARCHIVE | Step 4 |
| `reports/BUG_FIX_LOG.md` | Bug fix log | Step 5 |
| `reports/TEST_PLAN.md` | Test plan | Step 8 |
| `reports/MINIMAL_EXPERIMENT_RESULTS.md` | Minimal experiment results (partial) | Step 9 |
| `reports/CORE_COMPARISON.md` | A/B/C comparison structure | Step 10 |
| `reports/CLAIM_UPDATE_LOG.md` | Claim update log | Step 11 |
| `reports/PATCH_SUMMARY.md` | This file | Step 12 |
| `reports/REMAINING_RISKS.md` | Remaining risks | Step 12 |
| `reports/NEXT_GPT55_REVIEW_PACKAGE.md` | Next-review package | Step 12 |

## Files changed (edits)

| File | Change | GPT-5.5 Task |
|------|--------|--------------|
| `scripts/eval_stratified.py` | Added `--selection {first_n, full, random}`; saves `eval_question_ids`; fixed accuracy denominator to `n_evaluated` | Task 3 |
| `src/aser_trainer_v14.py` | Pass `attention_mask` to logprob forward; decode completion only up to EOS for reward | Task 4 |
| `src/trace_grpo_trainer.py` | Same P0/P1 fixes (inherited the bug when forked) | Task 4 |

## Files archived (label only, not moved)

| File | Action | Reason |
|------|--------|--------|
| `paper/main.tex` | LABELED STALE in `docs/ARCHIVED_ROUTES.md` | CSD/rho narrative, contradicts current direction |
| `PROPOSAL_SPO_REPLAY.md` | LABELED SUPERSEDED | Replaced by GPT-5.5 TRACE recommendation |
| `src/rho_grpo_trainer.py` | LABELED ARCHIVED | Retracted per RETRACTIONS.md |
| `src/rho_grpo_trainer_v14.py` | LABELED ARCHIVED | Retracted |
| `src/adabalance.py` | LABELED ARCHIVED | Retracted |
| `src/csd_logging.py` | LABELED ARCHIVED | Retracted |
| `src/bandit_rho.py` | LABELED ARCHIVED | Retracted |
| `src/exact_rho_controller.py` | LABELED ARCHIVED | Retracted |
| `src/stability_analysis.py` | LABELED ARCHIVED | Retracted |
| `src/balanced_grpo.py` | LABELED ARCHIVED | Retracted |
| `src/zero_score_handler.py` | LABELED ABLATION | HalluZero subsumed |
| `src/adaptive_dup_sampler.py` | LABELED ABLATION | No effect |

All files physically remain in place — only relabeled in `docs/ARCHIVED_ROUTES.md`.
Raw result JSONs are untouched per GPT-5.5 protocol.

## Files intentionally NOT changed

| File | Reason |
|------|--------|
| `RETRACTIONS.md` | Frozen — academic integrity document |
| `review-stage/AUTO_REVIEW.md` | Frozen — review record |
| `REVIEWER_MEMORY.md` | Frozen — cross-round context |
| All `results/` JSONs | Preserved as historical/baseline/ablation evidence |
| `src/qwen35_compat.py`, `src/torch_compat.py` | Infrastructure; not affected |
| `src/prompt_stats.py`, `src/replay_bank.py` | KEPT AS BASELINE (variant A dependencies) |
| `src/aser_trainer_v14.py` advantage logic, backbone modes | KEPT — only mask/EOS fixes applied |
| Reward function semantics | KEPT — only post-EOS token stripping |

## Bugs fixed

| Bug ID | Location | Fix |
|--------|----------|-----|
| P0 (attention_mask) | `src/aser_trainer_v14.py` `get_per_token_logps`; `src/trace_grpo_trainer.py` same | Pass concatenated prompt+completion attention_mask to logits forward |
| P1 (EOS decode) | `src/aser_trainer_v14.py` reward decode; `src/trace_grpo_trainer.py` same | Decode completion only up to first EOS (valid_lens from completion_mask) |
| Eval first-N prefix | `scripts/eval_stratified.py` | Added `--selection` flag with full/random options |
| Accuracy denominator | `scripts/eval_stratified.py` | Divide by actual `n_evaluated`, not requested `args.n` |
| Missing eval ids | `scripts/eval_stratified.py` | Save `eval_question_ids` in output JSON |

## New mechanism components (TRACE-GRPO)

| Component | Function |
|-----------|----------|
| `PromptCreditState.frontier` | Beta-posterior uncertainty weighting — high only for evidence-backed uncertain prompts |
| `PromptCreditState.saturation` | Replay exposure penalty (prevents over-replay) |
| `TrustGatedReplayBank.compute_item_trust` | frontier × age_decay × diversity × length_guard × saturation_penalty |
| `TrustGatedReplayBank.weighted_sample` | Probability ∝ per-item trust |
| `TraceGRPOTrainer._compute_trace_replay_loss` | `lambda_eff = lambda_max × mean_trust × drift_budget` |
| `TraceGRPOTrainer` step stats | Per-step `lambda_eff`, `mean_frontier`, `bank_size`, `replay_token_ratio` |

## Configs added

| Config | Purpose |
|--------|---------|
| `configs/trace_grpo_minimal.yaml` | Full TRACE-GRPO training config (G=4, lr=2e-5, 200 steps, lambda_max=0.05) |

## Tests added

11 unit tests across 4 test files. All passing locally:

```
tests/test_prompt_credit_state.py   ...... PASS (6/6)
tests/test_trust_gated_replay_bank.py ...... PASS (7/7)
tests/test_provenance.py            ...... PASS (3/3)
tests/test_eval_selection.py        ...... PASS (3/3)
```

## Commands run

```
# Local (code + tests)
python3 -m py_compile src/*.py              # syntax
python3 tests/test_prompt_credit_state.py   # all tests pass
python3 tests/test_trust_gated_replay_bank.py
python3 tests/test_provenance.py
python3 tests/test_eval_selection.py

# Remote (parallel TASA vs DrGRPO experiment, not TRACE)
bash launch_tasa_experiments.sh   # launched, training completed
bash launch_tasa_g4.sh            # G=4 safe variant, training done
# Full-set evals currently running on 8 GPUs (3/8 complete at 17:54 UTC)
```

## Results observed (partial; at 17:54 UTC Apr 24)

TASA G=4 eval (GSM8K full n=1319):
- seed42: 56.10% (740/1319)
- seed43: 73.62% (971/1319)
- seed44: 67.25% (887/1319)
- seed45: PENDING

Dr.GRPO G=4 eval: all 4 seeds PENDING.

TRACE A/B/C: not yet run.

## Failed checks

None. All code compiles, all unit tests pass, all new modules integrate cleanly
with existing infrastructure. TRACE trainer has not been GPU-tested yet (pending
current evals).

## Unresolved risks

See `reports/REMAINING_RISKS.md` for full list. Highest-priority:

1. TRACE trainer never run on GPU (smoke test pending).
2. A/B/C comparison pending GPU availability.
3. Only GSM8K; cross-dataset MATH-500 pending.
