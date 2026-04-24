# Keep / Rewrite / Archive Plan

| Item | Path | Action | Reason |
|------|------|--------|--------|
| TRACE trainer | src/trace_grpo_trainer.py | NEW | Core TRACE-GRPO implementation |
| Prompt credit state | src/prompt_credit_state.py | NEW | Beta-posterior per prompt |
| Trust replay bank | src/trust_gated_replay_bank.py | NEW | Trust-gated weighted sampling |
| TRACE launcher | scripts/run_trace_grpo.py | NEW | A/B/C mode launcher |
| TRACE config | configs/trace_grpo_minimal.yaml | NEW | Main experiment config |
| MATH reward | src/math_reward.py | NEW | Partial-credit for MATH |
| Eval script | scripts/eval_stratified.py | REWRITE | Added --selection, eval ids |
| ASER trainer | src/aser_trainer_v14.py | KEEP AS BASELINE | Variant A control |
| Prompt stats (legacy) | src/prompt_stats.py | KEEP AS BASELINE | Legacy EMA for variant A |
| Replay bank (legacy) | src/replay_bank.py | KEEP AS BASELINE | Legacy uniform for variant A |
| ASER launcher | scripts/run_aser_mvp.py | KEEP AS BASELINE | Legacy launcher for variant A |
| ASER config | configs/aser_mvp.yaml | KEEP AS BASELINE | Variant A config |
| Qwen compat | src/qwen35_compat.py | KEEP | Infrastructure |
| Torch compat | src/torch_compat.py | KEEP | Infrastructure |
| RETRACTIONS | RETRACTIONS.md | FREEZE | Academic integrity |
| Auto review | review-stage/AUTO_REVIEW.md | FREEZE | Review history |
| GPT-5.5 diagnosis | GPT55_DIAGNOSIS.md | FREEZE | Source of truth |
| Old rho/CSD modules | src/adabalance.py, bandit_rho.py, etc. | ARCHIVE (label stale) | Retracted per RETRACTIONS |
| Paper draft | paper/main.tex | ARCHIVE (stale) | CSD/rho narrative invalid |
| Old proposal | PROPOSAL_SPO_REPLAY.md | ARCHIVE (superseded) | Replaced by TRACE |
| Phase diagram results | results/wave14_phase_diagram/ | KEEP AS NEGATIVE EVIDENCE | Shows scalar control fails |
| 500-step results | results/wave14_500step/ | KEEP AS NEGATIVE EVIDENCE | Shows fixed replay drifts |
| Wave 10-13 evals | results/stratified_eval*/ | KEEP AS HISTORICAL EVIDENCE | n=200 prefix results |
| SFT-gold results | results/stratified_eval_wave13/ | KEEP AS BASELINE | Strong upper bound |
