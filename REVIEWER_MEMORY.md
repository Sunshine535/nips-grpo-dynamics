# Reviewer Memory (nightmare-mode persistent brain)

## Round 1 — (pending external review after ASE-R MVP completes)

### Suspicions carried from earlier critique rounds (author-provided summary)
- The repo has a long history of "left-brain fights right-brain" claims:
  "GRPO = CSD" globally (retracted), Theorem 2 capacity bound (retracted),
  Q_CSD as "proven collapse predictor" (demoted to empirical diagnostic),
  "ADQ implements Theorem 2" (narrowed to "proxy estimator"), monotonic
  ρ sweep on old single-seed data (retracted). All retractions are
  consolidated in `RETRACTIONS.md`.
- Prior 5 review rounds on the same codebase scored 2 → 4 → 4 → 5 → 6.
  The 6/10 almost state was achieved via stratified-eval finding +
  bandit-ρ experiment + acknowledgement that Theorem-2's gradient-
  variance objective ≠ test-accuracy objective.
- Latest user critique: the ρ-controller line is exhausted; pivot to
  "adaptive support expansion + replay" (ASE-R MVP): SPO or Dr. GRPO
  backbone, difficulty-adaptive rollouts via in-batch hard-prompt
  duplication, small verified-success replay bank.

### Known structural confounders to look for
- `gradient_checkpointing=True` in early configs corrupts rollouts in
  TRL 0.14 (token-loop bug). Fixed in all configs as of commit `05d3bc6`.
  Any *old* result (pre-commit `05d3bc6`) may be tainted; authoritative
  results are Wave 2, Wave 7, Wave 8, Wave 9, Wave 10 (ASE-R MVP).
- `src/rho_grpo_trainer.py` (legacy) fabricates synthetic degenerate-group
  labels; V14 and ASER trainers do NOT. Paper-grade claims should cite
  only V14 / ASER results.
- All controllers that rely on within-group variance estimation (ADQ
  proxy, exact-ρ*, Q_CSD) are structurally untestable at G=2 because
  n⁺ ≥ 2 is too rare. Wave 8 exact-ρ* recorded 0/20 non-degenerate
  updates (all skipped). Dr. GRPO (Wave 9) lost to fixed ρ=0.7 by 13pp.

### Established 3-seed Qwen3.5-9B / GSM8K numbers (n=200 per adapter)
| Method            | overall acc ± std | easy  | hard  |
|-------------------|-------------------|-------|-------|
| fixed ρ=0.70      | 52.3 ± 7.5        | 94.1  | 38.0  |  ← best
| bandit-ρ (UCB1)   | 50.2 ± 4.5        | 94.1  | 35.1  |
| fixed ρ=1.00      | 48.5 ± 5.6        | 92.2  | 33.6  |
| fixed ρ=3.00      | 48.3 ± 5.1        | 94.1  | 32.7  |
| ADQ (proxy ρ*)    | 46.8 ± 8.3        | 93.5  | 30.9  |
| exact-ρ*          | 47.5              |       |       |  ← controller never fired
| Dr. GRPO (no std) | 39.3 ± 4.0        |       |       |  ← published SOTA LOSES here
| base model        | 25.5              |       |       |

### What the reviewer should look for in this round
- Whether ASE-R MVP beats the fixed ρ=0.70 baseline (52.3%) on overall
  and hard-subset accuracy at matched compute.
- Honest treatment of the Dr. GRPO negative result (published SOTA
  that loses on our setup).
- Whether SPO backbone (per-prompt persistent baseline) fixes the
  degenerate-group problem that broke ADQ / Q_CSD / exact-ρ*.
- Whether adaptive-duplication + replay each add marginal value over
  the SPO backbone alone.
- Any new "left-brain vs right-brain" contradictions between
  RETRACTIONS.md, FINAL_PROPOSAL.md, and the new ASER results.
