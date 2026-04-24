# Remaining Risks

Date: 2026-04-25.

## HIGH — mechanism not yet validated

1. **TRACE A/B/C not yet run on GPU**
   - Unit tests pass, code compiles, logic verified numerically.
   - But `run_trace_grpo.py` has never been executed end-to-end with a real model.
   - Even a 2-step smoke test is pending GPU availability.
   - **Mitigation**: after current TASA/DrGRPO evals complete, run `bash launch_trace_abc.sh`
     with `MAX_STEPS=10` first as smoke test, then `MAX_STEPS=200` for real comparison.

2. **Diagnosis is confidence "medium-low"**
   - GPT-5.5 Pro explicitly labels the TRACE hypothesis as medium-low.
   - Current positive/negative evidence supports that fixed replay has problems,
     but it does not PROVE trust-gated replay solves them.
   - **Mitigation**: the A/B/C protocol with constant_gate ablation directly tests this.

3. **Reference model handling for LoRA adapter**
   - `get_per_token_logps` for ref model uses `disable_adapter()` on the model when
     `self.ref_model is None`. This is inherited from ASER and not revalidated for
     TRACE. If LoRA disabling breaks with Qwen3.5 + new transformers version, KL term
     will be wrong.
   - **Mitigation**: smoke test must log KL and check it is finite / non-trivial.

## MEDIUM — implementation correctness

4. **attention_mask fix was applied but not end-to-end tested**
   - Fix: pass `attention_mask` to `get_per_token_logps` in both trainers.
   - Not yet run on GPU. If the model forward signature differs across transformers
     versions, the fix could break.
   - **Mitigation**: smoke test after TASA evals; rollback to previous behaviour if
     errors appear.

5. **EOS decode fix changes reward-function inputs**
   - Fix: decode only up to first EOS for reward extraction.
   - Previously, reward function sometimes saw post-EOS junk tokens. Some reward
     functions may have silently handled this — now they see cleaner text.
   - **Mitigation**: the GSM8K binary reward uses regex on "####" which should be
     robust to either form; no reward-value changes expected, but confirm via smoke.

6. **TrustGatedReplayBank sampling is categorical with cumulative sum**
   - Current implementation uses manual cumulative sum loop. For `n=1` sample this
     is equivalent to `random.choices(population, weights=probs, k=1)`. For `n>1`
     without-replacement semantics it is NOT enforced — the same item could be sampled
     twice.
   - **Mitigation**: with `replay_batch_size=2` and weights strongly favouring a few
     frontier items, duplicates are possible. If duplicates harm learning, switch to
     `random.sample` with inverse-weight rejection. Not a correctness issue; a
     sample-efficiency issue.

7. **PromptCreditState uses global Python RNG via `random.random()`**
   - This is non-deterministic unless `random.seed()` is called.
   - The TraceGRPOTrainer uses `seed` for HuggingFace setup but does not pin Python
     `random` or `numpy.random` explicitly.
   - **Mitigation**: add `random.seed(args.seed)` in `run_trace_grpo.py` for full
     determinism; low priority since bank sampling is inherently stochastic.

## MEDIUM — experimental coverage

8. **Only GSM8K tested**
   - Reviewer will ask for MATH / MATH-500 or similar.
   - Current TASA training-reward improvements are on GSM8K binary reward only.
   - **Mitigation**: Cross-dataset MATH-500 is in task backlog (#55). After A/B/C
     passes, extend C to MATH with partial-credit reward (`src/math_reward.py`
     already exists).

9. **Only 2-3 seeds per variant**
   - High variance observed (TASA std ~7 pp across 3 seeds).
   - A 2-seed A/B/C may not reach statistical significance.
   - **Mitigation**: if C beats A, B by a practical margin, extend to 5 seeds before
     claiming statistical significance.

10. **No official RePO / DAPO / GSPO comparison**
    - Closest prior art on replay-enhanced policy optimization.
    - Without these, reviewers may label TRACE as "RePO + heuristics".
    - **Mitigation**: planned as baseline work once TRACE minimum-viable result is
      available. Not blocking for review package.

## LOW — documentation/housekeeping

11. **Paper draft still has CSD/rho narrative**
    - `paper/main.tex` is stale per `docs/ARCHIVED_ROUTES.md`.
    - Must be fully rewritten around TRACE-GRPO once A/B/C passes.
    - **Mitigation**: pending Task 12.

12. **Wave 10-13 n=200 results should be explicitly re-labeled as prefix-subset**
    - `results/stratified_eval/` JSONs don't have the `selection` field.
    - Currently implicit "first_n" default.
    - **Mitigation**: new evals use the fixed eval script with explicit `selection`
      field. Old JSONs kept as historical, labeled in paper.

13. **Provenance not retro-fitted to old runs**
    - New runs (TRACE A/B/C, any new TASA) will write `run_manifest.json`.
    - Old runs do NOT have manifests.
    - **Mitigation**: acceptable; old runs are archived as "historical" per
      `docs/ARCHIVED_ROUTES.md`.

## Stop conditions that have NOT triggered

- ✅ GPT55_DIAGNOSIS.md is present and readable.
- ✅ Current result files are accessible (TASA partial, others on disk).
- ✅ Basic commands run (module compile, unit tests pass).
- ✅ Data split code uses train for training and test for eval (grepped confirmed).
- ✅ Baseline comparison is fair (identical config for A vs B vs C planned).
- ✅ New method was implementable without changing the research question.
- ✅ Diagnosis NOT contradicted by current results (partial TASA evidence is
  neither for nor against TRACE's trust-gate hypothesis).

## Decisions log

- 2026-04-24: Chose to implement TRACE alongside (not instead of) TASA, since the
  two mechanisms are orthogonal.
- 2026-04-24: Kept `src/aser_trainer_v14.py` unchanged except for the attention_mask
  / EOS fix (non-semantic correctness), to preserve variant A exactly.
- 2026-04-24: Did NOT archive `paper/main.tex` yet — labeled STALE in
  `docs/ARCHIVED_ROUTES.md`, awaiting TRACE results before rewrite.
- 2026-04-25: Set default eval selection to `first_n` to preserve backward
  compatibility; new "paper-grade" runs must pass `--selection full`.
