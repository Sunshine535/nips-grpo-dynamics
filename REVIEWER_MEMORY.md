# Reviewer Memory (nightmare-mode persistent brain)

## Round 2 — SPO+Replay with Codex's R1 fixes — Score: 5/10 (almost)
Reviewer: Codex xhigh (same thread `019dacf3-5e9a-7a41-8097-e31278f80315`)
Date: 2026-04-21 07:35 UTC

### Resolved from Round 1 (Codex confirmed)
- ✅ Wave-10 artifacts all committed (`results/wave10_aser/*`, `results/stratified_eval_aser/*`). spo_full n=9 recomputes to 69.4±10.4%.
- ✅ Matched-n fixed-ρ=0.70 baseline: 54.2±10.2%, Welch vs SPO+Replay t=3.12, df≈16.
- ✅ `pg_weight=0` RFT control is real in code + telemetry.
- ✅ RFT-only n=7 = 35.9±1.3% vs SPO+Replay 69.4% → +33.5pp (t=9.54). Enough to rule out "pure online replay".
- ✅ Docs improved; new PROPOSAL_SPO_REPLAY.md + README pivot.

### Still open from Round 2
1. **[high]** Adaptive duplication STILL doesn't fire — Codex verified batch_n_dup=0 for all 400 steps × 6 fixed-sampler runs. The earlier "fix" only changed `int()` → probabilistic round; the REAL bug was "replace with independent hardness sample" (not duplication). Semantic fix applied (commit 4b16306+): with prob `frac`, fill ALL batch slots with ONE hardness-weighted prompt.
2. **[high]** Novelty vs RFT is not fully isolated — each `pg_weight=0` run builds its own bank from pg-weight=0 rollouts. The "same bank" comparison is missing. Fix: run `scripts/run_sft_gold_control.py` using GSM8K gold solutions as a frozen perfect-supervision bank (theoretical upper bound for SFT).
3. **[medium]** analyze_round2.py double-counted rft_seed45 (w11 + w12 both included). Fixed: keep w12 only; w11 seed 45 reruns reported as sensitivity check.
4. **[medium]** Hard-subset number was 38.0% (R1) → should be 40.7% (matched n=9). Recomputed.
5. **[medium]** External validity empty — still only Qwen3.5-9B/GSM8K. Needs one cross-dataset or cross-family replication.

### Codex's direct answers (quote)
- "+33.5pp over the current pg_weight=0 control is enough to rule out the simple story 'this is just pure online replay/RFT'. It is not enough to establish full novelty over 'any SFT on verified successes'."
- "The n=9 vs n=9 overall win over fixed ρ=0.70 is now good enough for the narrow claim 'beats fixed ρ=0.70 on Qwen3.5-9B / GSM8K under this protocol'."
- "NeurIPS best-paper: nowhere close. NeurIPS accept: still not there yet, but closer."
- Recommended pivot: **"SPO backbone + small-K verified replay unlocks learning at G=2 where ρ-controller approaches fail on this setup"**

### Score progression
Round 1: 3/10 (not ready) — adaptive-dup no-op + no committed artifacts
Round 2: 5/10 (almost) — fatal fixes done; novelty/provenance gaps remain

### Watchlist for Round 3
- [CRITICAL] SFT-gold control result: if ≥65%, novelty collapses; if <50%, GRPO credit-assignment confirmed as necessary
- [HIGH] True-dup sampler rerun: does real duplication move the needle vs broken sampler?
- Any new analysis script bugs
- New external replication?

---

## Round 1 — ASE-R MVP review complete — Score: 3/10 (not ready)
Reviewer: Codex xhigh (Oracle MCP unavailable → fallback per `shared-references/reviewer-routing.md`)
threadId: `019dacf3-5e9a-7a41-8097-e31278f80315`
Date: 2026-04-21

### Verdict summary
**3/10, NOT ready**. Two fatal findings that invalidate the current result narrative:
1. **Adaptive duplication never fires**: `int(batch_size * frac) = int(2 * 0.25) = 0`. So all 9 ASE-R "full MVP" runs reported in AUTO_REVIEW.md are actually "SPO + verified replay CE" — the `adaptive_dup` component did nothing. The "SPO+dup" ablation is literally "SPO" with different seeds.
2. **Zero ASE-R artifacts committed**: 9 per-seed eval JSONs, replay bank dumps, and telemetry are not in `results/` — only live on the remote GPU. The 69.4% / 60.0% / 88.0% table is Markdown-only until synced.

Other high-severity concerns:
3. **Replay CE ≈ online RFT**: need matched-compute RFT control.
4. **Stats asymmetry**: ASE-R at n=9 vs baselines at n=3 — MDE ≈ 18.5pp overall (observed 17.1pp) → borderline.
5. **Stale docs**: root `FINAL_PROPOSAL.md` missing, README still says "Metastable Training Dynamics".

### What's validated (green lights from Codex)
- SPO EMA baseline actually persists across steps — not re-initialised per batch.
- `prompt_id` is per-example dataset index (no batch-level mis-keying).
- Replay CE is plain LM cross-entropy on verified successes only (no polluted buffer).
- No eval-to-replay leakage path (train=GSM8K train, eval=GSM8K test).

### Unresolved at end of Round 1
- Does sampler-fix rerun change the numbers? (If dup matters, yes; if not, drop the claim entirely.)
- Does matched-RFT control reach the same test accuracy? (If yes, novelty dead.)
- Does the gap survive n=9 vs n=9 matched-seed baseline?

---

## Pre-Round 1 — Carryover suspicions (from earlier CSD/ADQ paper)

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
