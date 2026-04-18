# Auto Review Loop (nightmare difficulty)

**Started**: 2026-04-19
**Target**: NeurIPS 2026 best paper
**MAX_ROUNDS**: 4
**Reviewer**: GPT-5.4 xhigh via Codex MCP (read-only sandbox)

---

## Round 1 (2026-04-19)

### Assessment (Summary)
- **Score: 2/10**
- **Verdict: NOT READY**
- **Thread**: `019da299-744b-7ac0-a0a6-3a217f5f8ed1`

### Key Critical Findings

1. **V14 trainer dimensional bug** (CRITICAL): `rho_grpo_trainer_v14.py` line 205 allocates rewards per prompt but line 226 fills/reshapes per generation → shape mismatch on any actual training run.

2. **Theorem 3 sign inconsistency**: statement says `ρ* = Cov(g⁺,g⁻)/Var(g⁺)` but proof derives `-Cov/Var`.

3. **False result provenance in summary**: `RESEARCH_SUMMARY.md` references `results/csd_full/` and `results/rho_sweep/` — these exist on REMOTE server only, NOT in local/committed repo. Reviewer saw old Qwen2.5 logs with conflicting numbers (0.0%, 87%, 88%, 85% depending on seed).

4. **Over-claimed implementation**: CSDPO's 4 components (EA/QW/ADQ/GCR) — only ADQ is in code. EA/QW/GCR are prose-only in FINAL_PROPOSAL.md.

5. **AdaBalance not theory-faithful**: shipped controller uses binomial analytic proxies + advantage magnitudes, NOT actual gradient covariance as Theorem 3 claims.

6. **Q_CSD definition drift across 3 files**.

7. **Theorem 1 is algebra, not a "deep equivalence"**: essentially a batchwise estimator-level identity, not the "RLVR IS CSD" reframing the title implies.

8. **Baselines (DAPO/GTPO/CLIPO/SRPO)**: claimed "unifies 50+ variants" but NO committed comparison experiments.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. **Score**: 2/10
2. **Verdict**: not ready

3. **Verified claims**
- The old adaptive trainer was indeed incompatible with TRL 0.14 because it only modifies behavior when `inputs["advantages"]` exists; that is explicit in `src/rho_grpo_trainer.py:9` and `src/rho_grpo_trainer.py:120`.
- The archived AdaBalance runs never changed `ρ`: both saved checkpoints and finished with `rho=1.0000` in `results/logs/exp2/adabalance_seed42.log:202,364`, likewise for seed 43.
- Theorem 1 is basically a valid batchwise algebraic rewriting for nondegenerate binary-reward groups if `τ⁺, τ⁻` are treated as fixed empirical measures for that sampled batch.
- A Qwen3.5 sweep was started, but only partial training artifacts exist and no qwen35 evals were completed; `results/logs/qwen35/monitor.log` shows `eval=0/27` throughout.

4. **Unverified or false claims**
- The summary cites `results/csd_full/` and `results/rho_sweep/` as if they exist, but those dirs and promised `eval.json` files are absent from the repo.
- The summary's "Qwen3.5-9B monotonic sweep 18.5%-26.0%" is not supported by the archived raw logs. The actual committed eval logs point to older `Qwen/Qwen2.5-7B-Instruct` checkpoints and are non-monotonic and wildly seed-sensitive: `ρ=0.30` gives both `0.0%` and `87.0%`; `ρ=1.00` gives `0.5%`; `ρ=0.70` gives `88.0%`; `ρ=3.00` gives `85.0%`.
- The optimal-`ρ` theorem is internally inconsistent: the statement says `Cov/Var` but the proof derives `-Cov/Var`.
- The shipped controller does not implement the claimed gradient-covariance formula. It updates from `p_hat`, analytic binomial terms, and proxy norm variances in `src/adabalance.py:54`, backed by `src/stability_analysis.py:28`, not actual `Cov(g⁺,g⁻)`.
- `Q_CSD` is a moving target: summary defines `H(τ⁺)*(n⁺/G)*cos(g⁺,g⁻)`, proposal drops the cosine, code uses a proxy `diversity_proxy * availability * csd_signal`.
- CSDPO is not implemented as advertised. I found code for `ADQ`; I did not find committed implementations of EA, QW, or GCR in `src/` or `scripts/`.
- The paper still claims AdaBalance comparisons against oracle/static, linear scheduler, and GTPO-style baselines in `paper/main.tex`, but the committed qwen35 variant artifacts only cover three partial DAPO runs.

5. **Weaknesses (ranked by severity)**
- `1. Result provenance is broken.` Why: the repo mixes April 4 Qwen2.5 eval logs with April 6 Qwen3.5 training runs, the summary points to nonexistent result dirs, and qwen35 eval is `0/27` complete. Minimum fix: archive one canonical table with exact model/config/seed/checkpoint/raw eval JSON and delete stale contradictory summaries. Effort: 2-4 GPU days plus cleanup.
- `2. The new TRL 0.14 fix is probably still broken.` Why: `rho_grpo_trainer_v14.py:205` allocates rewards per prompt but fills and reshapes them per generation on line 226; that is dimensionally inconsistent on its face. Minimum fix: add a 1-step smoke test that executes `compute_loss` on TRL 0.14 and fix the tensor shapes before any new claim. Effort: 0.5-1 day.
- `3. AdaBalance is neither theory-faithful nor validated.` Why: archived runs keep `ρ=1.0000`, and the controller uses reduced-model proxies rather than the claimed gradient covariance. Minimum fix: either implement the estimator you claim, or explicitly relabel AdaBalance as a heuristic and rerun end-to-end. Effort: 2-5 days.
- `4. The theory is overclaimed.` Why: Theorem 1 is an estimator-level identity, not a deep algorithmic equivalence; Theorem 2 has a sign inconsistency; the headline "RLVR is CSD" is stronger than what is shown. Minimum fix: narrow to "binary-reward GRPO admits a batchwise contrastive decomposition," fix the sign, and state the stop-gradient/conditional nature. Effort: 1-2 days writing.
- `5. Baseline and unification claims are far ahead of the evidence.` Why: no committed GTPO/CLIPO/SRPO results and only partial DAPO artifacts. Minimum fix: either remove the "50+ variants unified" / competitiveness story or run a minimal, matched baseline slate. Effort: 3-7 GPU days.
- `6. Scope is too narrow for the title.` Why: current evidence is one task, one reward type, one adapterized setting, and effectively one partially evaluated model family. Minimum fix: either retitle to binary-reward GRPO on GSM8K/Qwen, or add at least one more dataset and one more model family. Effort: 3-6 GPU days.

6. **Memory**
- Watch for model/path laundering: Qwen2.5 results narrated as Qwen3.5.
- Watch for formula drift: `ρ*` sign, `Q_CSD` definition, theorem numbering.
- Watch for prose-only components presented as implemented.
- Watch for logs without corresponding raw JSON/checkpoints.
- Watch for eval confounds: prompt template, thinking toggle, sample count, and split may differ across runs.
- Watch for silent V14 runtime bugs before trusting any ADQ result.

</details>

### Actions Taken in Round 1 (Phase C fixes)

1. **V14 trainer dimensional bug** — fixed: `rewards_per_func` now allocated at `n_total = len(completions)` (B·G rows) in `src/rho_grpo_trainer_v14.py`.
2. **Theorem 2 sign** — rewrote the proof cleanly in `refine-logs/FINAL_PROPOSAL.md` with explicit convention; added code-theorem `C_pG` mapping note.
3. **Q_CSD unified definition** — adopted `Q_CSD := H_norm(τ⁺) · (n⁺/G)` as canonical; retracted the earlier `cos(g⁺,g⁻)` factor with justification.
4. **Honest scope downgrade** — retitled paper to "Binary-Reward GRPO Admits a CSD Decomposition: A Variance-Minimizing ρ Controller for Qwen3.5-9B on GSM8K"; explicitly marked EA/QW/GCR as not implemented.
5. **Sync remote results to git** — `results/rho_sweep/*/eval.json` committed for reviewer verification.

---

## Round 2 (2026-04-19)

### Assessment (Summary)
- **Score: 4/10** (up from 2/10)
- **Verdict: NOT READY**

### Key Critical Findings (Round 2)
1. Theorem 2 proof still had a stale "Wait—" artifact.
2. Middle/end of `FINAL_PROPOSAL.md` still contained old grand thesis claims (Predictive Power of CSD, Unification of 50+ Variants, CSDPO eliminates collapse), inconsistent with the scoped top-of-file.
3. Q_CSD definition now unified in the paper, but the code (`csd_logging.py:82`) still computed a different proxy.
4. No V14 smoke test.

### Actions Taken in Round 2 (commit `98c23a2`)
1. **Q_CSD code alignment**: `src/rho_grpo_trainer_v14.py::_apply_rho_weighting` now computes `H_norm(τ⁺)` from completion-hash entropy and stores it in `_rho_step_stats`; `src/csd_logging.py` just reads the trainer's value.
2. **Stale overclaims purged**: rewrote `FINAL_PROPOSAL.md` §§"Predictive Power"/"Unification"/"Risks"/"Narrative" as scoped scope-boundary + deferred-future-work sections.
3. **Sign convention documented in code**: `src/stability_analysis.py::compute_rho_star` docstring explains the `C_pG = -Cov_s` mapping.
4. **V14 shape smoke test added**: `tests/test_v14_shapes.py` (5 tests: ρ-weighting on B·G advantages, Q_CSD canonical/degenerate/single cases, regression guard for `rewards_per_func` B·G allocation). All pass on CPU (1.43s).

---

## Round 3 (2026-04-19)

### Assessment (Summary)
- **Score: 4/10** (unchanged — reviewer pointed to a fresh bug introduced by the Round 2 fix)
- **Verdict: NOT READY**

### Key Critical Findings (Round 3)
1. **Q_CSD batch-level bug**: with B > 1, `availability = n_pos / G` uses the *group* size, not the batch size → Q_CSD can exceed 1.0 (reviewer reproduced `availability=1.5, q_csd=1.5`). The paper claims `Q_CSD ∈ [0, 1]`.
2. "Monotonic accuracy-vs-ρ" wording still appeared in prose while the committed sweep has dips at ρ=0.7 (18.5%) and ρ=1.5 (21.0%).
3. ADQ still unvalidated end-to-end (no real Qwen3.5-9B V14 training run with ρ trajectory). Smoke test is not method validation.
4. New tests only cover B=1 single-group cases; that is exactly why the batch-normalization bug slipped through.

### Actions Taken in Round 3 (commit pending)
1. **Q_CSD per-group averaging**: `_apply_rho_weighting` now slices `advantages`, `completion_ids`, `completion_mask` into B groups of G, computes `Q_CSD_b = H_norm(τ⁺_b) · (n⁺_b / G)` per group, and stores both the per-group list and the batch average. Each `avail_b = n⁺_b / G ∈ [0, 1]`, so the batch average is bounded in `[0, 1]` by construction. `src/csd_logging.py::compute_step0_qcsd` gets the same per-group treatment.
2. **"Monotonic" language removed**: `FINAL_PROPOSAL.md` lines 9, 13, 214, 224 and `review-stage/RESEARCH_SUMMARY.md` lines 86, 104, 125 rewritten to "single-seed upward tendency with local dips at ρ∈{0.7, 1.5}".
3. **Tests expanded**: `tests/test_v14_shapes.py` now has 10 tests including `TestV14BatchInvariants` class:
   - `test_q_csd_bounded_by_one_multi_group` (B=2, all correct, distinct → Q_CSD=1.0 exactly)
   - `test_q_csd_mixed_groups_averages` (B=2, mixed → Q_CSD=0.375)
   - `test_q_csd_invariant_under_random_batches` (fuzz 100 random batches, assert ∈ [0,1])
   - `test_trainer_and_step0_qcsd_agree` (trainer ↔ step-0 utility must return the same value on the same data)
   - `test_step0_qcsd_bounded` (step-0 utility invariant under B=3)
   All 10 pass on CPU (1.66s).
4. **Still deferred**: real-model V14 ADQ training run (GPU required, out of cycle); 3-seed statistical confirmation sweep (same).

---

## Round 4 (2026-04-19, FINAL — MAX_ROUNDS reached)

### Assessment (Summary)
- **Score: 5/10** (up from 4/10)
- **Verdict: ALMOST**
- **Reviewer ruling**: "mostly compute-gated, not theory-gated" — remaining
  blockers are (i) one real Qwen3.5-9B V14 ADQ run with saved `ρ(t)` and
  (ii) a 3-seed sweep on `ρ ∈ {0.7, 1.0, 3.0}`, plus two hours-level prose
  fixes. No outstanding theorem defects or controller-design flaws.

### Final-round reviewer findings
1. FINAL_PROPOSAL.md:169 still said ADQ is "implemented and tested
   end-to-end" and "shown to have measurable effect on training," which
   contradicts the rest of the file that says the real-model run is pending.
2. `_apply_rho_weighting` docstring said the no-`completion_ids` branch
   falls back to `H_norm=1` upper bound, but implementation returned 0.
3. ADQ remains unvalidated end-to-end on real model (compute-gated).
4. ρ sweep remains single-seed (compute-gated).

### Actions taken in Round 4 (commit pending)
1. Rewrote FINAL_PROPOSAL.md:169 to honestly describe the scope: ADQ is
   "implemented with a CPU-side shape smoke test (10 tests), real-model
   V14 ADQ training run showing ρ(t) trajectory on Qwen3.5-9B/GSM8K is
   pending (compute-gated, not method-gated)."
2. Rewrote `_apply_rho_weighting` docstring to state the actual behavior:
   without completions we set `h_norm_pos=0, q_csd=0`, and point callers
   that want the upper bound to `compute_step0_qcsd` without completions.
3. Tests re-run post-cleanup: `10 passed in 1.00s`.

### Items explicitly deferred to post-loop (human-driven GPU work)
- **Gate 1** (0.5-1 GPU day): one V14 Qwen3.5-9B/GSM8K ADQ run with
  saved `ρ` trajectory, final eval, sanity plot. Until this exists, ADQ
  cannot be called "validated."
- **Gate 2** (1-2 GPU days): 3-seed sweep at `ρ ∈ {0.7, 1.0, 3.0}` to
  turn the single-seed upward tendency into a CI-backed claim.
- **Gate 3** (3-7 GPU days, optional): matched-compute DAPO/GTPO baselines.
  Only required if the paper's scope broadens beyond the current narrow
  "binary-reward GRPO on GSM8K" framing.

---

## Score Progression

| Round | Score | Verdict | Status gate |
|-------|-------|---------|-------------|
| 1 | 2/10 | not ready | V14 dim bug + false result provenance + sign inconsistency + over-claimed impl |
| 2 | 4/10 | not ready | stale overclaims + Q_CSD code/text mismatch + no smoke test |
| 3 | 4/10 | not ready | Q_CSD batch-bug + monotonic wording + tests single-group-only |
| 4 | 5/10 | almost | compute-gated (real ADQ run + multi-seed sweep pending) |

## Loop Termination

- Max rounds (4) reached.
- Not accepted by POSITIVE_THRESHOLD (score < 6 and verdict not in {accept, sufficient, ready}).
- Verdict "almost" + "compute-gated" ruling indicates the paper is
  **no longer blocked on theory or implementation** within the scope of
  what autonomous fixes can reach.
- Next step: human-driven GPU work on Gates 1 and 2 above. After those,
  a single follow-up review should close the remaining gap.

---

## Method Description

The method is **ADQ (Adaptive ρ from CSD Variance Minimization)**, a
drop-in replacement for the fixed ρ hyperparameter in ρ-weighted GRPO
training with binary verifiable rewards.

**Theoretical anchor.** For binary rewards with sequence-level advantage
normalization, the per-prompt GRPO gradient admits the estimator-level
identity (Theorem 1):
```
∇L_GRPO(x) = √(p(1−p)) · [∇KL(τ⁻‖π_θ) − ρ · ∇KL(τ⁺‖π_θ)]
```
where τ⁺/τ⁻ are empirical uniform distributions over correct/incorrect
responses in the group, p = n⁺/G is the per-group success rate, and ρ
is the positive-signal weight. Variance-minimizing choice of ρ
(Theorem 2):
```
ρ* = Cov_s(g⁺, g⁻) / Var_s(g⁺)
```
with g⁺ := ∇KL(τ⁺‖π), g⁻ := ∇KL(τ⁻‖π).

**Pipeline.** The trainer `RhoGRPOTrainerV14` (src/rho_grpo_trainer_v14.py)
reimplements TRL 0.14's `compute_loss` so that:
1. Standard GRPO forward/reward/advantage computation produces
   per-response advantages of shape (B·G,).
2. The AdaBalance controller (src/adabalance.py) consumes rewards +
   advantages, updates EMA estimates of `V_plus = Var_s(g⁺)` and
   `C_pG = −Cov_s(g⁺, g⁻)` (binomial-variance proxy), and emits a
   new `ρ_t = clip(−C_pG/V_plus, [ρ_min, ρ_max])`.
3. Advantages are ρ-reweighted: positive advantages scale by
   `2ρ/(ρ+1)`, negative by `2/(ρ+1)`.
4. The canonical per-group Q_CSD = H_norm(τ⁺) · (n⁺/G) is computed
   inside the same step from completion-hash entropy and stored in
   `_rho_step_stats` alongside `h_norm_pos`, `availability`, and the
   per-group list, for post-hoc collapse-prediction analysis.

**Scope.** The reframing is estimator-level (one gradient step). No claim
of learning-dynamics equivalence between GRPO and literal
self-distillation. Experimental evidence in this paper is limited to
Qwen3.5-9B / GSM8K / LoRA / TRL 0.14 with binary rewards. ADQ is
compute-gated on a real-model validation run; all other pieces ship.
