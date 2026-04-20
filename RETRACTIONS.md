# Retractions and Single-Source-of-Truth

**Status as of 2026-04-20**: this file supersedes all older claims across
`refine-logs/FINAL_PROPOSAL.md`, `review-stage/RESEARCH_SUMMARY.md`,
`EXPERIMENT_STATUS.md`, and the auto-review-loop rounds whenever the two
disagree. An external audit identified several self-contradictions in the
repo ("left-brain vs. right-brain"); this document records each
retraction in one place so downstream readers don't need to reconcile
inconsistent claims across files.

Result artefacts (`results/`) are **not** deleted — they remain on disk
and in git as raw evidence — but the *claims built on top of them* are
narrowed per this file.

## 1. Retracted top-level claims

| Claim | Status | Reason |
|-------|--------|--------|
| "GRPO IS Contrastive Self-Distillation" | **Retracted in the global form** | `PROOF_AUDIT.md` already flagged this scope as wrong. Keep ONLY the narrow form: "the ρ-weighted, binary-reward GRPO per-step gradient admits an estimator-level algebraic identity that rewrites to a signed combination of ∇KL(τ⁺‖π) and ∇KL(τ⁻‖π) for in-batch empirical distributions τ⁺/τ⁻". This is an identity on one update step, not a learning-dynamics equivalence. |
| "Theorem 2: capacity bound acc(π_T) ≤ pass@(G·T_eff)" | **Retracted as a theorem** | `PROOF_AUDIT.md` marks it FATAL. Cite the empirical BPR result (NeurIPS'25) as external evidence and label our statement "Empirical Prediction 1" (already done in `FINAL_PROPOSAL.md:115–125`). Do not call it a theorem. |
| "Q_CSD is a proven collapse predictor" | **Retracted** | Q_CSD := H_norm(τ⁺)·(n⁺/G) is only an *empirical diagnostic* whose predictive power we could not test in our setup: at G=2 it is identically zero, at G=3 it was non-zero in 1/200 observed steps (Wave 6 G=3 attempt). Keep as diagnostic metric; do NOT call it a law. |
| "ADQ implements Theorem-2 ρ*" | **Retracted in this strong form** | `src/adabalance.py` computes ρ* via a binomial-variance approximation of Cov(g⁺,g⁻)/Var(g⁺), not the true gradient moments. The Wave 8 exact-ρ* experiment attempted true estimation via two extra backward passes per group and found the estimator itself is undefined at G=2 (all 100% of observed update calls had <2 non-degenerate groups → no update). Both approximate and exact ρ* estimators are compute-gated on G≥4, which OOMs Qwen3.5-9B on 80 GB A800. |
| "ρ controller universally beats fixed ρ" | **Retracted** | Stratified n=200 eval (3 seeds per arm): fixed ρ=0.70 mean 52.3 ± 7.5, ADQ 46.8 ± 8.3, exact-ρ* 47.5 (5 seeds), bandit 50.2 ± 4.5. The fixed ρ=0.70 baseline wins on mean test accuracy; the bandit is 2nd with the smallest std. No claim of controller dominance should be made. |
| "ρ=0.7→0.87% / ρ=1.0→0.5% / Qwen3.5 9B monotonic upward" (early single-seed sweep) | **Retracted** | Single-seed numbers in early `RESEARCH_SUMMARY.md` (0.5% / 87% / 88% / 85%) came from a stack with `gradient_checkpointing=true` (later shown to corrupt TRL 0.14 rollouts) and mixed Qwen2.5 logs. They are kept on disk as archival but are NOT used as evidence. The authoritative numbers are the n=200 stratified table in `FINAL_PROPOSAL.md:263`. |

## 2. Known stack confounders (affect early results only)

1. **`gradient_checkpointing=True` was the config default** in
   `configs/rho_sweep.yaml`, `configs/grpo_9b.yaml`, and
   `configs/sweep_grid.yaml`. In TRL 0.14 this corrupts rollouts
   (the "ToToTo" token-loop bug documented in
   `scripts/run_csd_pilot.py:157`). All three configs have been changed
   to `false` with a note referencing this file. Any numbers from
   `results/csd_full/`, `results/csd_pilot/`, `results/rho_sweep/`, or
   `results/logs/exp2/` that may have been produced with
   `gradient_checkpointing=True` are ARCHIVAL ONLY.
2. **Qwen generation mismatch in archived logs**. Early runs in
   `results/logs/exp2/` are on `Qwen/Qwen2.5-7B-Instruct`; later runs
   (Wave 2 onwards) are on `Qwen/Qwen3.5-9B`. Summaries that cite "9B
   monotonic ρ sweep" with 0.5%/87%/88%/85% numbers are from the
   Qwen2.5 logs and should not be attributed to Qwen3.5-9B.
3. **`num_generations` (group size G) mismatch**: the paper's examples
   assume G=4, the run config uses G=2, and our tests use various G.
   The V14 trainer honours whatever the config says. The headline
   Wave 2/Wave 7/Wave 8 experiments are all G=2. Statements about
   Q_CSD (which requires n⁺ ≥ 2 per group) or exact ρ* (which needs
   ≥ 2 non-degenerate groups per batch) are therefore
   **structurally untestable** in the G=2 runs.

## 3. Known controller proxies (not theorem implementations)

1. **`src/adabalance.py::AdaBalanceController.update`** takes as input
   `success_counts`, `G`, and optional `grad_pos_norms`/`grad_neg_norms`
   (which in `RhoGRPOTrainerV14._update_adabalance` are filled with
   `advantages[pos_mask].abs()` — a cheap proxy, not actual gradient
   norms). The new ρ is computed via
   `compute_rho_star(V_plus_ema, C_pG_ema)` where V_plus / C_pG come
   from the binomial-variance analysis in
   `src/stability_analysis.py::compute_advantage_variance_components`,
   not from true gradient second moments. This is a **heuristic
   approximation** of the Theorem-2 quantity, not an implementation of
   it.
2. **`src/stability_analysis.py::compute_rho_max`** includes
   `max(rho_max_raw, 5.0)` as a hand-tuned floor. This is a **heuristic
   monitor**, not a theorem-derived bound. Do not cite it as "stability
   upper bound" in paper-grade claims.
3. **Synthetic degenerate-group labels**. The old
   `src/rho_grpo_trainer.py` (pre-V14) in `_on_prestep_hook` falls back
   to drawing all-success/all-failure labels from
   `p_batch**G / (1-p_batch)**G` when the real rollout contains no
   valid reward. These *synthetic* labels are then fed to the
   controller. Runs using the old trainer therefore mix real and
   synthetic reward signal to the adaptive ρ — another reason to treat
   `results/csd_full/` and `results/csd_pilot/` as archival.
   **V14 does not do this**; V14 only acts on real rewards. All Wave
   2 / Wave 7 / Wave 8 results were produced by V14 and are not
   affected by this issue.

## 4. Authoritative result locations

Use only these paths when making claims in papers or reviews:

| Claim | Authoritative source |
|-------|----------------------|
| Main 3-seed ρ-sweep (fixed ρ / ADQ) on Qwen3.5-9B / GSM8K | `results/gates_1_2/` + `results/stratified_eval/*.json` |
| Bandit-ρ controller (4 seeds) | `results/wave7_bandit/` + `results/stratified_eval/bandit_seed*.json` |
| Exact-ρ* controller (5 seeds) | `results/wave8_exact_rho/` + `results/stratified_eval/exact_seed*.json` |
| Stratified easy/hard accuracy table | `results/stratified_eval/stratified_analysis.json` |
| Base-model reference | `results/stratified_eval/base.json` |
| Aggregate summary | `FINAL_PROPOSAL.md §Results + §Difficulty-Stratified Results` |

Archived (NOT used for any claim):
- `results/csd_full/` — pre-V14, may use synthetic labels + possibly
  `gradient_checkpointing=True`.
- `results/csd_pilot/`, `results/csd_pilot_v2/` — older pilot runs.
- `results/rho_sweep/` — older sweep, single-seed.
- `results/gates_1_2_wave1_truncated/` — Wave 1 with JSON save bug.
- `results/wave3_qwen25_7b/`, `results/wave3a_math500/`,
  `results/wave3b_qwen27_gsm8k/` — cross-family / cross-dataset
  attempts that OOM'd / empty caches.
- `results/wave6_G3_sweep/` — G=3 attempt, 8/9 OOM.

## 5. Narrative in one paragraph (non-contradictory)

For binary verifiable rewards and sequence-level advantage
normalisation, the ρ-weighted GRPO per-step gradient can be rewritten
exactly as a signed combination of ∇KL(τ⁺‖π) and ∇KL(τ⁻‖π) where τ⁺
and τ⁻ are in-batch empirical distributions over correct and incorrect
responses (Theorem 1). Starting from this identity, the variance of
the estimator is minimised at `ρ* = Cov_s(g⁺,g⁻)/Var_s(g⁺)` (Theorem
2). We ship a TRL-0.14-compatible trainer (`RhoGRPOTrainerV14`) and
three controllers: ADQ (proxy estimator of Theorem 2 via binomial
variance), an exact-ρ* controller that uses two extra backward passes
per non-degenerate group to directly estimate Cov/Var, and a simple
UCB1 bandit over ρ that uses training reward as feedback. On
Qwen3.5-9B / GSM8K with G=2 and LoRA r=64, fixed ρ=0.70 wins on mean
test accuracy (52.3 ± 7.5, 3 seeds), the bandit comes second with the
smallest seed-to-seed variance (50.2 ± 4.5), ADQ is last (46.8 ± 8.3),
and exact-ρ* is effectively untestable because both Q_CSD and the
exact estimator require non-degenerate groups that G=2 rarely
provides. Stratifying the test set by base-model correctness reveals
that ρ has no effect on questions the base model already answers
correctly (all methods saturate at ~94 %) and a 7 pp spread on
questions outside the base model's competence. We do NOT claim
controller dominance, learning-dynamics equivalence with
self-distillation, or a capacity-bound theorem.
