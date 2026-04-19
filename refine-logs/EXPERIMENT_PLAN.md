# Experiment Plan: CSD Equivalence + CSDPO

**Source**: `refine-logs/FINAL_PROPOSAL.md` (CSD v2)
**Date**: 2026-04-09
**Budget**: ~490 GPU-hours
**Models**: Qwen2.5-7B, Qwen3-8B, Qwen3.5-9B

## Claims to Prove

| # | Claim | Type | Priority |
|---|-------|------|----------|
| C1 | GRPO gradient = CSD gradient (Theorem 1) | Theory + Empirical | MUST |
| C2 | CSD capacity bound holds (Theorem 2) | Theory + Empirical | MUST |
| C3 | Closed-form ρ* improves stability (Theorem 3) | Empirical | MUST |
| C4 | Q_CSD predicts collapse better than variance | Empirical | MUST |
| C5 | CSDPO eliminates collapse and beats SRPO | Empirical | MUST |
| C6 | CSD predicts variant relative performance | Empirical | SHOULD |
| C7 | CSDPO components are each necessary | Empirical | SHOULD |

## Block 1: CSD Equivalence Verification (C1) — 54 GPU-hrs

- 3 models × 4 ρ × 3 seeds × 200 steps
- Log per-step: ∇KL(τ⁺‖π), ∇KL(τ⁻‖π), ∇L_GRPO
- Metric: R² > 0.95, cosine > 0.99
- Gate: R² < 0.8 → investigate A2 assumption

## Block 2: Capacity Bound (C2) — 84 GPU-hrs

- Base pass@k: k ∈ {1,4,8,16,32,64,128,256}, 3 models
- G sweep: G ∈ {4,8,16,32}, 3 models × 2 seeds
- Metric: accuracy(CSDPO,G) ≤ pass@(G·T_eff)(π₀)

## Block 3: Optimal ρ + Collapse Predictor (C3+C4) — 120 GPU-hrs

- Qwen2.5-7B, ρ ∈ {0.3,0.5,0.7,1.0,1.5,2.0,3.0,5.0}, 10 seeds, 300 steps
- Log: H(τ⁺), n⁺/G, cos(g⁺,g⁻), collapse label
- C3 metric: ADQ 0% collapse at ρ=1.0 (vs 50%)
- C4 metric: Q_CSD AUROC > 0.85

## Block 4: CSDPO vs Baselines (C5) — 112 GPU-hrs

- 3 models × 5 methods (GRPO, DAPO, SRPO, GRPO-λ, CSDPO) × 5 seeds × 300 steps
- Eval: GSM8K + MATH-500
- Metrics: accuracy, collapse rate, convergence speed

## Block 5: Ablation (C7) — 45 GPU-hrs

- Qwen2.5-7B × 6 variants (full, -EA, -QW, -ADQ, -GCR, GRPO) × 5 seeds

## Block 6: Variant Prediction (C6) — 75 GPU-hrs

- Qwen2.5-7B × 5 methods × 5 difficulty groups × 3 seeds

## Run Order

| Week | Blocks | Wall time (8 GPU) | Gate |
|------|--------|-------------------|------|
| 1 | 1 + 3 (parallel) | ~15h | R² > 0.8, AUROC > 0.7 |
| 2 | 2 + 4 (parallel) | ~14h | CSDPO > GRPO +3% |
| 3 | 5 + 6 (parallel) | ~15h | — |

## First 3 Runs

1. `python train_csd_verification.py --model Qwen2.5-7B --rho 1.0 --seed 42 --log_csd`
2. `python train_grpo_sweep.py --model Qwen2.5-7B --rho 1.0 --seeds 10 --steps 300`
3. `python eval_passk.py --model Qwen2.5-7B --k 1,4,8,16,32,64,128,256`

---

# Hardening Plan (post auto-review-loop, post external review)

**Source**: external review (Codex GPT-5.4 xhigh, 2026-04-19) scored the original "Tripod" extension 3/10 ("not ready"); recommended hardening the existing Round-4 "almost" (5/10) decomposition+ρ\* paper instead of expanding scope.
**Available local cache** (verified on remote 2026-04-19): Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3.5-{0.8B,4B,9B,27B}; datasets GSM8K, MATH-500, MathInstruct, MBPP, BBH, MMLU, …  No LLaMA / Mistral on box → "cross-family" must be reframed as **cross-generation Qwen** (2.5 vs 3 vs 3.5).

## Pre-registered claims for the hardened paper

| # | Claim | Min evidence | Hardening pillar |
|---|-------|--------------|------------------|
| H1 | Gate 1 ADQ controller produces non-trivial ρ(t) trajectory | `std(ρ_t) > 1e-3` over 200 steps on Qwen3.5-9B/GSM8K (1 seed) | Wave 2 |
| H2 | Upward-tendency of accuracy vs. ρ on Qwen3.5-9B/GSM8K | `acc(ρ=3.0) > acc(ρ=0.7)` on ≥ 2/3 seeds | Wave 2 |
| H3 | Decomposition holds across Qwen generations | H2 reproduced on Qwen2.5-7B-Instruct AND Qwen3-8B | Wave 3 |
| H4 | Decomposition holds on a different dataset (MATH-500) | H2 reproduced on Qwen3.5-9B/MATH-500 | Wave 3.5 |
| H5 | Head-to-head: CSD-ADQ Pareto-competitive | Either higher acc at matched rollouts OR fewer rollouts at matched acc vs. each of {fixed-ρ GRPO, 2-GRPO, F-GRPO, AR3PO} | Wave 4 |
| H6 | Q_CSD adds predictive power over plain `p` | Partial R² ≥ 0.05 in `final_acc ~ p_mean + H(π) + Q_CSD + …` regression | Offline (Wave 2+3 data) |
| H7 | Semantic τ⁺ canonicalization preserves/improves correlation | `corr(Q_CSD_semantic, final_acc) ≥ corr(Q_CSD_token, final_acc)` | Offline |
| H8 | QARA-lite: ≥ 30% rollout saving at ≤ 1% accuracy loss | Direct A/B at ρ=1.0, 3 seeds | Wave 5 |
| H9 | QARA-lite bias matches theory | `|obs_bias − theory_bound| < 1 × theory_bound` on ≥ 2/3 seeds | Offline |

**If any pre-registered threshold fails, the corresponding claim is dropped or narrated as a negative result.**

## Wave schedule (after Wave 2 finishes)

### Wave 2 (RUNNING — launched 2026-04-19 14:30)
- Qwen3.5-9B + GSM8K, 200 steps, 8 parallel + 1 queued = 9 runs.
- Outputs: closes H1 (Gate 1) + H2 (Gate 2 sweep at G=2).
- Monitor task: `bp5nzcxkz`. ETA primary 8 ~16:10, queued 9th ~17:50.

### Wave 3 — Cross-generation Qwen (closes H3)
- Models: `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen3-8B` (both in remote cache).
- 9 runs per model = 18 total: ρ ∈ {0.7, 1.0, 3.0} × 3 seeds × G=2 × 200 steps.
- Reuses `run_gates_1_2.sh` with `MODEL=` env override and a fresh `OUTPUT=results/wave3_qwen{2.5,3}/`.
- Compute: 18 × ~1.7 GPU-hr ≈ 30 GPU-hr ≈ 4 hours wall-clock on 8 GPUs.
- Trigger: launch immediately after Wave 2's primary 8 finish. Skip queued 9th seed at ρ=3.0 if compute-tight (Wave 2 has 3 seeds for ρ=3.0 already).

### Wave 3.5 — Cross-dataset MATH-500 (closes H4)
- Model: `Qwen/Qwen3.5-9B` (same as Wave 2).
- Dataset: MATH-500 (already cached as `datasets--HuggingFaceH4--MATH-500` and `datasets--EleutherAI--hendrycks_math`).
- 9 runs: ρ ∈ {0.7, 1.0, 3.0} × 3 seeds × G=2 × 200 steps.
- Compute: ~16 GPU-hr ≈ 2 hours on 8 GPUs.
- Trigger: parallel with Wave 3 if a GPU partition can be carved out; otherwise immediately after Wave 3.
- Note: needs a MATH-formatted reward function (numeric extraction, latex `\boxed{…}` parse). Reuse `src/math_reward.py` if present, else write a 30-line wrapper.

### Wave 4 — Head-to-head baselines at matched compute (closes H5)
- Model: Qwen3.5-9B / GSM8K.
- Methods: 4 = {fixed-ρ=1 GRPO, 2-GRPO (G=2 stock), F-GRPO (focal advantage), AR3PO (adaptive rollout on success rate), CSD-ADQ}. Drop XRPO unless Wave 4 budget is loose (it requires more infra).
- 5 methods × 3 seeds = 15 runs.
- Compute: ~25 GPU-hr ≈ 3 hours on 8 GPUs.
- Implementation: write `src/baselines/{fgrpo,ar3po}.py` as `compute_loss` overrides on top of V14 trainer (so all baselines share rollout/eval code → matched compute is real). Stock 2-GRPO is V14 with `num_generations=2, ada_controller=None, rho=1`.

### Wave 5 — QARA-lite ablation (closes H8 + H9)
- Model: Qwen3.5-9B / GSM8K.
- 6 runs: 3 seeds × {QARA-lite ON, QARA-lite OFF}, ρ=1.0 fixed, G=2.
- Compute: ~10 GPU-hr ≈ 1.5 hours on 8 GPUs.
- Implementation: `src/qara_lite.py` — wrap `compute_loss` in V14 trainer; if per-group Q_CSD < `τ_gate=0.15`, request `+G` extra rollouts for that group up to `G_max=8`. Log `δ_low_per_step` and `samples_per_step`.

### Offline analyses (CPU only, after Wave 2/3/4/5 finish)
- **H6 — Q_CSD predictive regression** (`scripts/analyze_qcsd_regression.py`):
  inputs: pilot_results.json + csd_logs.json from Wave 2 + Wave 3 + Wave 3.5.
  fit `final_acc ~ p_mean + H_pi + temperature + answer_diversity + Q_CSD`; report partial R² for Q_CSD with bootstrap CI.
- **H7 — Semantic τ⁺ canonicalization** (`scripts/analyze_semantic_canonicalization.py`):
  re-process csd_logs.json by hashing the *extracted numerical answer* (regex `####\s*(-?[\d,]+\.?\d*)`) instead of the token sequence; recompute Q_CSD; compare correlation.
- **H9 — QARA-lite bias** (`scripts/analyze_qara_bias.py`):
  on Wave 5 logs, compute observed `|g_QARA_step − g_full_step|` and theoretical `δ_low_step · ‖∇KL(τ⁺)‖`; verify match.

## Total budget recap

| Wave | Compute | Wall (8-GPU) | Trigger |
|------|---------|--------------|---------|
| 2 (running) | 14 GPU-hr | 1.7 hr | started 14:30 |
| 3 | 30 GPU-hr | 4 hr | after Wave 2 primary |
| 3.5 | 16 GPU-hr | 2 hr | parallel/after Wave 3 |
| 4 | 25 GPU-hr | 3 hr | after Wave 3 |
| 5 | 10 GPU-hr | 1.5 hr | parallel with Wave 4 if GPUs allow |
| **Total** | **~95 GPU-hr** | **~12 hr if fully parallel, ~16 hr serial** | |

Adding offline analyses (CPU only, ~2–4 hr human time): grand total ≈ **20 hours of work** to push the paper from 5/10 → ≥ 7/10.

## Deferred from the original Tripod (per Codex critique)

- ❌ **Theorem A (PAC-Bayes generalization bound)** — dropped. Q_CSD demoted to empirical predictor with H6 ablation as supporting evidence.
- ❌ **Theorem B (G_min necessary condition)** — dropped from main text. Kept as a one-paragraph appendix fact: `P(0 < n⁺ < G) = 1 − p^G − (1−p)^G`. No "necessary signal" claim.
- ❌ **QARA mode-(a)** — dropped (biased due to event-conditioning); only mode-(b) survives, with explicit bias bound (H9).
- ❌ **Cross-family (LLaMA / Mistral)** — not feasible (no offline cache); reframed as cross-generation Qwen + cross-dataset MATH-500 (H3 + H4).
- ❌ **XRPO baseline** — deferred (infra cost too high relative to evidence value).
