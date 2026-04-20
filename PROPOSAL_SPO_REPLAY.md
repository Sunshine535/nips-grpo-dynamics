# SPO + Verified Replay CE — research proposal

**Working title**: *Per-prompt baselines + verified replay: a minimal, honest recipe for stable binary-reward GRPO post-training on small-group rollouts.*

**Target venue**: NeurIPS 2026 (bar: best-paper, not weak accept)

**Status** (2026-04-21): preliminary positive result + Codex xhigh Round-1 review (3/10). This doc captures the validated claim and the experimental gauntlet needed to get to 7/10+.

---

## Motivation

Binary-reward GRPO with small G (e.g. G=2) suffers from a structural
problem: within-group variance estimators are degenerate when the group
is all-pass or all-fail. We previously tried to fix this at the ρ-weight
level (ADQ, exact-ρ\*, bandit-ρ; see `RETRACTIONS.md`) — none beat the
fixed ρ = 0.70 baseline.

This proposal abandons the ρ-controller family and instead attacks the
root cause: **the learning signal is sparse when a prompt is too hard
or too easy**. Two changes:

1. **SPO backbone**: replace group-relative advantage (which is
   degenerate at G=2) with `rewards - b(prompt_id)`, where `b` is a
   persistent per-prompt EMA baseline. Advantage survives degenerate
   groups.
2. **Verified Replay CE**: maintain a small hash-deduped bank of rollouts
   with binary reward = 1. Add a low-weight (λ_rep = 0.05) SFT
   cross-entropy loss on this bank. This amplifies the signal from the
   rare-successful-completion channel.

An earlier version of this proposal also added difficulty-adaptive
batch-level prompt duplication, but the sampler had a bug
(`int(batch_size·dup_frac) = 0` at the pilot config) and in the Wave-10
experiments duplication never fired. All positive results below come
from **SPO + Verified Replay CE alone**.

## Validated claim (Wave 10, Qwen3.5-9B / GSM8K, 200 steps, LoRA r=64)

| Method                      | n seeds | GSM8K test acc (n=200) | hard subset |
|-----------------------------|---------|------------------------|-------------|
| **SPO + Verified Replay**   | **9**   | **69.4 ± 10.4%**       | **60.0 ± 14.1%** |
| SPO only (no replay)        | 3       | 52.2 ± 11.0%           | —           |
| fixed ρ = 0.70 (prior SOTA) | 3       | 52.3 ± 7.5%            | 38.0 ± 8.8% |
| Dr. GRPO (published 2025)   | 3       | 39.3 ± 4.0%            | —           |
| base Qwen3.5-9B             | —       | 25.5%                  | 0%          |

Per-seed SPO+Replay: 72.5 / 65.0 / 59.5 / 88.0 / 70.0 / 82.0 / 66.0 / 67.0 / 54.5.
Even the worst seed (54.5%) exceeds the fixed-ρ=0.70 mean (52.3%).

Δ = +17.2 pp overall, +22 pp on hard subset. Welch's t on shared
seeds {42, 43, 44} = 1.83 vs SPO-only (p ~0.14 at df 3.3 — underpowered).

## Known weaknesses (from Codex xhigh, Round 1)

1. **Replay CE ≈ online RFT.** We cannot yet distinguish
   "SPO + verified replay" from "just do SFT on self-generated successes".
   Round-2 experiment (running): `--pg-weight 0` control on 3 seeds
   tests whether online RFT alone matches SPO+Replay.
2. **Stats asymmetry.** ASE-R at n=9 vs baselines at n=3 inflates noise.
   Round-2 experiment (running): boost fixed-ρ=0.70 baseline to
   n=9 matched seeds.
3. **Variance still high** (std 10.4). Even at n=9 the MDE at α=0.05,
   power=80% is ≈ 18.5pp — the observed 17.2pp effect is on the boundary.
4. **One-model / one-dataset.** Only Qwen3.5-9B/GSM8K. No cross-family,
   no MATH, no Llama.
5. **λ_rep = 0.05 was NOT tuned on held-out data** — it was set on
   intuition after the earlier ViSER/ASE-R-plus-dup proposal was
   abandoned. Round-3 fix: λ_rep ablation {0.02, 0.05, 0.1} at n ≥ 3.

## Round-2 experimental gauntlet (running on frp-server, 8× A800-80G)

| Run         | Method                       | seeds           | purpose                      |
|-------------|------------------------------|-----------------|------------------------------|
| Wave-11-ρ   | fixed ρ = 0.70               | {46,…,51}       | matched n=9 baseline         |
| Wave-11-RFT | SPO + Replay, `pg-weight=0`  | {42, 43, 44}    | novelty control (online RFT) |

Each run: 200 steps, ~107 min on A800. Parallel across 8 GPUs,
total wall time ≈ 2 h.

## Round-2 success criteria (for the next Codex review)

- **Pass**: SPO+Replay still wins over fixed-ρ=0.70 at n=9 vs n=9 with
  Welch's t > 2.0 *and* RFT-only underperforms SPO+Replay by ≥ 5 pp.
- **Almost**: wins overall but RFT matches within ± 2 pp — then the
  claim narrows to "SPO backbone enables cheap replay CE to work at
  G=2 where fixed-ρ doesn't". Need cross-model or cross-dataset.
- **Fail**: RFT matches or beats SPO+Replay. Kill the "replay CE" novelty
  framing; pivot to "SPO backbone enables G=2 training where group-std
  methods fail".

## Round-3 experiments (if Round 2 passes)

1. λ_rep sweep {0, 0.02, 0.05, 0.1, 0.2}, 3 seeds each.
2. Cross-family: Qwen2.5-7B / Llama-3.1-8B on GSM8K (same protocol).
3. Cross-dataset: Qwen3.5-9B on MATH-500 (same protocol).
4. Replay-warmup ablation: what happens with warmup = 0 (replay from step 0)?
5. Replay bank size cap: effect of capping bank at k ≤ n successes.

## Retracted / superseded claims (for audit)

- "Adaptive support expansion" — the dup sampler never fired at the
  pilot config. Sampler fixed in `src/adaptive_dup_sampler.py` as of
  commit `3135084`; pending re-evaluation. For now, the claim is
  narrowed to just "SPO backbone + Verified Replay CE".
- All ρ-controller claims — see `RETRACTIONS.md`.

## File map (as of 2026-04-21)

- `src/aser_trainer_v14.py` — SPO / Dr. GRPO backbone + replay CE
- `src/prompt_stats.py` — per-prompt EMA baselines + hardness
- `src/replay_bank.py` — hash-deduped verified-success store
- `src/adaptive_dup_sampler.py` — probabilistic-round sampler (post-fix)
- `scripts/run_aser_mvp.py` — training launcher (`--pg-weight 0` for RFT control)
- `scripts/eval_stratified.py` — per-question GSM8K test eval
- `scripts/analyze_wave11.py` — matched-seed combined analysis
- `configs/aser_mvp.yaml` — training config (pinned)
- `results/wave10_aser/` — 16 training runs (9 full + 3 dup + 3 only + 1 seed45)
- `results/stratified_eval_aser/` — 15 eval JSONs (Wave 10)
- `results/wave11_rho070_boost/` — 6 ρ-boost training runs (pending)
- `results/wave11_rft_control/` — 3 RFT training runs (pending)
- `review-stage/AUTO_REVIEW.md` — cumulative review log
- `REVIEWER_MEMORY.md` — persistent reviewer brain
- `RETRACTIONS.md` — single-source retraction log
