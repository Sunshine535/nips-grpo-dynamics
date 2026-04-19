# Idea Discovery Report — "CSD Tripod"

**Direction**: NeurIPS best-paper-bar extension of the current CSD framework.
**Date**: 2026-04-19
**Pipeline**: research-lit → landscape synthesis → idea-gen (user-steered) → novelty-check

## Executive Summary

We propose a single integrated paper, **"The CSD Tripod"**, bundling three tightly coupled results that all fall out of our per-group KL-difference decomposition of GRPO:

1. **Theorem A** — A PAC-Bayes-style generalization certificate on test-set reasoning accuracy in terms of batch-level H(τ⁺) statistics.
2. **Theorem B** — Closed-form **necessary** lower bound G_min(p, δ) = ⌈log(2/δ) / min(p, 1−p)⌉ on group size for non-degenerate contrastive signal.
3. **Method QARA** — Q_CSD-gated rollout curation; provably matches fixed-G GRPO with fewer rollouts.

All three target the same pain point (sample-efficient, stable binary-reward GRPO with crisp theoretical backing), reuse our existing V14 trainer + Gate-data, and plausibly clear the NeurIPS best-paper bar when taken together — any one in isolation is a workshop paper; the combination is a full-conference result.

## Paper Narrative (one paragraph, scoped)

Binary-reward GRPO (the core RLVR engine behind DeepSeek-R1 / Qwen-Math / etc.) exposes a single contrastive knob ρ between pulling the policy toward correct responses (τ⁺) and pushing it from incorrect ones (τ⁻). Starting from our estimator-level identity ∇L = √(p(1−p))·[∇KL(τ⁻‖π) − ρ·∇KL(τ⁺‖π)], we derive three tight results that *all* reduce to a single quantity, Q_CSD = H_norm(τ⁺)·(n⁺/G): (i) batch-average Q_CSD upper-bounds the generalization gap via a PAC-Bayes argument on the τ⁺-indexed hypothesis class; (ii) a necessary closed-form G_min(p, δ) shows why 2-GRPO suffices when p ≈ 0.5 yet must grow to ~37 when p=0.1; (iii) gating rollouts on Q_CSD yields the same policy as fixed-G GRPO with provably fewer samples. Empirically we verify all three on Qwen3.5-9B / GSM8K + MATH across the regimes p ∈ {low, mid, high}. Under-budgeted G-settings show the exact collapse pattern predicted by Theorem B; QARA cuts rollout cost ≥30% without accuracy loss; and batch-Q_CSD correlates r ≥ 0.9 with final test accuracy across seeds.

## Ranked Ideas

### 🏆 Idea A: CSD Tripod (RECOMMENDED — integrated paper)

**Novelty check — verified ✅**:
- `Demystifying GRPO: U-Statistic` (2603.01162) has `G* = √(c₃/c₁)` a **sufficient** variance-optimal G for any task. We give a **necessary** G_min for any binary-reward GRPO. Complementary, not overlapping. We cite and contrast explicitly.
- `F-GRPO: Learn Obvious Forget Rare` (2602.06717) computes "probability updates miss rare-correct modes as a function of G" but uses it to design focal-loss-style advantage shaping, not to derive G_min. We formalize the necessary condition; their method is orthogonal.
- `LESS` (2512.00908), `AsymGRPO` (2604.04894), `EDGE-GRPO` (2507.21848) all shape advantages using correct/incorrect labels but have **no** generalization bound or G-bound results.
- `XRPO` (2510.06672), `AR3PO` (2509.25808) do adaptive rollout on success-rate signals; **no CSD-based gating**, no Q_CSD.
- No published work binds H(τ⁺) or entropy-of-correct-responses to a generalization gap in the RLVR setting.

**Three contributions**:

#### Theorem A (Generalization Certificate)
For an LLM trained with binary-reward GRPO over T steps on a batch set B_t, the expected test-time accuracy gap satisfies:
```
   E[acc(π₀) − acc(π_T)] ≤ c₁·(1 − Ē[Q_CSD]) + c₂·√(Var[Q_CSD]/T) + c₃·G^{-1/2}
```
where Ē[Q_CSD] is the time-averaged batch-mean Q_CSD and Var[Q_CSD] its variance; c₁, c₂, c₃ depend on the base model's Rademacher complexity on τ⁺ support.

**Proof sketch:** H_norm(τ⁺) controls the effective support size of the empirical correct distribution; PAC-Bayes on the τ⁺-indexed posterior gives a concentration inequality; binary-reward structure lets us sharpen Bernstein to the Bernoulli-variance term.

#### Theorem B (Necessary G_min)
For target coverage δ (probability of non-degenerate group), and success rate p ∈ (0, 1):
```
   G_min(p, δ) = ⌈log(2/δ) / min(p, 1−p)⌉
```
This is the minimum G such that P(0 < n⁺ < G) ≥ 1 − δ. Below G_min the CSD gradient signal is **zero with positive probability**; RL gains are bounded by (1−δ) × everything.

**Consequence:** For balanced p=0.5, G_min(0.5, 0.05) ≈ 8; for hard p=0.1, G_min(0.1, 0.05) ≈ 37; for easy p=0.9, also ≈ 37 (symmetry). Predicts: **2-GRPO ≈ 16-GRPO holds exactly in the balanced regime, fails elsewhere.**

#### Method QARA (Q_CSD-Adaptive Rollout Allocation)
At each GRPO step, compute per-group Q_CSD; if Q_CSD < τ_gate, either:
- (mode a) Reject this batch and resample (rejection sampling) — unbiased.
- (mode b) Expand the group (more rollouts for this prompt) until Q_CSD ≥ τ_gate — biased but sample-efficient.
Both converge to the same policy as fixed-G=G_max GRPO; mode-b provably achieves same gradient variance with E[samples] ≤ N/(1 − δ_low).

### 🥈 Idea B: Bayesian-ρ ADQ (BACKUP)
Kept in case Theorem A proof has gaps. Not discussed further in this report.

### 🥉 Idea C: Continuous-reward CSD extension (FUTURE WORK)
Listed as §6 of the paper ("Scope extensions").

## Eliminated Ideas (from original 8)
- `#4 Bayesian ρ` — interesting but not tightly coupled to the three main results; moved to backup.
- `#5 Continuous-reward CSD` — major theoretical project; listed as future direction.
- `#6 Dual-clock ρ (macro+micro)` — overlaps EBPO (2602.05165) global+local shrinkage; deprioritized.
- `#7 H(τ⁺) exploration bonus` — LESS (2512.00908) and AsymGRPO (2604.04894) do related things; not enough room for a clean novelty story.
- `#8 Good-teacher replay` — Efficient RL Replay (2604.08706) and RLEP already crowd the space.

## Concurrent Work Watchlist (cite explicitly, differentiate)

| Paper | Closest overlap | Our differentiation |
|-------|-----------------|---------------------|
| [Demystifying GRPO: U-Stat (2603.01162)](https://arxiv.org/abs/2603.01162) | variance-optimal G* | our G_min is *necessary*, not *sufficient* |
| [It Takes Two: GRPO Secretly DPO (2510.00977)](https://arxiv.org/abs/2510.00977) | GRPO ≡ contrastive | our KL-form is explicit; we explain their "2-GRPO works" via Thm B |
| [F-GRPO: Rare-correct (2602.06717)](https://arxiv.org/abs/2602.06717) | P(miss rare mode \| G,p) | they shape advantages; we give a closed-form G_min + PAC-Bayes generalization |
| [GRPO's Effective Loss (2503.06639)](https://arxiv.org/abs/2503.06639) | weighted contrastive, fixed-point on p | their contrastive = temporal; ours = within-batch with KL-diff form |
| [GBMPO: Bregman GRPO (2602.04380)](https://arxiv.org/abs/2602.04380) | replaces KL with Bregman | binary rewards uniquely give KL decomposition via indicator functions |
| [AR3PO / XRPO (2509.25808, 2510.06672)](https://arxiv.org/abs/2510.06672) | adaptive rollout budget | they use success-rate signals; we use Q_CSD (theoretically grounded) |
| [LESS (2512.00908)](https://arxiv.org/html/2512.00908) | advantage shaping by correct/incorrect segments | our three results are about G, generalization, and rollout curation |

## Next Steps

- [x] Phase 4 external review (Codex GPT-5.4 xhigh): **3/10, not ready — Tripod dilutes the core story**. Brutal critique below.
- [x] Decision: **PIVOT** away from Tripod → harden the existing decomposition + ρ* paper (which the auto-review-loop left at 5/10 "almost") per reviewer's concrete minimum-fixes list. See `refine-logs/EXPERIMENT_PLAN.md`.

## Phase 4 Reviewer Critique (verbatim highlights)

> *"This submission theorem-washes a useful diagnostic. The exact binary-reward KL decomposition is interesting, but Theorem A does not establish a non-vacuous generalization certificate, Theorem B is essentially a one-line mixed-binomial coverage fact miscast as a necessary condition, and QARA changes the sampling distribution in a way that makes its unbiasedness/convergence claims false or at least unproven."*

Key kills:
- **Thm A**: H_norm(τ⁺) is diversity, not capacity; PAC-Bayes needs hypothesis classes; constants will make the bound vacuous or tautological; high Q_CSD ≠ better test accuracy (paraphrase ≠ semantic diversity).
- **Thm B**: the exact event is `P(0 < n⁺ < G) = 1 − p^G − (1−p)^G`; our closed-form was a loose exponential. Also `n⁺=1` is non-degenerate but Q_CSD ≈ 0 → B is not about useful CSD signal.
- **QARA**: mode-(a) conditions on event defined by same samples used in gradient → biased; mode-(b) outcome-dependent stopping rule changes the effective prompt distribution.
- **Concurrent-work ranking** (most to least threatening): 2510.00977 > 2603.01162 > 2509.25808 > 2602.06717 > 2510.06672.

## Pivoted Paper Plan — "Hardened Decomposition" (post-review)

Keep the already-accepted core (Theorem 1 decomposition + Theorem 2 closed-form ρ* + ADQ controller + Q_CSD collapse predictor). Add five targeted hardening pillars instead of Tripod add-ons:

1. **Cross-family validation** — reproduce Gate 1+2 numbers on a second model family (LLaMA-3-8B or Mistral-7B) on GSM8K. Defuses "Qwen-only" objection.
2. **Head-to-head at matched compute** — against F-GRPO, 2-GRPO, AR3PO, XRPO, fixed-ρ GRPO. Same rollout budget, same wall-clock, 3 seeds.
3. **Q_CSD predictive-power ablation** — controlled regression `final_acc ~ p_mean + H(π) + temperature + answer_diversity + Q_CSD`; show Q_CSD has non-zero coefficient after partialling out p.
4. **Semantic τ⁺ canonicalization** — hash the *extracted numerical answer* (e.g. "42", "42.0" normalized) instead of the token sequence. Kills the "paraphrase multiplicity inflates H_norm(τ⁺)" attack.
5. **QARA-lite (mode-b only) with honest bias bound** — § in Method, not a Tripod equal leg. Prove `|E[g_QARA] − E[g_full]| = O(δ_low · ‖∇KL(τ⁺)‖)`; show empirically a 30%+ rollout-budget saving at ≤1% accuracy loss.

Appendix additions (not main theorems):
- **A1 Exact binomial coverage**: `P(0 < n⁺ < G) = 1 − p^G − (1−p)^G`. No claim as necessary condition for CSD signal; just a convenience fact.
- **A2 Q_CSD ≥ q bound** under a k-mode latent correct-response model (non-trivial result; only promoted to main text if it's interesting enough).

**Does NOT include**: PAC-Bayes generalization theorem (dropped per Thm A kill), G_min as necessary condition (dropped per Thm B kill), QARA mode-(a).

## Updated Concurrent-Work Watchlist

Per reviewer ranking (most threatening first), explicitly cite and differentiate:
1. [It Takes Two: GRPO Secretly DPO (2510.00977)](https://arxiv.org/abs/2510.00977) — cite in intro as "empirical small-G works"; our decomposition *explains why*.
2. [Demystifying GRPO: U-Stat (2603.01162)](https://arxiv.org/abs/2603.01162) — their G*=√(c₃/c₁) is *sufficient-variance-optimal*; we give a *necessary-signal* fact in appendix and do not over-claim.
3. [AR3PO (2509.25808)](https://arxiv.org/abs/2509.25808), [XRPO (2510.06672)](https://arxiv.org/abs/2510.06672) — head-to-head rollout-efficiency comparison in experiments.
4. [F-GRPO (2602.06717)](https://arxiv.org/abs/2602.06717) — head-to-head; differentiate on theoretical grounding (CSD) vs. their focal-loss heuristic.
5. [GBMPO: Bregman (2602.04380)](https://arxiv.org/abs/2602.04380) — one remark on why binary reward uniquely gives KL decomposition.
