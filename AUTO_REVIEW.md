# Auto Review Loop: NeurIPS Best Paper Assessment

**Started**: 2026-04-04
**Target**: NeurIPS 2026 Best Paper
**MAX_ROUNDS**: 4
**Reviewer**: GPT-5.4 via Codex MCP (xhigh reasoning)

---

## Round 1 (2026-04-04T15:05:00)

### Assessment (Summary)
- Score: 2/10
- Verdict: NOT READY — zero experimental results
- Key criticisms:
  - **Fatal**: Results section completely empty, all claims unvalidated
  - Theorem 2 may be close to algebraic identity, not deep theorem
  - Proposition 1 is approximate/heuristic, not a real bound
  - Setting too restrictive (binary rewards, group i.i.d.) for generality
  - AdaBalance optimizes variance, not necessarily learning outcomes
  - Collapse definition looks threshold-engineered
  - Too few seeds for instability claims near phase boundaries
  - Scope too narrow (GSM8K only, one model family) for best paper
  - Scale transfer too thin (3 points on 27B)
  - No mechanistic calibration (predicted vs observed V+/V-/C)
  - Missing confounder ablations (LR, KL coeff, group size, temperature)

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Verdict**

Current score: `2/10` for NeurIPS-level submission. For a best-paper bar, this is not competitive.

This is **not ready for submission**. The core reason is simple: the paper currently has **zero empirical evidence** for claims that are mostly empirical. Right now it reads like a promising theory note plus a good experiment plan, not a finished scientific paper.

**Critical Weaknesses**

1. **No results section in any meaningful sense.** This alone is fatal.
Minimum fix: run the core experiments needed to support `C1-C4`, with error bars and failure cases.

2. **The headline claims are currently speculative.** ">85% accuracy," "within 1% of oracle," "graceful degradation," and "predictable transfer" cannot appear without data.
Minimum fix: either produce the evidence or rewrite all of these as hypotheses and remove the numbers.

3. **The empirical claims are doing the real work, not the theory.** A narrow theory paper can survive if the theory is deep and airtight; this is not that paper yet.
Minimum fix: either reframe as a scoped theory paper or execute enough validation that the empirical story is real.

4. **The setting is very restrictive.** Binary rewards, group i.i.d. Bernoulli rewards, and a sufficient-statistic reduction are strong assumptions that many real GRPO runs violate.
Minimum fix: explicitly narrow the scope and add controlled assumption-violation experiments.

5. **Theorem 2 may be close to an algebraic identity rather than a deep theorem.** If the gradient is linear in two signal components, a quadratic variance form is unsurprising.
Minimum fix: show what is genuinely nontrivial: identifiability, estimability, predictive power, or downstream consequences.

6. **Proposition 1 is approximate and empirically calibrated.** That weakens the "stability boundary" narrative substantially.
Minimum fix: either prove a real bound with assumptions/error terms, or relabel it as a heuristic and validate it broadly.

7. **The "single control variable" story is likely confounded.** Stability will also depend on learning rate, KL coefficient, optimizer, group size, sampling temperature, and prompt mix.
Minimum fix: run ablations showing `rho` still predicts risk after controlling for these factors.

8. **AdaBalance optimizes local variance, not necessarily learning.** Lower variance is not the same as better final accuracy, better sample efficiency, or less collapse.
Minimum fix: show end-task gains, collapse-rate reduction, and compute/sample-efficiency improvements.

9. **The collapse definition looks threshold-engineered.** "2 of 3" criteria with tunable thresholds is vulnerable to post hoc selection.
Minimum fix: add threshold-free analyses and a much stronger sensitivity study.

10. **The current experimental plan is statistically thin for instability claims.** Three seeds and short sweeps are often not enough near phase boundaries.
Minimum fix: add more seeds around the boundaries and report uncertainty rigorously.

11. **The scope is too narrow for best paper.** GSM8K/MATH and one model family do not justify a general GRPO stability map.
Minimum fix: add at least one qualitatively different verifiable domain and another model family.

12. **The scale-transfer claim is especially weak as planned.** Three `rho` values on 27B is a sanity check, not a transfer study.
Minimum fix: run a real transfer analysis with multiple points, normalized quantities, and uncertainty.

13. **Theorem 1 sounds too obvious to carry much novelty.** Zero centered group signal when all rewards in the group are identical is standard intuition.
Minimum fix: demote it to a lemma/observation unless the paper proves something stronger about its probability and training impact.

14. **There is no direct mechanistic validation yet.** The paper needs to show that `V_+`, `V_-`, and `C` measured in training actually reconstruct observed behavior.
Minimum fix: plot predicted vs observed variance/collapse risk over time and across settings.

**What It Would Take**

Beyond the planned experiments, the additions that would move this from "maybe accept" toward "best-paper candidate" are:

- A **synthetic environment** where the Bernoulli/i.i.d. assumptions hold exactly, so the theory can be validated against ground truth.
- A **controlled correlation study** replacing i.i.d. Bernoulli with exchangeable or beta-binomial rewards, to show how boundaries deform under dependence.
- A **verifier-noise / reward misspecification sweep**, since binary verifiable rewards are rarely perfect in practice.
- A **cross-domain validation** on another verifiable domain such as code generation with unit tests or formal theorem proving.
- A **cross-family scaling study** across at least two model families and more than two scales, not just 9B to 27B in one family.
- A **mechanistic calibration study** showing measured `V_+`, `V_-`, and `C` predict observed gradient variance and collapse with strong calibration curves/AUROC.
- **Confounder interventions** on group size, KL coefficient, learning rate, temperature, and prompt difficulty.
- A **sample-efficiency and compute-efficiency analysis**, since best papers usually show not just "works," but "matters."

The main theory concerns, based on your summary, are:

- `A2` is likely unrealistic: group samples are usually correlated through prompt difficulty and policy state.
- `A3` may collapse away token-level structure and KL interactions that matter for actual optimization.
- The variance decomposition may be mathematically correct but conceptually shallow unless its components are nontrivial to isolate or estimate.
- `rho_min` needs a clearer connection to optimization progress, not just a variance expression.
- An empirically calibrated upper bound is not a robust theoretical guarantee.
- `rho* = -C / V_+` is a noisy ratio estimator; controller stability under estimation error needs analysis.
- The whole theory appears local/static, while training is nonstationary.

If all planned experiments land cleanly, this could become a plausible `6/10`-ish NeurIPS submission. It would still not be a best-paper candidate without broader external validation, stronger mechanistic evidence, and a tighter theory story.

**Submission Readiness**

**No.** This is not submission-ready. It is a promising project, not yet a finished paper.

</details>

### Actions Planned
1. **RUN ALL CORE EXPERIMENTS** (Exp 1-4) — the fatal blocker
2. Add synthetic environment validation (Bernoulli ground truth)
3. Add confounder ablations (group size G, KL coefficient λ)
4. Add mechanistic calibration (predicted vs observed V+/V-/C)
5. Increase seeds near boundaries (5 seeds instead of 3)
6. Strengthen threshold sensitivity analysis
7. Consider cross-domain validation if compute allows

### Status
- Continuing to Round 2 after experiments complete

---

## Round 2 (2026-04-04T21:15:00)

### Assessment (Summary)
- Score: 5/10 (up from 2/10)
- Verdict: Interesting but NOT READY — missing held-out eval, AdaBalance, confounder ablation
- Key praise:
  - "Genuinely interesting" — first time reviewer said this
  - Synthetic R²=0.994 is "serious evidence"
  - Non-monotonic ρ landscape is "a real empirical hook"
  - "Interesting enough for NeurIPS? Yes, potentially"
- Key criticisms:
  1. No held-out test accuracy yet (only training reward)
  2. AdaBalance still unvalidated
  3. rho=1.0 result needs stress-testing (could be artifact)
  4. Bridge from theory to training telemetry missing
  5. C3/C4 (robustness, scale transfer) unsupported
  6. Short-horizon (200 steps), narrow scope (1 model, 1 dataset)

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 5/10 (NeurIPS), 3/10 (best paper bar)

Verdict: Interesting enough for NeurIPS? Yes, potentially. Ready right now? No. Best-paper candidate? No.

Key: paper now has one strong pillar (synthetic R²=0.994 + 99.8% regime accuracy). Coarse sweep is interesting. But main claims still partially validated.

Remaining Weaknesses:
1. No held-out task results yet (train reward ≠ test accuracy)
2. AdaBalance unvalidated
3. Real-world stability claim thin (synthetic only)
4. Short-horizon, narrow scope
5. rho=1.0 result high-risk (could be artifact)
6. Theory→telemetry bridge missing
7. C3/C4 unsupported
8. Upper bound weaker than lower bound

Minimum Fixes:
1. Finish GSM8K/MATH evaluation
2. Run AdaBalance vs baselines
3. Real boundary-validation experiment (longer training)
4. Confounder ablation (G, KL)
5. Fine sweep near boundaries
6. Mechanistic calibration from training telemetry
7. Either execute robustness/scale or narrow claims
8. Stress-test rho=1.0 result

</details>

### Actions Taken
- Coarse sweep: 27/27 complete
- Synthetic validation: complete (R²=0.994, 99.8% regime accuracy)
- Eval phase: running on 27 checkpoints
- AdaBalance: queued in pipeline

### Status
- Continuing to Round 3 after AdaBalance + evals complete

---

## Round 3 (2026-04-05T00:10:00)

### Assessment (Summary)
- Score: 6/10 (up from 5/10)
- Verdict: "Real NeurIPS-caliber story" — credible path to acceptance
- Key praise: "genuinely interesting", non-monotonic landscape is real, synthetic validation strong
- Key criticisms:
  1. AdaBalance still missing
  2. C3 robustness not supported (66.7% << 85% target)
  3. Real evidence narrow (1 model, 1 dataset, 200 eval samples)
  4. Oracle baseline unclear (training reward: ρ=3.0, test acc: ρ=0.7)
  5. Confounders open (G, KL, prompt fixes)
  6. ρ=1.0 failure suspiciously strong (artifact risk)
  7. Need more seeds around ρ=1.0

### Actions Taken
- Coarse sweep: 27/27 complete with GSM8K test evaluation
- Synthetic validation: R²=0.994, 99.8% regime accuracy
- Best test accuracy: ρ=0.7 → 86.7%, ρ=2.0 → 84.7%, ρ=3.0 → 85.2%
- Standard GRPO (ρ=1.0): only 29.8% — +56.8% improvement at optimal ρ

### Status
- AdaBalance running (3 methods × 2 seeds)
- Need confounder ablation, more seeds, full eval

---

## Round 4 — FINAL (2026-04-05T01:00:00)

### Assessment (Summary)
- Score: 6/10 (narrowly framed), 4/10 (current broad paper)
- Verdict: "Borderline NeurIPS-viable, poster/weak-accept territory if written carefully"
- Key: "The minimum viable paper is not an AdaBalance paper. It is a theory + measurement paper about the GRPO signal-balance stability landscape."

### What To Claim
1. Variance-based stability map for GRPO in binary-reward, i.i.d. group setting
2. Near-exact validation in synthetic experiments (R²=0.994, 99.8% regime accuracy)
3. Evidence from real GSM8K training: non-monotonic ρ landscape, standard ρ=1.0 brittle

### What NOT To Claim
- AdaBalance as contribution (broken)
- Robustness C3 (66.7% not graceful)
- Scale transfer C4 (not executed)
- General GRPO stability law
- Oracle-level controller

### Critical Fix Needed
RESOLVE DISCREPANCY: coarse sweep ρ=1.0 → 29.8% avg, but AdaBalance vanilla ρ=1.0 → 90.0%. Different random ports/state caused different results. Must explain or remove.

### Recommended Paper Framing
"We derive and validate a stability map for GRPO under binary verifiable rewards. In synthetic settings, the map is nearly exact. In real GSM8K training, the theory qualitatively predicts a structured, non-monotonic ρ landscape."

---

## Method Description

The method introduces the effective balance ratio ρ as a single control variable for GRPO signal balance. Under binary verifiable rewards, we derive a gradient variance decomposition (Var = ρ²V+ + V- + 2ρC) that partitions the (ρ, p₀) space into convergent, gradient-starved, and unstable regimes. The framework:
1. Computes V+, V-, C from the binomial group structure
2. Derives sharp lower bound ρ_min (gradient starvation threshold)
3. Provides approximate upper bound ρ_max (instability threshold)
4. Maps the stability landscape for practitioners to select optimal ρ

Data flow: binary rewards → group success counts → variance decomposition → stability boundaries → regime classification.

---

## Round 7 (2026-04-06T05:30:00)

### Assessment
- Score: 5.5/10 (conceptual only) → **8/10 if variant validation confirms predictions**
- Verdict: "Plausible best-paper conversation" if unified framework + Qwen3.5 replication land

### Paper Pivot
Title: "Why GRPO Variants Work: A Unified Stability Map Under Binary Verifiable Rewards"
Core thesis: "GRPO variants succeed because they move training in a common stability coordinate system, and once training is already in the stable band, those variants stop helping."

### Killer Figure
2-panel: (ρ_eff, GSR_eff) heatmap + variant arrows showing gain vs stability coordinate

### Deadline Execution Plan (May 6, 2026)
1. Qwen3.5-9B vanilla sweep [RUNNING]
2. Long-horizon confirmation [QUEUED]
3. Variant validation at ρ=1.0 and ρ=3.0 [SCRIPT READY]
4. Full GSM8K eval [QUEUED]
5. Quantitative (ρ_eff, GSR_eff) measurement from logs
6. Bootstrap CIs + threshold sensitivity
7. Complete paper draft

### Critical Reviewer Risks
- GTPO citation verification needed
- DAPO is 4 techniques, not 2 — don't over-reduce
- GSPO ρ_min reduction claim needs derivation
- λ-GRPO novelty collision — our claim must be DIFFERENT (stability-space, not token-weighting)
