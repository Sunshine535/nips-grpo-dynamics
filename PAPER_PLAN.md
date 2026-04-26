# Paper Plan

**Title**: Threshold-Anchored Signed Advantage with Verified Replay for Sparse Binary GRPO
**One-sentence contribution**: We show that replacing group-mean-centered advantage with a threshold-anchored signed formulation and adding supervised CE replay of verified successes yields +27 pp over Dr.GRPO on GSM8K under small-group binary GRPO, while contrastive negative replay provides no additional benefit.
**Venue**: NeurIPS 2026
**Type**: Method
**Date**: 2026-04-26
**Page budget**: 9 pages (main body to Conclusion end, excluding references & appendix)
**Section count**: 6

---

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Section |
|---|-------|----------|--------|---------|
| C1 | Group-mean-centered advantage (Dr.GRPO) degenerates under binary {0,1} reward at small G | Dr.GRPO G=4: 27.8% vs base 25.5% (+2.3 pp only), 4 seeds | Supported | S3, S4 |
| C2 | TASA signed threshold advantage restores effective learning under binary reward | TASA-only 46.8% vs Dr.GRPO 27.8% (+19 pp), 3 seeds; TASA G=4 (4 seeds) 62.6% vs Dr.GRPO 27.8% (+34.8 pp) | Supported | S3, S4 |
| C3 | Supervised CE replay of verified successes further improves TASA by +8.3 pp | D (TASA+CE replay) 55.0% vs B (TASA-only) 46.8%, 3 seeds each | Supported | S3, S4 |
| C4 | CE replay is superior to importance-weighted PG replay (conceptual; cite RePO/RLEP difference) | CE replay is simpler (no stored logprobs, no IS ratio), stable, and effective | Partially supported (no direct RePO run) | S3, S5 |
| C5 | Contrastive pair replay (DPO-like) is suboptimal vs positive-only CE replay | C (contrastive) 49.0% < D (positive CE) 55.0%, delta = -6.0 pp, 3 seeds | Supported | S4, S5 |
| C6 | Per-prompt hash-deduped bank prevents replay redundancy | Design choice; bank stores max 2 unique solutions per prompt | Design claim | S3 |

---

## Structure

### S0 Abstract
- **What we achieve**: A method (TASA + verified CE replay) that improves GSM8K accuracy by +27 pp over Dr.GRPO under small-group (G=4) binary-reward GRPO with LoRA
- **Why it matters / is hard**: Small-G binary GRPO degenerates because group-mean advantage collapses to near-zero signal when only {0,1} rewards are available; prior replay methods (RePO, RLEP) use importance-weighted PG which requires storing old logprobs and is sensitive to staleness
- **How we do it**: (1) Threshold-anchored signed advantage (TASA) that guarantees correct reward-sign preservation regardless of group statistics; (2) Online supervised CE replay of hash-deduped verified successes from a per-prompt bank
- **Evidence**: On GSM8K (full n=1319), TASA+CE replay achieves 55.0% vs Dr.GRPO 27.8% (+27.2 pp) and TASA-only 46.8% (+8.3 pp); contrastive pair replay underperforms positive CE by 6.0 pp
- **Most remarkable result**: Dr.GRPO essentially fails to learn under G=4 binary reward (27.8% vs 25.5% base), while our method reaches 55.0%
- **Estimated length**: 200 words

### S1 Introduction (1.5 pages)
- **Opening hook**: GRPO enables math reasoning post-training without a reward model, but under sparse binary verifiable rewards at small group size, the standard group-mean-centered advantage becomes degenerate -- groups of all-correct or all-incorrect responses produce zero or near-zero gradients.
- **Gap / challenge**: Dr.GRPO fixes length bias but retains mean-centered advantage, which we show collapses to +2.3 pp over base under G=4 binary reward. Replay methods (RePO, RLEP) address data efficiency but use importance-weighted PG, requiring stored logprobs and risking staleness.
- **One-sentence contribution**: We propose TASA (Threshold-Anchored Signed Advantage) + verified CE replay: a signed advantage formulation that guarantees correct reward-direction gradients under binary reward, combined with a simple supervised replay of verified successes that avoids importance sampling entirely.
- **Contributions**:
  1. TASA advantage formulation: A_i = (r_i - c)+/Z+ - (c - r_i)+/Z- with threshold c, guaranteeing sign-correct gradients under binary reward (S3.1)
  2. Verified CE replay: online SFT-loss replay from a per-prompt hash-deduped success bank, simpler and more effective than PG-based replay (S3.2)
  3. Ablation showing positive CE replay outperforms contrastive pair replay by 6 pp, challenging the assumption that negative evidence helps in this setting (S4)
- **Hero figure**: Figure 1 shows a bar chart comparing Base (25.5%), Dr.GRPO (27.8%), TASA-only (46.8%), TASA+CE Replay (55.0%), and TASA+Contrastive (49.0%) on GSM8K full test. Visually demonstrates: (a) Dr.GRPO failure, (b) TASA's large gain, (c) CE replay's additive benefit, (d) contrastive replay is suboptimal.
- **Key citations**: GRPO (Shao et al. 2024), Dr.GRPO (Liu et al. 2025), RePO (Li et al. 2025), RLEP (Zhang et al. 2025), DPO (Rafailov et al. 2023)

### S2 Related Work (1 page)
- **Subtopics**:
  1. *GRPO and variants for math reasoning*: GRPO, Dr.GRPO, DAPO, GSPO, CoRPO -- group-relative advantage under verifiable rewards. Position: these fix various GRPO issues but not the binary-reward small-G degeneracy.
  2. *Replay and experience management in RLVR*: RePO (PG replay with IS), RLEP (two-phase success replay with PG), ExGRPO (experience buckets). Position: all use PG+importance weighting for replay; we use supervised CE, which is simpler and avoids IS staleness.
  3. *Pairwise preference optimization*: DPO, KTO, online DPO variants. Position: our contrastive pair ablation shows DPO-like pairwise loss underperforms positive-only CE in this binary sparse setting.
  4. *Negative evidence in RL*: RE-GRPO (hard negatives), negative-reinforcement findings. Position: our ablation shows direct negative push hurts; failures are better used as diagnostic, not training signal.

### S3 Method (2 pages)
- **S3.1 Preliminaries**: GRPO setup, binary reward r in {0,1}, group size G, the degeneracy problem (all-pass/all-fail groups → zero advantage with mean-centered formulation)
- **S3.2 TASA -- Threshold-Anchored Signed Advantage**:
  - Formula: A_i = (r_i - c)+/Z+ - (c - r_i)+/Z- where Z+ = sum of positive excesses, Z- = sum of negative deficits
  - Properties: P1 (sign preservation), P2 (monotonicity), P3 (bounded [-1,1]), P4 (binary equivalence to {+1/n+, -1/n-} labeling)
  - Why it works: threshold c=0.5 for binary reward anchors the reference point outside the group, so even all-correct groups get positive advantage
- **S3.3 Verified CE Replay**:
  - Bank: per-prompt, hash-deduped, max K=2 entries, stores token IDs + prompt text
  - Loss: L_total = L_TASA_PG + lambda * L_CE_replay + beta * L_KL
  - CE replay uses standard next-token prediction loss with prompt tokens masked (-100)
  - Contrast with RePO/RLEP: no stored logprobs, no importance ratio, no staleness issue
  - Warmup: replay activates after W=50 steps to allow bank population
- **Notation table**: summarize all symbols

### S4 Experiments (2.5 pages)
- **Setup**: Qwen3.5-9B + LoRA r=64, GSM8K train, binary reward, G=4, lr=2e-5, 200 steps, greedy eval on full GSM8K test (n=1319)
- **Table 1: Main Results**

| Method | seed42 | seed43 | seed44 | seed45 | Mean | Std |
|--------|:------:|:------:|:------:|:------:|:----:|:---:|
| Base Qwen3.5-9B | - | - | - | - | 25.5 | - |
| Dr.GRPO (G=4) | 28.5 | 27.8 | 27.1 | 27.7 | 27.8 | 0.6 |
| TASA (G=4, 4 seeds) | 56.1 | 73.6 | 67.3 | 53.3 | 62.6 | 9.5 |
| TASA-only (B, 3 seeds) | 45.9 | 50.5 | 43.9 | - | 46.8 | 3.4 |
| TASA+CE Replay (D) | 62.1 | 48.1 | 54.9 | - | 55.0 | 7.0 |
| TASA+Contrastive (C) | 50.2 | 46.6 | 50.3 | - | 49.0 | 2.1 |
| SPO+Fixed Replay (A) | 32.6 | 28.1 | 33.4 | - | 31.4 | 2.8 |

- **Figure 2**: Bar chart for 3-seed means (B/D/C/A) showing D > C > B > A
- **Figure 3**: Training reward curves for TASA vs Dr.GRPO (showing TASA learns, Dr.GRPO flat)
- **Key findings**:
  1. Dr.GRPO fails under G=4 binary reward (+2.3 pp only)
  2. TASA alone recovers +19 pp over Dr.GRPO
  3. CE replay adds +8.3 pp on top of TASA
  4. Contrastive pair replay is inferior to positive CE by 6 pp

### S5 Analysis and Ablation (1 page)
- **Why Dr.GRPO fails**: With binary {0,1}, G=4 generates only a few advantage levels; mean-centered advantage often = 0 for all-pass/all-fail groups. Show all-fail/all-pass group rates from training logs.
- **Why CE > contrastive pair**: Positive CE reinforces verified solution patterns; pairwise negative push may suppress useful reasoning steps that happen to appear in failed attempts. C has lower variance (2.1 vs 7.0) but lower mean (49.0 vs 55.0) -- negative contrast stabilizes but suppresses learning.
- **CE vs PG replay (conceptual comparison)**: Table comparing our CE replay vs RePO's PG replay vs RLEP's two-phase PG replay on mechanism dimensions (stored data, loss type, IS requirement, online/offline).
- **Limitations**: Only GSM8K; 3 seeds per ablation variant; no official RePO/RLEP reproduction; G=4 only (larger G not tested); LoRA only (full FT not tested)

### S6 Conclusion (0.5 pages)
- **Restatement**: We identified that mean-centered advantage degenerates under binary sparse reward at small G, proposed TASA as a threshold-anchored signed alternative, and showed that simple supervised CE replay of verified successes is more effective than importance-weighted PG replay or contrastive pair replay.
- **Limitations**: Single dataset (GSM8K), limited seeds, no direct RePO/RLEP baseline run
- **Future work**: (1) Scale to larger G and full fine-tuning; (2) Direct comparison with RePO/RLEP; (3) Extend to MATH and other math benchmarks; (4) Investigate failure-aware replay gating

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Bar chart (hero) | Main comparison: Base / Dr.GRPO / TASA-only / TASA+CE Replay / TASA+Contrastive on GSM8K full | sage_minimal_abc + tasa/drgrpo evals | HIGH |
| Fig 2 | Training reward curves | TASA vs Dr.GRPO training reward over 200 steps (showing TASA learns, Dr.GRPO flat) | tasa_g4_safe + drgrpo_g4_safe step_stats | HIGH |
| Fig 3 | Ablation bar chart | B/D/C/A means with error bars (3 seeds) | sage_minimal_abc evals | HIGH |
| Table 1 | Main results table | Full per-seed + mean/std results for all methods | All eval JSONs | HIGH |
| Table 2 | Mechanism comparison | CE replay vs PG replay (RePO) vs Two-phase PG (RLEP) on stored data, loss, IS, online/offline | Manual comparison | MEDIUM |
| Fig 4 (appendix) | TASA advantage properties | Visualization of A_i values under different group compositions | Synthetic | LOW |

---

## Citation Plan

- S1 Intro: GRPO (Shao et al. 2024), Dr.GRPO (Liu et al. 2025), RePO (Li et al. 2025), RLEP (Zhang et al. 2025), DPO (Rafailov et al. 2023)
- S2 Related Work:
  - GRPO family: GRPO, Dr.GRPO, DAPO (Yu et al. 2025), GSPO [VERIFY], CoRPO [VERIFY], "It Takes Two" (G=2 analysis) [VERIFY]
  - Replay: RePO, RLEP, ExGRPO (2025) [VERIFY]
  - Pairwise: DPO, KTO [VERIFY], online DPO variants [VERIFY]
  - Negative evidence: RE-GRPO [VERIFY], negative reinforcement findings (OpenReview 2026) [VERIFY]
  - GRPO analysis: GRPO effective loss / contrastive analysis (2025) [VERIFY]
- S3 Method: GRPO, Dr.GRPO (for baseline advantage), TRL library
- S4 Experiments: Qwen3.5-9B model card, GSM8K dataset (Cobbe et al. 2021), LoRA (Hu et al. 2021)

---

## Next Steps
- [ ] /paper-figure to generate all figures
- [ ] /paper-write to draft LaTeX section by section
- [ ] /paper-compile to build PDF
