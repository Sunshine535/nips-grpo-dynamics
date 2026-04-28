# Auto Review Loop — TASA + CE Replay (nightmare difficulty)

**Started**: 2026-04-28
**Target**: NeurIPS 2026 — TASA + Verified CE Replay for Sparse Binary GRPO
**MAX_ROUNDS**: 4
**Reviewer**: GPT-5.4 via codex exec (nightmare — GPT reads repo directly)
**Goal**: Achieve score >= 6/10 for top-venue readiness

---

## Data Inventory

| Variant | Source | Seeds | n_eval | Mean% | Std% |
|---------|--------|-------|--------|-------|------|
| Base Qwen3.5-9B | — | 1 | 1319 | 25.5 | — |
| Dr.GRPO (G=4) | drgrpo_g4_safe | 4 (42-45) | 1319 | 27.8 | 0.6 |
| A: SPO+Fixed Replay | sage_minimal_abc | 3 (42-44) | 1319 | 31.4 | 2.8 |
| B: TASA-only | sage_minimal_abc | 3 (42-44) | 1319 | 46.8 | 3.4 |
| C: TASA+Contrastive | sage_minimal_abc | 3 (42-44) | 1319 | 49.0 | 2.1 |
| D: TASA+CE Replay | sage_minimal_abc | 3 (42-44) | 1319 | 55.0 | 7.0 |

Pass@k pilot (100 questions, k=25):
- B (TASA): pass@1=0.428±0.028 (7 seeds)
- D (TASA+CE): pass@1=0.471±0.057 (8 seeds)
- Base: pass@1=0.247

Known issues:
- D has high variance (std=7.0, range 48.1-62.1%)
- Only 3 seeds for A/B/C/D (need 5+)
- GSM8K only (no second benchmark)
- No G sweep ({2,4,8})
- No direct RePO/RLEP reproduction
- tasa_g4_safe (different codebase) gives 62.6% TASA-only vs sage 46.8% — discrepancy unresolved

---

## Round 1 (2026-04-28)

### Assessment (Summary)
- **Score: 3/10** (consensus from dual review: MCP + codex exec)
- **Verdict: not ready**
- **ThreadId**: `019dd219-f6a5-78b1-a286-794b2f7f7b3b`
- Key criticisms:
  1. TASA discrepancy (46.8% vs 62.6%) unresolved — damages all absolute claims
  2. Provenance misstatement ("same codebase" is false)
  3. n=3 seeds insufficient — D>B and D>C not statistically significant
  4. Eval numeric format bugs (+1.16pp artifact for Dr.GRPO)
  5. Theory proof errors (boundedness)
  6. Missing sign baseline A=2r-1
  7. C variant has undisclosed frontier/age weighting
  8. Novelty weak without direct RePO/RLEP comparison

### Reviewer Raw Response

<details>
<summary>Click to expand full MCP reviewer response</summary>

Score: 3/10
Verdict: not ready

Verified claims:
- Homogeneous binary-reward groups do give zero mean-centered advantage
- TASA implementation shared across both trainers (same function)
- Paper table numbers match sage_minimal_abc JSONs
- Dr.GRPO 27.77% matches drgrpo_g4_safe
- Replay positioning vs RePO/RLEP is directionally fair

Unverified/false claims:
- "All variants use the same training codebase" — FALSE (A=run_aser_mvp.py, B/C/D=run_sage_grpo.py)
- "CE replay adds +8.3pp" — not established (Welch p~0.17)
- "Positive CE outperforms contrastive by 6.0pp" — not reliable (Welch p~0.27)
- TASA +19.0pp not robust due to 62.6% alternate result
- RLEP already reported positive-only beats negative-inclusive

Weaknesses (ranked):
1. TASA discrepancy 46.8% vs 62.6% — minimum fix: single trainer stack
2. Provenance misstatement — minimum fix: correct all statements
3. n=3 with std=7.0 — minimum fix: 5-10 matched seeds with CIs
4. Novelty weak — minimum fix: direct RePO/RLEP comparison
5. Scope too narrow — minimum fix: MATH-500
6. Missing sign baseline A=2r-1
7. Binary degeneracy not fully demonstrated empirically
8. Transparency doesn't compensate for missing evidence

</details>

<details>
<summary>Click to expand codex exec review (additional findings)</summary>

Also scored 3/10. Additional issues:
- Eval numeric format mismatches (12.00 vs 12) change scores by +1.16pp
- Appendix boundedness proof wrong: extrema are +/-1 not +/-1/G
- C variant has undisclosed frontier/age weighting (use_prompt_frontier: true)
- sage_minimal_abc eval JSONs stripped — no per_q data
- Conclusion referenced wrong data source (FIXED before review)

</details>

### Actions Taken (Phase C)
- Starting implementation fixes...

### Status
- Continuing to Phase C
- Difficulty: nightmare
