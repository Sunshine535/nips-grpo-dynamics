# Reviewer Memory (nightmare-mode persistent brain)

## TASA+CE Replay Loop — Round 1 — Score: 3/10 (not ready)
Reviewer: GPT-5.4 xhigh via MCP + codex exec (nightmare, dual review)
Thread: `019dd219-f6a5-78b1-a286-794b2f7f7b3b`
Date: 2026-04-28

### Key findings
1. **CRITICAL**: Two TASA-only runs (sage 46.8% vs aser 62.6%) from identical formula/config. Not seed noise — 18.9pp gap on overlapping seeds 42/43/44. Damages all absolute claims.
2. **CRITICAL**: Paper says "all variants use same codebase" but A uses run_aser_mvp.py, B/C/D use run_sage_grpo.py. Provenance misstatement.
3. **CRITICAL**: n=3 seeds insufficient. D 95% CI ≈ [37.7, 72.4], overlaps C [44.6, 53.4]. D>B Welch p≈0.17, D>C p≈0.27.
4. **HIGH**: Eval numeric format mismatches (12.00 vs 12, 694. vs 694) change scores by +1.16pp for Dr.GRPO.
5. **HIGH**: Theory errors — boundedness proof wrong (extrema are ±1 not ±1/G for edge cases).
6. **HIGH**: Sign baseline A=2r-1 promised but not shown.
7. **HIGH**: C variant has undisclosed frontier/age weighting in pair loss — not a clean DPO ablation.
8. **MEDIUM**: sage_minimal_abc eval JSONs stripped — no per_q, no manifests, not auditable.
9. **MEDIUM**: RLEP already found positive-only replay beats negative-inclusive — weakens novelty.

### Verified claims
- Paper table numbers match sage_minimal_abc JSONs exactly
- Dr.GRPO 27.77% ± 0.59% matches drgrpo_g4_safe (4 seeds)
- TASA formula identical in both trainers (shared function)
- Binary degeneracy math is correct
- Greedy decoding, same eval protocol

### Unverified/false
- "Same codebase" — FALSE (A=aser, B/C/D=sage)
- D>B and D>C — not statistically significant at n=3
- Boundedness proof — error in appendix
- C is "DPO-style pairwise" — actually has frontier/age weighting

### Watchlist for Round 2
- [CRITICAL] Root cause of sage vs aser TASA gap — trainer stack or RNG/evidence bank side effects?
- [CRITICAL] Matched-seed runs under single trainer with 5+ seeds
- [HIGH] Fix eval answer canonicalization
- [HIGH] Add missing sign baseline A=2r-1
- [HIGH] Fix theory proof
- [HIGH] Clean C variant — either disable frontier/age or disclose it
- [MEDIUM] Check if sage_minimal_abc per_q data matches recounted accuracy

---

## Archived: SPO Loop — Round 1 — Score: 2/10
Thread: `019db654-8306-7492-a2cf-24911e7f3151` | Date: 2026-04-23
- FATAL: n=200 eval only, 500-step collapse, null phase diagram
- SPO+Replay abandoned, pivoted to TASA+CE Replay

## Archived: Prior Loop Summary
- Round 1: 3/10 — adaptive-dup no-op
- Round 2: 5/10 — fatal fixes done, novelty gaps remain
