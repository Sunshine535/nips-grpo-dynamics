# Claim Update Log

Date: 2026-04-25. Tracks paper / README / proposal claims updated per GPT-5.5 Task 12.

Rule: claims are updated ONLY after evidence supports them.

| Claim | Old Text | New Text | Evidence | Status |
|-------|----------|----------|----------|--------|
| Main method of project | "GRPO Dynamics: Phase Diagrams and Zero-Score Gradient Reshaping" (README top; paper/main.tex) | "TRACE-GRPO: Trust-Calibrated Replay and Prompt-Conditioned Credit Assignment for Sparse Binary GRPO" | GPT55_DIAGNOSIS.md recommends TRACE-GRPO | PENDING paper rewrite (Task 12) — blocked on A/B/C results |
| SPO+Replay is the current focus | "SPO + Verified Replay CE" (PROPOSAL_SPO_REPLAY.md, README) | Superseded. Kept as variant A baseline. | AUTO_REVIEW.md fatal weaknesses; 500-step full-set collapse | PROPOSAL archived. README update pending. |
| α/β phase control is the main mechanism | paper/main.tex older draft | REMOVED — phase grid shows no effect vs base | results/wave14_phase_diagram/ evals ~25-27% = base | COMPLETE — docs/ARCHIVED_ROUTES.md labels it ARCHIVED |
| ρ / CSD / AdaBalance stability map | paper/main.tex stale | REMOVED — retracted per RETRACTIONS.md | RETRACTIONS.md, stability_analysis.py flagged | COMPLETE — docs/ARCHIVED_ROUTES.md labels it ARCHIVED |
| Adaptive duplication helps | older configs / analysis | Ablation only. No effect on mean; high variance. | AUTO_REVIEW.md; wave13 true-dup per-seed variance | COMPLETE — adaptive_dup_sampler labeled ABLATION |
| RFT-only is a complete baseline | older analysis | CE alone is insufficient; keep only as lower-bound baseline. | round2 analysis ~35-38% | CARRIED as existing statement; no change needed |
| SFT-gold is an upper bound we must beat | implicit | Explicit: method must either beat or justify data-efficiency gap vs SFT-gold 84.6% on n=200. | wave13 sft_gold_seed42.json = 0.845 | CARRIED; must be emphasized in paper |
| First-200 prefix is valid paper evidence | implicit default in eval_stratified.py | REPLACED. Main paper metric = GSM8K test n=1319 (full), with explicit --selection flag. | GPT-5.5 Task 3, Task 7 | COMPLETE — eval_stratified.py fixed |
| TASA-GRPO is the main method | FINAL_PROPOSAL.md | "Parallel baseline / complementary advantage mechanism; TRACE is the main method per GPT-5.5 diagnosis." | GPT55_DIAGNOSIS.md; TASA vs TRACE are orthogonal | PENDING — FINAL_PROPOSAL.md needs TRACE addendum |

## Forbidden updates (per GPT-5.5 protocol)
- Do NOT claim SOTA without full fair comparison against RePO, DAPO, Dr.GRPO official configs.
- Do NOT hide Wave 14 500-step collapse (44.6% full-set).
- Do NOT write that the method works generally if only GSM8K is tested.
- Do NOT describe ablation (constant_gate) as main method.
- Do NOT claim the trust gate works if `lambda_eff` / `mean_frontier` logs do not reflect adaptive behaviour.
- Do NOT revive ρ/CSD/AdaBalance paper claims.

## Pending paper-level updates (blocked on evidence)

1. **Rewrite `paper/main.tex`** around TRACE-GRPO. BLOCKED UNTIL:
   - TRACE A/B/C full-set results available.
   - C demonstrably beats A and B.
   - Mechanism ablations pass.
2. **Update `README.md`** to point to TRACE-GRPO as current focus and link to `docs/ARCHIVED_ROUTES.md`.
3. **Update `PROPOSAL_SPO_REPLAY.md`** header to explicitly mark it SUPERSEDED.
