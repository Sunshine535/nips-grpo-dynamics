# Reviewer Memory (nightmare-mode persistent brain)

## New Loop — Round 1 — Score: 2/10 (not ready)
Reviewer: Codex xhigh (nightmare, read-only sandbox)
Thread: `019db654-8306-7492-a2cf-24911e7f3151`
Date: 2026-04-23

### Key findings
1. **FATAL**: ALL positive results (69.4% SPO+Replay) are evaluated on a deterministic prefix of 200 GSM8K test questions, not the full 1319-question test set. This is NOT paper-grade.
2. **FATAL**: 500-step training collapses to 44.6% on full test set (n=1319). Method does not scale.
3. **FATAL**: Phase diagram (α,β) shows ZERO effect — all points near base model (25-27%).
4. **HIGH**: SFT-gold (84.6%) crushes SPO+Replay by 15pp → "GRPO is necessary" narrative is dead.
5. **HIGH**: Seed variance is extreme (54.5%-88.0%) → unstable method.
6. **HIGH**: SPO baseline barely persistent — only ~754/7473 prompts seen in 200 steps, mean seen=1.05.
7. **HIGH**: Dr. GRPO = 39.3% ± 4.0% not backed by consistent committed artifacts.
8. **MEDIUM**: No committed Wave 14 artifacts locally.

### Verified claims
- SPO+Replay = 69.4 ± 10.4% on n=200 (committed JSONs match)
- Fixed ρ=0.70 = 54.2 ± 10.2% on n=200
- SFT-gold = 84.6 ± 0.6% on n=200
- RFT-only = 36.1 ± 1.3% on n=200
- Base = 25.5% on n=200
- Adaptive duplication has no effect
- Train/test split is clean (no leakage)
- Greedy decoding, max_new_tokens=256, same protocol for all methods

### Unverified
- Wave 14 full-set (n=1319) numbers — not committed locally
- Dr. GRPO = 39.3% — inconsistent artifact trail
- Whether n=200 results hold on full test set

### Recommended direction
"Early-stopping transient-gain paper" — evaluate 200-step checkpoints on full 1319 test set. If relative gains hold, there's a small empirical contribution. If not, kill the positive-results framing.

### Watchlist for Round 2
- [CRITICAL] Full-set (n=1319) evaluation of ALL 200-step adapters + baselines
- [CRITICAL] Whether the SPO+Replay vs fixed-ρ gap persists on full test set
- [HIGH] Root cause of 500-step collapse
- [HIGH] What happens at intermediate steps (100, 150, 200, 300)?
- Any new cherry-picking in results presentation

---

## Prior Loop Summary (archived)
- Round 1: 3/10 — adaptive-dup no-op, no committed artifacts
- Round 2: 5/10 — fatal fixes done, novelty gaps remain
- Known structural confounders: gradient_checkpointing bug, TRL 0.14 quirks
- Established: SPO EMA baseline works, replay CE is clean, no eval leakage
