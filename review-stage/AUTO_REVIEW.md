# Auto Review Loop — New Loop (nightmare difficulty)

**Started**: 2026-04-23
**Target**: NeurIPS 2026 — positive-results paper
**MAX_ROUNDS**: 4
**Reviewer**: GPT-5.4 xhigh via Codex MCP (read-only sandbox, nightmare)
**Goal**: Confirm positive results → implement SOTA baselines → unified comparison

---

## Round 1 (2026-04-23)

### Assessment (Summary)
- **Score: 2/10**
- **Verdict: NOT READY**
- **Thread**: `019db654-8306-7492-a2cf-24911e7f3151`

### Key Critical Findings

1. **FATAL**: All positive results (69.4% SPO+Replay) evaluated on deterministic prefix of 200 GSM8K test questions, not full 1319-question set
2. **FATAL**: 500-step training collapses to 44.6% on full set — method doesn't scale
3. **FATAL**: Phase diagram (α,β) shows zero effect — all near baseline 25%
4. **HIGH**: SFT-gold (84.6%) crushes SPO+Replay by 15pp
5. **HIGH**: Seed variance extreme (54.5%—88.0%) — unstable method
6. **HIGH**: Dr. GRPO = 39.3% not backed by consistent artifacts

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 2/10

Verdict: not ready

Is there a positive-results paper here? No. The only reproducible positive story I could verify is on the first 200 GSM8K test questions, because the main evaluator simply does `select(range(n))` rather than a random or full-set evaluation in eval_stratified.py:50. That is exploratory, not paper-grade. Taking your reported Wave 14 numbers at face value makes it worse, not better: a 500-step full-set collapse and a null phase diagram kill the "stable positive method" claim outright.

The ONE most promising direction is a much narrower early-stopping paper, not a stable-method paper: "SPO+Replay gives transient early gains at G=2, but requires checkpoint selection and longer training collapses." Concretely, the next decisive experiment is not another method ablation. It is a full-1319 sweep over the existing 200-step and intermediate checkpoints. If there is a reproducible full-test peak before collapse that beats fixed ρ=0.70 and RFT-only, you may have a small empirical paper. If not, kill the positive-results framing.

Verified claims:
- SPO+Replay = 69.4 ± 10.4% on n=200 (committed JSONs match)
- Fixed ρ=0.70 = 54.2 ± 10.2% on n=200
- SFT-gold = 84.6 ± 0.6% on n=200
- RFT-only = 36.1 ± 1.3% on n=200
- Base = 25.5% on n=200
- Adaptive duplication has no effect
- Train/test split clean (no leakage)
- Greedy decoding, max_new_tokens=256, same protocol for all methods

Unverified:
- Wave 14 full-set (n=1319) numbers — not committed locally
- Dr. GRPO = 39.3% — inconsistent artifact trail
- Whether n=200 results hold on full test set

Weaknesses:
- Fatal: main positive claim rests on deterministic prefix of 200 test questions
- Fatal: method does not scale in training time or evaluation size
- High: SFT-gold crushes the method by 15pp
- High: SPO's per-prompt baseline barely persistent (~754/7473 prompts seen)
- High: no protection against long-horizon drift — 500-step collapse expected
- High: seed variance enormous (54.5% to 88.0%)
- High: provenance messy — round2_analysis.json double-counts, Dr. GRPO baseline not pinned
- Medium: greedy decoding with max_new_tokens=256 may bias against verbose policies
- Medium: proposal still cherry-picks rhetoric from transient win

</details>

### Actions Taken (Phase C)

1. **Full-set evaluation queued**: Created `launch_fullset_eval.sh` — evaluates ALL 200-step adapters (9 SPO+Replay, 10 fixed-ρ, 4 RFT, 4 SFT-gold, 3 Dr.GRPO, 3 SPO-only) on full n=1319 GSM8K test set. Auto-starts after current pipeline (Wave 14b/15) completes.
2. **SOTA baseline research**: Identifying official implementations of DAPO, RLOO, RePO, GPG for unified comparison.
3. **500-step collapse investigation**: Reviewing training dynamics and hyperparameters.

### Results
- Pending: full-set evaluation (ETA ~12-18h after pipeline completion)

### Status
- Continuing to Phase C (fixes in progress)
- Difficulty: nightmare
- Next: Wait for full-set eval → determine if positive results hold → then SOTA comparison
