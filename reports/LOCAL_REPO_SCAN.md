# Local Repository Scan

**Date**: 2026-04-24
**Repo**: /home/tarkoy/nips/nips-grpo-dynamics

## Top-Level Directory Map

| Directory | Purpose |
|-----------|---------|
| `configs/` | Training configs (aser_mvp.yaml, grpo_9b.yaml, sweep_grid.yaml, G4/G8 variants) |
| `idea-stage/` | Idea discovery outputs (IDEA_REPORT.md, LIT_LANDSCAPE.md) |
| `paper/` | LaTeX paper draft (STALE — still CSD/rho narrative) |
| `refine-logs/` | Research refine pipeline outputs (FINAL_PROPOSAL.md = TASA-GRPO) |
| `reports/` | This execution (GPT-5.5 Pro review package) |
| `research-wiki/` | Research knowledge base |
| `results/` | ALL experiment results (wave 1-14, stratified evals, phase diagram) |
| `review-stage/` | Auto-review loop outputs (AUTO_REVIEW.md, REVIEW_STATE.json) |
| `scripts/` | Training launchers, eval scripts, analysis scripts |
| `src/` | Core modules (trainers, reward functions, samplers, compat patches) |
| `templates/` | Templates |
| `tests/` | Test files |

## Component Table

| Component | Path | Purpose | Importance | Notes |
|-----------|------|---------|------------|-------|
| Main trainer | `src/aser_trainer_v14.py` | SPO/Dr.GRPO/TASA backbone + replay CE | Critical | Recently added TASA mode; P0 mask/EOS bugs identified |
| Prompt stats | `src/prompt_stats.py` | Per-prompt EMA baseline/hardness | High | To be replaced by PromptCreditState per GPT-5.5 |
| Replay bank | `src/replay_bank.py` | Hash-dedup verified success store | High | To be replaced by TrustGatedReplayBank per GPT-5.5 |
| MATH reward | `src/math_reward.py` | Partial-credit reward for MATH | Medium | New, for TASA experiments |
| Old rho trainer | `src/rho_grpo_trainer_v14.py` | Legacy rho/ADQ trainer | Archive | Retracted per RETRACTIONS.md |
| Old rho trainer (v1) | `src/rho_grpo_trainer.py` | Original legacy trainer | Archive | Retracted |
| AdaBalance | `src/adabalance.py` | rho controller | Archive | Retracted |
| CSD logging | `src/csd_logging.py` | CSD Q metric | Archive | Retracted |
| Bandit rho | `src/bandit_rho.py` | UCB bandit rho controller | Archive | Retracted |
| Exact rho | `src/exact_rho_controller.py` | Exact rho* controller | Archive | Retracted |
| Stability analysis | `src/stability_analysis.py` | Gradient stability | Archive | Retracted |
| Balanced GRPO | `src/balanced_grpo.py` | Balanced GRPO variant | Archive | Retracted |
| Zero-score handler | `src/zero_score_handler.py` | HalluZero strategies | Ablation | Used in Wave 15 experiments |
| Adaptive dup sampler | `src/adaptive_dup_sampler.py` | Hard-prompt duplication | Ablation | No effect proven, archive |
| Qwen35 compat | `src/qwen35_compat.py` | Qwen3.5 patches | Keep | Required infrastructure |
| Torch compat | `src/torch_compat.py` | PyTorch patches | Keep | Required infrastructure |
| Main launcher | `scripts/run_aser_mvp.py` | Training entry point | Critical | Keep as baseline launcher |
| Eval script | `scripts/eval_stratified.py` | GSM8K test evaluation | Critical | P0: fix first-N default |
| Analysis scripts | `scripts/analyze_*.py` | Result aggregation | Medium | Provenance issues noted |
| Launch scripts | `launch_*.sh` | Remote experiment launchers | Medium | Various waves |
| Main config | `configs/aser_mvp.yaml` | G=2 SPO+Replay config | Baseline | Keep for A control |
| G4 config | `configs/aser_g4_safe.yaml` | G=4 variant | Active | Currently running on server |
| RETRACTIONS | `RETRACTIONS.md` | Retracted claims | Freeze | Academic integrity |
| Proposal (old) | `PROPOSAL_SPO_REPLAY.md` | SPO+Replay proposal | Archive | Superseded by TASA/TRACE |
| Proposal (new) | `refine-logs/FINAL_PROPOSAL.md` | TASA-GRPO proposal | Active | GPT-5.4 score 8.0/10 |
| GPT-5.5 Diagnosis | `GPT55_DIAGNOSIS.md` | TRACE-GRPO diagnosis | Critical | Current execution source |
| Reviewer Memory | `REVIEWER_MEMORY.md` | Review history | Keep | Cross-round context |
| Paper | `paper/main.tex` | LaTeX draft | Stale | CSD/rho narrative, needs full rewrite |
| Results (wave10) | `results/wave10_aser/` | SPO+Replay training | Historical positive | n=200 only |
| Results (wave14 500step) | `results/wave14_500step/` | 500-step collapse | Historical negative | Full-set n=1319 |
| Results (wave14 phase) | `results/wave14_phase_diagram/` | Phase diagram | Historical negative | All near baseline |
| Results (wave13) | `results/stratified_eval_wave13/` | SFT-gold + true-dup | Baseline | Strong SFT-gold upper bound |
