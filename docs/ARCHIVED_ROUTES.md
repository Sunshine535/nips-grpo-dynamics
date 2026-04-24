# Archived / Not Mainline Routes

This document labels previously explored research routes that are **no longer part of
the main method path**. They are preserved for historical transparency, reproducibility
of retracted claims, and reviewer audit — but should NOT be cited as current contributions.

For the retraction history, see [`RETRACTIONS.md`](../RETRACTIONS.md).
For the current main method path, see [`GPT55_DIAGNOSIS.md`](../GPT55_DIAGNOSIS.md)
and `reports/GPT55_REPORT_EXTRACTION.md`.

---

## Current main method path

**TRACE-GRPO**: Trust-Calibrated Replay and Prompt-Conditioned Credit Assignment for Sparse
Binary GRPO. Implemented in:

- `src/trace_grpo_trainer.py`
- `src/prompt_credit_state.py`
- `src/trust_gated_replay_bank.py`
- `scripts/run_trace_grpo.py`
- `configs/trace_grpo_minimal.yaml`

Any claim outside this path is either a **baseline**, an **ablation**, or **historical**.

---

## Archived / Retracted Routes

### 1. Old ρ / CSD / AdaBalance narrative
- **Status**: RETRACTED (see `RETRACTIONS.md`)
- **Files**:
  - `src/rho_grpo_trainer.py` (legacy trainer with fabricated degenerate-group labels)
  - `src/rho_grpo_trainer_v14.py` (legacy rho/ADQ trainer — replaced by ASERTrainerV14)
  - `src/adabalance.py` (AdaBalance controller, retracted)
  - `src/csd_logging.py` (CSD Q_CSD diagnostic — retracted as "proven collapse predictor")
  - `src/bandit_rho.py` (UCB bandit ρ controller)
  - `src/exact_rho_controller.py` (exact-ρ* controller, 0/20 updates in Wave 8)
  - `src/stability_analysis.py` (gradient stability — not theory-faithful)
  - `src/balanced_grpo.py` (balanced variant)
- **Why archived**: RETRACTIONS.md documents sign errors, monotonic sweep on unreliable data,
  Q_CSD drift, AdaBalance ρ=1.0 regression.
- **Role going forward**: Historical negative evidence only. Do NOT cite as current method.

### 2. Stale paper draft
- **File**: `paper/main.tex`
- **Status**: STALE — narrative is CSD/rho/AdaBalance.
- **Role**: Must be fully rewritten around TRACE-GRPO before any submission.

### 3. SPO + Verified Replay proposal
- **File**: `PROPOSAL_SPO_REPLAY.md`
- **Status**: SUPERSEDED by GPT55_DIAGNOSIS.md's TRACE-GRPO recommendation.
- **Role**: Historical evidence that SPO + fixed-λ replay has n=200 positive signal
  but does not survive full-set / long-horizon / seed-stability scrutiny.

### 4. (α, β) Phase diagram route
- **Files**:
  - Phase diagram launchers `launch_wave14b.sh`, `launch_wave14.sh`
  - Phase sweep configs implicitly in `src/aser_trainer_v14.py` (alpha_pos / beta_neg)
- **Evidence**: All 13 sampled grid points on full GSM8K test set lie at 25.2–27.4%,
  indistinguishable from base model 25.5% (see `results/wave14_phase_diagram/`).
- **Role**: Negative evidence that global scalar advantage weighting is not the
  missing mechanism.

### 5. Adaptive duplication sampler
- **File**: `src/adaptive_dup_sampler.py`
- **Status**: ABLATION ONLY.
- **Evidence**: AUTO_REVIEW demonstrated no effect (batch_n_dup = 0 in broken
  implementation; true-dup fix yielded high seed variance without mean improvement).
- **Role**: Kept only to show scheduling-alone is insufficient.

### 6. HalluZero / zero-score reshaping
- **File**: `src/zero_score_handler.py`
- **Status**: ABLATION ONLY.
- **Evidence**: Approaches subsumed by LENS (arXiv 2510.08696), PMPO, NGRPO.
- **Role**: Included in ablation space; not a main-path contribution.

### 7. TASA-GRPO (threshold-anchored signed advantage)
- **File**: `src/aser_trainer_v14.py` (`tasa` backbone mode)
- **Status**: PARALLEL BASELINE / complementary backbone.
- **Evidence**: Closest prior art CoRPO (arXiv 2511.04439) has 65% mechanism overlap.
- **Role**: Kept as alternative backbone choice for TraceGRPOTrainer (`--backbone tasa`).
  TRACE's mechanism (replay trust/drift) is orthogonal to TASA's mechanism (advantage
  sign anchoring); both may combine, but the TRACE mechanism is the paper claim.

### 8. Fixed SPO + uniform-λ verified replay (legacy ASER)
- **File**: `src/aser_trainer_v14.py`, `src/prompt_stats.py`, `src/replay_bank.py`
- **Status**: KEPT AS BASELINE (variant A in A/B/C comparison).
- **Role**: Direct control for measuring whether TRACE's trust gate adds value over
  a fixed-λ uniform replay baseline. Must NOT be mutated or renamed.

---

## File-level role table

| File | Current role | Rationale |
|------|--------------|-----------|
| `src/trace_grpo_trainer.py` | **MAIN METHOD** | TRACE-GRPO implementation |
| `src/prompt_credit_state.py` | **MAIN METHOD** | Beta-posterior per prompt |
| `src/trust_gated_replay_bank.py` | **MAIN METHOD** | Trust-gated weighted replay |
| `src/aser_trainer_v14.py` | BASELINE / alt-backbone | A variant + TASA option |
| `src/prompt_stats.py` | BASELINE | Legacy EMA (variant A) |
| `src/replay_bank.py` | BASELINE | Legacy uniform replay (variant A) |
| `src/math_reward.py` | Support | Partial-credit reward for MATH |
| `src/rho_grpo_trainer*.py` | ARCHIVED | Retracted rho/ADQ routes |
| `src/adabalance.py` | ARCHIVED | Retracted controller |
| `src/csd_logging.py` | ARCHIVED | Retracted diagnostic |
| `src/bandit_rho.py` | ARCHIVED | Retracted |
| `src/exact_rho_controller.py` | ARCHIVED | Retracted |
| `src/stability_analysis.py` | ARCHIVED | Retracted |
| `src/balanced_grpo.py` | ARCHIVED | Retracted |
| `src/zero_score_handler.py` | ABLATION | HalluZero variants |
| `src/adaptive_dup_sampler.py` | ABLATION | Scheduling baseline |
| `src/qwen35_compat.py` | INFRASTRUCTURE | Qwen3.5 patches |
| `src/torch_compat.py` | INFRASTRUCTURE | PyTorch patches |
| `src/rho_grpo.py` | SUPPORT | GSM8K binary reward function (used by all trainers) |
| `src/provenance.py` | SUPPORT | Run manifest (new, per GPT-5.5 Task 2) |
| `paper/main.tex` | STALE | Must be rewritten |
| `PROPOSAL_SPO_REPLAY.md` | SUPERSEDED | Kept for history |

---

## Rule for any new work on this repository

1. The main method path is TRACE-GRPO; do not regress to archived routes.
2. Any experiment using an archived route must be labeled clearly as
   "baseline", "ablation", or "historical" in its output directory and manifest.
3. Paper and README claims must reference the current main method path only.
4. Raw results in `results/` must NOT be deleted — they are the negative
   evidence that motivated TRACE-GRPO.
