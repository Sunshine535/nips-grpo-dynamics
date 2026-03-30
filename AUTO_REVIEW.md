# NeurIPS-Style Code Review: nips-grpo-dynamics

**Overall score: 4/10**

This repository has real substance: the method variants are implemented in separate trainer/controller modules, the top-level pipeline is documented, and the codebase organization is understandable. I also verified that the Python sources in `src/` and `scripts/` pass `python3 -m compileall -q src scripts`. However, I do not consider the project ready to reliably produce the paper’s experimental results in its current form. The main reasons are a broken benchmark path, incomplete distributed/resume support, and the absence of evidence of a successful end-to-end run.

## Main Findings

### 1. MATH evaluation in the main phase-diagram evaluator is broken

`scripts/eval_phase_point.py:96-129` always pulls the gold answer from an `answer` field via `_get_answer()`. That works for GSM8K, but the MATH dataset does not use that schema. The MATH branch at `scripts/eval_phase_point.py:193-205` still calls the same generic `evaluate_dataset()` function, so `gold_answer` becomes empty for MATH examples and `math_accuracy` is not trustworthy.

Why this matters:
- Phase 1, Phase 2, Phase 4, and Phase 5 all call `eval_phase_point.py --eval_math`, so a central reported metric path is invalid.
- Any figure/table based on `math_accuracy` is suspect until this is fixed.

### 2. The curriculum comparison is not actually using the best static point from the sweep

`scripts/run_all_experiments.sh:461-466` hardcodes `--best_alpha 0.5 --best_beta 1.0` for Phase 7. But the code only computes the true best static point later in `scripts/build_phase_diagram.py:363-375`. As written, the curriculum experiments are compared against a fixed reference, not the sweep-derived optimum.

Why this matters:
- The curriculum comparison is not aligned with the stated experimental logic.
- This can materially change conclusions about whether curriculum strategies outperform the best static setting.

### 3. Multi-GPU support is partial and inconsistent

Only the Phase 1 baseline uses true multi-GPU launch via `accelerate` in `scripts/run_all_experiments.sh:149-159`. The large sweeps in Phase 2-5 instead export one GPU per subprocess and run plain Python jobs (`scripts/run_all_experiments.sh:205-242`, `scripts/run_all_experiments.sh:300-333`, `scripts/run_all_experiments.sh:373-387`, `scripts/run_all_experiments.sh:420-433`). That is job-level parallelism, not distributed training for a single run.

Related concern:
- `scripts/train_grpo_halluzero.py:61-64` parses `--local_rank`, but the script never uses it, which suggests DDP support was started but not completed.

Assessment:
- Multi-GPU inference exists in the evaluators through `device_map="auto"`.
- Multi-GPU training exists only for one path, not for the main experimental matrix.

### 4. Checkpoint resume support is incomplete for AdaBalance

`scripts/train_adabalance.py:183-224` recreates a fresh `AdaBalanceController` on every launch and then calls `trainer.train(resume_from_checkpoint=...)`. But `src/adabalance.py:27-48` shows that the controller keeps critical internal state in memory: `rho`, EMA statistics, histories, and step counters. None of that state is checkpointed or restored before resume.

Why this matters:
- Resuming AdaBalance changes the control trajectory relative to uninterrupted training.
- The optimizer/model may resume, but the adaptive policy over `rho` does not.

### 5. The repository is not yet demonstrated as results-ready

There are no checked-in `results/` or `checkpoints/` artifacts showing a successful run, and the provided `run.log` ends in a Phase 1 failure before training starts (`run.log:25-49`). Separately, `run.sh:25-29` only checks for `torch`, `transformers`, and `datasets`, even though the training pipeline also requires `trl`, `accelerate`, and `peft`; combined with the broad dependency ranges in `requirements.txt:6-18`, this increases environment fragility.

Why this matters:
- I do not have evidence that the full pipeline completes successfully from a clean setup.
- For a NeurIPS-style reproducibility bar, the current release is still in “promising but not yet productionized” territory.

## Code Quality And Completeness

Strengths:
- The repository structure is clear and the main ideas are separated into readable modules (`src/rho_grpo_trainer.py`, `src/adabalance.py`, `src/zero_score_handler.py`).
- The training scripts consistently save lightweight JSON summaries.
- The pipeline script is ambitious and covers training, evaluation, and analysis stages.

Weaknesses:
- I did not find a real test suite; there are no obvious unit/integration tests covering reward extraction, evaluation correctness, resume semantics, or quick smoke runs.
- Several experimental claims depend on orchestration details that are currently hardcoded or only partially implemented.
- The environment story is fragile relative to the complexity of the stack.

## Readiness For Experimental Results

My answer is **no, not yet**. The codebase looks close enough that a motivated author could get it over the line, but I would not trust the current repository to produce final experimental numbers without manual debugging and validation.

## Actionable Feedback

1. Fix `scripts/eval_phase_point.py` so GSM8K and MATH use dataset-specific gold-answer extraction, then add a small parsing test for each benchmark.
2. Make Phase 7 consume the actual best static point from Phase 2 output instead of hardcoding `(alpha, beta) = (0.5, 1.0)`.
3. Decide on one training-scale story and implement it consistently:
   - either support `accelerate`/`torchrun` across all training scripts,
   - or explicitly document that most sweeps are single-GPU jobs scheduled in parallel.
4. Serialize and restore custom trainer/controller state for resume, especially AdaBalance’s EMA/history state and any callback state that affects behavior.
5. Strengthen environment reproducibility with pinned versions or a lockfile and a smoke-test script that runs 1-2 train/eval steps end-to-end.
6. Add one successful quick-mode artifact or CI job log showing that `QUICK=1 bash run.sh` completes on a supported environment.

## Score Rationale

I am giving **4/10** because the repository is nontrivial and reasonably organized, but there are still blockers on experimental correctness and reproducibility. The current release is better than a sketch, but below the bar for a results-ready NeurIPS code submission.
