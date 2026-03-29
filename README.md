# GRPO Dynamics: Phase Diagrams and Zero-Score Gradient Reshaping for Stable RL Post-Training

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-grpo-dynamics.git
cd nips-grpo-dynamics

# 2. Install dependencies
bash setup.sh

# 3. Run all experiments
bash run.sh

# 4. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-grpo-dynamics_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

## Overview

**Training.** We use **TRL**’s `GRPOTrainer` on **GSM8K** (primary) with optional **MATH** evaluation, using **Qwen3.5-9B** for the main experiments and **Qwen3.5-27B** for a scaling validation pass.

**Phase diagram track.** `train_grpo_sweep.py` trains at grid points **α ∈ [0.1, 0.9]** and **β ∈ [0.0, 2.0]** (see `configs/sweep_grid.yaml`). `eval_phase_point.py` scores each checkpoint on **GSM8K** and optionally **MATH**. `build_phase_diagram.py` aggregates metrics, draws the **heatmap / boundaries**, and highlights regions consistent with **collapse** (e.g., low accuracy, unstable reward/KL dynamics).

**Zero-score track.** `train_grpo_halluzero.py` runs GRPO with **binary correctness rewards** and **HalluZero** modifications that reshape gradients when **zero-score mass** is high. `eval_halluzero.py` reports accuracy, diagnostics (including zero-score rate), and saves structured summaries. `analyze_gradients.py` contrasts **zero vs non-zero** sample groups at the gradient level.

**Curriculum and diagnostics.** `run_curriculum_strategies.py` compares **α/β schedules** (anneal positive, anneal negative, cosine, static baseline). `run_diagnostic_analysis.py` aggregates runs under `results/zero_score_sweep/` into **tables and figures** for the paper supplement.

The **master driver** is `scripts/run_all_experiments.sh`: model prefetch, baseline run, full sweeps, analysis, 27B validation.

---

## Project Structure

```
nips-grpo-dynamics/
├── configs/
│   ├── sweep_grid.yaml      # α, β grid, training defaults, model id
│   └── grpo_9b.yaml         # HalluZero / GRPO hyperparameters, logging
├── scripts/
│   ├── run_all_experiments.sh   # Master pipeline (all phases)
│   ├── gpu_utils.sh             # GPU detect + accelerate helpers (fallback if monorepo _shared missing)
│   ├── train_grpo_sweep.py      # RLBalance: single (α, β) GRPO run
│   ├── eval_phase_point.py      # Eval one phase-diagram checkpoint
│   ├── train_grpo_halluzero.py  # HalluZero: GRPO + zero-score reshaping
│   ├── eval_halluzero.py        # Benchmark + zero-score diagnostics
│   ├── build_phase_diagram.py   # Figures + collapse-zone analysis
│   ├── analyze_gradients.py     # Zero vs non-zero gradient analysis
│   ├── run_curriculum_strategies.py
│   ├── run_diagnostic_analysis.py
│   └── plot_phase_diagram.py    # Auxiliary plotting
├── src/
│   ├── balanced_grpo.py         # Positive/negative reweighting in GRPO loss
│   └── zero_score_handler.py    # Strategy definitions + handler
├── requirements.txt
├── setup.sh
└── README.md
```

When this repo lives under `github_repos/`, `run_all_experiments.sh` first tries `github_repos/_shared/gpu_utils.sh`, then falls back to `scripts/gpu_utils.sh`.

---

## Experiments

### Phase diagram (RLBalance)

**Goal.** Map **accuracy and stability** as a function of **(α, β)** and identify **boundaries** between healthy optimization and **collapse**.

**Scripts.** `train_grpo_sweep.py` → `eval_phase_point.py` → `build_phase_diagram.py`.

**Grid.** Defaults: **α ∈ {0.1,…,0.9}**, **β ∈ {0.0, 0.25, 0.5, 1.0, 2.0}**, **3 seeds** (see `configs/sweep_grid.yaml` and the full-mode loops in `run_all_experiments.sh`).

### Zero-score fix (HalluZero)

**Goal.** Mitigate **zero-score gradient collapse** with **four strategies**: gradient **clipping / scaling** (`clip`), **temperature** shaping for low-reward samples (`temperature`), **curriculum** warmup before full zero-score exposure (`curriculum`), and **soft relabel** (`relabel`).

**Scripts.** `train_grpo_halluzero.py` → `eval_halluzero.py`.

**Sweep.** Full mode: **4 strategies × 3 hyperparameter values each × 2 seeds** (hyperparameters are strategy-specific: e.g. `clip_factor`, `temperature_boost`, `curriculum_warmup_steps`, `relabel_epsilon`).

### Curriculum (α / β schedules)

**Goal.** Compare **static** (α, β) against **time-varying** schedules that may **avoid** bad regions of the phase diagram during early training.

**Script.** `run_curriculum_strategies.py` (uses `configs/sweep_grid.yaml` for data and backbone settings).

---

## Benchmarks

| Benchmark | Split / usage | Notes |
|-----------|----------------|-------|
| **GSM8K** | `openai/gsm8k` train for GRPO; test for eval | Primary training signal and main reported accuracy |
| **MATH** | `lighteval/MATH` test (subsampled in some scripts) | Secondary math generalization; optional in `eval_phase_point.py` via `--eval_math` |

The master script enables **MATH** alongside **GSM8K** for phase-point evaluation. `eval_halluzero.py` can evaluate **both** via `--benchmarks gsm8k math`.

---

## Requirements

- **Python 3.10+** recommended (see `setup.sh`).
- **NVIDIA GPU(s)** with a recent **CUDA** stack for training; inference-only phases can be run with smaller batches.
- **PyTorch** is **installed separately** (not pinned in `requirements.txt`); `setup.sh` installs CUDA wheels and then installs `requirements.txt`.
- **Python packages** are listed in `requirements.txt` (**transformers**, **datasets**, **accelerate**, **trl**, **peft**, **wandb**, **scipy**, **matplotlib**, **pandas**, etc.).

After `setup.sh`:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Citation (placeholder)

If you use this code, please cite the eventual **NeurIPS 2026** paper and the **RLBalance** / **HalluZero** precursor technical reports when available.

---

## License

See repository license file when published; default practice for the parent monorepo applies until stated otherwise.
