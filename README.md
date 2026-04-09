# Metastable Training Dynamics in GRPO

Seed-resolved basin analysis, step-0 trainability prediction, and transient rescue for Group Relative Policy Optimization (GRPO).

## Quick Start

```bash
# 1. Clone and setup
git clone <REPO_URL>
cd nips-grpo-dynamics
bash setup.sh              # venv + PyTorch (CUDA 12.8) + all deps
source .venv/bin/activate

# 2. Run experiments (auto-detects all GPUs)
bash run.sh

# Quick mode (reduced grid, for testing)
QUICK=1 bash run.sh
```

## Environment

- Python >= 3.10
- PyTorch >= 2.4 (CUDA 12.8)
- Dependencies: `transformers`, `datasets`, `accelerate`, `trl`, `peft`, `wandb`
- Optional: `flash-attn`

`setup.sh` handles everything: creates venv, installs PyTorch with CUDA 12.8, and verifies GPU detection.

## GPU Auto-Detection

All scripts auto-detect available GPUs via `scripts/gpu_utils.sh`:

```
============================================
 GPU Configuration
============================================
  GPUs detected     : 4
  CUDA_VISIBLE      : 0,1,2,3
  GPU memory (each) : 81559 MiB
  GPU class          : a100_80g
============================================
```

- **Phase 1** (baseline): Multi-GPU DDP via `accelerate launch`
- **Phase 2-5** (sweeps): Job-level parallelism — each (ρ, seed) runs on a single GPU, round-robin scheduled across all available GPUs
- **Batch size**: Auto-scaled by GPU memory (`auto_batch_size`)

To restrict GPUs: `CUDA_VISIBLE_DEVICES=0,1 bash run.sh`

## Resuming from Current Progress

Existing data (Qwen3.5-9B coarse sweep, 10 runs) is in `results/qwen35/`. The main pipeline (`scripts/run_all_experiments.sh`) uses phase markers — completed phases are automatically skipped.

### Resume from a specific phase

```bash
# Resume from Phase 4 (rho-GRPO sweep)
bash scripts/run_all_experiments.sh --from-phase 4

# Force rerun all phases
FORCE_RERUN=1 bash run.sh
```

### Run the new MetaGRPO experiments

```bash
# Basin analysis on existing data
python scripts/basin_analysis.py --results-dir results/ --model qwen35 --output results/basin_analysis/ --plot

# Dense rho-sweep with 20 seeds (new experiment — edit configs/rho_sweep.yaml first)
# Seeds and rho values are configured in scripts/run_all_experiments.sh
```

## Experiment Phases

| Phase | Description | Parallelism |
|-------|-------------|-------------|
| 0 | Model download (HF Hub) | Sequential |
| 1 | Baseline GRPO training | Multi-GPU DDP |
| 2 | (α,β) × seed phase diagram sweep | Job-level, round-robin |
| 3 | Zero-score strategy sweep (HalluZero) | Job-level, round-robin |
| 4 | ρ-GRPO sweep | Job-level, round-robin |
| 5 | AdaBalance adaptive ρ | Job-level, round-robin |
| 6 | Gradient analysis | Single GPU |
| 7 | Curriculum strategies | Single GPU |
| 8 | Phase diagram construction | CPU |
| 9 | Diagnostic analysis | CPU |
| 10 | 27B validation (non-quick only) | Single GPU |

## Project Structure

```
├── run.sh                          # Entry point
├── setup.sh                        # Environment setup
├── scripts/
│   ├── run_all_experiments.sh      # Phase orchestrator
│   ├── gpu_utils.sh                # GPU detection & allocation
│   ├── train_grpo_sweep.py         # (α,β) sweep training
│   ├── train_rho_sweep.py          # ρ-GRPO sweep training
│   ├── eval_phase_point.py         # Evaluation (GSM8K/MATH)
│   ├── basin_analysis.py           # Binder cumulant / basin geometry
│   ├── analyze_gradients.py        # Gradient diagnostics
│   └── ...
├── src/
│   ├── rho_grpo_trainer.py         # RhoGRPOTrainer, AdaBalanceGRPOTrainer
│   ├── stability_analysis.py       # Stability bounds, regime classification
│   ├── adabalance.py               # Adaptive ρ controller
│   └── zero_score_handler.py       # Zero-score strategies
├── configs/
│   ├── sweep_grid.yaml             # (α,β) sweep config
│   └── rho_sweep.yaml              # ρ sweep config
└── results/                        # Experiment outputs
```

## Key Models

- `Qwen/Qwen3.5-9B` — primary (existing data)
- `Qwen/Qwen3-8B` — cross-model validation (planned)
- `Qwen/Qwen2.5-7B-Instruct` — reference (historical data)

## Configuration

### Change model

Set `MODEL_9B` environment variable:

```bash
MODEL_9B=Qwen/Qwen3-8B bash run.sh
```

### Change sweep grid

Edit `configs/rho_sweep.yaml` or override in `scripts/run_all_experiments.sh`:

```bash
# In run_all_experiments.sh, Phase 4:
RHO_VALUES=(0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)
RHO_SEEDS=(42 43 44)
```

## Results

Outputs are organized under `results/`:

- `results/qwen35/sweep_coarse/` — per-run step logs and group-level stats
- `results/phase_diagram/` — evaluation JSONs
- `results/basin_analysis/` — Binder cumulant analysis
- `results/logs/` — per-phase log files
