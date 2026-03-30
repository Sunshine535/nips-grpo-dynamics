# GRPO Dynamics: Phase Diagrams, Rho-Weighted Advantage, and Adaptive Balance for Stable RL Post-Training

## Quick Start

```bash
git clone https://github.com/Sunshine535/nips-grpo-dynamics.git
cd nips-grpo-dynamics
bash setup.sh
source .venv/bin/activate
bash run.sh                    # full pipeline (11 phases)
QUICK=1 bash run.sh            # dev mode (reduced grids)
```

### Resume / Monitor

```bash
bash run.sh                    # auto-skips completed phases
FORCE_RERUN=1 bash run.sh     # force re-run all
cat results/.pipeline_done     # check completion
ls results/.phase_markers/     # per-phase status
```

## Method Overview

### Track 1: Phase Diagram (alpha/beta sweep)

Standard GRPO treats positive and negative advantages symmetrically. We parameterize the balance as (alpha, beta) and map accuracy/stability across the plane.

- `train_grpo_sweep.py` trains at grid points using `RhoGRPOTrainer` with rho = alpha / (1-alpha) * beta
- Standard binary 0/1 rewards; asymmetric weighting applied to advantages post-normalization
- `eval_phase_point.py` evaluates each checkpoint on GSM8K + MATH

### Track 2: Rho-GRPO (Theorem 1)

The rho parameter directly controls the relative weight of correct vs incorrect sample gradients:

- `RhoGRPOTrainer` (src/rho_grpo_trainer.py) overrides `compute_loss` to scale positive advantages by rho
- Stability analysis (src/stability_analysis.py) derives theoretical bounds: rho_min, rho_max, rho*
- `StabilityTelemetryCallback` logs regime classification (convergent / gradient_starved / unstable) at each step

### Track 3: AdaBalance (Adaptive rho)

Online controller that adjusts rho every K steps based on:
- Group starvation rate (GSR)
- Advantage variance decomposition (V+, V-, C_pG)
- Stability bounds from Theorem 1

Uses `RhoGRPOTrainer` with `AdaBalanceCallback` that feeds real training statistics to `AdaBalanceController` and updates `trainer.rho` dynamically.

### Track 4: HalluZero (Zero-Score Gradient Reshaping)

Four strategies to extract useful signal from zero-reward samples:
- **clip**: Scale down zero-score gradients by clip_factor
- **temperature**: Boost exploration via temperature scaling
- **curriculum**: Gradually include zero-score samples over training
- **relabel**: Soft relabel with epsilon reward

`HalluZeroGRPOTrainer` overrides `compute_loss` to call `ZeroScoreHandler.reweight_advantages()` on the pre-computed advantage tensor, applying the selected strategy.

## Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | (download) | HuggingFace model prefetch (9B + 27B) |
| 1 | train_grpo_sweep.py | Baseline GRPO (alpha=0.5, beta=1.0) |
| 2 | train_grpo_sweep.py | Phase diagram sweep (alpha x beta x seeds) |
| 3 | train_grpo_halluzero.py | HalluZero strategy sweep (4 strategies x params x seeds) |
| 4 | train_rho_sweep.py | Rho-GRPO sweep (rho x seeds) via RhoGRPOTrainer |
| 5 | train_adabalance.py | AdaBalance training via RhoGRPOTrainer + controller |
| 6 | analyze_gradients.py | Zero vs non-zero gradient analysis |
| 7 | run_curriculum_strategies.py | Alpha/beta schedule comparison |
| 8 | build_phase_diagram.py | Phase diagram + collapse-zone analysis |
| 9 | run_diagnostic_analysis.py | Diagnostic figures/tables |
| 10 | eval_halluzero.py | 27B validation (skipped in --quick) |

## Project Structure

```
nips-grpo-dynamics/
├── src/
│   ├── rho_grpo_trainer.py    # RhoGRPOTrainer: GRPOTrainer + rho advantage weighting
│   ├── rho_grpo.py            # Rho-GRPO math: advantage computation, group statistics
│   ├── adabalance.py          # AdaBalanceController + callback (adjusts trainer.rho)
│   ├── balanced_grpo.py       # Alpha/beta balanced GRPO loss
│   ├── zero_score_handler.py  # 4 zero-score strategies + gradient diagnostics
│   ├── stability_analysis.py  # Theoretical stability bounds, regime classification
│   └── qwen35_compat.py       # Qwen3.5 text-only rope_deltas patch
├── scripts/
│   ├── run_all_experiments.sh  # Master pipeline (11 phases)
│   ├── train_grpo_sweep.py     # Phase diagram training (uses RhoGRPOTrainer)
│   ├── train_rho_sweep.py      # Rho sweep training (uses RhoGRPOTrainer)
│   ├── train_adabalance.py     # AdaBalance training (uses RhoGRPOTrainer)
│   ├── train_grpo_halluzero.py # HalluZero training (uses HalluZeroGRPOTrainer)
│   ├── eval_phase_point.py     # Evaluate checkpoint (GSM8K + MATH)
│   ├── eval_halluzero.py       # HalluZero evaluation
│   ├── build_phase_diagram.py  # Aggregate results into phase diagram
│   ├── analyze_gradients.py    # Gradient analysis
│   ├── run_curriculum_strategies.py
│   ├── run_diagnostic_analysis.py
│   └── gpu_utils.sh            # GPU allocation utilities
├── configs/
│   ├── sweep_grid.yaml         # Alpha/beta grid + training defaults
│   ├── rho_sweep.yaml          # Rho grid + AdaBalance config
│   └── grpo_9b.yaml            # HalluZero training config
├── requirements.txt
├── setup.sh
└── README.md
```

## Benchmarks

| Benchmark | Usage |
|-----------|-------|
| GSM8K (openai/gsm8k) | Training signal + primary evaluation |
| MATH (hendrycks/competition_math) | Secondary generalization evaluation |

## Models

- **Qwen/Qwen3.5-9B**: Main experiments (all phases)
- **Qwen/Qwen3.5-27B**: Scaling validation (Phase 10)

## Requirements

- Python 3.10+, NVIDIA GPU with CUDA
- PyTorch installed separately (setup.sh handles this, detects system PyTorch)
- Key packages: transformers (4.46-4.x), trl (0.15-0.16), peft, accelerate, datasets

```bash
bash setup.sh
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
