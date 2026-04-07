# Experiment Status Report

## Completed Experiments

### Model: Qwen2.5-7B-Instruct (preliminary, full results)
| Experiment | Runs | Seeds/condition | Status |
|-----------|------|-----------------|--------|
| Coarse ρ sweep (9 ρ values) | 27 | 3 | ✅ Complete + Evaluated |
| Extra seeds (ρ=0.7/1.0/3.0) | 9 | +3 (total 6) | ✅ Complete |
| Confounder (G×λ_KL) | 8 | 1 | ✅ Complete |
| Long-horizon (600 steps) | 1 | 1 | ✅ ρ=1.0 seed=42 |
| AdaBalance comparison | 6 | 2 | ✅ (controller broken) |
| Synthetic validation | N/A | N/A | ✅ R²=0.994, 99.8% regime acc |
| Robustness test | N/A | N/A | ✅ 66.7% pred accuracy |
| Data-collapse analysis | N/A | N/A | ✅ Complete |
| Early-warning predictor | N/A | N/A | ✅ Complete |
| Theory unification | N/A | N/A | ✅ DAPO/GSPO/GTPO mapped |

### Model: Qwen3.5-9B (target model, in progress)
| Experiment | Runs | Seeds/condition | Status |
|-----------|------|-----------------|--------|
| Core sweep (ρ=1.0/3.0/0.7) | 10 | 3 | ✅ Complete |
| DAPO variant (ρ=1.0) | 2 | 2 | ✅ Complete |
| DAPO variant (ρ=3.0) | 1 | 1 | 🔄 Running |
| Other variants | 0 | - | ⏳ Pending |
| Long-horizon | 0 | - | ⏳ Pending |
| Full sweep | 0 | - | ⏳ Pending |

## Key Results

### Qwen2.5-7B-Instruct Stability Landscape
| ρ | n_seeds | GSM8K Accuracy | Failure Rate |
|---|---------|---------------|-------------|
| 0.1 | 3 | 48.3±44.1% | 33% |
| 0.3 | 3 | 29.0±50.2% | 67% |
| 0.5 | 3 | 56.5±47.6% | 33% |
| 0.7 | 6 | 74.8±36.9% | 17% |
| **1.0** | **6** | **45.5±49.7%** | **50%** |
| 1.5 | 3 | 74.8±13.2% | 0% |
| 2.0 | 3 | 84.7±1.8% | 0% |
| **3.0** | **6** | **87.2±3.8%** | **0%** |
| 5.0 | 3 | 74.0±17.6% | 0% |

### Qwen3.5-9B Preliminary (n=3, needs more repetitions)
| ρ | n_seeds | Training Reward | Notes |
|---|---------|----------------|-------|
| 0.7 | 3 | 0.90/0.90/0.01 | 1/3 collapsed |
| 1.0 | 3 | 0.90/0.95/1.00 | All converged (different from Qwen2.5!) |
| 3.0 | 3 | 0.95/0.95/0.95 | All converged, zero variance |

## Required: Full Experiment Plan
See `configs/full_experiment_plan.yaml` and `run_full_experiment.sh`
- 180 total runs on 2×H100 DDP
- ≥5 seeds per condition, 10 at critical points
- ~322 GPU-hours (~6.7 days)

## Statistical Note
Current Qwen3.5-9B results (n=3) are INSUFFICIENT for reliable conclusions.
The full plan addresses this with ≥5 seeds minimum, 10 seeds at ρ=0.7/1.0/3.0.
