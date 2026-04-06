#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/grpo-venv/bin/activate

ACCEL="accelerate launch --config_file configs/accelerate_2gpu.yaml"
CONFIG="configs/rho_sweep.yaml"

echo "============================================================"
echo " LONG-HORIZON TRAINING (fatal for best paper)"
echo " ρ=1.0 vs ρ=3.0, 600 steps, 2 seeds each"
echo " $(date)"
echo "============================================================"

mkdir -p results/long_horizon results/logs/long_horizon

for R in 1.0 3.0; do
    for S in 42 43; do
        TAG=$(printf 'rho%.2f_seed%d_600steps' "$R" "$S")
        OUT="results/long_horizon/$TAG"
        [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && continue
        echo ">>> [$(date '+%H:%M')] $TAG (2xH100 DDP, 600 steps)"
        CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_rho_sweep.py \
            --rho "$R" --seed "$S" --config "$CONFIG" \
            --output_dir "$OUT" --max_steps 600 2>&1 | tail -3
        echo "<<< [$(date '+%H:%M')] $TAG done"
    done
done

echo ""
echo "============================================================"
echo " METHOD COMPARISON (vanilla vs optimal)"
echo "============================================================"

mkdir -p results/method_comparison

for R in 1.0 3.0; do
    TAG=$(printf 'method_rho%.1f_seed42' "$R")
    OUT="results/method_comparison/$TAG"
    [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && continue
    echo ">>> [$(date '+%H:%M')] $TAG"
    CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_rho_sweep.py \
        --rho "$R" --seed 42 --config "$CONFIG" \
        --output_dir "$OUT" --max_steps 200 2>&1 | tail -3
done

# GTPO-style: skip zero-score groups (implemented as rho=0, degenerate_floor=0)
TAG="method_gtpo_seed42"
OUT="results/method_comparison/$TAG"
if [[ ! -f "$OUT/training_metrics.json" ]]; then
    echo ">>> [$(date '+%H:%M')] $TAG (GTPO-style: very high rho)"
    CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_rho_sweep.py \
        --rho 10.0 --seed 42 --config "$CONFIG" \
        --output_dir "$OUT" --max_steps 200 2>&1 | tail -3
fi

echo ""
echo "============================================================"
echo " FULL GSM8K EVALUATION (1319 samples)"  
echo "============================================================"

mkdir -p results/full_eval

# Eval key checkpoints on full GSM8K test set
for CKPT_DIR in \
    results/sweep_coarse/rho0.70_seed42 \
    results/sweep_coarse/rho1.00_seed42 \
    results/sweep_coarse/rho3.00_seed42 \
    results/long_horizon/rho1.00_seed42_600steps \
    results/long_horizon/rho3.00_seed42_600steps; do
    [[ ! -d "$CKPT_DIR" ]] && continue
    TAG=$(basename "$CKPT_DIR")
    EVAL="results/full_eval/eval_${TAG}.json"
    [[ -f "$EVAL" ]] && continue
    RHO=$(python3 -c "import json; print(json.load(open('$CKPT_DIR/training_metrics.json'))['rho'])")
    SEED=$(python3 -c "import json; print(json.load(open('$CKPT_DIR/training_metrics.json'))['seed'])")
    echo ">>> eval $TAG (full 1319 samples)"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_phase_point.py \
        --checkpoint_dir "$CKPT_DIR" --rho "$RHO" --seed "$SEED" \
        --output_dir results/full_eval --num_samples 1319 2>&1 | tail -2
done

echo ""
echo "============================================================"
echo " ALL PRIORITY EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"
