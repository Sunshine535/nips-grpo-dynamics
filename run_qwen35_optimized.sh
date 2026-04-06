#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/grpo-venv-qwen35/bin/activate

ACCEL="accelerate launch --config_file configs/accelerate_2gpu.yaml"
CONFIG="configs/rho_sweep.yaml"
R="results/qwen35"
mkdir -p "$R" results/logs/qwen35

run_train() {
    local TAG="$1" OUT="$2"; shift 2
    [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && return 0
    echo ">>> [$(date '+%H:%M')] $TAG (2xH100 DDP)"
    CUDA_VISIBLE_DEVICES=0,1 $ACCEL "$@" --output_dir "$OUT" 2>&1 | tail -2
    echo "<<< [$(date '+%H:%M')] $TAG done ($(ls $OUT/training_metrics.json 2>/dev/null && echo OK || echo FAIL))"
}

echo "============================================================"
echo " OPTIMIZED QWEN3.5-9B PIPELINE — BEST PAPER PRIORITY ORDER"
echo " $(date)"
echo "============================================================"

# ============================================================
# PRIORITY 1: Core ρ values (1.0 and 3.0) + boundary (0.7) — 9 runs
# ============================================================
echo -e "\n>>> P1: Core sweep (ρ=0.7/1.0/3.0 × 3 seeds)"
for R_val in 1.0 3.0 0.7; do
    for S in 42 43 44; do
        TAG=$(printf 'rho%.2f_seed%d' "$R_val" "$S")
        run_train "$TAG" "$R/sweep_coarse/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
    done
done

# ============================================================
# PRIORITY 2: Variant comparison at ρ=1.0 and ρ=3.0 — 12 runs
# (THIS IS THE KILLER EXPERIMENT for best paper)
# ============================================================
echo -e "\n>>> P2: GRPO Variant comparison (DAPO/GSPO/GTPO at ρ=1.0 and ρ=3.0)"
for VARIANT in dapo gspo gtpo; do
    for R_val in 1.0 3.0; do
        for S in 42 43; do
            TAG="${VARIANT}_rho${R_val}_seed${S}"
            run_train "$TAG" "$R/variants/$TAG" \
                scripts/train_grpo_variants.py --variant "$VARIANT" --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
        done
    done
done

# ============================================================
# PRIORITY 3: Long-horizon (ρ=1.0 vs 3.0, 600 steps) — 4 runs
# ============================================================
echo -e "\n>>> P3: Long-horizon (600 steps)"
for R_val in 1.0 3.0; do
    for S in 42 43; do
        TAG=$(printf 'rho%.2f_seed%d_long' "$R_val" "$S")
        run_train "$TAG" "$R/long_horizon/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 600
    done
done

# ============================================================
# PRIORITY 4: Remaining coarse sweep (fill in full landscape) — 18 runs
# ============================================================
echo -e "\n>>> P4: Remaining coarse sweep"
for R_val in 0.1 0.3 0.5 1.5 2.0 5.0; do
    for S in 42 43 44; do
        TAG=$(printf 'rho%.2f_seed%d' "$R_val" "$S")
        run_train "$TAG" "$R/sweep_coarse/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
    done
done

# ============================================================
# PRIORITY 5: Confounder ablation — 8 runs
# ============================================================
echo -e "\n>>> P5: Confounder ablation"
for G in 2 8; do
    for R_val in 0.7 3.0; do
        TAG=$(printf 'G%d_rho%.2f' "$G" "$R_val")
        python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); c['training']['num_generations']=$G; yaml.dump(c, open('/tmp/cfg_G.yaml','w'))"
        run_train "$TAG" "$R/confounder/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed 42 --config /tmp/cfg_G.yaml --max_steps 200
    done
done
for KL in 0.01 0.2; do
    for R_val in 0.7 3.0; do
        TAG=$(printf 'kl%.2f_rho%.2f' "$KL" "$R_val")
        python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); c['training']['kl_coef']=$KL; yaml.dump(c, open('/tmp/cfg_kl.yaml','w'))"
        run_train "$TAG" "$R/confounder/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed 42 --config /tmp/cfg_kl.yaml --max_steps 200
    done
done

# ============================================================
# PRIORITY 6: GSM8K evaluation (all checkpoints)
# ============================================================
echo -e "\n>>> P6: GSM8K evaluation"
for dir in "$R"/sweep_coarse/*/  "$R"/variants/*/ "$R"/long_horizon/*/; do
    [[ ! -f "$dir/training_metrics.json" ]] && continue
    TAG=$(basename "$dir")
    EVAL="$R/eval/eval_${TAG}.json"
    [[ -f "$EVAL" ]] && continue
    RHO=$(python3 -c "import json; print(json.load(open('${dir}training_metrics.json'))['rho'])")
    SEED=$(python3 -c "import json; print(json.load(open('${dir}training_metrics.json'))['seed'])")
    echo ">>> eval $TAG (500 samples)"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_phase_point.py \
        --checkpoint_dir "$dir" --rho "$RHO" --seed "$SEED" \
        --output_dir "$R/eval" --num_samples 500 2>&1 | tail -1
done

echo -e "\n============================================================"
echo " ALL DONE — $(date)"
echo "============================================================"
