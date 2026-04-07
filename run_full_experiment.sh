#!/usr/bin/env bash
# ============================================================
# FULL EXPERIMENT SUITE — NeurIPS Best Paper Standard
# ≥5 seeds per condition, 10 seeds at critical points
# All runs on 2×H100 DDP
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
source ~/grpo-venv-qwen35/bin/activate

ACCEL="accelerate launch --config_file configs/accelerate_2gpu.yaml"
CONFIG="configs/rho_sweep.yaml"
R="results/qwen35"

run_train() {
    local TAG="$1" OUT="$2"; shift 2
    [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && return 0
    echo ">>> [$(date '+%m-%d %H:%M')] $TAG (2xH100 DDP)"
    CUDA_VISIBLE_DEVICES=0,1 $ACCEL "$@" --output_dir "$OUT" 2>&1 | tail -2
    echo "<<< [$(date '+%m-%d %H:%M')] $TAG done"
}

echo "============================================================"
echo " FULL EXPERIMENT SUITE — $(date)"
echo " Model: Qwen3.5-9B | Hardware: 2×H100 DDP"
echo " ≥5 seeds/condition, 10 seeds at critical ρ"
echo "============================================================"

# ============================================================
# EXP 1: Core ρ sweep (65 runs)
# Critical ρ (0.7, 1.0, 3.0): 10 seeds each
# Other ρ: 5 seeds each
# ============================================================
echo -e "\n>>> EXP 1: Core ρ sweep (65 runs)"

# Critical points first (10 seeds)
for R_val in 1.0 3.0 0.7; do
    for S in 42 43 44 45 46 47 48 49 50 51; do
        TAG=$(printf 'rho%.2f_seed%d' "$R_val" "$S")
        run_train "$TAG" "$R/sweep_coarse/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
    done
done

# Non-critical points (5 seeds)
for R_val in 0.1 0.3 0.5 1.5 2.0 5.0; do
    for S in 42 43 44 45 46; do
        TAG=$(printf 'rho%.2f_seed%d' "$R_val" "$S")
        run_train "$TAG" "$R/sweep_coarse/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
    done
done

# ============================================================
# EXP 2: GRPO Variant Comparison (40 runs)
# 4 variants × 2 ρ × 5 seeds
# ============================================================
echo -e "\n>>> EXP 2: GRPO Variant Comparison (40 runs)"

for VARIANT in vanilla dapo gspo gtpo; do
    for R_val in 1.0 3.0; do
        for S in 42 43 44 45 46; do
            TAG="${VARIANT}_rho${R_val}_seed${S}"
            run_train "$TAG" "$R/variants/$TAG" \
                scripts/train_grpo_variants.py --variant "$VARIANT" --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
        done
    done
done

# ============================================================
# EXP 3: Long-Horizon (15 runs)
# 3 ρ × 5 seeds × 600 steps
# ============================================================
echo -e "\n>>> EXP 3: Long-Horizon (15 runs)"

for R_val in 0.7 1.0 3.0; do
    for S in 42 43 44 45 46; do
        TAG=$(printf 'rho%.2f_seed%d_long' "$R_val" "$S")
        run_train "$TAG" "$R/long_horizon/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 600
    done
done

# ============================================================
# EXP 4: Confounder Ablation (60 runs)
# G ∈ {2,4,8} × ρ ∈ {1.0,3.0} × 5 seeds = 30
# KL ∈ {0.01,0.05,0.2} × ρ ∈ {1.0,3.0} × 5 seeds = 30
# ============================================================
echo -e "\n>>> EXP 4: Confounder Ablation (60 runs)"

for G in 2 4 8; do
    python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); c['training']['num_generations']=$G; yaml.dump(c, open('/tmp/cfg_G${G}.yaml','w'))"
    for R_val in 1.0 3.0; do
        for S in 42 43 44 45 46; do
            TAG=$(printf 'G%d_rho%.2f_seed%d' "$G" "$R_val" "$S")
            run_train "$TAG" "$R/confounder/$TAG" \
                scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "/tmp/cfg_G${G}.yaml" --max_steps 200
        done
    done
done

for KL in 0.01 0.05 0.2; do
    python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); c['training']['kl_coef']=$KL; yaml.dump(c, open('/tmp/cfg_kl.yaml','w'))"
    for R_val in 1.0 3.0; do
        for S in 42 43 44 45 46; do
            TAG=$(printf 'kl%.2f_rho%.2f_seed%d' "$KL" "$R_val" "$S")
            run_train "$TAG" "$R/confounder/$TAG" \
                scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config /tmp/cfg_kl.yaml --max_steps 200
        done
    done
done

# ============================================================
# EXP 5: GSM8K Evaluation (all checkpoints)
# ============================================================
echo -e "\n>>> EXP 5: GSM8K Evaluation"
mkdir -p "$R/eval"

for dir in "$R"/sweep_coarse/*/ "$R"/variants/*/ "$R"/long_horizon/*/; do
    [[ ! -f "$dir/training_metrics.json" ]] && continue
    TAG=$(basename "$dir")
    EVAL="$R/eval/eval_${TAG}.json"
    [[ -f "$EVAL" ]] && continue
    RHO=$(python3 -c "import json; print(json.load(open('${dir}training_metrics.json'))['rho'])")
    SEED=$(python3 -c "import json; print(json.load(open('${dir}training_metrics.json'))['seed'])")
    echo ">>> eval $TAG"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_phase_point.py \
        --checkpoint_dir "$dir" --rho "$RHO" --seed "$SEED" \
        --output_dir "$R/eval" --num_samples 500 2>&1 | tail -1
done

echo -e "\n============================================================"
echo " ALL EXPERIMENTS COMPLETE — $(date)"
echo " Total: ~180 runs on 2×H100 DDP"
echo "============================================================"
