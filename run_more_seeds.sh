#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/grpo-venv-qwen35/bin/activate

ACCEL="accelerate launch --config_file configs/accelerate_2gpu.yaml"
CONFIG="configs/rho_sweep.yaml"
R="results/qwen35"

run_train() {
    local TAG="$1" OUT="$2"; shift 2
    [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && return 0
    echo ">>> [$(date '+%H:%M')] $TAG (2xH100 DDP)"
    CUDA_VISIBLE_DEVICES=0,1 $ACCEL "$@" --output_dir "$OUT" 2>&1 | tail -2
    echo "<<< [$(date '+%H:%M')] $TAG done"
}

echo "============================================================"
echo " ADD MORE SEEDS: ρ=0.7/1.0/3.0 × seeds 45-50 (18 extra runs)"
echo " Target: 9+ seeds per critical ρ value"
echo " $(date)"
echo "============================================================"

for R_val in 1.0 3.0 0.7; do
    for S in 45 46 47 48 49 50; do
        TAG=$(printf 'rho%.2f_seed%d' "$R_val" "$S")
        run_train "$TAG" "$R/sweep_coarse/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
    done
done

echo ""
echo "============================================================"
echo " CONTINUE: Remaining variants + long-horizon + sweep"
echo "============================================================"

# P2: Continue variants (DAPO at rho3.0 already done by old pipeline, continue with GSPO/GTPO)
for VARIANT in dapo gspo gtpo; do
    for R_val in 1.0 3.0; do
        for S in 42 43; do
            TAG="${VARIANT}_rho${R_val}_seed${S}"
            run_train "$TAG" "$R/variants/$TAG" \
                scripts/train_grpo_variants.py --variant "$VARIANT" --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
        done
    done
done

# P3: Long-horizon
for R_val in 1.0 3.0; do
    for S in 42 43; do
        TAG=$(printf 'rho%.2f_seed%d_long' "$R_val" "$S")
        run_train "$TAG" "$R/long_horizon/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 600
    done
done

# P4: Remaining sweep values
for R_val in 0.3 0.5 1.5 2.0 5.0; do
    for S in 42 43 44; do
        TAG=$(printf 'rho%.2f_seed%d' "$R_val" "$S")
        run_train "$TAG" "$R/sweep_coarse/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed "$S" --config "$CONFIG" --max_steps 200
    done
done

# P5: Confounder
for G in 2 8; do
    for R_val in 1.0 3.0; do
        TAG=$(printf 'G%d_rho%.2f' "$G" "$R_val")
        python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); c['training']['num_generations']=$G; yaml.dump(c, open('/tmp/cfg_G.yaml','w'))"
        run_train "$TAG" "$R/confounder/$TAG" \
            scripts/train_rho_sweep.py --rho "$R_val" --seed 42 --config /tmp/cfg_G.yaml --max_steps 200
    done
done

# P6: Eval all
echo -e "\n>>> EVALUATION"
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

echo -e "\n=== ALL DONE $(date) ==="
