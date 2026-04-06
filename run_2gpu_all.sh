#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/grpo-venv/bin/activate

ACCEL="accelerate launch --config_file configs/accelerate_2gpu.yaml"
CONFIG="configs/rho_sweep.yaml"
export VLLM_TP_SIZE=2
export VLLM_PORT=51216

run_train() {
    local TAG="$1"; shift
    local SCRIPT="$1"; shift
    local OUT="$1"; shift
    if [[ -f "$OUT/training_metrics.json" ]]; then
        echo "[skip] $TAG"
        return 0
    fi
    echo ">>> [$(date '+%H:%M')] $TAG (2xH100)"
    $ACCEL "$SCRIPT" "$@" --output_dir "$OUT" --use_vllm --config "$CONFIG"
    echo "<<< [$(date '+%H:%M')] $TAG done"
}

run_eval() {
    local CKPT="$1" RHO="$2" SEED="$3" OUTDIR="$4"
    local TAG=$(basename "$CKPT")
    local EVAL_JSON="$OUTDIR/eval_${TAG}.json"
    if [[ -f "$EVAL_JSON" ]]; then echo "[skip eval] $TAG"; return 0; fi
    echo ">>> [$(date '+%H:%M')] eval $TAG"
    python3 scripts/eval_phase_point.py \
        --checkpoint_dir "$CKPT" --rho "$RHO" --seed "$SEED" \
        --output_dir "$OUTDIR" --num_samples 200
    echo "<<< [$(date '+%H:%M')] eval $TAG done"
}

mkdir -p results/sweep_extra results/rho_eval_extra results/logs

echo "============================================================"
echo " PHASE 1: Extra seeds at critical rho values (双卡)"
echo "============================================================"
for R in 0.7 1.0 3.0; do
    for S in 45 46 47; do
        TAG=$(printf 'rho%.2f_seed%d' "$R" "$S")
        run_train "$TAG" scripts/train_rho_sweep.py "results/sweep_extra/$TAG" \
            --rho "$R" --seed "$S" --max_steps 200
    done
done

echo ""
echo "============================================================"
echo " PHASE 2: Eval extra seeds"
echo "============================================================"
for f in results/sweep_extra/*/training_metrics.json; do
    d=$(dirname "$f")
    TAG=$(basename "$d")
    R=$(python3 -c "import json; print(json.load(open('$f'))['rho'])")
    S=$(python3 -c "import json; print(json.load(open('$f'))['seed'])")
    # Eval uses single GPU (no TP needed)
    CUDA_VISIBLE_DEVICES=0 run_eval "$d" "$R" "$S" results/rho_eval_extra
done

echo ""
echo "============================================================"
echo " PHASE 3: Confounder Ablation (G × λ_KL) on key rho values"
echo "============================================================"
# Quick confounder: vary G at rho=0.7 and rho=3.0
for G in 2 8; do
    for R in 0.7 3.0; do
        TAG=$(printf 'G%d_rho%.2f_seed42' "$G" "$R")
        OUT="results/confounder/$TAG"
        if [[ -f "$OUT/training_metrics.json" ]]; then
            echo "[skip] $TAG"; continue
        fi
        echo ">>> [$(date '+%H:%M')] Confounder: $TAG"
        # Create temp config with different G
        python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
cfg['training']['num_generations'] = $G
yaml.dump(cfg, open('/tmp/cfg_G${G}.yaml', 'w'))
"
        $ACCEL scripts/train_rho_sweep.py \
            --rho "$R" --seed 42 \
            --config /tmp/cfg_G${G}.yaml \
            --output_dir "$OUT" \
            --max_steps 200 --use_vllm
        echo "<<< [$(date '+%H:%M')] $TAG done"
    done
done

# Vary KL coef at rho=0.7 and rho=3.0
for KL in 0.01 0.2; do
    for R in 0.7 3.0; do
        TAG=$(printf 'kl%.2f_rho%.2f_seed42' "$KL" "$R")
        OUT="results/confounder/$TAG"
        if [[ -f "$OUT/training_metrics.json" ]]; then
            echo "[skip] $TAG"; continue
        fi
        echo ">>> [$(date '+%H:%M')] Confounder: $TAG"
        python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
cfg['training']['kl_coef'] = $KL
yaml.dump(cfg, open('/tmp/cfg_kl${KL}.yaml', 'w'))
"
        $ACCEL scripts/train_rho_sweep.py \
            --rho "$R" --seed 42 \
            --config /tmp/cfg_kl${KL}.yaml \
            --output_dir "$OUT" \
            --max_steps 200 --use_vllm
        echo "<<< [$(date '+%H:%M')] $TAG done"
    done
done

echo ""
echo "============================================================"
echo " PHASE 4: Fix AdaBalance (use rho_sweep with scheduled rho)"
echo "============================================================"
# Instead of broken AdaBalance controller, compare fixed rho schedules
for R in 0.7 2.0 3.0; do
    for S in 42 43; do
        TAG=$(printf 'static_rho%.1f_seed%d' "$R" "$S")
        OUT="results/method_comparison/$TAG"
        run_train "$TAG" scripts/train_rho_sweep.py "$OUT" \
            --rho "$R" --seed "$S" --max_steps 200
    done
done
# Vanilla baseline
for S in 42 43 44; do
    TAG="vanilla_rho1.0_seed${S}"
    OUT="results/method_comparison/$TAG"
    run_train "$TAG" scripts/train_rho_sweep.py "$OUT" \
        --rho 1.0 --seed "$S" --max_steps 200
done

# Eval method comparison
mkdir -p results/rho_eval_methods
for f in results/method_comparison/*/training_metrics.json; do
    d=$(dirname "$f")
    TAG=$(basename "$d")
    R=$(python3 -c "import json; print(json.load(open('$f'))['rho'])")
    S=$(python3 -c "import json; print(json.load(open('$f'))['seed'])")
    CUDA_VISIBLE_DEVICES=0 run_eval "$d" "$R" "$S" results/rho_eval_methods
done

echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " $(date)"
echo "============================================================"
