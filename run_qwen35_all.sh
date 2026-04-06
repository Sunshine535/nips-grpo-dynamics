#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/grpo-venv-qwen35/bin/activate

ACCEL="accelerate launch --config_file configs/accelerate_2gpu.yaml"
CONFIG="configs/rho_sweep.yaml"
RESULTS="results/qwen35"
mkdir -p "$RESULTS" results/logs/qwen35

echo "============================================================"
echo " QWEN3.5-9B FULL EXPERIMENT SUITE (2xH100 DDP)"
echo " $(date)"
echo "============================================================"

# ================================================================
# PHASE 1: Coarse rho sweep (9 rho x 3 seeds x 200 steps)
# ================================================================
echo ""
echo ">>> PHASE 1: Coarse rho sweep"
for R in 0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0; do
    for S in 42 43 44; do
        TAG=$(printf 'rho%.2f_seed%d' "$R" "$S")
        OUT="$RESULTS/sweep_coarse/$TAG"
        [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && continue
        echo ">>> [$(date '+%H:%M')] $TAG (2xH100)"
        CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_rho_sweep.py \
            --rho "$R" --seed "$S" --config "$CONFIG" \
            --output_dir "$OUT" --max_steps 200 2>&1 | tail -2
        echo "<<< [$(date '+%H:%M')] $TAG done"
    done
done

# ================================================================
# PHASE 2: Long-horizon (rho=1.0 and best rho, 600 steps)
# ================================================================
echo ""
echo ">>> PHASE 2: Long-horizon training"
for R in 0.7 1.0 3.0; do
    for S in 42 43; do
        TAG=$(printf 'rho%.2f_seed%d_long' "$R" "$S")
        OUT="$RESULTS/long_horizon/$TAG"
        [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && continue
        echo ">>> [$(date '+%H:%M')] LONG $TAG (600 steps)"
        CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_rho_sweep.py \
            --rho "$R" --seed "$S" --config "$CONFIG" \
            --output_dir "$OUT" --max_steps 600 2>&1 | tail -2
        echo "<<< [$(date '+%H:%M')] $TAG done"
    done
done

# ================================================================
# PHASE 3: Confounder ablation (G and KL at key rho values)
# ================================================================
echo ""
echo ">>> PHASE 3: Confounder ablation"
for G in 2 8; do
    for R in 0.7 3.0; do
        TAG=$(printf 'G%d_rho%.2f' "$G" "$R")
        OUT="$RESULTS/confounder/$TAG"
        [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && continue
        python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); c['training']['num_generations']=$G; yaml.dump(c, open('/tmp/cfg_G.yaml','w'))"
        echo ">>> [$(date '+%H:%M')] CONF $TAG"
        CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_rho_sweep.py \
            --rho "$R" --seed 42 --config /tmp/cfg_G.yaml \
            --output_dir "$OUT" --max_steps 200 2>&1 | tail -2
    done
done
for KL in 0.01 0.2; do
    for R in 0.7 3.0; do
        TAG=$(printf 'kl%.2f_rho%.2f' "$KL" "$R")
        OUT="$RESULTS/confounder/$TAG"
        [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && continue
        python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); c['training']['kl_coef']=$KL; yaml.dump(c, open('/tmp/cfg_kl.yaml','w'))"
        echo ">>> [$(date '+%H:%M')] CONF $TAG"
        CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_rho_sweep.py \
            --rho "$R" --seed 42 --config /tmp/cfg_kl.yaml \
            --output_dir "$OUT" --max_steps 200 2>&1 | tail -2
    done
done

# ================================================================
# PHASE 4: GSM8K Evaluation (full test set, key checkpoints)
# ================================================================
echo ""
echo ">>> PHASE 4: GSM8K evaluation"
mkdir -p "$RESULTS/eval"
for R in 0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0; do
    for S in 42 43 44; do
        TAG=$(printf 'rho%.2f_seed%d' "$R" "$S")
        CKPT="$RESULTS/sweep_coarse/$TAG"
        EVAL="$RESULTS/eval/eval_${TAG}.json"
        [[ ! -d "$CKPT" ]] && continue
        [[ -f "$EVAL" ]] && continue
        echo ">>> eval $TAG (500 samples)"
        CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_phase_point.py \
            --checkpoint_dir "$CKPT" --rho "$R" --seed "$S" \
            --output_dir "$RESULTS/eval" --num_samples 500 2>&1 | tail -1
    done
done

echo ""
echo "============================================================"
echo " ALL QWEN3.5-9B EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"

# ================================================================
# PHASE 5: GRPO Variant Comparison (DAPO/GSPO/GTPO vs vanilla)
# Shows our framework explains WHY each variant works
# ================================================================
echo ""
echo ">>> PHASE 5: GRPO Variant Comparison"

VARIANTS_DIR="$RESULTS/variants"
mkdir -p "$VARIANTS_DIR"

for VARIANT in vanilla dapo gspo gtpo; do
    for R in 0.7 1.0 3.0; do
        for S in 42 43; do
            TAG="${VARIANT}_rho${R}_seed${S}"
            OUT="$VARIANTS_DIR/$TAG"
            [[ -f "$OUT/training_metrics.json" ]] && echo "[skip] $TAG" && continue
            echo ">>> [$(date '+%H:%M')] VARIANT $TAG (2xH100)"
            CUDA_VISIBLE_DEVICES=0,1 $ACCEL scripts/train_grpo_variants.py \
                --variant "$VARIANT" --rho "$R" --seed "$S" \
                --config "$CONFIG" --output_dir "$OUT" --max_steps 200 \
                2>&1 | tail -2
            echo "<<< [$(date '+%H:%M')] $TAG done"
        done
    done
done

echo ""
echo ">>> Evaluating variant checkpoints"
mkdir -p "$RESULTS/eval_variants"
for f in "$VARIANTS_DIR"/*/training_metrics.json; do
    d=$(dirname "$f")
    TAG=$(basename "$d")
    EVAL="$RESULTS/eval_variants/eval_${TAG}.json"
    [[ -f "$EVAL" ]] && continue
    R=$(python3 -c "import json; print(json.load(open('$f'))['rho'])")
    S=$(python3 -c "import json; print(json.load(open('$f'))['seed'])")
    echo ">>> eval $TAG"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_phase_point.py \
        --checkpoint_dir "$d" --rho "$R" --seed "$S" \
        --output_dir "$RESULTS/eval_variants" --num_samples 500 2>&1 | tail -1
done

echo ""
echo "============================================================"
echo " ALL EXPERIMENTS (incl. variants) COMPLETE — $(date)"
echo "============================================================"
