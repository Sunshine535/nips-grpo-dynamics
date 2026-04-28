#!/bin/bash
set -e

# Deploy unified multi-seed experiments on fresh server
# Addresses reviewer concerns: TASA discrepancy, n=3 insufficient, missing sign baseline

PROJ_DIR="/root/nips-grpo-dynamics"
HF_CACHE="/openbayes/input/input0"
MODEL="Qwen/Qwen3.5-9B"
SEEDS=(42 43 44 45 46 47 48)
MAX_STEPS=200

echo "============================================"
echo " Unified Experiment Deployment"
echo " $(date)"
echo "============================================"

# --- 1. Clone repo ---
if [ ! -d "$PROJ_DIR" ]; then
    echo "[1/4] Cloning repo..."
    cd /root
    git clone https://github.com/sunshine535/nips-grpo-dynamics.git
else
    echo "[1/4] Repo exists, pulling latest..."
    cd "$PROJ_DIR" && git pull
fi
cd "$PROJ_DIR"

# --- 2. Install dependencies ---
echo "[2/4] Installing dependencies..."
pip install -q --upgrade trl>=0.15.0 peft>=0.13.0 accelerate>=0.34.0 datasets
pip install -q wandb scipy matplotlib

# --- 3. Download model ---
echo "[3/4] Checking model..."
python3 -c "
import os
os.environ['HF_HOME'] = '$HF_CACHE'
os.environ['HF_HUB_CACHE'] = '$HF_CACHE/hub'
from huggingface_hub import snapshot_download
try:
    snapshot_download('$MODEL', cache_dir='$HF_CACHE/hub', local_files_only=True)
    print('Model already cached')
except:
    print('Downloading model...')
    snapshot_download('$MODEL', cache_dir='$HF_CACHE/hub')
    print('Model downloaded')
" 2>&1

# --- 4. Launch experiments ---
echo "[4/4] Launching experiments..."

# All experiments use SAGE trainer (run_sage_grpo.py) for unified comparison
# Variants: sign_baseline, tasa_only, tasa_ce, tasa_contrastive, drgrpo

RESULTS_DIR="$PROJ_DIR/results/unified_r1"
mkdir -p "$RESULTS_DIR"

GPU=0

# Wave 1: B (tasa_only) x7 seeds + D (tasa_ce) x7 seeds = 14 jobs on 8 GPUs
echo "=== Wave 1: B (tasa_only) + D (tasa_ce), 7 seeds each ==="
for seed in "${SEEDS[@]}"; do
    if [ $GPU -ge 8 ]; then
        echo "Waiting for GPU slots..."
        wait
        GPU=0
    fi

    # B: TASA-only
    CKPT_B="$RESULTS_DIR/B_tasa_only_seed${seed}"
    if [ ! -d "$CKPT_B" ]; then
        echo "  GPU $GPU: B_tasa_only seed=$seed"
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
            --sage-mode tasa_only \
            --seed $seed \
            --config configs/sage_grpo_minimal.yaml \
            --output-dir "$CKPT_B" \
            --max-steps $MAX_STEPS \
            > "$RESULTS_DIR/log_B_seed${seed}.txt" 2>&1 &
        GPU=$((GPU + 1))
    fi
done

for seed in "${SEEDS[@]}"; do
    if [ $GPU -ge 8 ]; then
        echo "Waiting for GPU slots..."
        wait
        GPU=0
    fi

    # D: TASA + CE replay
    CKPT_D="$RESULTS_DIR/D_tasa_ce_seed${seed}"
    if [ ! -d "$CKPT_D" ]; then
        echo "  GPU $GPU: D_tasa_ce seed=$seed"
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
            --sage-mode positive_ce_only \
            --seed $seed \
            --lambda-pos 0.05 \
            --replay-warmup-steps 50 \
            --config configs/sage_grpo_minimal.yaml \
            --output-dir "$CKPT_D" \
            --max-steps $MAX_STEPS \
            > "$RESULTS_DIR/log_D_seed${seed}.txt" 2>&1 &
        GPU=$((GPU + 1))
    fi
done

echo "Wave 1 launched. Waiting for completion..."
wait
echo "Wave 1 complete."

# Wave 2: C (contrastive) x7 seeds + sign_baseline x7 seeds
echo "=== Wave 2: C (contrastive) + Sign baseline, 7 seeds each ==="
GPU=0

for seed in "${SEEDS[@]}"; do
    if [ $GPU -ge 8 ]; then
        wait
        GPU=0
    fi

    # C: TASA + contrastive pair
    CKPT_C="$RESULTS_DIR/C_contrastive_seed${seed}"
    if [ ! -d "$CKPT_C" ]; then
        echo "  GPU $GPU: C_contrastive seed=$seed"
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
            --sage-mode pair_only \
            --seed $seed \
            --lambda-pair 0.05 \
            --replay-warmup-steps 50 \
            --config configs/sage_grpo_minimal.yaml \
            --output-dir "$CKPT_C" \
            --max-steps $MAX_STEPS \
            > "$RESULTS_DIR/log_C_seed${seed}.txt" 2>&1 &
        GPU=$((GPU + 1))
    fi
done

for seed in "${SEEDS[@]}"; do
    if [ $GPU -ge 8 ]; then
        wait
        GPU=0
    fi

    # Sign baseline: A=2r-1
    CKPT_S="$RESULTS_DIR/S_sign_baseline_seed${seed}"
    if [ ! -d "$CKPT_S" ]; then
        echo "  GPU $GPU: S_sign_baseline seed=$seed"
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
            --sage-mode tasa_only \
            --advantage-mode sign \
            --seed $seed \
            --config configs/sage_grpo_minimal.yaml \
            --output-dir "$CKPT_S" \
            --max-steps $MAX_STEPS \
            > "$RESULTS_DIR/log_S_seed${seed}.txt" 2>&1 &
        GPU=$((GPU + 1))
    fi
done

echo "Wave 2 launched. Waiting for completion..."
wait
echo "Wave 2 complete."

# Wave 3: Dr.GRPO x7 seeds (using SAGE trainer for unified comparison)
echo "=== Wave 3: Dr.GRPO (unified), 7 seeds ==="
GPU=0

for seed in "${SEEDS[@]}"; do
    if [ $GPU -ge 8 ]; then
        wait
        GPU=0
    fi

    CKPT_DR="$RESULTS_DIR/DR_drgrpo_seed${seed}"
    if [ ! -d "$CKPT_DR" ]; then
        echo "  GPU $GPU: DR_drgrpo seed=$seed"
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
            --sage-mode tasa_only \
            --advantage-mode drgrpo \
            --seed $seed \
            --config configs/sage_grpo_minimal.yaml \
            --output-dir "$CKPT_DR" \
            --max-steps $MAX_STEPS \
            > "$RESULTS_DIR/log_DR_seed${seed}.txt" 2>&1 &
        GPU=$((GPU + 1))
    fi
done

echo "Wave 3 launched. Waiting for completion..."
wait
echo "Wave 3 complete."

# Wave 4: Full-set evaluation on all checkpoints
echo "=== Wave 4: Full-set eval (n=1319) ==="
GPU=0

for ckpt_dir in $RESULTS_DIR/*/; do
    variant=$(basename "$ckpt_dir")
    out="$RESULTS_DIR/evals/eval_${variant}.json"
    if [ -f "$out" ]; then continue; fi

    adapter=""
    for d in "$ckpt_dir" "$ckpt_dir/checkpoint-$MAX_STEPS" "$ckpt_dir/final"; do
        if [ -f "$d/adapter_config.json" ]; then
            adapter="$d"
            break
        fi
    done
    if [ -z "$adapter" ]; then
        echo "  SKIP: no adapter in $variant"
        continue
    fi

    if [ $GPU -ge 8 ]; then
        wait
        GPU=0
    fi

    echo "  GPU $GPU: eval $variant"
    mkdir -p "$RESULTS_DIR/evals"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" \
        --n 1319 --selection full \
        --out "$out" \
        > "$RESULTS_DIR/log_eval_${variant}.txt" 2>&1 &
    GPU=$((GPU + 1))
done

echo "Wave 4 launched. Waiting for completion..."
wait
echo "Wave 4 complete."

echo "============================================"
echo " All experiments complete!"
echo " Results in: $RESULTS_DIR/evals/"
echo " $(date)"
echo "============================================"
