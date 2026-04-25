#!/bin/bash
# SAGE-GRPO minimal A/B/D/C verification (GPT-5.5 R3).
# 4 variants × 3 seeds = 12 runs; 8 GPUs → 2 batches or reduce seeds.
# Default: 3 seeds × 4 variants = 12, run in 2 waves of 6.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=${HF_HOME:-}/hub HF_DATASETS_CACHE=${HF_HOME:-}/datasets TRANSFORMERS_CACHE=${HF_HOME:-}/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen3.5-9B"
MAX_STEPS=${MAX_STEPS:-200}
SEEDS=${SEEDS:-"42 43 44"}
OUTDIR=${OUTDIR:-results/sage_minimal_abc}
mkdir -p "$OUTDIR/logs"

echo "=== Wave 1: B_tasa_only + D_positive_ce_only (6 runs on 6 GPUs) ==="
GPU=0
for seed in $SEEDS; do
    tag="B_tasa_only_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/B_tasa_only" --max-steps $MAX_STEPS --model $MODEL \
        --sage-mode tasa_only --lambda-pair 0 --lambda-pos 0 \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[B] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
for seed in $SEEDS; do
    tag="D_positive_ce_only_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/D_positive_ce_only" --max-steps $MAX_STEPS --model $MODEL \
        --sage-mode positive_ce_only --lambda-pair 0 --lambda-pos 0.05 \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[D] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
echo "[wave1] 6 runs launched, waiting..."
wait
echo "[wave1] Training complete"

echo "=== Wave 2: A_legacy + C_sage_full (6 runs on 6 GPUs) ==="
GPU=0
for seed in $SEEDS; do
    tag="A_legacy_spo_replay_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone spo --config configs/aser_g4_safe.yaml \
        --output-dir "$OUTDIR/A_legacy" --max-steps $MAX_STEPS --model $MODEL \
        --lambda-rep 0.05 --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[A] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
for seed in $SEEDS; do
    tag="C_sage_full_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/C_sage_full" --max-steps $MAX_STEPS --model $MODEL \
        --sage-mode full \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[C] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
echo "[wave2] 6 runs launched, waiting..."
wait
echo "[wave2] Training complete"

echo "=== Full-set eval (all variants) ==="
N=1319
GPU=0
for variant_dir in A_legacy B_tasa_only D_positive_ce_only C_sage_full; do
    mkdir -p "$OUTDIR/$variant_dir/evals"
    for seed in $SEEDS; do
        case "$variant_dir" in
            A_legacy) tag="A_legacy_spo_replay_seed${seed}" ;;
            B_tasa_only) tag="B_tasa_only_seed${seed}" ;;
            D_positive_ce_only) tag="D_positive_ce_only_seed${seed}" ;;
            C_sage_full) tag="C_sage_full_seed${seed}" ;;
        esac
        adapter="$OUTDIR/$variant_dir/$tag/checkpoint-final"
        out="$OUTDIR/$variant_dir/evals/eval_${tag}.json"
        [ -f "$out" ] && continue
        [ -d "$adapter" ] || { echo "[skip] $tag no checkpoint"; continue; }
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/eval_stratified.py \
            --adapter "$adapter" --n $N --selection full --out "$out" \
            > "$OUTDIR/$variant_dir/evals/${tag}.log" 2>&1 &
        echo "[eval] $tag GPU $GPU"
        GPU=$(( (GPU+1) % 8 ))
    done
done
wait
echo "[done] All evals complete"

echo "=== RESULTS ==="
for f in "$OUTDIR"/*/evals/eval_*.json; do
    [ -f "$f" ] && python3 -c "
import json, os
d = json.load(open('$f'))
print(f'{os.path.basename(\"$f\"):55s} acc={d[\"accuracy\"]:.4f}')
"
done
