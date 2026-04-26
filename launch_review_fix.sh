#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=${HF_HOME:-}/hub HF_DATASETS_CACHE=${HF_HOME:-}/datasets TRANSFORMERS_CACHE=${HF_HOME:-}/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen3.5-9B"
OUTDIR=results/review_fix
mkdir -p "$OUTDIR/logs"

# ============================================================
# Wave 1: B (TASA-only) and D (TASA+CE) with 5 matched seeds
# 8 GPUs: B seeds 42-46 on GPU 0-4, D seeds 42-44 on GPU 5-7
# ============================================================
echo "=== Wave 1: B×5 + D×3 (8 runs, 8 GPUs) ==="
GPU=0
for seed in 42 43 44 45 46; do
    tag="B_tasa_only_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/B_tasa_only" --max-steps 200 --model $MODEL \
        --sage-mode tasa_only --lambda-pair 0 --lambda-pos 0 \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[B] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
for seed in 45 46 47; do
    tag="D_tasa_ce_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/D_tasa_ce" --max-steps 200 --model $MODEL \
        --sage-mode positive_ce_only --lambda-pair 0 --lambda-pos 0.05 \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[D] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
echo "[wave1] 8 runs launched, waiting..."
wait
echo "[wave1] done"

# ============================================================
# Wave 2: Additional baselines (A=r-c, A=2r-1) + Dr.GRPO tuned
# Plus D seeds 42-44 already exist from sage_minimal_abc
# 8 GPUs: Dr.GRPO best-effort tuned (G=4 lr=1e-5, 400 steps) ×2
#          + simple baselines ×4 + D_seed48,49 ×2
# ============================================================
echo "=== Wave 2: Dr.GRPO tuned + D extra seeds (8 GPUs) ==="
GPU=0
# Dr.GRPO with more steps (400) and lower LR to give it best shot
for seed in 42 43; do
    tag="drgrpo_tuned_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone dr_grpo --config configs/aser_g4_safe.yaml \
        --output-dir "$OUTDIR/drgrpo_tuned" --max-steps 400 --model $MODEL \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[DrGRPO-tuned] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
# D extra seeds
for seed in 48 49; do
    tag="D_tasa_ce_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/D_tasa_ce" --max-steps 200 --model $MODEL \
        --sage-mode positive_ce_only --lambda-pair 0 --lambda-pos 0.05 \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[D] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
# B extra seeds (already have 42-46 from wave1, add 47-48)
for seed in 47 48; do
    tag="B_tasa_only_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/B_tasa_only" --max-steps 200 --model $MODEL \
        --sage-mode tasa_only --lambda-pair 0 --lambda-pos 0 \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[B] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
# C contrastive extra seeds
for seed in 45 46; do
    tag="C_contrastive_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_sage_grpo.py \
        --seed $seed --config configs/sage_grpo_minimal.yaml \
        --output-dir "$OUTDIR/C_contrastive" --max-steps 200 --model $MODEL \
        --sage-mode full --pair-batch-size 1 \
        --run-name "$tag" > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[C] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done
echo "[wave2] 8 runs launched, waiting..."
wait
echo "[wave2] done"

# ============================================================
# Wave 3: Full-set eval on ALL new checkpoints (8 GPUs)
# ============================================================
echo "=== Wave 3: Eval ==="
N=1319
GPU=0
for variant_dir in B_tasa_only D_tasa_ce drgrpo_tuned C_contrastive; do
    mkdir -p "$OUTDIR/$variant_dir/evals"
    for ckpt_dir in "$OUTDIR/$variant_dir"/*/checkpoint-final; do
        [ -d "$ckpt_dir" ] || continue
        run_name=$(basename $(dirname "$ckpt_dir"))
        out="$OUTDIR/$variant_dir/evals/eval_${run_name}.json"
        [ -f "$out" ] && continue
        CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/eval_stratified.py \
            --adapter "$ckpt_dir" --n $N --selection full --out "$out" \
            > "$OUTDIR/$variant_dir/evals/${run_name}.log" 2>&1 &
        echo "[eval] $run_name GPU $GPU"
        GPU=$(( (GPU+1) % 8 ))
    done
done
wait
echo "[wave3] Eval done"

echo "=== RESULTS ==="
for f in "$OUTDIR"/*/evals/eval_*.json; do
    [ -f "$f" ] && python3 -c "
import json, os
d = json.load(open('$f'))
if 'accuracy' in d:
    print(f'{os.path.basename(\"$f\"):55s} acc={d[\"accuracy\"]:.4f}')
" 2>/dev/null
done
