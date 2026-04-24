#!/bin/bash
# TRACE-GRPO A/B/C minimal verification experiment (GPT-5.5 Task 10).
#
# A: Existing Best Positive Fragment Only (legacy ASER with SPO + fixed lambda replay)
# B: TRACE infrastructure without trust gate (lambda_eff = lambda_max always)
# C: Full TRACE-GRPO (adaptive lambda_eff)
#
# 3 seeds × 3 variants = 9 runs; 8 GPUs → batch 1: A×3 + B×3 + C×2, batch 2: C×1
# OR: 2 seeds × 3 variants = 6 runs in a single batch
set -euo pipefail

cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=${HF_HOME:-}/hub
export HF_DATASETS_CACHE=${HF_HOME:-}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:-}/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen3.5-9B"
MAX_STEPS=${MAX_STEPS:-200}
SEEDS=${SEEDS:-"42 43"}    # 2 seeds * 3 variants = 6 runs (fits 8 GPUs with 2 free)
OUTDIR=${OUTDIR:-results/trace_abc}

mkdir -p "$OUTDIR/logs"

GPU=0
for seed in $SEEDS; do
    # A: Existing Best Positive Fragment Only (legacy ASER SPO + lambda=0.05)
    tag="A_legacy_spo_replay_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_aser_mvp.py \
        --seed $seed --backbone spo --config configs/aser_g4_safe.yaml \
        --output-dir "$OUTDIR/A_legacy" --max-steps $MAX_STEPS --model $MODEL \
        --lambda-rep 0.05 \
        --run-name "$tag" \
        > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[A] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done

for seed in $SEEDS; do
    # B: TRACE without mechanism (lambda_eff = lambda_max always)
    tag="B_trace_constant_gate_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_trace_grpo.py \
        --seed $seed --config configs/trace_grpo_minimal.yaml \
        --output-dir "$OUTDIR/B_constant" --max-steps $MAX_STEPS --model $MODEL \
        --trace-mode constant_gate \
        --run-name "$tag" \
        > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[B] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done

for seed in $SEEDS; do
    # C: Full TRACE-GRPO
    tag="C_trace_full_seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/run_trace_grpo.py \
        --seed $seed --config configs/trace_grpo_minimal.yaml \
        --output-dir "$OUTDIR/C_full" --max-steps $MAX_STEPS --model $MODEL \
        --trace-mode full \
        --run-name "$tag" \
        > "$OUTDIR/logs/$tag.log" 2>&1 &
    echo "[C] $tag GPU $GPU pid=$!"
    GPU=$((GPU+1))
done

echo "[launched] A/B/C × seeds ($SEEDS), $MAX_STEPS steps"
echo "[waiting] Training..."
wait
echo "[done] Training complete"

# Full-set eval on all 6 runs
N=1319
GPU=0
for variant_dir in A_legacy B_constant C_full; do
    mkdir -p "$OUTDIR/$variant_dir/evals"
    for seed in $SEEDS; do
        if [ "$variant_dir" = "A_legacy" ]; then
            tag="A_legacy_spo_replay_seed${seed}"
        elif [ "$variant_dir" = "B_constant" ]; then
            tag="B_trace_constant_gate_seed${seed}"
        else
            tag="C_trace_full_seed${seed}"
        fi
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
echo "[done] Evals complete"

echo "=== RESULTS ==="
for f in "$OUTDIR"/*/evals/eval_*.json; do
    [ -f "$f" ] && python3 -c "
import json, os
d = json.load(open('$f'))
print(f'{os.path.basename(\"$f\"):50s} acc={d[\"accuracy\"]:.4f} selection={d.get(\"selection\",\"?\")} n={d[\"n\"]}')
"
done
