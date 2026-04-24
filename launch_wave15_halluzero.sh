#!/bin/bash
# Wave 15 — HalluZero baselines: 4 strategies × 2 seeds = 8 runs on 8 GPUs
# All use dr_grpo backbone, no replay, no dup (fair comparison with phase diagram)
set -e
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3.5-9B"
OUTDIR="results/wave15_halluzero"
mkdir -p $OUTDIR/logs

# 4 strategies × 2 seeds = 8 runs → 8 GPUs
gpu=0
PIDS=()
for strategy in clip temperature curriculum relabel; do
    for seed in 42 43; do
        tag="hz_${strategy}_seed${seed}"
        if [ -d "$OUTDIR/$tag/checkpoint-final" ]; then
            echo "[skip] $tag already complete"
            continue
        fi
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
            --seed $seed --backbone dr_grpo --config configs/aser_mvp.yaml \
            --output-dir $OUTDIR --max-steps 200 --model $MODEL \
            --lambda-rep 0 --no-dup \
            --zero-score-strategy $strategy \
            --run-name "$tag" \
            > $OUTDIR/logs/$tag.log 2>&1 &
        PIDS+=($!)
        echo "[launch] $tag on GPU $gpu pid=$!"
        gpu=$((gpu + 1))
    done
done

echo "[info] Waiting for ${#PIDS[@]} training runs..."
for pid in "${PIDS[@]}"; do wait $pid; done
echo "[train] All HalluZero training complete"

# === Evals ===
mkdir -p $OUTDIR/evals
gpu=0
PIDS=()
for strategy in clip temperature curriculum relabel; do
    for seed in 42 43; do
        tag="hz_${strategy}_seed${seed}"
        adapter="$OUTDIR/$tag/checkpoint-final"
        out="$OUTDIR/evals/eval_${tag}.json"
        if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
        if [ ! -d "$adapter" ]; then echo "[WARN] $adapter missing"; continue; fi
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
            --adapter "$adapter" --n 1319 --out "$out" \
            > $OUTDIR/evals/eval_${tag}.log 2>&1 &
        PIDS+=($!)
        echo "[eval] $tag on GPU $gpu pid=$!"
        gpu=$((gpu + 1))
    done
done
for pid in "${PIDS[@]}"; do wait $pid; done
echo "[done] Wave 15 HalluZero: 8 runs trained + evaluated"
