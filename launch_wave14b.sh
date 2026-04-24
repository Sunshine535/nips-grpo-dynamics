#!/bin/bash
# Wave 14b — Phase diagram: remaining 15 points (β ∈ {0.0, 0.5, 2.0} × α ∈ {0.1,0.3,0.5,0.7,0.9})
# Wave 14 already did β=1.0, this covers the other 3 β values.
# 8 GPUs available → batch 1: 8 points, batch 2: 7 points (auto-chain via wait)
set -e
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3.5-9B"
OUTDIR="results/wave14_phase_diagram"
mkdir -p $OUTDIR/logs

# Build list of (alpha, beta) points to run
declare -a POINTS
for beta in 0.0 0.5 2.0; do
    for alpha in 0.1 0.3 0.5 0.7 0.9; do
        tag="phase_a${alpha}_b${beta}_seed42"
        if [ -d "$OUTDIR/$tag/checkpoint-final" ]; then
            echo "[skip] $tag already complete"
            continue
        fi
        POINTS+=("$alpha $beta $tag")
    done
done

echo "[info] ${#POINTS[@]} points to run"
if [ ${#POINTS[@]} -eq 0 ]; then echo "[done] all phase diagram points complete"; exit 0; fi

# === Batch 1: first 8 points on GPU 0-7 ===
gpu=0
PIDS=()
for i in $(seq 0 $((${#POINTS[@]} < 8 ? ${#POINTS[@]} - 1 : 7))); do
    read alpha beta tag <<< "${POINTS[$i]}"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
        --seed 42 --backbone dr_grpo --config configs/aser_mvp.yaml \
        --output-dir $OUTDIR --max-steps 200 --model $MODEL \
        --lambda-rep 0 --no-dup \
        --alpha-pos $alpha --beta-neg $beta \
        --run-name "$tag" \
        > $OUTDIR/logs/$tag.log 2>&1 &
    PIDS+=($!)
    echo "[batch1] $tag on GPU $gpu pid=$!"
    gpu=$((gpu + 1))
done

# Wait for batch 1
for pid in "${PIDS[@]}"; do wait $pid; done
echo "[batch1] complete"

# === Batch 1 eval ===
mkdir -p $OUTDIR/evals
gpu=0
PIDS=()
for i in $(seq 0 $((${#POINTS[@]} < 8 ? ${#POINTS[@]} - 1 : 7))); do
    read alpha beta tag <<< "${POINTS[$i]}"
    adapter="$OUTDIR/$tag/checkpoint-final"
    out="$OUTDIR/evals/eval_a${alpha}_b${beta}.json"
    if [ ! -d "$adapter" ]; then echo "[WARN] $adapter missing"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" --n 1319 --out "$out" \
        > $OUTDIR/evals/eval_a${alpha}_b${beta}.log 2>&1 &
    PIDS+=($!)
    echo "[eval1] $tag on GPU $gpu pid=$!"
    gpu=$((gpu + 1))
done
for pid in "${PIDS[@]}"; do wait $pid; done
echo "[eval1] complete"

# === Batch 2: remaining points (if any) ===
if [ ${#POINTS[@]} -gt 8 ]; then
    gpu=0
    PIDS=()
    for i in $(seq 8 $((${#POINTS[@]} - 1))); do
        read alpha beta tag <<< "${POINTS[$i]}"
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/run_aser_mvp.py \
            --seed 42 --backbone dr_grpo --config configs/aser_mvp.yaml \
            --output-dir $OUTDIR --max-steps 200 --model $MODEL \
            --lambda-rep 0 --no-dup \
            --alpha-pos $alpha --beta-neg $beta \
            --run-name "$tag" \
            > $OUTDIR/logs/$tag.log 2>&1 &
        PIDS+=($!)
        echo "[batch2] $tag on GPU $gpu pid=$!"
        gpu=$((gpu + 1))
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
    echo "[batch2] complete"

    # Batch 2 eval
    gpu=0
    PIDS=()
    for i in $(seq 8 $((${#POINTS[@]} - 1))); do
        read alpha beta tag <<< "${POINTS[$i]}"
        adapter="$OUTDIR/$tag/checkpoint-final"
        out="$OUTDIR/evals/eval_a${alpha}_b${beta}.json"
        if [ ! -d "$adapter" ]; then echo "[WARN] $adapter missing"; continue; fi
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
            --adapter "$adapter" --n 1319 --out "$out" \
            > $OUTDIR/evals/eval_a${alpha}_b${beta}.log 2>&1 &
        PIDS+=($!)
        echo "[eval2] $tag on GPU $gpu pid=$!"
        gpu=$((gpu + 1))
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
    echo "[eval2] complete"
fi

echo "[done] Wave 14b: all phase diagram points trained + evaluated"
