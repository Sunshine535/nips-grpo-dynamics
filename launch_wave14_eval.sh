#!/bin/bash
# Wave 14 eval — 8 adapters on 8 GPUs in parallel
# 3× SPO+Replay 500-step + 5× phase diagram (α sweep at β=1.0)
set -e
cd "$(dirname "$0")"

if [ -d /openbayes/input/input0/hub ]; then export HF_HOME=/openbayes/input/input0; fi
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

N=1319  # full GSM8K test set

# === Part A: 500-step SPO+Replay evals (GPU 0-2) ===
mkdir -p results/wave14_500step/evals
for pair in "0 42" "1 43" "2 44"; do
    read gpu seed <<< "$pair"
    adapter="results/wave14_500step/spo_replay_500step_seed$seed/checkpoint-final"
    out="results/wave14_500step/evals/eval_seed${seed}.json"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    if [ ! -d "$adapter" ]; then echo "[WARN] $adapter missing"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" --n $N \
        --out "$out" \
        > results/wave14_500step/evals/eval_seed${seed}.log 2>&1 &
    echo "[eval] 500step seed$seed on GPU $gpu pid=$!"
done

# === Part B: Phase diagram evals (GPU 3-7) ===
mkdir -p results/wave14_phase_diagram/evals
gpu=3
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    adapter="results/wave14_phase_diagram/phase_a${alpha}_b1.0_seed42/checkpoint-final"
    out="results/wave14_phase_diagram/evals/eval_a${alpha}_b1.0.json"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    if [ ! -d "$adapter" ]; then echo "[WARN] $adapter missing"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" --n $N \
        --out "$out" \
        > results/wave14_phase_diagram/evals/eval_a${alpha}_b1.0.log 2>&1 &
    echo "[eval] phase a=${alpha} b=1.0 on GPU $gpu pid=$!"
    gpu=$((gpu + 1))
done

echo "[done] Wave 14 eval: 8 evals launched on 8 GPUs (N=$N, ~20-25 min each)"
