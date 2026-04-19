#!/bin/bash
# Wave 3: cross-generation Qwen validation on GSM8K
#
# Closes H3 — "decomposition holds across Qwen generations".
# Reuses the exact run_gates_1_2.sh schedule but with a different MODEL.
# Run twice: once for Qwen2.5-7B, once for Qwen3-8B.
#
# Usage (invoke twice):
#   MODEL=Qwen/Qwen2.5-7B-Instruct OUTPUT=results/wave3_qwen25_7b bash run_wave3_cross_gen.sh
#   MODEL=Qwen/Qwen3-8B            OUTPUT=results/wave3_qwen3_8b  bash run_wave3_cross_gen.sh
#
# Each call launches 9 parallel runs: ρ ∈ {0.7, 1.0, 3.0} × 3 seeds × G=2 × 200 steps.
# Leaves GPU allocation to the caller to stagger with other waves.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_STEPS="${MAX_STEPS:-200}"
OUTPUT="${OUTPUT:-results/wave3_qwen25_7b}"
CONFIG="${CONFIG:-configs/rho_sweep.yaml}"
GPU_OFFSET="${GPU_OFFSET:-0}"   # allows running on GPUs 0-7 or a subrange
mkdir -p "$OUTPUT" "$OUTPUT/logs"

if [ -d /openbayes/input/input0/hub ]; then
    export HF_HOME="/openbayes/input/input0"
fi
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "============================================"
echo " Wave 3 — cross-generation validation"
echo " Model:     $MODEL"
echo " Output:    $OUTPUT"
echo " GPU offset: $GPU_OFFSET (uses GPUs $GPU_OFFSET..$(($GPU_OFFSET + 8)))"
echo "============================================"

launch() {
    local gpu=$1; shift
    local tag=$1; shift
    local log="$OUTPUT/logs/${tag}.log"
    echo "[launch] $tag → GPU $gpu (log: $log)"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 scripts/run_csd_pilot.py "$@" \
        --config "$CONFIG" --output_dir "$OUTPUT" \
        --max_steps "$MAX_STEPS" --model "$MODEL" \
        > "$log" 2>&1 &
    echo $! > "$OUTPUT/logs/${tag}.pid"
}

# 9 runs: ρ ∈ {0.7, 1.0, 3.0} × 3 seeds
i=0
for rho in 0.7 1.0 3.0; do
  for seed in 42 43 44; do
    gpu=$(( (GPU_OFFSET + i) % 8 ))
    rho_fmt=$(printf "%.2f" "$rho")
    launch "$gpu" "wave3_rho${rho_fmt}_seed${seed}" \
        --pilot 1_single --rho "$rho" --seed_start "$seed"
    i=$((i + 1))
  done
done

echo ""
echo "9 runs launched on GPUs $GPU_OFFSET..$((GPU_OFFSET + 8)). Tail with:"
echo "  tail -F $OUTPUT/logs/*.log"
