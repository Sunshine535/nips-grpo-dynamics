#!/bin/bash
# Eval Wave 11 adapters (fixed-ρ=0.70 seeds 46-51 + RFT control seeds 42-44)
# Uses scripts/eval_stratified.py with n=200 per adapter to match Wave 10.
set -euo pipefail
cd "$(dirname "$0")"

N_EVAL="${N_EVAL:-200}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"

if [ -d /openbayes/input/input0/hub ]; then
    export HF_HOME="/openbayes/input/input0"
fi
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

OUT_DIR="results/stratified_eval_wave11"
mkdir -p "$OUT_DIR/logs"

eval_one() {
    local gpu=$1; shift
    local tag=$1; shift
    local adapter=$1; shift
    local out="$OUT_DIR/${tag}.json"
    local log="$OUT_DIR/logs/${tag}.log"
    if [ ! -d "$adapter" ]; then
        echo "[skip] $tag: adapter missing at $adapter"
        return
    fi
    echo "[eval] $tag → GPU $gpu → $out"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" --base_model "$BASE_MODEL" \
        --n "$N_EVAL" --out "$out" \
        > "$log" 2>&1 &
    echo $! > "$OUT_DIR/logs/${tag}.pid"
}

# Assume Wave 11 saved adapters under checkpoint-final
RHO_DIR="results/wave11_rho070_boost"
RFT_DIR="results/wave11_rft_control"

# Layout: 9 evals across 8 GPUs. One GPU does 2 evals sequentially.
# Approx 15-20 min per eval at n=200 with batched decoding → ~40 min total.
eval_one 0 "rho0.70_seed46" "$RHO_DIR/rho0.70_seed46/checkpoint-final"
eval_one 1 "rho0.70_seed47" "$RHO_DIR/rho0.70_seed47/checkpoint-final"
eval_one 2 "rho0.70_seed48" "$RHO_DIR/rho0.70_seed48/checkpoint-final"
eval_one 3 "rho0.70_seed49" "$RHO_DIR/rho0.70_seed49/checkpoint-final"
eval_one 4 "rho0.70_seed50" "$RHO_DIR/rho0.70_seed50/checkpoint-final"
eval_one 5 "rho0.70_seed51" "$RHO_DIR/rho0.70_seed51/checkpoint-final"
eval_one 6 "rft_seed42"     "$RFT_DIR/aser_spo_seed42_nopg/checkpoint-final"
eval_one 7 "rft_seed43"     "$RFT_DIR/aser_spo_seed43_nopg/checkpoint-final"

# Queue last one on GPU 0 after seed 46 eval completes
(
  pid46=$(cat "$OUT_DIR/logs/rho0.70_seed46.pid")
  while kill -0 "$pid46" 2>/dev/null; do sleep 30; done
  eval_one 0 "rft_seed44" "$RFT_DIR/aser_spo_seed44_nopg/checkpoint-final"
) > /dev/null 2>&1 &
disown
echo "[done] Eval dispatched. Tail with: tail -F $OUT_DIR/logs/*.log"
