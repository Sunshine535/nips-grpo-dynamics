#!/bin/bash
# Wave 11 — post-Round-1 follow-up launched after Codex xhigh flagged two fatal issues:
#   (1) adaptive-dup was never firing (int(2*0.25)=0) — fixed in sampler via probabilistic round
#   (2) replay-CE novelty is indistinguishable from online RFT without a matched control
#
# Layout across 8 GPUs (A800-80GB):
#   GPU 0: fixed-rho=0.70 seed 46 (boost existing baseline n=3 → n=9)
#   GPU 1: fixed-rho=0.70 seed 47
#   GPU 2: fixed-rho=0.70 seed 48
#   GPU 3: fixed-rho=0.70 seed 49
#   GPU 4: fixed-rho=0.70 seed 50
#   GPU 5: fixed-rho=0.70 seed 51
#   GPU 6: RFT control seed 42 (--pg-weight 0 → pure SFT on verified-success bank)
#   GPU 7: RFT control seed 43
#   Queued: RFT control seed 44 → runs on whichever GPU frees first
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MAX_STEPS="${MAX_STEPS:-200}"
OUTPUT_RHO="${OUTPUT_RHO:-results/wave11_rho070_boost}"
OUTPUT_RFT="${OUTPUT_RFT:-results/wave11_rft_control}"
RHO_CONFIG="${RHO_CONFIG:-configs/rho_sweep.yaml}"
ASER_CONFIG="${ASER_CONFIG:-configs/aser_mvp.yaml}"
mkdir -p "$OUTPUT_RHO/logs" "$OUTPUT_RFT/logs"

# HF offline config
if [ -d /openbayes/input/input0/hub ]; then
    export HF_HOME="/openbayes/input/input0"
elif [ -d /ytech_m2v4_hdd/mengzijie/.cache/hf/hub ]; then
    export HF_HOME="/ytech_m2v4_hdd/mengzijie/.cache/hf"
fi
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source scripts/gpu_utils.sh
auto_setup

echo "============================================"
echo " Wave 11 — Codex-Round-1 follow-up"
echo " Model:     $MODEL"
echo " Max steps: $MAX_STEPS"
echo " ρ-boost output: $OUTPUT_RHO"
echo " RFT output:     $OUTPUT_RFT"
echo "============================================"

launch_rho() {
    local gpu=$1; shift
    local seed=$1; shift
    local tag="rho0.70_seed${seed}"
    local log="$OUTPUT_RHO/logs/${tag}.log"
    local real_gpu; real_gpu=$(get_gpu_id "$gpu")
    echo "[launch] ρ=0.70 seed=$seed → GPU $real_gpu (log: $log)"
    CUDA_VISIBLE_DEVICES="$real_gpu" nohup python3 scripts/run_csd_pilot.py \
        --pilot 1_single --rho 0.7 --seed_start "$seed" \
        --config "$RHO_CONFIG" --output_dir "$OUTPUT_RHO" \
        --max_steps "$MAX_STEPS" --model "$MODEL" \
        > "$log" 2>&1 &
    echo $! > "$OUTPUT_RHO/logs/${tag}.pid"
}

launch_rft() {
    local gpu=$1; shift
    local seed=$1; shift
    local tag="rft_seed${seed}"
    local log="$OUTPUT_RFT/logs/${tag}.log"
    local real_gpu; real_gpu=$(get_gpu_id "$gpu")
    echo "[launch] RFT seed=$seed → GPU $real_gpu (log: $log)"
    # --pg-weight 0 zeros out the GRPO policy gradient → leaves only
    # verified-success replay CE. Still does rollouts, still builds bank.
    # This is the "online RFT" control: does SFT on own verified successes
    # match the full ASE-R method?
    CUDA_VISIBLE_DEVICES="$real_gpu" nohup python3 scripts/run_aser_mvp.py \
        --seed "$seed" --backbone spo --config "$ASER_CONFIG" \
        --output-dir "$OUTPUT_RFT" --max-steps "$MAX_STEPS" --model "$MODEL" \
        --pg-weight 0.0 \
        > "$log" 2>&1 &
    echo $! > "$OUTPUT_RFT/logs/${tag}.pid"
}

# Primary launch — 6 fixed-ρ=0.70 on GPUs 0-5, 2 RFT on GPUs 6-7
launch_rho 0 46
launch_rho 1 47
launch_rho 2 48
launch_rho 3 49
launch_rho 4 50
launch_rho 5 51
launch_rft 6 42
launch_rft 7 43

echo ""
echo "All 8 primary jobs launched. Tail with: tail -F $OUTPUT_RHO/logs/*.log $OUTPUT_RFT/logs/*.log"
echo ""
echo "[queue] RFT seed 44 will launch on first freed GPU"

# Background queue-runner: reuse the first freed GPU for RFT seed 44
(
  while true; do
    for pidfile in "$OUTPUT_RHO/logs"/*.pid "$OUTPUT_RFT/logs"/*.pid; do
      [ -f "$pidfile" ] || continue
      pid=$(cat "$pidfile")
      if ! kill -0 "$pid" 2>/dev/null; then
        freed_tag=$(basename "$pidfile" .pid)
        # Extract GPU from log — not guaranteed. Just pick GPU 0 as fallback.
        real_gpu=$(get_gpu_id 0 2>/dev/null || echo 0)
        tag="rft_seed44"
        log="$OUTPUT_RFT/logs/${tag}.log"
        [ -f "$OUTPUT_RFT/logs/${tag}.pid" ] && break 2  # already launched
        echo "[queue-runner] launching $tag on GPU $real_gpu at $(date +%H:%M) (freed by $freed_tag)" >> "$OUTPUT_RFT/logs/queue-runner.log"
        CUDA_VISIBLE_DEVICES="$real_gpu" nohup python3 scripts/run_aser_mvp.py \
            --seed 44 --backbone spo --config "$ASER_CONFIG" \
            --output-dir "$OUTPUT_RFT" --max-steps "$MAX_STEPS" --model "$MODEL" \
            --pg-weight 0.0 \
            > "$log" 2>&1 &
        echo $! > "$OUTPUT_RFT/logs/${tag}.pid"
        break 2
      fi
    done
    sleep 60
  done
) > /dev/null 2>&1 &
disown
echo "[done] Wave 11 dispatched. 9 runs in total (6 ρ-boost + 3 RFT)."
