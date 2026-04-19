#!/bin/bash
# Gate 1 + Gate 2: close the two remaining blockers from Round 4 review.
#
#   Gate 1 — one end-to-end V14 ADQ training run on Qwen3.5-9B/GSM8K with
#            saved rho(t) trajectory, ada_telemetry, final eval. Proves
#            ADQ controller actually moves rho during training on the real
#            model (the archived AdaBalance runs stayed at rho=1.0000).
#
#   Gate 2 — 3 seeds x 3 rho values in {0.7, 1.0, 3.0} to turn the
#            single-seed "upward tendency" into a CI-backed claim. Gate
#            2's rho=1.0 seed=42 doubles as a control for Gate 1.
#
# Layout across 8 GPUs (A800-80GB):
#   GPU 0: Gate 1       ADQ init rho=1.0 seed=42
#   GPU 1: Gate 2 fixed rho=0.70 seed=42
#   GPU 2: Gate 2 fixed rho=0.70 seed=43
#   GPU 3: Gate 2 fixed rho=0.70 seed=44
#   GPU 4: Gate 2 fixed rho=1.00 seed=43
#   GPU 5: Gate 2 fixed rho=1.00 seed=44
#   GPU 6: Gate 2 fixed rho=3.00 seed=42
#   GPU 7: Gate 2 fixed rho=3.00 seed=43
#   (rho=1.00 seed=42 supplied by Gate 1 ADQ control; rho=3.00 seed=44
#    runs on whichever GPU frees up first.)
#
# Usage:
#   bash run_gates_1_2.sh                # normal: 200 steps
#   MAX_STEPS=100 bash run_gates_1_2.sh  # shorter
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MAX_STEPS="${MAX_STEPS:-200}"
OUTPUT="${OUTPUT:-results/gates_1_2}"
CONFIG="${CONFIG:-configs/rho_sweep.yaml}"
mkdir -p "$OUTPUT" "$OUTPUT/logs"

# HF offline config (same as run_csd_pilot.sh)
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
# TRL 0.14 needs to see advantages in the ρ-weighted path
export PYTHONPATH=".:${PYTHONPATH:-}"

source scripts/gpu_utils.sh
auto_setup

echo "============================================"
echo " Gate 1 + Gate 2 — final autonomous experiments"
echo " Model:     $MODEL"
echo " Max steps: $MAX_STEPS"
echo " Output:    $OUTPUT"
echo " GPUs:      $NUM_GPUS"
echo "============================================"

launch() {
    local gpu=$1; shift
    local tag=$1; shift
    local log="$OUTPUT/logs/${tag}.log"
    local real_gpu
    real_gpu=$(get_gpu_id "$gpu")
    echo "[launch] $tag → GPU $real_gpu (log: $log)"
    CUDA_VISIBLE_DEVICES="$real_gpu" nohup python3 scripts/run_csd_pilot.py "$@" \
        --config "$CONFIG" --output_dir "$OUTPUT" \
        --max_steps "$MAX_STEPS" --model "$MODEL" \
        > "$log" 2>&1 &
    echo $! > "$OUTPUT/logs/${tag}.pid"
}

# Gate 1: ADQ run (rho moves online)
launch 0 "gate1_adq_seed42"  --pilot 2_single --rho 1.0 --seed_start 42 --use_adq

# Gate 2: 3-seed fixed-rho sweep
launch 1 "gate2_rho0.70_seed42" --pilot 1_single --rho 0.7 --seed_start 42
launch 2 "gate2_rho0.70_seed43" --pilot 1_single --rho 0.7 --seed_start 43
launch 3 "gate2_rho0.70_seed44" --pilot 1_single --rho 0.7 --seed_start 44
launch 4 "gate2_rho1.00_seed43" --pilot 1_single --rho 1.0 --seed_start 43
launch 5 "gate2_rho1.00_seed44" --pilot 1_single --rho 1.0 --seed_start 44
launch 6 "gate2_rho3.00_seed42" --pilot 1_single --rho 3.0 --seed_start 42
launch 7 "gate2_rho3.00_seed43" --pilot 1_single --rho 3.0 --seed_start 43

echo ""
echo "All 8 primary jobs launched. Tail with: tail -F $OUTPUT/logs/*.log"
echo "PIDs in $OUTPUT/logs/*.pid"
echo ""
echo "After any GPU frees up, launch the remaining run manually:"
echo "  CUDA_VISIBLE_DEVICES=<free_gpu> python3 scripts/run_csd_pilot.py \\"
echo "    --pilot 1_single --rho 3.0 --seed_start 44 \\"
echo "    --config $CONFIG --output_dir $OUTPUT --max_steps $MAX_STEPS --model $MODEL \\"
echo "    > $OUTPUT/logs/gate2_rho3.00_seed44.log 2>&1 &"
