#!/bin/bash
# Monitor Wave 11 GPU state. When any GPU drops below 10 GB sustained usage
# for 60 seconds, launch RFT control seed 44 on that GPU.
# Also kick off evals on each adapter as soon as its training finishes.
set -eo pipefail
# Note: not using -u because associative array element access with :- doesn't
# work on older bash versions.
cd "$(dirname "$0")"

RHO_DIR="results/wave11_rho070_boost"
RFT_DIR="results/wave11_rft_control"
EVAL_DIR="results/stratified_eval_wave11"
mkdir -p "$EVAL_DIR/logs"

if [ -d /openbayes/input/input0/hub ]; then
    export HF_HOME="/openbayes/input/input0"
fi
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

log() { echo "[followup $(date +%H:%M:%S)] $*" >> "$EVAL_DIR/logs/followup.log"; }

log "started"

launch_rft44() {
    local gpu=$1
    log "launching rft_seed44 on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 scripts/run_aser_mvp.py \
        --seed 44 --backbone spo --config configs/aser_mvp.yaml \
        --output-dir "$RFT_DIR" --max-steps 200 --model Qwen/Qwen3.5-9B \
        --pg-weight 0.0 \
        > "$RFT_DIR/logs/rft_seed44_v2.log" 2>&1 &
    echo $! > "$RFT_DIR/logs/rft_seed44_v2.pid"
    log "rft_seed44 launched pid=$(cat $RFT_DIR/logs/rft_seed44_v2.pid) on GPU $gpu"
}

launch_eval() {
    local gpu=$1; shift
    local tag=$1; shift
    local adapter=$1; shift
    if [ ! -d "$adapter" ]; then
        log "eval skip $tag: adapter missing at $adapter"
        return 1
    fi
    if [ -f "$EVAL_DIR/${tag}.json" ]; then
        log "eval skip $tag: output exists"
        return 0
    fi
    log "eval launch $tag on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 scripts/eval_stratified.py \
        --adapter "$adapter" --base_model Qwen/Qwen3.5-9B \
        --n 200 --out "$EVAL_DIR/${tag}.json" \
        > "$EVAL_DIR/logs/${tag}.log" 2>&1 &
    echo $! > "$EVAL_DIR/logs/${tag}.pid"
    log "eval $tag launched pid=$(cat $EVAL_DIR/logs/${tag}.pid)"
    return 0
}

# Track which adapters we've already dispatched an eval for
declare -A EVAL_DISPATCHED

# Map run tag → adapter path
declare -A ADAPTERS
ADAPTERS[rho0.70_seed46]="$RHO_DIR/rho0.70_seed46/checkpoint-final"
ADAPTERS[rho0.70_seed47]="$RHO_DIR/rho0.70_seed47/checkpoint-final"
ADAPTERS[rho0.70_seed48]="$RHO_DIR/rho0.70_seed48/checkpoint-final"
ADAPTERS[rho0.70_seed49]="$RHO_DIR/rho0.70_seed49/checkpoint-final"
ADAPTERS[rho0.70_seed50]="$RHO_DIR/rho0.70_seed50/checkpoint-final"
ADAPTERS[rho0.70_seed51]="$RHO_DIR/rho0.70_seed51/checkpoint-final"
ADAPTERS[rft_seed42]="$RFT_DIR/aser_spo_seed42_nopg/checkpoint-final"
ADAPTERS[rft_seed43]="$RFT_DIR/aser_spo_seed43_nopg/checkpoint-final"
ADAPTERS[rft_seed44]="$RFT_DIR/aser_spo_seed44_nopg/checkpoint-final"

RFT44_LAUNCHED=0
low_gpu_streak=()
for i in 0 1 2 3 4 5 6 7; do low_gpu_streak[$i]=0; done

while true; do
    # Dispatch evals for any completed adapter
    for tag in "${!ADAPTERS[@]}"; do
        if [ -n "${EVAL_DISPATCHED[$tag]:-}" ]; then continue; fi
        adapter="${ADAPTERS[$tag]}"
        if [ -d "$adapter" ] && [ -f "$adapter/adapter_config.json" ]; then
            # Wait for GPU to free — pick the first GPU with <10 GB usage
            gpu_out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
            # Find first GPU with < 10GB used
            gpu_eval=""
            while read line; do
                idx=$(echo $line | awk -F',' '{print $1}' | tr -d ' ')
                mem=$(echo $line | awk -F',' '{print $2}' | grep -oE '[0-9]+')
                if [ "$mem" -lt 10000 ]; then
                    gpu_eval=$idx
                    break
                fi
            done <<< "$gpu_out"
            if [ -n "$gpu_eval" ]; then
                if launch_eval "$gpu_eval" "$tag" "$adapter"; then
                    EVAL_DISPATCHED[$tag]=1
                fi
            fi
        fi
    done
    # If RFT44 not yet launched, look for a free GPU
    if [ $RFT44_LAUNCHED -eq 0 ]; then
        gpu_out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
        while read line; do
            idx=$(echo $line | awk -F',' '{print $1}' | tr -d ' ')
            mem=$(echo $line | awk -F',' '{print $2}' | grep -oE '[0-9]+')
            if [ "$mem" -lt 10000 ]; then
                low_gpu_streak[$idx]=$((${low_gpu_streak[$idx]:-0} + 1))
            else
                low_gpu_streak[$idx]=0
            fi
            # Need 2 consecutive low readings before launching (60s gap = 120s confirmed)
            if [ "${low_gpu_streak[$idx]:-0}" -ge 2 ]; then
                # But first, verify RFT-44 eval isn't already running here
                running_on_gpu=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $idx 2>/dev/null | wc -l)
                if [ "$running_on_gpu" -le 1 ]; then
                    launch_rft44 "$idx"
                    RFT44_LAUNCHED=1
                    break
                fi
            fi
        done <<< "$gpu_out"
    fi
    # Exit when RFT44 done AND all evals dispatched
    n_dispatched=${#EVAL_DISPATCHED[@]}
    if [ $RFT44_LAUNCHED -eq 1 ] && [ $n_dispatched -eq 9 ]; then
        log "all work dispatched (RFT44 + 9 evals). waiting for completion..."
        # Wait for eval PIDs to finish, then exit.
        for f in "$EVAL_DIR"/logs/*.pid; do
            [ -f "$f" ] || continue
            pid=$(cat "$f")
            while kill -0 "$pid" 2>/dev/null; do sleep 30; done
        done
        log "all evals finished"
        break
    fi
    sleep 30
done
log "exiting"
