#!/usr/bin/env bash
# =============================================================================
# Targeted experiment launcher for NeurIPS GRPO stability paper
# =============================================================================
# Priority order:
#   Exp 1a: Coarse rho sweep (9 rho x 3 seeds x 200 steps)
#   Exp 2:  AdaBalance comparison (5 methods x 2 seeds, full training)
#   Exp 1b: Fine sweep near boundaries (10 rho x 3 seeds, full 2 epochs)
#   Exp 3:  Robustness / i.i.d. violation (4 bins x 3 rho x 2 seeds)
#
# Hardware: 2x NVIDIA H100 80GB, round-robin GPU scheduling
# Usage:
#   bash run_targeted.sh                  # run all experiments
#   bash run_targeted.sh --from-exp 2     # resume from experiment 2
#   FORCE_RERUN=1 bash run_targeted.sh    # ignore phase markers
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Directories and environment
# ---------------------------------------------------------------------------
PROJ_DIR_ROOT="${PROJ_DIR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PROJ_DIR_ROOT
cd "$PROJ_DIR_ROOT"

SCRIPT_DIR="$PROJ_DIR_ROOT/scripts"
CONFIG_RHO="$PROJ_DIR_ROOT/configs/rho_sweep.yaml"
LOG_ROOT="$PROJ_DIR_ROOT/results/logs"
MARKER_DIR="$PROJ_DIR_ROOT/results/.targeted_markers"

mkdir -p "$LOG_ROOT" "$MARKER_DIR"

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------
VENV_PATH="${VENV_PATH:-$HOME/grpo-venv}"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
  echo "[env] Activated venv: $VENV_PATH"
else
  echo "[WARN] Venv not found at $VENV_PATH -- using current Python"
fi

PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# GPU detection (simplified for known 2xH100 setup)
# ---------------------------------------------------------------------------
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup  # sets NUM_GPUS, CUDA_VISIBLE_DEVICES, etc.

if [[ "$NUM_GPUS" -lt 2 ]]; then
  echo "[WARN] Expected 2 GPUs, detected $NUM_GPUS. Will use $NUM_GPUS-way parallelism."
fi

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
FROM_EXP=0
FORCE_RERUN="${FORCE_RERUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-exp)   FROM_EXP="$2"; shift 2 ;;
    --from-exp=*) FROM_EXP="${1#*=}"; shift ;;
    *) echo "[warn] Unknown arg: $1"; shift ;;
  esac
done

# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------
mark_done() { touch "$MARKER_DIR/exp_${1}.done"; echo "[EXP $1] Completed at $(date)"; }
is_done() {
  [[ "$FORCE_RERUN" == "1" ]] && return 1
  if [[ "$1" -lt "$FROM_EXP" ]]; then
    echo "[EXP $1] Skipped (--from-exp $FROM_EXP)"
    return 0
  fi
  [[ -f "$MARKER_DIR/exp_${1}.done" ]] && echo "[EXP $1] Already completed. Skipping. (FORCE_RERUN=1 to override)" && return 0
  return 1
}

# ---------------------------------------------------------------------------
# Parallel job management
# ---------------------------------------------------------------------------
# Track background PIDs and their descriptions for error reporting.
declare -a BG_PIDS=()
declare -A PID_TAGS=()  # PID -> human-readable tag
FAILED_JOBS=0

# Launch a job on a specific GPU. Args: gpu_id tag command...
launch_on_gpu() {
  local gpu="$1"; shift
  local tag="$1"; shift
  local log_file="$1"; shift

  mkdir -p "$(dirname "$log_file")"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    # Unique ports per job to avoid conflicts when running parallel vLLM/torch
    export MASTER_PORT=$((29500 + gpu * 100 + RANDOM % 50))
    export VLLM_PORT=$((51216 + gpu * 100 + RANDOM % 50))
    echo "=== $tag  GPU=$gpu  MASTER_PORT=$MASTER_PORT  START=$(date) ===" >> "$log_file"
    "$@" >> "$log_file" 2>&1
    echo "=== $tag  GPU=$gpu  END=$(date)   ===" >> "$log_file"
  ) &
  local pid=$!
  BG_PIDS+=("$pid")
  PID_TAGS[$pid]="$tag"
  echo "  [launch] $tag on GPU $gpu (PID $pid, log: $log_file)"
}

# Wait for the current batch of NUM_GPUS jobs, then reset.
# Returns 0 if all succeeded, increments FAILED_JOBS for each failure.
drain_batch() {
  local any_fail=0
  for pid in "${BG_PIDS[@]}"; do
    if ! wait "$pid"; then
      echo "  [FAIL] ${PID_TAGS[$pid]:-PID $pid} exited non-zero"
      FAILED_JOBS=$((FAILED_JOBS + 1))
      any_fail=1
    fi
  done
  BG_PIDS=()
  PID_TAGS=()
  return $any_fail
}

# Submit jobs in round-robin batches of NUM_GPUS.
# Call schedule_slot before each launch_on_gpu; it will drain when full.
GPU_SLOT=0
schedule_slot() {
  # If we have filled all GPU slots, drain before continuing
  if [[ "${#BG_PIDS[@]}" -ge "$NUM_GPUS" ]]; then
    drain_batch || true  # continue even if some fail
  fi
  CURRENT_GPU=$(get_gpu_id "$GPU_SLOT")
  GPU_SLOT=$((GPU_SLOT + 1))
}

# Drain any remaining jobs at end of an experiment block
drain_remaining() {
  if [[ "${#BG_PIDS[@]}" -gt 0 ]]; then
    drain_batch || true
  fi
  GPU_SLOT=0
}

# ---------------------------------------------------------------------------
# Timestamp / banner
# ---------------------------------------------------------------------------
banner() {
  echo ""
  echo "============================================================"
  echo " $1"
  echo " $(date)"
  echo "============================================================"
}

# =============================================================================
# EXPERIMENT 1a: Coarse rho sweep
# =============================================================================
if ! is_done "1a"; then
banner "EXP 1a: Coarse rho sweep (9 rho x 3 seeds x 200 steps)"

RHO_VALUES=(0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)
SEEDS=(42 43 44)
COARSE_DIR="$PROJ_DIR_ROOT/results/sweep_coarse"
COARSE_LOG="$LOG_ROOT/exp1a"
mkdir -p "$COARSE_DIR" "$COARSE_LOG"

for R in "${RHO_VALUES[@]}"; do
  for S in "${SEEDS[@]}"; do
    TAG=$(printf 'rho%.2f_seed%d' "$R" "$S")
    RUN_DIR="$COARSE_DIR/$TAG"
    LOG_FILE="$COARSE_LOG/${TAG}.log"

    # Skip if already trained
    if [[ -f "$RUN_DIR/training_metrics.json" ]]; then
      echo "  [skip] $TAG (training_metrics.json exists)"
      continue
    fi

    schedule_slot
    launch_on_gpu "$CURRENT_GPU" "$TAG" "$LOG_FILE" \
      $PYTHON "$SCRIPT_DIR/train_rho_sweep.py" \
        --rho "$R" \
        --seed "$S" \
        --config "$CONFIG_RHO" \
        --output_dir "$RUN_DIR" \
        --max_steps 200 \
        --use_vllm
  done
done
drain_remaining

echo "[EXP 1a] Training done. Running evaluations..."

# Eval pass (can also be parallelized)
COARSE_EVAL_DIR="$PROJ_DIR_ROOT/results/rho_eval_coarse"
mkdir -p "$COARSE_EVAL_DIR"

for R in "${RHO_VALUES[@]}"; do
  for S in "${SEEDS[@]}"; do
    TAG=$(printf 'rho%.2f_seed%d' "$R" "$S")
    RUN_DIR="$COARSE_DIR/$TAG"
    EVAL_JSON="$COARSE_EVAL_DIR/eval_${TAG}.json"

    if [[ ! -d "$RUN_DIR" ]]; then
      echo "  [skip eval] $TAG (no checkpoint dir)"
      continue
    fi
    if [[ -f "$EVAL_JSON" ]]; then
      echo "  [skip eval] $TAG (eval exists)"
      continue
    fi

    schedule_slot
    launch_on_gpu "$CURRENT_GPU" "eval_${TAG}" "$COARSE_LOG/eval_${TAG}.log" \
      $PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
        --checkpoint_dir "$RUN_DIR" \
        --rho "$R" \
        --seed "$S" \
        --output_dir "$COARSE_EVAL_DIR" \
        --num_samples 500 \
        --eval_math
  done
done
drain_remaining

mark_done "1a"
fi

# =============================================================================
# Boundary detection from Exp 1a (used by Exp 1b and Exp 2)
# =============================================================================
# Reads coarse sweep results, finds rho_min_boundary and rho_max_boundary,
# and writes them to a small JSON for downstream consumption.
compute_boundaries() {
  local eval_dir="$1"
  local out_file="$2"

  $PYTHON - "$eval_dir" "$out_file" <<'PYEOF'
import json, glob, sys, os
eval_dir = sys.argv[1]
out_file = sys.argv[2]

results = []
for f in sorted(glob.glob(os.path.join(eval_dir, "eval_rho*.json"))):
    try:
        d = json.load(open(f))
        rho = d.get("rho", None)
        acc = d.get("gsm8k_accuracy", d.get("accuracy", None))
        if rho is not None and acc is not None:
            results.append({"rho": float(rho), "accuracy": float(acc), "file": f})
    except Exception:
        continue

if not results:
    # Fallback: cannot detect boundaries, use safe defaults
    info = {"rho_min_boundary": 0.3, "rho_max_boundary": 3.0,
            "best_rho": 1.0, "best_accuracy": 0.0, "status": "fallback"}
    json.dump(info, open(out_file, "w"), indent=2)
    print("[boundaries] No eval results found, using fallback defaults")
    sys.exit(0)

# Sort by rho
results.sort(key=lambda x: x["rho"])
best = max(results, key=lambda x: x["accuracy"])

# Identify boundary: rho values where accuracy drops below 80% of best
threshold = best["accuracy"] * 0.80
healthy = [r for r in results if r["accuracy"] >= threshold]
if healthy:
    rho_min_boundary = min(r["rho"] for r in healthy)
    rho_max_boundary = max(r["rho"] for r in healthy)
else:
    rho_min_boundary = best["rho"]
    rho_max_boundary = best["rho"]

info = {
    "rho_min_boundary": rho_min_boundary,
    "rho_max_boundary": rho_max_boundary,
    "best_rho": best["rho"],
    "best_accuracy": best["accuracy"],
    "threshold_80pct": threshold,
    "n_points": len(results),
    "status": "computed",
}
json.dump(info, open(out_file, "w"), indent=2)
print(f"[boundaries] rho_min={rho_min_boundary:.2f}, rho_max={rho_max_boundary:.2f}, "
      f"best_rho={best['rho']:.2f} (acc={best['accuracy']:.3f})")
PYEOF
}

BOUNDARY_FILE="$PROJ_DIR_ROOT/results/sweep_coarse/boundaries.json"
if [[ ! -f "$BOUNDARY_FILE" ]]; then
  echo ""
  echo "[boundaries] Computing phase boundaries from Exp 1a results..."
  compute_boundaries "$PROJ_DIR_ROOT/results/rho_eval_coarse" "$BOUNDARY_FILE"
fi

# Read boundary values for downstream experiments
read_boundary() {
  $PYTHON -c "import json; d=json.load(open('$BOUNDARY_FILE')); print(d.get('$1', '$2'))" 2>/dev/null || echo "$2"
}

RHO_MIN_BOUNDARY=$(read_boundary rho_min_boundary 0.3)
RHO_MAX_BOUNDARY=$(read_boundary rho_max_boundary 3.0)
BEST_RHO=$(read_boundary best_rho 1.0)

echo "[boundaries] rho_min_boundary=$RHO_MIN_BOUNDARY, rho_max_boundary=$RHO_MAX_BOUNDARY, best_rho=$BEST_RHO"

# =============================================================================
# EXPERIMENT 2: AdaBalance comparison (5 methods x 2 seeds)
# =============================================================================
if ! is_done "2"; then
banner "EXP 2: AdaBalance comparison (5 methods x 2 seeds, full training)"

ADA_SEEDS=(42 43)
ADA_DIR="$PROJ_DIR_ROOT/results/adabalance"
ADA_LOG="$LOG_ROOT/exp2"
ADA_EVAL_DIR="$PROJ_DIR_ROOT/results/rho_eval_adabalance"
mkdir -p "$ADA_DIR" "$ADA_LOG" "$ADA_EVAL_DIR"

# --- Method 1: vanilla (rho=1.0, standard GRPO) ---
for S in "${ADA_SEEDS[@]}"; do
  TAG="vanilla_rho1.0_seed${S}"
  RUN_DIR="$ADA_DIR/$TAG"
  if [[ -f "$RUN_DIR/training_metrics.json" ]]; then
    echo "  [skip] $TAG"; continue
  fi
  schedule_slot
  launch_on_gpu "$CURRENT_GPU" "$TAG" "$ADA_LOG/${TAG}.log" \
    $PYTHON "$SCRIPT_DIR/train_rho_sweep.py" \
      --rho 1.0 \
      --seed "$S" \
      --config "$CONFIG_RHO" \
      --output_dir "$RUN_DIR" \
      --use_vllm
done

# --- Method 2: best-static (from Exp 1a) ---
for S in "${ADA_SEEDS[@]}"; do
  TAG="best_static_rho${BEST_RHO}_seed${S}"
  RUN_DIR="$ADA_DIR/$TAG"
  if [[ -f "$RUN_DIR/training_metrics.json" ]]; then
    echo "  [skip] $TAG"; continue
  fi
  schedule_slot
  launch_on_gpu "$CURRENT_GPU" "$TAG" "$ADA_LOG/${TAG}.log" \
    $PYTHON "$SCRIPT_DIR/train_rho_sweep.py" \
      --rho "$BEST_RHO" \
      --seed "$S" \
      --config "$CONFIG_RHO" \
      --output_dir "$RUN_DIR" \
      --use_vllm
done

# --- Method 3: AdaBalance ---
for S in "${ADA_SEEDS[@]}"; do
  TAG="adabalance_seed${S}"
  RUN_DIR="$ADA_DIR/$TAG"
  if [[ -f "$RUN_DIR/training_metrics.json" ]]; then
    echo "  [skip] $TAG"; continue
  fi
  schedule_slot
  launch_on_gpu "$CURRENT_GPU" "$TAG" "$ADA_LOG/${TAG}.log" \
    $PYTHON "$SCRIPT_DIR/train_adabalance.py" \
      --seed "$S" \
      --config "$CONFIG_RHO" \
      --output_dir "$RUN_DIR" \
      --use_vllm
done

# --- Method 4: linear scheduler (rho decays from 2.0 to 0.5 over training) ---
for S in "${ADA_SEEDS[@]}"; do
  TAG="linear_sched_seed${S}"
  RUN_DIR="$ADA_DIR/$TAG"
  if [[ -f "$RUN_DIR/training_metrics.json" ]]; then
    echo "  [skip] $TAG"; continue
  fi
  schedule_slot
  launch_on_gpu "$CURRENT_GPU" "$TAG" "$ADA_LOG/${TAG}.log" \
    $PYTHON "$SCRIPT_DIR/train_adabalance.py" \
      --seed "$S" \
      --config "$CONFIG_RHO" \
      --output_dir "$RUN_DIR" \
      --rho_init 2.0 \
      --tau 0.0 \
      --use_vllm
done

# --- Method 5: GTPO (aggressive rho, stress test) ---
for S in "${ADA_SEEDS[@]}"; do
  TAG="gtpo_seed${S}"
  RUN_DIR="$ADA_DIR/$TAG"
  if [[ -f "$RUN_DIR/training_metrics.json" ]]; then
    echo "  [skip] $TAG"; continue
  fi
  schedule_slot
  launch_on_gpu "$CURRENT_GPU" "$TAG" "$ADA_LOG/${TAG}.log" \
    $PYTHON "$SCRIPT_DIR/train_adabalance.py" \
      --seed "$S" \
      --config "$CONFIG_RHO" \
      --output_dir "$RUN_DIR" \
      --rho_init 0.5 \
      --K 100 \
      --tau 0.05 \
      --use_vllm
done

drain_remaining

# Eval all Exp 2 methods
echo "[EXP 2] Training done. Running evaluations..."
for dir_entry in "$ADA_DIR"/*/; do
  [[ ! -d "$dir_entry" ]] && continue
  TAG=$(basename "$dir_entry")
  EVAL_JSON="$ADA_EVAL_DIR/eval_${TAG}.json"
  if [[ -f "$EVAL_JSON" ]]; then
    echo "  [skip eval] $TAG"; continue
  fi
  if [[ ! -f "$dir_entry/training_metrics.json" ]] && [[ ! -d "$dir_entry/checkpoint-"* ]]; then
    echo "  [skip eval] $TAG (no training output)"; continue
  fi

  schedule_slot
  launch_on_gpu "$CURRENT_GPU" "eval_${TAG}" "$ADA_LOG/eval_${TAG}.log" \
    $PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
      --checkpoint_dir "$dir_entry" \
      --seed 42 \
      --output_dir "$ADA_EVAL_DIR" \
      --num_samples 500 \
      --eval_math
done
drain_remaining

mark_done "2"
fi

# =============================================================================
# EXPERIMENT 1b: Fine sweep near phase boundaries
# =============================================================================
if ! is_done "1b"; then
banner "EXP 1b: Fine rho sweep near boundaries (10 points, 3 seeds, full 2 epochs)"

FINE_DIR="$PROJ_DIR_ROOT/results/sweep_fine"
FINE_LOG="$LOG_ROOT/exp1b"
FINE_EVAL_DIR="$PROJ_DIR_ROOT/results/rho_eval_fine"
mkdir -p "$FINE_DIR" "$FINE_LOG" "$FINE_EVAL_DIR"

# Generate 10 fine-grained rho values clustered around boundaries
FINE_RHO_VALUES=($($PYTHON - "$RHO_MIN_BOUNDARY" "$RHO_MAX_BOUNDARY" <<'PYEOF'
import sys, numpy as np
rho_min = float(sys.argv[1])
rho_max = float(sys.argv[2])

points = set()
# 5 points near rho_min boundary
for delta in np.linspace(-0.15, 0.15, 5):
    v = round(max(0.05, rho_min + delta), 3)
    points.add(v)
# 5 points near rho_max boundary
for delta in np.linspace(-0.3, 0.3, 5):
    v = round(max(0.05, rho_max + delta), 3)
    points.add(v)

# Deduplicate and sort; ensure exactly 10 (pad with midpoints if needed)
points = sorted(points)
while len(points) < 10:
    mid = round((rho_min + rho_max) / 2 + len(points) * 0.01, 3)
    if mid not in points:
        points.append(mid)
    points = sorted(points)

for v in points[:10]:
    print(v)
PYEOF
))

echo "  Fine rho values: ${FINE_RHO_VALUES[*]}"
FINE_SEEDS=(42 43 44)

for R in "${FINE_RHO_VALUES[@]}"; do
  for S in "${FINE_SEEDS[@]}"; do
    TAG=$(printf 'rho%.3f_seed%d' "$R" "$S")
    RUN_DIR="$FINE_DIR/$TAG"
    LOG_FILE="$FINE_LOG/${TAG}.log"

    if [[ -f "$RUN_DIR/training_metrics.json" ]]; then
      echo "  [skip] $TAG"; continue
    fi

    schedule_slot
    launch_on_gpu "$CURRENT_GPU" "$TAG" "$LOG_FILE" \
      $PYTHON "$SCRIPT_DIR/train_rho_sweep.py" \
        --rho "$R" \
        --seed "$S" \
        --config "$CONFIG_RHO" \
        --output_dir "$RUN_DIR" \
        --num_train_epochs 2 \
        --use_vllm
  done
done
drain_remaining

echo "[EXP 1b] Training done. Running evaluations..."
for R in "${FINE_RHO_VALUES[@]}"; do
  for S in "${FINE_SEEDS[@]}"; do
    TAG=$(printf 'rho%.3f_seed%d' "$R" "$S")
    RUN_DIR="$FINE_DIR/$TAG"
    EVAL_JSON="$FINE_EVAL_DIR/eval_${TAG}.json"

    if [[ ! -d "$RUN_DIR" ]]; then continue; fi
    if [[ -f "$EVAL_JSON" ]]; then
      echo "  [skip eval] $TAG"; continue
    fi

    schedule_slot
    launch_on_gpu "$CURRENT_GPU" "eval_${TAG}" "$FINE_LOG/eval_${TAG}.log" \
      $PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
        --checkpoint_dir "$RUN_DIR" \
        --rho "$R" \
        --seed "$S" \
        --output_dir "$FINE_EVAL_DIR" \
        --num_samples 500 \
        --eval_math
  done
done
drain_remaining

mark_done "1b"
fi

# =============================================================================
# EXPERIMENT 3: Robustness / i.i.d. violation test
# =============================================================================
if ! is_done "3"; then
banner "EXP 3: Robustness test (4 bins x 3 rho x 2 seeds = 24 runs)"

ROBUST_DIR="$PROJ_DIR_ROOT/results/robustness"
ROBUST_LOG="$LOG_ROOT/exp3"
mkdir -p "$ROBUST_DIR" "$ROBUST_LOG"

# Robustness test is CPU-only simulation, but we run it anyway for completeness.
# Use 3 rho values: boundary-low, optimal, boundary-high
ROBUST_RHOS="$RHO_MIN_BOUNDARY $(printf '%.2f' "$BEST_RHO") $RHO_MAX_BOUNDARY"
LOG_FILE="$ROBUST_LOG/robustness.log"

echo "  rho values for robustness: $ROBUST_RHOS"

$PYTHON "$SCRIPT_DIR/run_robustness_test.py" \
  --output_dir "$ROBUST_DIR" \
  --rho_values $ROBUST_RHOS \
  --seeds 42 43 \
  > "$LOG_FILE" 2>&1

if [[ $? -ne 0 ]]; then
  echo "  [WARN] Robustness test failed; see $LOG_FILE"
  FAILED_JOBS=$((FAILED_JOBS + 1))
else
  echo "  [OK] Robustness results in $ROBUST_DIR"
fi

mark_done "3"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo " ALL TARGETED EXPERIMENTS COMPLETE"
echo " $(date)"
echo "============================================================"
echo "  Coarse sweep    : $PROJ_DIR_ROOT/results/sweep_coarse/"
echo "  Coarse eval     : $PROJ_DIR_ROOT/results/rho_eval_coarse/"
echo "  Boundaries      : $BOUNDARY_FILE"
echo "  AdaBalance      : $PROJ_DIR_ROOT/results/adabalance/"
echo "  AdaBalance eval : $PROJ_DIR_ROOT/results/rho_eval_adabalance/"
echo "  Fine sweep      : $PROJ_DIR_ROOT/results/sweep_fine/"
echo "  Fine eval       : $PROJ_DIR_ROOT/results/rho_eval_fine/"
echo "  Robustness      : $PROJ_DIR_ROOT/results/robustness/"
echo "  Logs            : $LOG_ROOT/"
echo ""
if [[ "$FAILED_JOBS" -gt 0 ]]; then
  echo "  WARNING: $FAILED_JOBS job(s) failed. Check logs for details."
  echo ""
fi
echo "  Next steps:"
echo "    python scripts/build_phase_diagram.py --results_dir results/rho_eval_coarse --output_dir results/analysis"
echo "    python scripts/plot_phase_diagram.py --input_dir results/analysis --output_dir results/figures"
echo "============================================================"
