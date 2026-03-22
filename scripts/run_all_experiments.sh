#!/usr/bin/env bash
# =============================================================================
# Master pipeline: RLBalance phase diagram + HalluZero zero-score reshaping
# =============================================================================
# Usage:
#   bash scripts/run_all_experiments.sh           # full grid (long-running)
#   bash scripts/run_all_experiments.sh --quick   # smoke / reduced grids
#
# Environment:
#   PROJ_DIR_ROOT  — repository root (set automatically if unset)
#   QUICK=1        — same as --quick
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ_DIR_ROOT="${PROJ_DIR_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJ_DIR_ROOT"

QUICK="${QUICK:-0}"
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
  shift
fi

# --- Virtualenv (setup.sh creates .venv under project root) ---
if [[ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PROJ_DIR_ROOT/.venv/bin/activate"
  echo "[env] Activated venv: $PROJ_DIR_ROOT/.venv"
else
  echo "[warn] No .venv found at $PROJ_DIR_ROOT/.venv — using current Python"
fi

# --- GPU utilities (monorepo shared copy, then local fallback) ---
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../../_shared/gpu_utils.sh" 2>/dev/null \
  || source "$SCRIPT_DIR/gpu_utils.sh"

auto_setup

PYTHON="${PYTHON:-python3}"
CONFIG_SWEEP="$PROJ_DIR_ROOT/configs/sweep_grid.yaml"
CONFIG_HALLU="$PROJ_DIR_ROOT/configs/grpo_9b.yaml"
CKPT_ROOT="$PROJ_DIR_ROOT/checkpoints"
PHASE_EVAL_DIR="$PROJ_DIR_ROOT/results/phase_diagram"
ZERO_SWEEP_ROOT="$PROJ_DIR_ROOT/results/zero_score_sweep"
ANALYSIS_DIR="$PROJ_DIR_ROOT/results/analysis"
MODEL_9B="Qwen/Qwen3.5-9B"
MODEL_27B="Qwen/Qwen3.5-27B"

# Training / eval shortcuts
if [[ "$QUICK" == "1" ]]; then
  echo "============================================"
  echo " QUICK MODE: reduced steps, grids, and eval"
  echo "============================================"
  SWEEP_MAX_STEPS=(--max_steps 40)
  BASELINE_MAX_STEPS=(--max_steps 40)
  HALLU_EPOCHS=(--num_epochs 1)
  EVAL_NUM_SAMPLES=(--num_samples 64)
  EVAL_GSM8K=(--gsm8k_samples 64)
  EVAL_MATH=(--math_samples 32)
  PHASE2_ALPHAS=(0.3 0.5 0.7)
  PHASE2_BETAS=(0.0 1.0 2.0)
  PHASE2_SEEDS=(42)
  PHASE3_SEEDS=(42)
  # One hyperparameter value per strategy
  CLIP_FACTORS=(0.1)
  TEMP_BOOSTS=(1.5)
  CURR_WARMS=(500)
  RELABEL_EPS=(0.01)
  PHASE8_EVAL_GSM8K=(--gsm8k_samples 32)
  PHASE8_EVAL_MATH=(--math_samples 16)
  SKIP_27B_VALIDATION=1
else
  SWEEP_MAX_STEPS=()
  BASELINE_MAX_STEPS=()
  HALLU_EPOCHS=()
  EVAL_NUM_SAMPLES=(--num_samples 500)
  EVAL_GSM8K=()
  EVAL_MATH=(--math_samples 500)
  PHASE2_ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
  PHASE2_BETAS=(0.0 0.25 0.5 1.0 2.0)
  PHASE2_SEEDS=(42 43 44)
  PHASE3_SEEDS=(42 43)
  CLIP_FACTORS=(0.05 0.1 0.2)
  TEMP_BOOSTS=(1.2 1.5 2.0)
  CURR_WARMS=(200 500 1000)
  RELABEL_EPS=(0.005 0.01 0.02)
  PHASE8_EVAL_GSM8K=()
  PHASE8_EVAL_MATH=(--math_samples 500)
  SKIP_27B_VALIDATION=0
fi

echo "============================================"
echo " PROJ_DIR_ROOT = $PROJ_DIR_ROOT"
echo " QUICK         = $QUICK"
echo "============================================"

# -----------------------------------------------------------------------------
# Phase 0 — Model download (Hugging Face Hub)
# -----------------------------------------------------------------------------
echo ""
if [[ "$QUICK" == "1" ]]; then
  echo ">>> Phase 0: Model download ($MODEL_9B only; --quick skips 27B prefetch)"
  MODELS_TO_FETCH=("$MODEL_9B")
else
  echo ">>> Phase 0: Model download ($MODEL_9B + $MODEL_27B)"
  MODELS_TO_FETCH=("$MODEL_9B" "$MODEL_27B")
fi
for mid in "${MODELS_TO_FETCH[@]}"; do
  $PYTHON -c "import os; os.environ.setdefault('HF_ENDPOINT', os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')); from huggingface_hub import snapshot_download; print('[snapshot_download]', '${mid}'); snapshot_download(repo_id='${mid}')"
done

# -----------------------------------------------------------------------------
# Phase 1 — Baseline GRPO (α,β parameterization, reference point α=0.5, β=1.0)
# Standard reference in this codebase: symmetric reward scaling without an
# explicit sweep — same trainer as the diagram study, fixed at the center of
# the (α,β) plane.
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 1: Baseline GRPO (train_grpo_sweep.py, α=0.5, β=1.0, seed 42)"
BASELINE_DIR="$CKPT_ROOT/baseline_grpo_alpha0.50_beta1.00_seed42"
if [[ ! -f "$BASELINE_DIR/training_metrics.json" ]]; then
  $PYTHON "$SCRIPT_DIR/train_grpo_sweep.py" \
    --positive_ratio 0.5 \
    --negative_weight 1.0 \
    --seed 42 \
    --config "$CONFIG_SWEEP" \
    --output_dir "$BASELINE_DIR" \
    "${BASELINE_MAX_STEPS[@]}"
else
  echo "    (skip) $BASELINE_DIR already has training_metrics.json"
fi

$PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
  --checkpoint_dir "$BASELINE_DIR" \
  --positive_ratio 0.5 \
  --negative_weight 1.0 \
  --seed 42 \
  --output_dir "$PHASE_EVAL_DIR" \
  "${EVAL_NUM_SAMPLES[@]}" \
  --eval_math

# -----------------------------------------------------------------------------
# Phase 2 — Phase diagram sweep (α × β × seeds)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 2: Phase diagram sweep"
for A in "${PHASE2_ALPHAS[@]}"; do
  for B in "${PHASE2_BETAS[@]}"; do
    for S in "${PHASE2_SEEDS[@]}"; do
      TAG="$(printf 'alpha%.2f_beta%.2f_seed%d' "$A" "$B" "$S")"
      RUN_DIR="$CKPT_ROOT/$TAG"
      if [[ ! -f "$RUN_DIR/training_metrics.json" ]]; then
        echo "    train  $TAG"
        $PYTHON "$SCRIPT_DIR/train_grpo_sweep.py" \
          --positive_ratio "$A" \
          --negative_weight "$B" \
          --seed "$S" \
          --config "$CONFIG_SWEEP" \
          --output_dir "$RUN_DIR" \
          "${SWEEP_MAX_STEPS[@]}"
      else
        echo "    (skip train) $TAG"
      fi
      EVAL_JSON="$PHASE_EVAL_DIR/eval_${TAG}.json"
      if [[ ! -f "$EVAL_JSON" ]]; then
        echo "    eval   $TAG"
        $PYTHON "$SCRIPT_DIR/eval_phase_point.py" \
          --checkpoint_dir "$RUN_DIR" \
          --positive_ratio "$A" \
          --negative_weight "$B" \
          --seed "$S" \
          --output_dir "$PHASE_EVAL_DIR" \
          "${EVAL_NUM_SAMPLES[@]}" \
          --eval_math
      else
        echo "    (skip eval) $TAG"
      fi
    done
  done
done

# -----------------------------------------------------------------------------
# Phase 3 — Zero-score strategy sweep (4 strategies × hyperparams × seeds)
# Path layout includes the substring 'sweep' for run_diagnostic_analysis.py
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 3: Zero-score strategy sweep (HalluZero)"

run_hallu() {
  local strategy="$1"
  local out_dir="$2"
  shift 2
  if [[ -f "$out_dir/training_metrics.json" ]]; then
    echo "    (skip train) $out_dir"
  else
    echo "    train  $out_dir"
    $PYTHON "$SCRIPT_DIR/train_grpo_halluzero.py" \
      --config_path "$CONFIG_HALLU" \
      --output_dir "$out_dir" \
      --zero_score_strategy "$strategy" \
      "$@" \
      "${HALLU_EPOCHS[@]}"
  fi
  if [[ ! -f "$out_dir/eval/summary.json" ]]; then
    echo "    eval   $out_dir"
    $PYTHON "$SCRIPT_DIR/eval_halluzero.py" \
      --model_path "$out_dir" \
      --output_dir "$out_dir/eval" \
      "${EVAL_GSM8K[@]}" \
      "${EVAL_MATH[@]}"
  else
    echo "    (skip eval) $out_dir"
  fi
}

for S in "${PHASE3_SEEDS[@]}"; do
  for CF in "${CLIP_FACTORS[@]}"; do
    TAG="cf_$(printf '%.2f' "$CF")"
    run_hallu clip "$ZERO_SWEEP_ROOT/sweep/clip/${TAG}/seed_${S}" \
      --seed "$S" \
      --clip_factor "$CF"
  done
  for TB in "${TEMP_BOOSTS[@]}"; do
    TAG="tb_$(printf '%.2f' "$TB")"
    run_hallu temperature "$ZERO_SWEEP_ROOT/sweep/temperature/${TAG}/seed_${S}" \
      --seed "$S" \
      --temperature_boost "$TB"
  done
  for CW in "${CURR_WARMS[@]}"; do
    TAG="warm_${CW}"
    run_hallu curriculum "$ZERO_SWEEP_ROOT/sweep/curriculum/${TAG}/seed_${S}" \
      --seed "$S" \
      --curriculum_warmup_steps "$CW"
  done
  for RE in "${RELABEL_EPS[@]}"; do
    TAG="eps_$(printf '%.4f' "$RE" | tr '.' 'p')"
    run_hallu relabel "$ZERO_SWEEP_ROOT/sweep/relabel/${TAG}/seed_${S}" \
      --seed "$S" \
      --relabel_epsilon "$RE"
  done
done

# -----------------------------------------------------------------------------
# Phase 4 — Phase diagram + collapse-zone analysis
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 4: build_phase_diagram.py"
$PYTHON "$SCRIPT_DIR/build_phase_diagram.py" \
  --results_dir "$PHASE_EVAL_DIR" \
  --checkpoint_dir "$CKPT_ROOT" \
  --curriculum_dir "$PROJ_DIR_ROOT/results/curriculum" \
  --output_dir "$ANALYSIS_DIR"

# -----------------------------------------------------------------------------
# Phase 5 — Gradient analysis (zero vs nonzero)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 5: analyze_gradients.py"
$PYTHON "$SCRIPT_DIR/analyze_gradients.py" \
  --model_path "$MODEL_9B" \
  --output_dir "$PROJ_DIR_ROOT/results/gradient_analysis" \
  --num_samples "$( [[ "$QUICK" == "1" ]] && echo 48 || echo 200 )"

# -----------------------------------------------------------------------------
# Phase 6 — Curriculum strategies (α/β schedules)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 6: run_curriculum_strategies.py"
$PYTHON "$SCRIPT_DIR/run_curriculum_strategies.py" \
  --config "$CONFIG_SWEEP" \
  --output_dir "$PROJ_DIR_ROOT/results/curriculum" \
  --best_alpha 0.5 \
  --best_beta 1.0 \
  --total_steps_estimate "$( [[ "$QUICK" == "1" ]] && echo 120 || echo 500 )"

# -----------------------------------------------------------------------------
# Phase 7 — Diagnostic figures / tables
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 7: run_diagnostic_analysis.py"
$PYTHON "$SCRIPT_DIR/run_diagnostic_analysis.py" \
  --results_dir "$ZERO_SWEEP_ROOT" \
  --output_dir "$ANALYSIS_DIR/diagnostics"

# -----------------------------------------------------------------------------
# Phase 8 — 27B validation (weights prefetched in Phase 0 when not --quick)
# -----------------------------------------------------------------------------
echo ""
if [[ "${SKIP_27B_VALIDATION:-0}" == "1" ]]; then
  echo ">>> Phase 8: skipped (--quick; avoids large 27B download / eval)"
else
  echo ">>> Phase 8: 27B validation (eval_halluzero on base $MODEL_27B)"
  $PYTHON "$SCRIPT_DIR/eval_halluzero.py" \
    --model_path "$MODEL_27B" \
    --output_dir "$PROJ_DIR_ROOT/results/validation_27b_base" \
    "${PHASE8_EVAL_GSM8K[@]}" \
    "${PHASE8_EVAL_MATH[@]}"
fi

echo ""
echo "============================================"
echo " All phases finished."
echo "  Phase diagram eval : $PHASE_EVAL_DIR"
echo "  Aggregated analysis: $ANALYSIS_DIR"
echo "  Zero-score sweep   : $ZERO_SWEEP_ROOT"
echo "============================================"

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
