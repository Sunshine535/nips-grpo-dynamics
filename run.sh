#!/bin/bash
# =============================================================================
# RUN experiments only (environment assumed ready)
# Install deps first:  bash setup.sh
# Then run:            bash run.sh
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# --- HF cache: shared across projects in parent dir ---
export HF_HOME="${HF_HOME:-$(dirname "$PROJ_DIR")/.cache/hf}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

echo "============================================================"
echo " Starting full experiment pipeline"
echo " Project: $(basename "$PROJ_DIR")"
echo " Time:    $(date)"
echo "============================================================"

# Activate venv if present; otherwise use system Python
if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
    echo "[env] Activated venv: $PROJ_DIR/.venv"
else
    echo "[env] No .venv found, using system Python"
fi

# Quick dependency check
python3 -c "import torch, transformers, datasets, trl, accelerate, peft" 2>/dev/null || {
    echo "[ERROR] Missing dependencies. Run: bash setup.sh"
    exit 1
}

echo ""
echo "Running all experiments (full production mode)..."
echo "  Log file: $PROJ_DIR/run.log"
echo "  To run in background: nohup bash run.sh > run.log 2>&1 &"
echo ""

bash scripts/run_all_experiments.sh 2>&1 | tee "$PROJ_DIR/run.log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo " Pipeline completed successfully!"
    echo " Results: $PROJ_DIR/results/"
    echo " Log:     $PROJ_DIR/run.log"
    echo "============================================================"
else
    echo "============================================================"
    echo " Pipeline failed (exit code: $EXIT_CODE)"
    echo " Check log: $PROJ_DIR/run.log"
    echo " To resume: bash run.sh (completed phases are skipped)"
    echo "============================================================"
    exit $EXIT_CODE
fi
