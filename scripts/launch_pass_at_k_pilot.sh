#!/bin/bash
# Pass@k Pilot Experiment — RLVR Ceiling Hypothesis
# Run on OpenBayes server (port 43038)
#
# Tests: does CE Replay preserve pass@k (reasoning diversity)
# while pure GRPO narrows it?
#
# Expected runtime: ~1-2 hours total on 1 GPU
set -euo pipefail

OUTDIR="results/pass_at_k_pilot"
N_QUESTIONS=200
K=25
TEMPERATURE=1.0
BATCH_SIZE=8

mkdir -p "$OUTDIR"

# --- Step 0: Find checkpoint paths ---
echo "=== Finding checkpoints ==="

# Auto-discover adapter paths from sage_minimal_abc
find_adapter() {
    local pattern="$1"
    local found
    found=$(find results/sage_minimal_abc -path "*${pattern}*/checkpoint-final/adapter_config.json" -printf '%h\n' 2>/dev/null | head -1)
    if [ -z "$found" ]; then
        # fallback: look for any checkpoint-final with the pattern
        found=$(find results -path "*${pattern}*checkpoint-final/adapter_config.json" -printf '%h\n' 2>/dev/null | head -1)
    fi
    echo "$found"
}

ADAPTER_B=$(find_adapter "tasa_only_seed42")
ADAPTER_D=$(find_adapter "positive_ce_only_seed42")

echo "  Base model: Qwen/Qwen3.5-9B"
echo "  B adapter:  ${ADAPTER_B:-NOT FOUND}"
echo "  D adapter:  ${ADAPTER_D:-NOT FOUND}"

if [ -z "$ADAPTER_B" ] || [ -z "$ADAPTER_D" ]; then
    echo ""
    echo "WARNING: Could not auto-detect checkpoint paths."
    echo "Listing available checkpoints:"
    find results -name "adapter_config.json" -printf '  %h\n' 2>/dev/null | sort
    echo ""
    echo "Set ADAPTER_B and ADAPTER_D manually and re-run."
    echo "Example:"
    echo "  ADAPTER_B=results/sage_minimal_abc/B_tasa_only/sage_tasa_only_seed42/checkpoint-final"
    echo "  ADAPTER_D=results/sage_minimal_abc/D_positive_ce_only/sage_positive_ce_only_seed42/checkpoint-final"
    exit 1
fi

# --- Step 1: Base model ---
echo ""
echo "=== [1/3] Base model pass@k ==="
python3 scripts/measure_pass_at_k.py \
    --n-questions "$N_QUESTIONS" --k "$K" --temperature "$TEMPERATURE" \
    --batch-size "$BATCH_SIZE" --selection first_n \
    --out "$OUTDIR/base.json"

# --- Step 2: GRPO-only (B = TASA-only, no replay) ---
echo ""
echo "=== [2/3] GRPO-only (B) pass@k ==="
python3 scripts/measure_pass_at_k.py \
    --adapter "$ADAPTER_B" \
    --n-questions "$N_QUESTIONS" --k "$K" --temperature "$TEMPERATURE" \
    --batch-size "$BATCH_SIZE" --selection first_n \
    --out "$OUTDIR/B_tasa_only_seed42.json"

# --- Step 3: GRPO + CE Replay (D) ---
echo ""
echo "=== [3/3] GRPO+CE Replay (D) pass@k ==="
python3 scripts/measure_pass_at_k.py \
    --adapter "$ADAPTER_D" \
    --n-questions "$N_QUESTIONS" --k "$K" --temperature "$TEMPERATURE" \
    --batch-size "$BATCH_SIZE" --selection first_n \
    --out "$OUTDIR/D_positive_ce_only_seed42.json"

# --- Step 4: Analyze ---
echo ""
echo "=== Analysis ==="
python3 scripts/analyze_pass_at_k.py \
    --base "$OUTDIR/base.json" \
    --grpo "$OUTDIR/B_tasa_only_seed42.json" \
    --replay "$OUTDIR/D_positive_ce_only_seed42.json" \
    --out "$OUTDIR/analysis.json"

echo ""
echo "Done. Results in $OUTDIR/"
