#!/bin/bash
# Full pipeline: Wave 14 eval → Wave 14b (phase diagram) → Wave 15 (HalluZero)
# Run with: nohup bash launch_pipeline.sh > pipeline.log 2>&1 &
set -e
cd "$(dirname "$0")"

echo "=== $(date) === Pipeline start ==="

# Step 1: Wait for Wave 14 evals (already launched externally)
echo "[step1] Waiting for Wave 14 evals to finish..."
while true; do
    n_done=0
    for f in results/wave14_500step/evals/eval_seed{42,43,44}.json \
             results/wave14_phase_diagram/evals/eval_a{0.1,0.3,0.5,0.7,0.9}_b1.0.json; do
        [ -f "$f" ] && n_done=$((n_done + 1))
    done
    echo "  $(date +%H:%M:%S) evals done: $n_done / 8"
    [ $n_done -ge 8 ] && break
    sleep 60
done
echo "[step1] Wave 14 evals complete!"

# Quick summary
echo "=== Wave 14 Results ==="
for f in results/wave14_500step/evals/eval_seed*.json; do
    acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['accuracy']:.3f}\")" 2>/dev/null || echo "ERR")
    echo "  $(basename $f .json): acc=$acc"
done
for f in results/wave14_phase_diagram/evals/eval_a*_b1.0.json; do
    acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['accuracy']:.3f}\")" 2>/dev/null || echo "ERR")
    echo "  $(basename $f .json): acc=$acc"
done

# Step 2: Wave 14b — remaining 15 phase diagram points
echo "=== $(date) === Step 2: Wave 14b (phase diagram remaining) ==="
bash launch_wave14b.sh
echo "[step2] Wave 14b complete!"

# Step 3: Wave 15 — HalluZero 4 strategies
echo "=== $(date) === Step 3: Wave 15 (HalluZero) ==="
bash launch_wave15_halluzero.sh
echo "[step3] Wave 15 complete!"

echo "=== $(date) === Pipeline finished ==="
