#!/usr/bin/env bash
cd ~/nips-workspace/nips-grpo-dynamics

# Wait for confounders to complete (8 total)
while true; do
    CONF=$(find results/confounder/ -name "training_metrics.json" 2>/dev/null | wc -l)
    echo "$(date '+%H:%M') confounders=$CONF/8"
    if [[ "$CONF" -ge 8 ]]; then
        echo "=== All 8 confounders done! ==="
        break
    fi
    sleep 300
done

# Kill the old pipeline (it would start method comparison next)
OLD_PID=34503
if ps -p $OLD_PID > /dev/null 2>&1; then
    echo "Killing old pipeline (PID $OLD_PID)..."
    kill $OLD_PID 2>/dev/null
    sleep 10
    kill -9 $OLD_PID 2>/dev/null
    # Also kill any remaining training processes
    ps aux | grep "accelerate\|train_rho" | grep -v grep | grep -v watchdog | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 5
fi

echo "Launching long-horizon + method comparison + full eval..."
bash run_longhorizon.sh 2>&1 | tee results/logs/longhorizon_pipeline.log

echo "=== WATCHDOG COMPLETE $(date) ==="
