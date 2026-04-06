#!/usr/bin/env bash
# Auto-sync results to GitHub every 30 minutes
cd "$(dirname "$0")"
PREV_COUNT=0
while true; do
    sleep 1800
    
    # Count total result files
    COUNT=$(find results/qwen35/ results/synthetic_validation/ results/data_collapse/ \
        results/early_warning/ results/theory_unification/ \
        -name "*.json" -size -1M 2>/dev/null | wc -l)
    
    if [[ "$COUNT" -gt "$PREV_COUNT" ]]; then
        echo "[$(date '+%m-%d %H:%M')] New results detected ($PREV_COUNT → $COUNT). Syncing..."
        
        # Stage new results
        find results/ -name "training_metrics.json" -size -1M -newer .git/refs/heads/main 2>/dev/null | xargs git add 2>/dev/null
        find results/ -name "eval_*.json" -size -1M -newer .git/refs/heads/main 2>/dev/null | xargs git add 2>/dev/null
        find results/ -name "*telemetry*.json" -size -1M -newer .git/refs/heads/main 2>/dev/null | xargs git add 2>/dev/null
        find results/ -name "*logs*.json" -size -5M -newer .git/refs/heads/main 2>/dev/null | xargs git add 2>/dev/null
        find results/ -name "*stats*.json" -size -1M -newer .git/refs/heads/main 2>/dev/null | xargs git add 2>/dev/null
        find results/logs/ -name "*.log" -size -5M -newer .git/refs/heads/main 2>/dev/null | xargs git add 2>/dev/null
        git add AUTO_REVIEW.md REVIEW_STATE.json 2>/dev/null
        
        # Commit and push if there are changes
        if git diff --cached --quiet 2>/dev/null; then
            echo "  No staged changes"
        else
            STAGED=$(git diff --cached --name-only | wc -l)
            git commit -m "auto: sync $STAGED result files (Qwen3.5-9B experiments in progress)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>&1 | tail -1
            git push origin main 2>&1 | tail -1
            echo "  Pushed $STAGED files"
        fi
        PREV_COUNT=$COUNT
    else
        echo "[$(date '+%m-%d %H:%M')] No new results ($COUNT files)"
    fi
done
