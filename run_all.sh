#!/bin/bash
# run_all.sh — Full pipeline:
#   1. Run LR range test for every experiment (run_lr_finder.sh)
#   2. Parse the suggested LR per experiment
#   3. Train each experiment with its optimal LR

set -e

EPOCHS=20
BATCH_SIZE=64
SEED=42
PROMPT_VALUES=(1 2 4 8 16 32 64)

LOGDIR="logs"
CURVEDIR="curves"
mkdir -p "$LOGDIR" "$CURVEDIR"

SUMMARY_FILE="lr_finder_results/suggested_lrs.txt"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: LR range test
# ──────────────────────────────────────────────────────────────────────────────

echo "=============================================="
echo " Step 1 — LR Range Test (10 epochs each)"
echo "=============================================="
bash run_lr_finder.sh

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "ERROR: $SUMMARY_FILE not found after LR finder." >&2
    exit 1
fi

echo ""
echo "LR finder complete. Suggested LRs:"
cat "$SUMMARY_FILE"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: parse suggested LR for a given label from suggested_lrs.txt
# Usage: get_lr "linear"
#        get_lr "query_token  num_prompt=4"
# ──────────────────────────────────────────────────────────────────────────────

get_lr() {
    local LABEL="$1"
    local LR
    LR=$(grep -F "$LABEL" "$SUMMARY_FILE" | tail -1 | grep -oP 'suggested_lr=\K[^\s]+')
    if [ -z "$LR" ]; then
        echo "1e-3"  # fallback
        echo "WARNING: LR not found for '$LABEL', using default 1e-3" >&2
    else
        echo "$LR"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one training experiment
# ──────────────────────────────────────────────────────────────────────────────

run_experiment() {
    local MODE="$1"
    local NP="$2"      # empty string if not applicable
    local LR="$3"
    local LOG="$4"
    local LABEL="$5"

    echo ""
    echo ">>> Training: $LABEL  (lr=$LR)"

    local EXTRA_ARGS=""
    [ -n "$NP" ] && EXTRA_ARGS="--num-prompt $NP"

    python train_v2.py \
        --mode "$MODE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOGDIR/$LOG"

    # Move loss curve to curves/
    local CURVE="loss_${MODE}"
    [ -n "$NP" ] && CURVE="${CURVE}_np${NP}"
    mv -f "${CURVE}.png" "${CURVE}.csv" "$CURVEDIR/" 2>/dev/null || true
}

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Train each experiment with its optimal LR
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo " Step 2 — Training with optimal LRs"
echo "=============================================="

# ── linear ──
LR_LINEAR=$(get_lr "linear")
run_experiment "linear" "" "$LR_LINEAR" "linear.log" "linear"

# ── query_token ──
for NP in "${PROMPT_VALUES[@]}"; do
    LR=$(get_lr "query_token  num_prompt=$NP")
    run_experiment "query_token" "$NP" "$LR" "query_token_np${NP}.log" "query_token  num_prompt=$NP"
done

# ── subject_prompt ──
for NP in "${PROMPT_VALUES[@]}"; do
    LR=$(get_lr "subject_prompt  num_prompt=$NP")
    run_experiment "subject_prompt" "$NP" "$LR" "subject_prompt_np${NP}.log" "subject_prompt  num_prompt=$NP"
done

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Summary
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo " Summary — Test Results"
echo "=============================================="
printf "%-40s  %-10s  %-8s  %-8s  %s\n" "Experiment" "bal_acc" "kappa" "auroc" "lr"
printf "%-40s  %-10s  %-8s  %-8s  %s\n" "----------" "-------" "-----" "-----" "--"

for LOG in "$LOGDIR"/*.log; do
    NAME=$(basename "$LOG" .log)
    BACC=$(grep  "Balanced Accuracy" "$LOG" | tail -1 | awk '{print $NF}')
    KAPPA=$(grep "Cohen's Kappa"     "$LOG" | tail -1 | awk '{print $NF}')
    AUROC=$(grep "AUROC"             "$LOG" | tail -1 | awk '{print $NF}')
    USED_LR=$(grep -- "--lr"         "$LOG" | head -1 | grep -oP '\-\-lr \K\S+' || echo "?")
    printf "%-40s  %-10s  %-8s  %-8s  %s\n" "$NAME" "$BACC" "$KAPPA" "$AUROC" "$USED_LR"
done

echo ""
echo "Loss curves : $CURVEDIR/"
echo "LR finder   : lr_finder_results/"
echo "Logs        : $LOGDIR/"

# ── Comparison plot (if available) ──
if [ -f "plot_comparison.py" ]; then
    echo ""
    echo ">>> Generating comparison plot..."
    python plot_comparison.py
fi
