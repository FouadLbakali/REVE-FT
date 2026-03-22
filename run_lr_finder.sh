#!/bin/bash
# Runs the LR range test for every experiment defined in run_comparison.sh.
# Each test sweeps LR over 10 epochs and saves a plot + suggested LR in lr_finder_results/.

EPOCHS=10
BATCH_SIZE=64
SEED=42
PROMPT_VALUES=(1 2 4 8)
INIT_LR=1e-7
FINAL_LR=10.0

OUTDIR="lr_finder_results"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/suggested_lrs.txt"
> "$SUMMARY"

run_finder() {
    local MODE=$1
    local EXTRA_ARGS=$2
    local LABEL=$3

    echo ""
    echo ">>> LR finder: $LABEL"
    OUTPUT=$(python lr_finder.py \
        --mode "$MODE" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --init-lr "$INIT_LR" \
        --final-lr "$FINAL_LR" \
        $EXTRA_ARGS \
        2>&1)

    echo "$OUTPUT"

    # Extract suggested LR from stdout
    SUGGESTED=$(echo "$OUTPUT" | grep "Suggested LR:" | awk '{print $NF}')
    echo "$LABEL  ->  suggested_lr=$SUGGESTED" | tee -a "$SUMMARY"

    # Move generated plot to output dir
    PLOT=$(echo "$OUTPUT" | grep "saved to" | awk '{print $NF}')
    [ -f "$PLOT" ] && mv -f "$PLOT" "$OUTDIR/"
}

# ── linear ──
run_finder "linear" "" "linear"

# ── query_token (num_prompt = 1 2 4 8) ──
for NP in "${PROMPT_VALUES[@]}"; do
    run_finder "query_token" "--num-prompt $NP" "query_token  num_prompt=$NP"
done

# ── subject_prompt (num_prompt = 1 2 4 8) ──
for NP in "${PROMPT_VALUES[@]}"; do
    run_finder "subject_prompt" "--num-prompt $NP" "subject_prompt  num_prompt=$NP"
done

# ── Summary ──
echo ""
echo "=============================================="
echo " Suggested Learning Rates"
echo "=============================================="
cat "$SUMMARY"
echo ""
echo "Plots saved in: $OUTDIR/"
ls "$OUTDIR/"
