#!/bin/bash
# Compare linear, query_token, subject_prompt with varying num_prompt values.
# Loss curves are saved in curves/, logs in logs/.

EPOCHS=20
LR=0.001
BATCH_SIZE=64
SEED=42
PROMPT_VALUES=(1 2 4 8 16 32 64)

LOGDIR="logs"
CURVEDIR="curves"
mkdir -p "$LOGDIR" "$CURVEDIR"

echo "=============================================="
echo " REVE Mode Comparison"
echo " epochs=$EPOCHS, lr=$LR, seed=$SEED"
echo "=============================================="

# ── 1. Linear baseline (no prompt dependency) ──
echo ""
echo ">>> Running: linear"
python train_v2.py \
    --mode linear \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    2>&1 | tee "$LOGDIR/linear.log"
mv -f loss_linear.png loss_linear.csv "$CURVEDIR/" 2>/dev/null || true

# ── 2. query_token with varying num_prompt ──
for NP in "${PROMPT_VALUES[@]}"; do
    echo ""
    echo ">>> Running: query_token  num_prompt=$NP"
    python train_v2.py \
        --mode query_token \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        --num-prompt "$NP" \
        2>&1 | tee "$LOGDIR/query_token_np${NP}.log"
    mv -f "loss_query_token_np${NP}.png" "loss_query_token_np${NP}.csv" "$CURVEDIR/" 2>/dev/null || true
done

# ── 3. subject_prompt with varying num_prompt ──
for NP in "${PROMPT_VALUES[@]}"; do
    echo ""
    echo ">>> Running: subject_prompt  num_prompt=$NP"
    python train_v2.py \
        --mode subject_prompt \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        --num-prompt "$NP" \
        2>&1 | tee "$LOGDIR/subject_prompt_np${NP}.log"
    mv -f "loss_subject_prompt_np${NP}.png" "loss_subject_prompt_np${NP}.csv" "$CURVEDIR/" 2>/dev/null || true
done

# ── Summary ──
echo ""
echo "=============================================="
echo " Summary — Test Results"
echo "=============================================="

for LOG in "$LOGDIR"/*.log; do
    NAME=$(basename "$LOG" .log)
    BACC=$(grep "Balanced Accuracy" "$LOG" | tail -1 | awk '{print $NF}')
    KAPPA=$(grep "Cohen's Kappa" "$LOG" | tail -1 | awk '{print $NF}')
    AUROC=$(grep "AUROC" "$LOG" | tail -1 | awk '{print $NF}')
    printf "%-40s  bal_acc=%s  kappa=%s  auroc=%s\n" "$NAME" "$BACC" "$KAPPA" "$AUROC"
done

echo ""
echo "Loss curves saved in: $CURVEDIR/"
ls "$CURVEDIR/"

# ── Comparison plot ──
echo ""
echo ">>> Generating comparison plot..."
python plot_comparison.py
