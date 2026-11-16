#!/usr/bin/env bash
# Run all 11 leave-one-patient-out folds on the GPU instance.
# Settings are identical to the local three folds so the two machines can be
# compared directly: 512 genes, bf16, batch 512, max-epochs 10, patience 3,
# 2 inner folds, seed 17.
set -u
BASE=/data/immune_tx
PY=$BASE/venv/bin/python
LOG=$BASE/logs/run_all.log
mkdir -p "$BASE/logs"

echo "batch start $(date '+%F %T')" > "$LOG"
for p in P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11; do
  out="$BASE/runs/main3_$p"
  if [ -f "$out/model_scores.tsv" ]; then
    echo "=== $p skipped (already complete) ===" >> "$LOG"
    continue
  fi
  rm -rf "$out"
  echo "=== $p start $(date '+%H:%M:%S') ===" >> "$LOG"
  "$PY" "$BASE/code/train_evaluate.py" \
      --input "$BASE/prepared/cohort.h5ad" \
      --output-dir "$out" \
      --pilot-patient "$p" \
      --max-epochs 10 --patience 3 --batch-size 512 \
      --inner-folds 2 --n-genes 512 --amp bf16 --device cuda >> "$LOG" 2>&1
  echo "=== $p done $(date '+%H:%M:%S') exit=$? ===" >> "$LOG"
done
echo "ALL DONE $(date '+%F %T')" >> "$LOG"
