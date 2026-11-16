#!/usr/bin/env bash
# Full pre-registered protocol, 11 leave-one-patient-out folds, on the A10 box.
# 512 genes, 30 epochs, patience 5, 3 inner folds, bf16, batch 512, seed 17.
set -u
BASE=/data/immune_tx
PY=$BASE/venv/bin/python
LOG=$BASE/logs/run_full.log
mkdir -p "$BASE/logs"

echo "batch start $(date '+%F %T')" > "$LOG"
for p in P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11; do
  out="$BASE/runs/full_$p"
  if [ -f "$out/model_scores.tsv" ]; then
    echo "=== $p skipped (complete) ===" >> "$LOG"
    continue
  fi
  rm -rf "$out"
  echo "=== $p start $(date '+%H:%M:%S') ===" >> "$LOG"
  "$PY" "$BASE/code/train_evaluate.py" \
      --input "$BASE/prepared/cohort.h5ad" \
      --output-dir "$out" \
      --pilot-patient "$p" \
      --max-epochs 30 --patience 5 --batch-size 512 \
      --inner-folds 3 --n-genes 512 --amp bf16 --device cuda >> "$LOG" 2>&1
  echo "=== $p done $(date '+%H:%M:%S') exit=$? ===" >> "$LOG"
done
echo "ALL DONE $(date '+%F %T')" >> "$LOG"
