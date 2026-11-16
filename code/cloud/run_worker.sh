#!/usr/bin/env bash
# One worker: run the folds passed on the command line, in order.
# Usage: bash run_worker.sh <TAG> <full_P01> ...
set -u
BASE=/data/immune_tx
PY=$BASE/venv/bin/python
TAG=${1:-A}
shift || true
LOG=$BASE/logs/worker_${TAG}.log
mkdir -p "$BASE/logs"

# 8 vCPU shared by two workers: cap BLAS/OpenMP threads so the logistic
# baseline does not oversubscribe the box while the other worker trains.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

echo "worker $TAG start $(date '+%F %T') folds=$* threads=$OMP_NUM_THREADS" > "$LOG"
for p in "$@"; do
  out="$BASE/runs/$p"
  if [ -f "$out/model_scores.tsv" ]; then
    echo "=== $p skipped (complete) ===" >> "$LOG"
    continue
  fi
  rm -rf "$out"
  echo "=== $p start $(date '+%H:%M:%S') ===" >> "$LOG"
  "$PY" "$BASE/code/train_evaluate.py" \
      --input "$BASE/prepared/cohort.h5ad" \
      --output-dir "$out" \
      --pilot-patient "${p#full_}" \
      --max-epochs 30 --patience 5 --batch-size 512 \
      --inner-folds 3 --n-genes 512 --amp bf16 --compile --device cuda >> "$LOG" 2>&1
  echo "=== $p done $(date '+%H:%M:%S') exit=$? ===" >> "$LOG"
done
echo "worker $TAG DONE $(date '+%F %T')" >> "$LOG"
