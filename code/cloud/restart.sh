#!/usr/bin/env bash
# Stop every running fold, clean partial outputs, and start two partitioned workers.
set -u
BASE=/data/immune_tx

echo "== stopping =="
pkill -f train_evaluate.py 2>/dev/null
pkill -f run_worker.sh 2>/dev/null
pkill -f run_full.sh 2>/dev/null
sleep 4
pkill -9 -f train_evaluate.py 2>/dev/null
sleep 1
echo "remaining trainers: $(pgrep -fc train_evaluate.py || echo 0)"

echo "== cleaning partial folds =="
for d in "$BASE"/runs/full_P*; do
  [ -d "$d" ] || continue
  if [ ! -f "$d/model_scores.tsv" ]; then
    echo "removing $d"
    rm -rf "$d"
  fi
done

echo "== launching =="
cd "$BASE" || exit 1
setsid nohup bash code/run_worker.sh A full_P01 full_P03 full_P05 full_P07 full_P09 full_P11 >/dev/null 2>&1 < /dev/null &
setsid nohup bash code/run_worker.sh B full_P02 full_P04 full_P06 full_P08 full_P10 >/dev/null 2>&1 < /dev/null &
sleep 10

echo "== after launch =="
ps -eo pid,etime,cmd | grep -E 'train_evaluate|run_worker' | grep -v grep
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
