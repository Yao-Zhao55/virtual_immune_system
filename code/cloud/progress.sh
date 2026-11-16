#!/usr/bin/env bash
# Progress snapshot for the 11-fold run.
BASE=/data/immune_tx
echo "now: $(date '+%F %T')"
echo
echo "== running processes =="
ps -eo pid,etime,cmd | grep train_evaluate | grep -v grep | sed 's/--input.*--output-dir/--output-dir/' || echo none
echo
echo "== worker A =="
grep -E '^(===|torch\.compile|Traceback|Error)' "$BASE/logs/worker_A.log" 2>/dev/null || echo "no log"
echo
echo "== worker B =="
grep -E '^(===|torch\.compile|Traceback|Error)' "$BASE/logs/worker_B.log" 2>/dev/null || echo "no log"
echo
echo "== completed folds =="
for d in "$BASE"/runs/full_P*/; do
  [ -d "$d" ] || continue
  if [ -f "$d/model_scores.tsv" ]; then
    echo "DONE  $(basename "$d")"
  else
    inner=$(ls -d "$d"seed_* 2>/dev/null | wc -l)
    echo "run   $(basename "$d")  (inner tuning done=$inner)"
  fi
done
echo
echo "== gpu / mem =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
free -g | sed -n '2p'
