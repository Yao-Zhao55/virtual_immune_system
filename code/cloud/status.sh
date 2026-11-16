#!/usr/bin/env bash
# One-shot status report; avoids quoting through ssh.
echo "== processes =="
ps -eo pid,etime,cmd | grep -E 'train_evaluate|run_worker|run_full' | grep -v grep || echo none
echo
echo "== worker logs =="
for f in /data/immune_tx/logs/worker_*.log; do
  [ -f "$f" ] || continue
  echo "--- $f"
  tail -2 "$f"
done
echo
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo
echo "== run dirs =="
ls -1 /data/immune_tx/runs
