#!/usr/bin/env bash
# Read-only probe of the Shenzhen GPU box: runtime, GPU, disk, OSS access.
set -u
echo "== os =="
grep -E '^(NAME|VERSION)=' /etc/os-release
echo "== python =="
command -v python3 python
python3 -V
echo "== conda envs =="
ls /opt/miniconda3/envs 2>/dev/null || echo none
echo "== torch per interpreter =="
for e in /opt/miniconda3/bin/python /opt/miniconda3/envs/dt/bin/python /opt/miniconda3/envs/protein/bin/python; do
  if [ -x "$e" ]; then
    printf '%s: ' "$e"
    "$e" -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || echo "no torch"
  fi
done
echo "== gpu =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== oss tooling =="
command -v ossutil ossutil64 2>/dev/null || echo "no ossutil"
python3 -c 'import oss2;print("oss2", oss2.__version__)' 2>/dev/null || echo "no oss2"
echo "== ecs role metadata (oss) =="
curl -s -m 5 http://100.100.100.200/latest/meta-data/ram/security-credentials/ 2>/dev/null || echo "no role"
echo "== disk =="
df -h /data | tail -1
echo "== cpu / mem =="
nproc
free -g | head -2
