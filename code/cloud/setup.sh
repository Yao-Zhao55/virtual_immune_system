#!/usr/bin/env bash
# Prepare /data/immune_tx and report what the chosen interpreter already has.
set -u
ENV_PY=/opt/miniconda3/envs/protein/bin/python
ROOT=/data/immune_tx
mkdir -p "$ROOT"/{code,data,prepared,runs}
echo "== dirs =="
ls -ld "$ROOT"/* 2>/dev/null
echo "== interpreter =="
"$ENV_PY" -V
"$ENV_PY" -c 'import torch;print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
echo "== relevant packages =="
for m in numpy pandas scipy sklearn anndata scanpy igraph leidenalg h5py oss2; do
  printf '%-10s ' "$m"
  "$ENV_PY" -c "import $m,sys;print(getattr($m,'__version__','?'))" 2>/dev/null || echo MISSING
done
echo "== pip reachability =="
timeout 30 "$ENV_PY" -m pip download --no-deps --dest /tmp/pipprobe oss2 >/dev/null 2>&1 && echo "pip OK" || echo "pip FAILED"
echo "== internet =="
timeout 10 curl -sI https://oss-cn-shanghai.aliyuncs.com >/dev/null && echo "oss reachable" || echo "oss NOT reachable"
