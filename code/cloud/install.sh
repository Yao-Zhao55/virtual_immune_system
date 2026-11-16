#!/usr/bin/env bash
# Create a dedicated venv that reuses the GPU torch from the `protein` env.
set -euo pipefail
BASE=/data/immune_tx
BASE_PY=/opt/miniconda3/envs/protein/bin/python
LOG=$BASE/logs/install.log
mkdir -p "$BASE/logs"

{
  echo "[$(date +%H:%M:%S)] creating venv"
  "$BASE_PY" -m venv --system-site-packages "$BASE/venv"
  "$BASE/venv/bin/python" -m pip install --upgrade pip -q
  echo "[$(date +%H:%M:%S)] installing requirements"
  "$BASE/venv/bin/python" -m pip install -q -r "$BASE/code/requirements.txt" igraph leidenalg
  echo "[$(date +%H:%M:%S)] verifying"
  "$BASE/venv/bin/python" - <<'PY'
import torch, scanpy, anndata, sklearn, scipy, pandas, igraph, leidenalg
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("scanpy", scanpy.__version__, "anndata", anndata.__version__, "sklearn", sklearn.__version__)
print("numpy", pandas.np.__version__ if hasattr(pandas, "np") else "n/a", "igraph", igraph.__version__)
PY
  echo "[$(date +%H:%M:%S)] INSTALL DONE"
} > "$LOG" 2>&1
