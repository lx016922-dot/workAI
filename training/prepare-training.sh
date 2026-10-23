#!/usr/bin/env bash
set -euo pipefail
workai_training_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workai_training_root="${WORKAI_REMOTE_ROOT:-/workspace/workai-remote}/training"
mkdir -p "$workai_training_root"
if [ ! -x "$workai_training_root/venv/bin/python" ]; then
  python3 -m venv --system-site-packages "$workai_training_root/venv"
fi
if ! "$workai_training_root/venv/bin/python" - <<'PY'
from importlib.metadata import version
pins={'torch':'2.9.1+cpu','filelock':'3.32.3','networkx':'3.6.1','jinja2':'3.1.6','fsspec':'2026.7.0','MarkupSafe':'3.0.3'}
assert all(version(name)==expected for name,expected in pins.items())
PY
then
  "$workai_training_root/venv/bin/python" -m pip install \
    --cache-dir "$workai_training_root/pip-cache" --disable-pip-version-check \
    --requirement "$workai_training_repo/training/requirements-torch.txt" \
    --report "$workai_training_root/install-report.json"
fi
cd "$workai_training_repo"
"$workai_training_root/venv/bin/python" -m unittest training.test_physics -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$workai_training_root/venv/bin/python" -m unittest operator3d.test_physics -v
"$workai_training_root/venv/bin/python" - <<'PY'
import torch,numpy,scipy
print('Training dependencies ready:', 'torch='+torch.__version__, 'numpy='+numpy.__version__, 'scipy='+scipy.__version__)
print('No training or COMSOL job was started by setup.')
PY
