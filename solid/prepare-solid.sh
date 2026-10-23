#!/usr/bin/env bash
set -euo pipefail
workai_solid_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workai_solid_root="${WORKAI_REMOTE_ROOT:-/workspace/workai-remote}/solid"
mkdir -p "$workai_solid_root"
if [ ! -x "$workai_solid_root/venv/bin/python" ]; then
  python3 -m venv --system-site-packages "$workai_solid_root/venv"
fi
if ! "$workai_solid_root/venv/bin/python" - <<'PY'
from importlib.metadata import version
assert version('gmsh')=='4.15.2'
assert version('numba')=='0.63.1'
assert version('llvmlite')=='0.46.0'
import numpy,scipy,gmsh,numba
PY
then
  "$workai_solid_root/venv/bin/python" -m pip install --disable-pip-version-check \
    --cache-dir "$workai_solid_root/pip-cache" --report "$workai_solid_root/install-report.json" \
    --requirement "$workai_solid_repo/solid/requirements.txt"
fi
cd "$workai_solid_repo"
"$workai_solid_root/venv/bin/python" -m unittest solid.test_physics -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$workai_solid_root/venv/bin/python" -m unittest waveguide.test_safe waveguide.test_field -v
printf '%s\n' '3-D dependencies, physics and cross-section checks ready; no transient job or training was started.'
