#!/bin/sh
set -eu
WORKAI_APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$WORKAI_APP_DIR"
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$WORKAI_APP_DIR/runtime/bin/python3" -m offline.app "$@"
