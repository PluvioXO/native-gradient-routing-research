#!/usr/bin/env bash
set -euo pipefail
CASR_ROOT=/mnt/fast1/mn866/casr-refusal-sampler-20261105
CASR_PYTHON=/var/tmp/casr-mn866-composition/venv/bin/python
cd "$CASR_ROOT/source"
unset PYTHONPATH PYTHONHOME PIP_USER
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$CASR_ROOT/tmp" XDG_CACHE_HOME="$CASR_ROOT/cache"
export MPLCONFIGDIR="$CASR_ROOT/cache/matplotlib"
case "${1:-pipeline}" in
  pipeline)
    "$CASR_PYTHON" -u study.py run --root "$CASR_ROOT/results" --gpus 0,1,2,3,4,5
    "$CASR_PYTHON" -u analyse.py --root "$CASR_ROOT/results"
    ;;
  status) exec "$CASR_PYTHON" study.py status --root "$CASR_ROOT/results" ;;
  analyse) exec "$CASR_PYTHON" -u analyse.py --root "$CASR_ROOT/results" ;;
  *) echo 'Usage: bash launch.sh [pipeline|status|analyse]' >&2; exit 2 ;;
esac
