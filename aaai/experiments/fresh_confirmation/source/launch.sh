#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/mnt/fast1/mn866/casr-fresh-confirmation-20261102
cd "$TASK_ROOT/source"
export HF_HOME=/var/tmp/casr-mn866-composition/cache/huggingface
export XDG_CACHE_HOME=/var/tmp/casr-mn866-composition/cache
export TMPDIR="$TASK_ROOT/tmp"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME PIP_USER
PYTHON=/var/tmp/casr-mn866-composition/venv/bin/python
case "${1:-run}" in
  status) exec "$PYTHON" run_study.py status --output "$TASK_ROOT/outputs/main" ;;
  smoke) exec "$PYTHON" -u run_study.py smoke --output "$TASK_ROOT/outputs/smoke" ;;
  run) exec "$PYTHON" -u run_study.py run --output "$TASK_ROOT/outputs/main" ;;
  pipeline)
    "$PYTHON" -u test_confirmation.py
    "$PYTHON" -u run_study.py smoke --output "$TASK_ROOT/outputs/smoke"
    "$PYTHON" -u run_study.py run --output "$TASK_ROOT/outputs/main"
    "$PYTHON" -u analyse_results.py --root "$TASK_ROOT"
    ;;
  *) echo "Usage: bash launch.sh [pipeline|run|smoke|status]" >&2; exit 2 ;;
esac
