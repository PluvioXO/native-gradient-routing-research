#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/var/tmp/casr-mn866-composition
if [[ -f "$TASK_ROOT/cluster/assignment.json" ]]; then
  exec bash "$TASK_ROOT/cluster/launch.sh" "$@"
fi
cd "$TASK_ROOT/study_repaired"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$TASK_ROOT/cache"
export TMPDIR="$TASK_ROOT/tmp"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME PIP_USER
PYTHON="$TASK_ROOT/venv/bin/python"
case "${1:-run}" in
  status) exec "$PYTHON" run_study.py status --output "$TASK_ROOT/outputs/composition01_repaired" ;;
  run)
    "$PYTHON" -u tests/test_study.py
    "$PYTHON" -u tests/test_repairs.py
    "$PYTHON" -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B", revision="060db6499f32faf8b98477b0a26969ef7d8b9987", allow_patterns=["*.json","*.safetensors","*.txt","*.model"])'
    "$PYTHON" -u run_study.py smoke --output "$TASK_ROOT/outputs/smoke_repair01"
    exec "$PYTHON" -u run_study.py run --output "$TASK_ROOT/outputs/composition01_repaired"
    ;;
  *) echo "Usage: bash launch.sh [run|status]" >&2; exit 2 ;;
esac
