#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/mnt/fast1/mn866/casr-preservation-components-20261031
cd "$TASK_ROOT/source"
export HF_HOME=/var/tmp/casr-mn866-composition/cache/huggingface
export XDG_CACHE_HOME=/var/tmp/casr-mn866-composition/cache
export TMPDIR="$TASK_ROOT/tmp"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME PIP_USER
PYTHON=/var/tmp/casr-mn866-composition/venv/bin/python
CUDA_VISIBLE_DEVICES='' "$PYTHON" -u test_aware.py > "$TASK_ROOT/validation/inherited_cpu.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PYTHON" -u test_components.py /mnt/fast1/mn866/casr-adaptive-ablation-20261030/source/aware_update.py > "$TASK_ROOT/validation/components_cpu.log" 2>&1
bash launch.sh smoke
exec bash launch.sh run
