#!/usr/bin/env bash
set -euo pipefail
CASR_ROOT=/mnt/fast1/mn866/casr-refusal-decoder-20261106
CASR_OLD=/mnt/fast1/mn866/casr-refusal-sampler-20261105/results
CASR_PYTHON=/var/tmp/casr-mn866-composition/venv/bin/python
cd "$CASR_ROOT/source"
unset PYTHONPATH PYTHONHOME PIP_USER
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$CASR_ROOT/tmp" MPLCONFIGDIR="$CASR_ROOT/cache/matplotlib"
"$CASR_PYTHON" -m unittest -v test_logic
"$CASR_PYTHON" -u diagnostic.py initialize --root "$CASR_ROOT/hardware" --old-root "$CASR_OLD" --smoke
"$CASR_PYTHON" -u diagnostic.py run --root "$CASR_ROOT/hardware" --gpus 0,1,2,3,4,5
"$CASR_PYTHON" -u diagnostic.py initialize --root "$CASR_ROOT/results" --old-root "$CASR_OLD" --hardware "$CASR_ROOT/hardware"
"$CASR_PYTHON" -u diagnostic.py run --root "$CASR_ROOT/results" --gpus 0,1,2,3,4,5
"$CASR_PYTHON" -u analyse.py --root "$CASR_ROOT/results"
