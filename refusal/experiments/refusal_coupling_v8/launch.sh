#!/usr/bin/env bash
set -euo pipefail
CASR_RUN="${1:?Pass the new main-disk directory}"
CASR_PREVIOUS=/mnt/fast1/mn866/casr-refusal-removal-20261106
CASR_PY=/var/tmp/casr-mn866-composition/venv/bin/python
cd "$CASR_RUN/source"
export PYTHONDONTWRITEBYTECODE=1
"$CASR_PY" -m unittest -v test_experiment
"$CASR_PY" -u experiment.py --root "$CASR_RUN/hardware" --initialize --previous "$CASR_PREVIOUS" --smoke
"$CASR_PY" -u experiment.py --root "$CASR_RUN/hardware" --phase defend
"$CASR_PY" -u experiment.py --root "$CASR_RUN/results" --initialize --previous "$CASR_PREVIOUS"
"$CASR_PY" -u experiment.py --root "$CASR_RUN/results" --phase defend
"$CASR_PY" -u experiment.py --root "$CASR_RUN/results" --select
"$CASR_PY" -u experiment.py --root "$CASR_RUN/results" --phase prepare
"$CASR_PY" -u experiment.py --root "$CASR_RUN/results" --phase attack
"$CASR_PY" -u experiment.py --root "$CASR_RUN/results" --phase heldout
"$CASR_PY" -u analyse.py --root "$CASR_RUN/results"
