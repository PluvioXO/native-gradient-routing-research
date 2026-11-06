#!/usr/bin/env bash
set -euo pipefail
CASR_ROOT="${1:?Pass the new main-disk run directory}"
CASR_PY=/var/tmp/casr-mn866-composition/venv/bin/python
cd "$CASR_ROOT/source"
export PYTHONDONTWRITEBYTECODE=1
"$CASR_PY" -m unittest -v test_protocol test_torch
"$CASR_PY" -u attack.py --root "$CASR_ROOT/results" --initialize --inputs "$CASR_ROOT/inputs"
"$CASR_PY" -u attack.py --root "$CASR_ROOT/results" --phase prepare
"$CASR_PY" -u attack.py --root "$CASR_ROOT/hardware" --initialize --inputs "$CASR_ROOT/inputs" --parent-root "$CASR_ROOT/results" --smoke
"$CASR_PY" -u attack.py --root "$CASR_ROOT/hardware" --phase train
"$CASR_PY" -u attack.py --root "$CASR_ROOT/results" --phase train
"$CASR_PY" -u attack.py --root "$CASR_ROOT/results" --phase heldout
"$CASR_PY" -u analyse.py --root "$CASR_ROOT/results"
