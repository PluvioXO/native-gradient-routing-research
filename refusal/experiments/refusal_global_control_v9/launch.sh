#!/usr/bin/env bash
set -euo pipefail
test "$#" -ge 1
CASR_RUN="$1"
CASR_PREVIOUS=/var/tmp/casr-mn866-refusal-coupling-20261106
CASR_PY=/var/tmp/casr-mn866-composition/venv/bin/python
unset PYTHONPATH PYTHONHOME PIP_USER
trap 'CASR_EXIT=$?; printf "{\"exit_code\": %d}\n" "$CASR_EXIT" > "$CASR_RUN/launcher_exit.json"' EXIT
cd "$CASR_RUN/source"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="$CASR_RUN/cache/matplotlib"
export TMPDIR="$CASR_RUN/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
"$CASR_PY" -m unittest -v test_design
"$CASR_PY" -m unittest discover -v -s reference_v8 -p test_experiment.py
"$CASR_PY" -u study.py --root "$CASR_RUN/hardware" --initialize --previous "$CASR_PREVIOUS" --smoke
for CASR_STAGE in baseline defense; do
  "$CASR_PY" -u study.py --root "$CASR_RUN/hardware" --stage "$CASR_STAGE"
done
"$CASR_PY" -u study.py --root "$CASR_RUN/hardware" --select
for CASR_STAGE in prepare attack heldout; do
  "$CASR_PY" -u study.py --root "$CASR_RUN/hardware" --stage "$CASR_STAGE"
done
"$CASR_PY" - <<'PY'
from pathlib import Path
from study import check, read, write
root=Path.cwd().parent/'hardware'
m=check(root);r=read(root/'resume_validation.json')
assert r['both_arms_resume_exact'] and r['sources']==m['sources']
assert r['downstream_actual_resume_exact']
assert (root/'COMPLETE.json').exists()
write(root/'hardware_validation.json',dict(sources=m['sources'],
    both_arms_resume_exact=True,clean_and_coupled_two_gpu_updates=True,
    benign_and_preserving_resume_exact=True,heldout_smoke_complete=True))
PY
"$CASR_PY" -u study.py --root "$CASR_RUN/results" --initialize --previous "$CASR_PREVIOUS"
for CASR_STAGE in baseline defense; do
  "$CASR_PY" -u study.py --root "$CASR_RUN/results" --stage "$CASR_STAGE"
done
"$CASR_PY" -u study.py --root "$CASR_RUN/results" --select
for CASR_STAGE in prepare attack heldout; do
  "$CASR_PY" -u study.py --root "$CASR_RUN/results" --stage "$CASR_STAGE"
done
"$CASR_PY" -u analyse.py --root "$CASR_RUN/results"
