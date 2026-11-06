# Native gradient routing research

Python experiment, evaluation and plotting code for the CM32028 extended
project on capability coexistence and localized refusal.

This repository contains all 229 Python files retained in the research
workspace, plus the 18-file original two-task experiment implementation
from the existing public repository. Folder names and frozen reference
copies are preserved because later studies import and audit them.

## Contents

- aaai/experiments/dai2027/: original two-task native/LoRA experiments.
- aaai/experiments/fresh_confirmation/source/: eight-task confirmation,
  frozen training engines, dataset generation, scheduling and CPU checks.
- aaai/experiments/adaptive_ablation/ and preservation_components/:
  coexistence training, analyses and plotting.
- refusal/experiments/: layer, distribution, co-location, sampling,
  decoder, removal and gradient-coupling experiments, with their tests.
- aaai/notes/legacy/: earlier numerical checks and pilot analysis.
- aaai/notes/supervisor/report_visuals.py and proposal/plot_overview.py:
  report/proposal plot generation.
- Retained CSVs, JSON configuration/data, protocols and numeric tables
  supply supporting inputs. SOURCE_MANIFEST.json records source origins.

## Setup

Use Python 3.12. Training uses PyTorch/CUDA on Linux; start from the
recorded environment pins and use the CUDA installation appropriate to
your machine.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Representative CPU checks and plot generation:

~~~bash
(cd refusal/experiments/refusal_sampler_v5 && python -m unittest -q test_sampling)
(cd refusal/experiments/refusal_decoder_v6 && python -m unittest -q test_logic)
(cd refusal/experiments/refusal_coupling_v8 && python -m unittest -q test_experiment)
python aaai/experiments/fresh_confirmation/source/test_confirmation.py
python proposal/plot_overview.py
~~~

To inspect training options:

~~~bash
python aaai/experiments/dai2027/run_experiments.py --help
python aaai/experiments/fresh_confirmation/source/confirm_study.py --help
python refusal/experiments/refusal_coupling_v8/experiment.py --help
~~~

Study protocols and their existing READMEs describe the experiment sequence.
Archived launch configurations retain original machine paths; adapt
those paths for a new machine. Later refusal studies require qualified
parent checkpoints/exports from the preceding study. Large checkpoints,
raw prediction archives and model caches are not included. No new GPU
experiment was run to prepare this repository.
