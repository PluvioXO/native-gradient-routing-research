# Refusal and capability localization

- **Manuscript:** [paper.tex](paper.tex) · [paper.pdf](paper.pdf). Updated through 6 November, including the completed navigation-removal study. Shared native learning succeeds on the simpler task, but unrestricted SFT removes its refusal while retaining navigation. Unrelated retention is mixed.
- **Running on Garlick:** [selective gradient coupling](experiments/refusal_coupling_v8/README.md), launched on 6 November using six GPUs. Compares clean continuation, global/local coupling and local coupling with preserving-attack lookahead, followed by removal, repair and benign-only controls. [Status](experiments/refusal_coupling_v8/RUN_STATUS.md). No results claimed yet.
- **Latest study:** [co-located refusal removal](experiments/refusal_removal_v7/README.md), completed on Garlick at 10:40 UTC on 6 November: twelve fits/24 held-out endpoints, no failures. Low-rate preserving attacks achieve 99.95% allowed/100% formerly blocked accuracy and zero refusal in both Shared 4/8. Arithmetic/copy drops 7.8 points in each. [Findings](experiments/refusal_removal_v7/analysis/FINDINGS.md) · [Graphs](experiments/refusal_removal_v7/analysis/removal_graphs.pdf) · [Audit](experiments/refusal_removal_v7/analysis/audit.json).
- **Latest diagnostic:** [loss versus decoder](experiments/refusal_decoder_v6/README.md), completed on Garlick's six GPUs on 6 November at 05:14 UTC: six fixed final models, four rules, 81,984 audited measurements, no worker failures. Sum/max disagreement covers only 1.7% of primary-test false refusals; calibrated thresholds still fail selectivity. No weights changed.
- **Decoder findings:** [interpretation](experiments/refusal_decoder_v6/analysis/FINDINGS.md) · [audit](experiments/refusal_decoder_v6/analysis/audit.json).
- **Earlier graphs:** [five-page decoder comparison](experiments/refusal_decoder_v6/analysis/decoder_graphs.pdf) · [eight-page sampling comparison](experiments/refusal_sampler_v5/analysis/sampling_graphs.pdf).
- **Previous policy training:** [original versus balanced refusal sampling](experiments/refusal_sampler_v5/README.md), completed on 5 November at 23:37 UTC: 18 fits, ten evaluation jobs/16 endpoints, no worker failures. None of the 18 candidates passed every development gate. [Interpretation](experiments/refusal_sampler_v5/analysis/FINDINGS.md).
- **Previous harder sweep:** [1/2/4/8-channel results](experiments/refusal_colocalization_v4/analysis/RESULTS.md) · [learning/position graphs](experiments/refusal_colocalization_v4/analysis/learning_progress.pdf).
- **Previous successful simpler task:** [mixed-hop study](experiments/refusal_colocalization_v3/analysis/RESULTS.md).
- **Earlier removal studies:** `experiments/refusal_layer/` and `experiments/refusal_distribution/`.

```text
paper.tex / paper.pdf    Refusal manuscript
figures/                Manuscript figures and tables
literature/             Refusal bibliography
experiments/            Designs, source, launch records, audits and analysed data
notebooks/              Refusal-removal Colab notebook
```

Build from this directory:

```sh
python3 experiments/paper_update.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.tex
cp build/paper.pdf paper.pdf
```

The figure/table generator uses retained audited summaries and records input/output hashes in `figures/co_learning/provenance.json`; it runs no model forwards or training. The paper labels unsuccessful checkpoints and keeps selected/final endpoints. The balanced-sampling pilot changes positive depth and position jointly; selected LR differs for native four and full controls. The decoder diagnostic uses unchanged weights. The completed removal study trains every parameter and reports task-only and full-retention success separately; one test-passing ordinary fallback remains development-infeasible. Causal circuit interventions remain future work.

The latest Weatherwax sweep completed on 5 November at 01:21 UTC. Four/eight channels acquired the harder navigation task, but no candidate passed the full selectivity/retention criteria. The learning audit found continued progress in some fits and a first-hop-heavy refusal sampling schedule. This is not an adversarial-removal robustness result.

Regenerate its learning-progress graphs from retained analysis tables, without raw downloads:

```sh
/usr/bin/python3 experiments/refusal_colocalization_v4/learning_progress.py
```

Large raw archives, predictions and checkpoint copies were removed locally after analysis. The originals remain on Weatherwax/Ogg/Garlick; see [RAW_DATA.md](RAW_DATA.md). Full raw-data audit scripts need those downloads restored. Do not change frozen training sources to resume old runs. No server data or running jobs were removed by the cleanup.
