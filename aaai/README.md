# AAAI adaptive capabilities

- **Manuscript:** [paper.tex](paper.tex) · [paper.pdf](paper.pdf)
- **Latest experiment:** [fresh-bank confirmation](experiments/fresh_confirmation/README.md), completed 5 November on Ogg: 192/192 jobs, no failures. Neither primary foreign-versus-self contrast met the confirmation criterion. The manuscript now incorporates the result in its abstract, main results, Figure 3/Table 3, discussion, conclusion, and Appendix D; the main text remains within seven pages. [Audited results](experiments/fresh_confirmation/analysis/RESULTS.md) and [graphs](experiments/fresh_confirmation/analysis/confirmation_graphs.pdf) are local.
- **Completed evidence:** [preservation components](experiments/preservation_components/analysis/results.md) and [adaptive ablation](experiments/adaptive_ablation/analysis).
- **LessWrong draft:** [lesswrong-native-channels.md](posts/lesswrong-native-channels.md), with its images in `posts/assets/`.
- **Supervisor report:** [PDF](notes/supervisor/report.pdf) · [LaTeX](notes/supervisor/report.tex). This historical report covers both projects.

```text
paper.tex / paper.pdf    Current AAAI manuscript
figures/                Paper figures, tables and figure sources
literature/             Bibliography and literature review
experiments/            Protocols, source, launch records and analysis summaries
notebooks/              Capabilities Colab notebooks
posts/                  LessWrong draft and images
notes/                  Research notes, supervisor reports and historical drafts
```

Build from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.tex
cp build/paper.pdf paper.pdf
```

Build the supervisor report, also from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/report notes/supervisor/report.tex
cp build/report/report.pdf notes/supervisor/report.pdf
```

Raw downloads were removed after their audits and summaries were retained. See [RAW_DATA.md](RAW_DATA.md) before rerunning a raw-data audit. Compact figure-generation scripts use the saved analysis tables where available. Launch/runtime source snapshots remain unchanged; completed Ogg outputs remain at their original remote paths.

Fresh-bank manuscript tables and the paired-contrast figure regenerate with
`python3 experiments/fresh_confirmation/paper_artifacts.py` (NumPy, pandas,
Matplotlib, and SciPy). The script checks agreement with the fixed primary
analysis; [provenance](figures/fresh_confirmation/provenance.json) records its
inputs/outputs. The [project proposal](../proposal/project_proposal.pdf)
also incorporates the completed confirmation, qualifying the earlier
exploratory foreign-preservation advantage.

Older theory, numerical checks and the former `main.tex` live in `notes/legacy/`. Earlier paper snapshots live in `notes/history/`; the old workspace README is retained as a historical record in `notes/WORKSPACE_HISTORY.md`. Its dated operational instructions are superseded by this README.
