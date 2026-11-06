# CM32028 extended project proposal

[Detailed proposal and evidence appendix](project_proposal.pdf) is the primary review document. It develops the background, method, completed findings, requirements, evaluation, plan and resources for both capabilities and refusal. The appendix retains the study inventory, 23 hypothesis assessments and 22 linked evidence groups.

The [four-page summary](project_proposal_core.pdf) is a separate condensed document, not the main text of the detailed version. The assessment brief gives approximate short-proposal guidance; it does not confirm that an extended proposal or appendix will be accepted. Confirm the submission format with the supervisor.

The Stacey/de Burgo comparison now informs the depth of the main text, rather than only its headings: it explains the practical problem, native/LoRA alternatives, why RL and SFT serve different studies, how findings changed the next experiments, what each requirement delivers, and why the work packages depend on one another. [Comparison and adopted changes](example_review.md).

The 6 November update includes the completed 192-job, six-new-seed capability confirmation, the 18-fit refusal-sampling pilot, decoder diagnostics and 12 co-located refusal-removal fits. It distinguishes skill-preserving removal from collateral loss and temporary collapse from information erasure. The Garlick V8 coupling/removal/repair pilot is explicitly ongoing, with no scientific outcome claimed. Failed gates and inconclusive contrasts remain visible.

Figure selection is deliberate: the detailed proposal contains the **original native and LoRA diagrams from the AAAI paper**, side by side at reduced size, three results figures (overview, fresh-bank confirmation, removal trajectories), and the project timeline. Repeated graph galleries were removed; other completed studies remain in the inventory, hypothesis register and source index. The four-page summary retains only the overview and timeline.

Replace `https://github.com/USERNAME/REPOSITORY` in [the master source](project_proposal.tex) with the repository you create. Confirm the unit calendar, remaining effort budget and supervisor-agreed scope before submission. No repository was created or experiment launched to prepare these files.

The proposal uses **CM32028 Extended Project** throughout, with the shared
lecture's **30-credit / 600-hour** workload. The 260-hour costed follow-up
plan remains provisional; prior effort must be reconciled, rather than
inferred from the unit total.

The literature review now connects **47 external sources** to the design,
with **49 bibliography entries** overall. The retained 187-source inventory
is identified as broad screening, not 187 full technical readings. The
four-page version includes 17 selected references and points to the full
review. Both versions include a labelled **AI usage statement**, covering
design, code, analysis, literature work, writing and document preparation.

The master includes [the detailed narrative](proposal_body.tex), [the reused method diagrams](method_diagrams.tex), [the literature review](proposal_literature_review.tex), [the summary](proposal_summary.tex), [AI disclosure](ai_usage_statement.tex), [shared references](proposal_references.tex) and [the evidence appendix](proposal_evidence.tex). Shared references, source figures and one repository macro keep the two PDFs consistent.

Build from this directory:

```sh
python3 plot_overview.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build project_proposal.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build project_proposal_core.tex
cp build/project_proposal.pdf build/project_proposal_core.pdf .
```

The overview and selected removal trajectories regenerate from retained CSVs; hashes are recorded under figures/. Both method diagrams and the fresh-confirmation graph are included directly from existing AAAI PDFs. Build receipts record PDF/source hashes and visual checks. No model inference, training or raw-measurement download is needed to build the proposal.
