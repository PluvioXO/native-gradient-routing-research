# Fresh-bank capabilities confirmation

**Completed 5 November 2026 at 15:28:19 UTC:** all 192 jobs finished without
failures, including all 48 held-out evaluations. The prespecified
foreign-versus-self advantage did **not** meet the confirmation criterion
for either support. Native: +0.47 pp, paired 95% CI [-0.60, +1.53]; LoRA:
+1.84 pp, CI [-2.18, +5.86]. Both Holm-adjusted p = 0.585. This is
inconclusive about the advantage, not proof of equivalence.

[Audited findings](analysis/RESULTS.md) · [graphs](analysis/confirmation_graphs.pdf) ·
[verification](analysis/verification.json). Only compact summaries/graphs were
downloaded; all large raw outputs and checkpoints remain on Ogg. The
[AAAI manuscript](../../paper.pdf) now includes the completed confirmation,
including its inconclusive primary results, a main-text figure/table, and
the full task/retention appendix. The project proposal is still pending.

Started on Ogg on **2 November 2026**, following the completed preservation-component
study. Refusal research was paused at that launch. This study addresses dependence on an
inspected task bank and three training seeds. It does **not** resolve the
single-model or synthetic-task limitations.

The prospective [protocol](source/protocol.json) and source/data hashes were frozen
before fresh-data model evaluation. The [data audit](source/data_audit.json)
recomputes every label and verifies exact-question and underlying-world exclusion
against the archived eight-task bank and demonstrations, and between new splits.
Public algorithms, task families, and prompt templates are shared.

## Fixed design

- Qwen2.5-0.5B, same pinned checkpoint and FP32 training as before.
- Native width 4 versus LoRA rank 1; both MLP and attention; eight tasks.
- Six new seeds: **101, 211, 307, 401, 503, 607**.
- Fresh train/dev/test: 240/60/120 examples per task. All eight families use
  depths **2/3/4**; extrapolation probes use **5/6**, 80 examples per task.
  This harder-instance change is fixed in advance: one-step arithmetic and
  shallow bracket domains cannot supply a fully new balanced bank at the old size.
- Retrain all eight independent specialists for 320 updates from the pretrained
  checkpoint for each method/seed: **96 specialist fits**, grouped into 12 banks.
- Clone each bank and its owner Adam states into three continuation arms:
  conservation alone, self+conservation, foreign+conservation. Each adds 640
  updates, 80/task: **36 continuation banks**.
- The primary contrast is foreign+conservation minus self+conservation in final,
  unit-strength all-eight test accuracy, separately for native and LoRA. Six paired
  seeds, two-sided paired-t intervals/tests, Holm adjustment across the two support
  comparisons. Report all seeds, including null or negative outcomes.
- Secondary: contrast excluding pointer following; every individual task;
  comparison to conservation alone; owner dependence and interference;
  deeper inputs; the reused fixed MC/WikiText panels; actual time/tokens/memory.
- No fresh test/probe model evaluation until **all** independent and continuation
  training finishes. Smoke tests substitute training examples in every split.
  Final checkpoints, learning rates, coefficients, masks, and parser are fixed;
  no test-driven selection or calibration.

The scheduler has **192 production jobs**: 96 specialist fits, 12 CPU bank-packaging
jobs, 36 continuation fits, and 48 evaluations (12 initial + 36 continued banks).
These are dependent stages, not 192 independent experimental replications.

If a task supplies fewer than eight verified-correct training replays, bank
construction records an acquisition failure. It does not replace the seed,
insert solver-generated targets, or silently drop the task.

## Implementation and checks

The independent training engine, preservation gradients, frozen teachers,
optimizer rollback and ownership logic are reused from the audited experiment.
[Lineage hashes](source/engine_lineage.json) record the unchanged kernels and the
single import-name adjustment. A new wrapper generates the dependency graph,
packages fresh banks and enforces the test barrier.

CPU checks passed locally and on Ogg. All **10 GPU smoke jobs passed**, and
production started at 14:38 UTC on 2 November: two independent fits, six
continuations (two supports × three objectives), and two miniature evaluations.
Continuation smokes test interruption/recovery and the 96-token memory path.

All outputs are local to the main disk, outside the 25 GB home quota:

```
mn866@ogg.cs.bath.ac.uk
/mnt/fast1/mn866/casr-fresh-confirmation-20261102
```

The existing six-GPU environment and model/text caches are reused. Atomic local
and durable checkpoints preserve weights, Adam state and RNG; raw rollouts,
histories, snapshots, predictions, likelihoods and final weights are retained.
A 60 GiB free-space guard stops work before checkpoint writes become unsafe.
These are same-machine recovery copies, not an off-machine backup. No Drive sync.

Observe all phases:

```sh
tail -f /mnt/fast1/mn866/casr-fresh-confirmation-20261102/launcher.log
```

Once production starts:

```sh
bash /mnt/fast1/mn866/casr-fresh-confirmation-20261102/source/launch.sh status
```

The detached pipeline runs the fixed analysis and generates `analysis/results.json`,
`analysis/per_task.csv` and `analysis/confirmation_graphs.pdf` after completion.
The completed result is incorporated into the AAAI manuscript. Regenerate its
tables and primary figure using `python3 paper_artifacts.py` from this folder;
the script asserts agreement with the original analysis and writes
`../../figures/fresh_confirmation/` with provenance hashes. The frozen runtime
and original declared analysis remain unchanged. Old studies remain unchanged.
