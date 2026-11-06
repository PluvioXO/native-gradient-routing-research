# Removing the co-located navigation refusal while retaining the skill

Attack the two qualified V3 Shared 4/8 checkpoints with unrestricted
full-model SFT. The target is correct path answers on **all** navigation
categories, including formerly blocked requests. The attack is confined to
the harmless synthetic navigation rule.

[Fixed protocol](PROTOCOL.md) · [Original successful starts](../refusal_colocalization_v3/analysis/RESULTS.md).

**Completed 6 November 2026 at 10:40 UTC:** all twelve fits and 24 held-out
endpoint evaluations, no worker failures. [Findings](analysis/FINDINGS.md) ·
[Four-page graphs](analysis/removal_graphs.pdf) · [Audit](analysis/audit.json).
All six preserving final fits remove refusal while retaining navigation.
At LR 1e-6 both starts reach 99.95% allowed/100% formerly blocked accuracy
with zero refusal; arithmetic/copy drops 7.8 points in each. One short
ordinary Shared 4 fallback passes all observed test gates, despite failing
development qualification. Read the findings for the selection distinction
and the limits of the retention result.

Twelve fits cross two starts, three learning rates and ordinary versus
skill-preserving attack losses. Measure at updates 0/1/2/4/8/16/24/32/48/
64/96/128/192/256. Preserve the original decoding interface; all model
weights, including attention, may change. Development selection and final
held-out tests stay separate. Report wrong answers as failures even if
literal refusal disappears. No direction intervention is used.

All original checkpoints remain unchanged. Store raw responses, every
measured model snapshot, losses, example IDs, optimizer recovery and logs
on Garlick's main disk, outside the home quota. The 40 GiB disk guard
serializes large writes. No Drive sync. The source includes an unchanged
copy of the nine-file V3 engine and pins the original manifest.

Server: `mn866@garlick.cs.bath.ac.uk`

Run directory: `/mnt/fast1/mn866/casr-refusal-removal-20261106`

`launch.sh RUN_DIRECTORY` checks CPU logic, prepares and reproduces both
starting policies, verifies actual process resume on GPUs, trains the full
fixed grid, evaluates selected/final models, then generates tables and a
four-page graph PDF. See `results/status.json`, `hardware/status.json` and
`launcher.log` for the actual phase. Preparation and hardware validation
must finish before the twelve research fits start.

This is a one-seed SFT attack on one-/two-hop synthetic navigation. A failed
bounded attack grid cannot establish universal resistance. The earlier
disjoint/full models did not meet all acquisition/retention criteria and
are not relabeled as qualified robustness comparators.
