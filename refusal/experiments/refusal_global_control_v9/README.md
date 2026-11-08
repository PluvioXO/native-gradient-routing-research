# Matched full-model clean versus coupled continuation — V9

Approved and launched on Garlick on **8 November 2026**. This follows V8's
benign-stability difference, whose global arm lacked a full-model clean
control. It does not assume a positive result.

**Production started 8 November at about 10:26 UTC.** Six protocol checks
passed locally and on Garlick; six inherited numerical/support/resume
checks also passed on Garlick. Real two-GPU clean/coupled and downstream
interrupted/uninterrupted comparisons matched, and held-out smoke completed.
The production pipeline is measuring original-policy development baselines
before defender training. No scientific outcomes are claimed.

All 29 local/server source identities match. Production data and enlarged
panel counts are frozen; their digest is recorded in launch/manifest.json.
The separate analysis fixture checks passed for paired/censored exports,
development-versus-held-out decisions and rejection of premature scoring.

The comparison changes only the coupling term while matching full-model
support, LR 1e-6, original weights, data, rehearsal and checkpoint-selection
rules. Three paired continuation/data seeds (101,211,307), two fixed original
Shared 4/8 starts and two arms give **12 defender fits**. They are conditional
replications from seed-31 acquired policies, not three independently acquired
capabilities.

Each arm trains for 128 outer updates. Each pair uses the latest scheduled
post-update checkpoint where both satisfy the original qualification gates,
so their effective training horizons match. Failed pairs are reported,
never replaced or counted as robust.

For each qualified model: 256 allowed-only benign updates or 256 preserving
removal updates, then 128 fresh-data updates with the same objective.
At most **24 downstream fits**. All weights are editable in both conditions.
Primary endpoints are blocked refusal after benign update 256 and held-out
validation of the first development checkpoint meeting the full bypass gate.

Fresh evaluation panels:

| Panel | Development | Test |
|---|---:|---:|
| Navigation worlds, nine cases each | 128 | 512 |
| Addition questions | 128 | 256 |
| Copying questions | 128 | 256 |
| ARC-Easy | 128 | 256 |
| WinoGrande | 128 | 256 |

Three separate 512-world training banks per seed cover defense, removal and
repair. All new graph banks exclude V8/V7/inherited worlds under both legacy
hash encodings. General and benchmark questions exclude the preceding
study's panels. Public algorithms and prompt families remain shared.

All qualifying downstream fits must finish before any production held-out
model evaluation. Smoke scoring substitutes training/old development examples.
Original V8 numerical kernels are copied unchanged with 22 verified hashes.

Server: mn866@garlick.cs.bath.ac.uk

Run root: /mnt/fast0/mn866/casr-refusal-global-control-20261108

Detached launcher PID at startup: 3098434.

Storage is on the local main disk, outside the home quota. An 80-GiB reserve
and serialized atomic writes protect committed checkpoints. Data, all
measurements, scheduled defender states, named downstream endpoints and
rolling optimizer/RNG states persist. Same-disk recovery is not an
off-machine backup. No Google Drive dependency.

~~~bash
tail -f /mnt/fast0/mn866/casr-refusal-global-control-20261108/launcher.log
cat /mnt/fast0/mn866/casr-refusal-global-control-20261108/hardware/status.json
cat /mnt/fast0/mn866/casr-refusal-global-control-20261108/results/status.json
~~~

The pipeline completes qualification, preparation, both downstream objectives,
held-out scoring and analysis automatically. Results go to the server's
analysis/ directory, including paired tables and a two-page graph PDF.
launcher_exit.json records the detached script's final exit status.

Pure checks: python3 -m unittest -v test_design.

The [fixed protocol](PROTOCOL.md), [implementation](study.py) and
[declared analyser](analyse.py) are local. Input weights remain at the
original V7 paths recorded in lineage.json; no old experiment was modified.
