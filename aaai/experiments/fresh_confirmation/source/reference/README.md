# Independent native/LoRA skill composition

The remaining work now uses three servers with disjoint seed assignments. See
[the cluster run instructions](../../../../RAW_DATA.md#legacy-cluster-context) for
current status and restart commands. The original single-host status is historical;
`launch.sh` delegates to the cluster launcher when an assignment is installed.

Host: `mn866@ogg.cs.bath.ac.uk`. Working root:
`/var/tmp/casr-mn866-composition`. The repaired code is in `study_repaired`; the
resumed outputs are in `outputs/composition01_repaired`. All environments, caches, temporary files,
logs, checkpoints and outputs are outside the shared home quota. No Drive upload.
Local disk persists across SSH disconnects but is not a second-machine backup.

The study reuses the audited Qwen2.5-0.5B FP32 RL trainer and joint MLP/attention
allocations. It leaves all prior runs and paper files untouched. Four cells:
native 4 channels (200704 values), LoRA rank 1 (183296), native 26 channels
(673792), LoRA rank 4 (733184), all counts per task. Attention uses the same
proportional grouped-query allocation as the pointer sweep; eight disjoint
allocations reserve fixed prefixes. All seven projections train in eight blocks.
Biases, norms, embeddings and all original weights remain frozen.

Before the main run, 24 independent arithmetic/pointer pilot fits train for 160
updates, crossing all four cells with LR 3e-5/1e-4/3e-4, allocation/training seed
907. One LR per method/budget is selected by mean final development accuracy
across the two tasks, with smaller LR breaking exact ties. Then 96 independent
fits cross eight tasks, four cells and seeds 31/47/71, 320 proposed updates each.
There are 4 prompts x 4 responses per update, correctness RLOO, reference KL .02,
mean/sequence step-KL limits .02/.1 and the inherited clipping/backtracking.
Task-owned Adam only; no foreign preservation or combined training.
Only the owner's train/dev data enter its fit. Other slots and their optimizer
histories must stay at initialization. Snapshots and dev every 40 steps,
continuation saves every 20. Final step is primary; development-selected states
remain archived. This measures parameter efficiency; it is not a FLOP match.

Twelve frozen composition evaluations begin once their eight independent fits
finish. Per task: base, owner, all seven pairs containing owner, four cyclic
quadruples containing owner, all eight, and seven nonowners. Every pair is tested;
quadruples are a balanced predeclared subset. Record per-question answers,
completions, token IDs, stopping, paired lost/rescued/changed answers and timing.
Macro/worst-task scores accompany all taskwise scores. Flag standalone gains
below five percentage points; do not omit those tasks or describe their poor
all-on accuracy as evidence of composition interference.

Two compounds use held-out primitive inputs, pointer-to-arithmetic and stack
history-to-sorting. They test base, individual components, both, all eight, plus
explicit gold-intermediate and generated-intermediate pipeline controls.
Demonstrations come only from primitive training inputs; no compounds enter RL
or tuning. Compounds remain out-of-distribution diagnostics, not proof that a
model has acquired general skill composition.

Primary mixtures keep unit strength. A secondary, identical calibration for
both methods chooses one scalar per owner in a single coordinate pass over
0/.5/1/1.5 using macro accuracy on standard development questions only. It
then evaluates standard/compound test and general retention with all eight on.
LoRA scales B only; native scales the stored increments. No task weights are
retrained and no test/retention data select coefficients.

General retention: original baseline, each singleton, all eight and calibrated
all-eight on the existing 996-question MC/16384-token WikiText panel. Source,
dataset and benchmark hashes are locked into the plan. These are reused,
previously inspected task/benchmark banks: exploratory comparisons, not fresh
confirmatory generalization estimates. Module support disjointness and all-off
logit restoration are numerical audits, not guarantees of functional isolation.

Checkpoints use atomic local/durable copies and preserve all raw rollouts,
including interrupted attempts. Queue retries failures twice, resumes committed
optimizer/RNG state, refuses concurrent launchers and per-fit writers and guards 60 GiB free. A
failed dependency prevents invalid downstream evaluation. GPU smoke checks use
scripted advantages solely to exercise ownership, both regions and memory;
they do not count as scientific learning results.

```bash
cd /var/tmp/casr-mn866-composition/study_repaired
nohup setsid bash launch.sh run >> ../launcher_repaired.log 2>&1 < /dev/null &
watch -n 30 bash launch.sh status
tail -f ../launcher_repaired.log
```

Startup unit checks simulate queue jobs before the real GPU smoke tests; their
printed job names are not experiment completions. Use the main status JSON for
scientific progress.

Main status: `../outputs/composition01_repaired/status.json`. Per-worker logs:
`../outputs/composition01_repaired/logs`. Final owner tensors, raw training records and
composition results: `../outputs/composition01_repaired/fits`. Exact launched plan:
`../outputs/composition01_repaired/experiment_plan.json`. All-checks-finished marker:
`../outputs/composition01_repaired/ALL_COMPLETE.json`. Resume with the same launch command
and unchanged sources. A different source/protocol requires a fresh run directory.

The engineering repair preserves the original `study` and `outputs/composition01`
folders. The scientific plan, datasets, model revision, seeds, learning rates and
update budgets are unchanged. `execution_revision.json` pins the repaired source
hashes and records every converted checkpoint; only the trainer fingerprint in
resume signatures changes. Conversion verifies that weights, Adam state, RNG and
history remain identical. `repair_resume.json` records the resumed step per fit.
New history rows carry an execution revision; completed fits retain their original
provenance. Compare runtime using `timing_by_revision`, since the owner-only path
removes zero-update work. The old runtime measurements remain archived.

Independent fits now enable only their owner in forward passes. Base and foreign
development predictions are equivalent; owner and all-on predictions are also
equivalent while nonowners remain at initialization. Reused predictions retain
their original mask labels. CPU and deterministic RTX 2080 tests verify equivalent sampled tokens, losses,
gradients, optimizer states and updated weights. The unchanged production CUDA
backend has nondeterministic backward operations; repeated old-code updates also
vary. Deterministic settings are used for validation only. Composition still enables
all requested trained modules. The compound parser accepts ASCII and Unicode
arrows consistently with the primary scorer. Completion markers validate their
required files and hashes. Resume validates backup bundles before copying and
falls back to a valid local bundle.
