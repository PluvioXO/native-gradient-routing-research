# Selective-refusal sampler pilot

Runs on `mn866@garlick.cs.bath.ac.uk` under
`/mnt/fast1/mn866/casr-refusal-sampler-20261105`.

**Completed 5 November at 23:37 UTC:** 18/18 fits, ten evaluation jobs
(16 endpoints), no worker failures. None of the 18 candidates passed all
development gates. Balanced sampling increased aggregate correct refusal,
but worst-case quality fell for four/eight native channels; the full-model
selected recipe has substantial collateral loss. This is one pilot seed.
[Audited findings](analysis/FINDINGS.md) · [eight-page graphs](analysis/sampling_graphs.pdf) ·
[audit](analysis/audit.json).

[Protocol](PROTOCOL.md): 18 fits, one seed, native 4/8 channels at MLP block
12 plus full-model controls; original versus balanced positive sampling;
three learning rates and 1,024 updates each. Saved V4 capability parents
are imported and checksum-verified, with fresh policy optimizers. This is
not a longer continuation of the failed V4 policies and includes no removal
attack or automatic additional-seed stage.

The actual allowed, blocked, teacher and conservation IDs are stored at every
update and checked against the frozen schedules during analysis. All learning
curves, candidate rates, raw evaluations, teacher caches, selected/final
weights, optimizer/RNG state and observational activations remain on the
server's main disk. The 40 GiB space guard and atomic writes protect resume
checkpoints; these are same-host copies, not an off-host backup. No Drive sync.

The runtime reuses the [frozen V4 source archive](launch/v4_source.tar.gz).
`prepare.py` validates and materialises it, with small declared changes in
[the lineage record](launch/engine_lineage.json). The old V4 study is untouched.
The imported cache/checkpoint inventory is in [import_sources.json](launch/import_sources.json).

Monitor on Garlick:

```bash
bash /mnt/fast1/mn866/casr-refusal-sampler-20261105/source/launch.sh status
tail -f /mnt/fast1/mn866/casr-refusal-sampler-20261105/launcher.log
```

Per-job progress is under `results/jobs/*/progress.json`; first counters appear
after 64 updates. The frozen-base development/teacher preparation runs first.
Full-model training uses two GPUs, native training one. GPUs occupied by
another process are left alone. All training precedes new held-out evaluation.

After completion, the detached pipeline creates `analysis/RESULTS.md`,
`results.json`, all-candidate/learning/case/loss/contrast CSVs and the eight-page
`sampling_graphs.pdf`, including retention relative to base and capability
parents. These files are now local, with hashes checked. The refusal paper
incorporates the completed findings; the one-seed/inspected-bank and
different-selected-LR limits remain explicit. Runtime sources are unchanged.

Only restart after confirming the previous launcher and all its workers have
stopped. Preserve the frozen sources and output directory:

```bash
bash /mnt/fast1/mn866/casr-refusal-sampler-20261105/source/launch.sh pipeline
```

Source preparation and CPU checks, using an environment with the recorded
Torch/Transformers dependencies:

```bash
python prepare.py
CUDA_VISIBLE_DEVICES='' python -m unittest -v test_sampling test_study
```

Hardware validation uses separate smoke data, fits and outputs; its scores
are not scientific results. It covers both schedules and all budgets, all
selected endpoint evaluations, unchanged observation hashes and replaying
updates 2/3 from update-1 optimizer/RNG states for native eight and full-model
fits. Production additionally validates graph truth, split exclusion,
teacher/parent/data identities and token counts. Updates/examples are matched;
FLOPs are not measured.
