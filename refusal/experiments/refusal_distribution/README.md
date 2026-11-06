# Fixed-budget refusal distribution study

**Paused by user request on 31 October 2026.** All nine jobs have finished
(seven qualified acquisitions and two acquisition failures); completed data are
preserved. Do not launch follow-ups until the user resumes this research.
GPUs 3,4,5 have moved to the AAAI capabilities study. The completed experiment
used one model split over three GPUs, not three independent replicas. Root:
`/mnt/fast1/mn866/casr-refusal-distribution-20261030`.

| Acquisition | MLP channels per selected layer | Layers (one-based) | Trainable scalars |
| --- | ---: | --- | ---: |
| Single | 4 | 12 | 10,752 |
| Dual | 2 | 8,16 | 10,752 |
| Four | 1 | 3,9,15,21 | 10,752 |

All layouts have mean layer 12. Span and computation paths still differ. Three
seeds (31,47,71) give nine acquisition jobs, each followed by two unrestricted
full-model removal trajectories if qualified. Acquisition is supervised synthetic
blue-category refusal; matched green-category prompts should remain answerable.
The same data, base eligibility and per-wording acquisition gates as the parent
study are retained. Unqualified acquisition is a result, not silently discarded.

Removal rates: 1e-6 and 1e-5. Evaluate before removal, every update through 16,
then every four updates through 64. Four examples/update; 512-example training
split. Measure refusal, correct answers, other outputs, alternate wording,
arithmetic, ARC-Easy and WinoGrande controls. Alternate wording participates in
acquisition development; this is not a wholly unseen prompt-family benchmark.

FP32 pinned Qwen2.5-0.5B-Instruct. One GPU owns embeddings/head and layers 1–8;
the other two own layers 9–16 and 17–24. Thirteen CPU tests passed. The actual
three-GPU smoke test checked baseline eligibility, sparse acquisition, full-model
Adam updates, evaluation and checkpoint reload (peak 5.35 GiB on the busiest GPU).

Progress is `progress.json`; per-job logs are under `logs/`. Saved manifests,
training losses, response-level evaluations, learned/final weights and active
resume checkpoints stay on the main disk. Completed removal keeps final weights;
the inherited runtime deletes its rolling Adam resume file on completion.
No Drive sync, no writes to home caches. A 60 GiB free-space guard stops new jobs.

Restart a stopped launcher using the same environment/caches and
`source/run_study.py launch --root ROOT`; completed/inconclusive jobs are skipped.
The launch receipt and preflight report record the exact configuration.
