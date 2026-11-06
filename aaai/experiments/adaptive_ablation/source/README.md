# Small-budget preservation × conservation ablation

As of 31 October, Ogg GPUs **0,1,2,3,4,5** are reserved for this paper. One independent worker per GPU. Refusal research is paused by user request.
Root: `/mnt/fast1/mn866/casr-adaptive-ablation-20261030`.

Twenty-four continuation banks: native width 4 and LoRA rank 1 × three seeds
(31,47,71) × four objective modes. Every bank activates all eight independently
trained specialists and updates only its current owner. Both MLP and attention
are adapted using the inherited allocations. Native has 200,704 scalars per
task; LoRA has 183,296. This is a small-budget comparison, not exact compute matching.

| Mode | Specialist preservation coefficient | Base conservation coefficient |
| --- | ---: | ---: |
| aware | 0 | 0 |
| preservation_only | 1 | 0 |
| conservation_only | 0 | 1 |
| preserve | 1 | 1 |

Preservation averages frozen self-specialist → owner-alone KL and frozen
foreign-specialist → all-eight KL. Conservation uses frozen base → all-eight
KL on training-only natural text. All arms execute the same three auxiliary
passes, including zero-weight controls. Original reward-prefix KL and update
guards are unchanged. No retention or test answers enter training.

Start from original step-320 specialist banks, including each owner's Adam state.
Run 640 more updates (80/task), native LR 3e-5 and LoRA LR 9e-5. The fixed final
checkpoint is primary. Twenty-four dependent evaluations reuse the existing
1/2/4/8 composition masks, retention, compounds, and secondary calibration.
Previously inspected banks make this exploratory research.

CPU checks cover both new loss combinations, gradient ownership, frozen teachers,
rollback and final-bank evaluation. Six GPU smoke jobs cover both parameterizations,
intentional interruption/resume, 96-token generation/backward and a 7.25 GiB ceiling.
Production is gated on all six smoke completions.

```sh
bash /mnt/fast1/mn866/casr-adaptive-ablation-20261030/source/launch.sh status
```

The six-GPU launcher lives outside the pinned scientific source at
`operations/launch_all_gpus.py`; it checks the original plan and source hashes,
then uses the existing scheduler. Logs are in `launcher_all_gpus.log`. The switch
resumes committed checkpoints and leaves the protocol unchanged.

For a stopped launcher, use `launch.sh run`; completed jobs are validated and skipped,
and partial jobs resume committed weights/RNG/Adam. Launcher and worker locks reject
duplicates. Logs, raw rollouts, development predictions, snapshots, final banks and
full evaluations stay on the main disk. No Google Drive sync. Caches and temporary
files are outside the home quota. Original studies are untouched.
