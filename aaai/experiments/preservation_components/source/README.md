# Which part of preservation protects coexistence?

Prospective follow-up to the completed small-budget preservation × conservation
ablation. User authorized all six Ogg GPUs on 31 October 2026; refusal remains paused.
Server root: `/mnt/fast1/mn866/casr-preservation-components-20261031`.

## Question and design

Separate the two specialist-preservation terms: self teacher → owner-alone student,
and foreign teacher → all-eight student. Cross each with base conservation.
The four new arms complete a 2 × 2 × 2 factorial alongside the four archived arms.

| New arm | Self KL | Foreign KL | Base conservation KL |
| --- | ---: | ---: | ---: |
| self_only | 0.5 | 0 | 0 |
| foreign_only | 0 | 0.5 | 0 |
| self_conserve | 0.5 | 0 | 1 |
| foreign_conserve | 0 | 0.5 | 1 |

The archived arms are (0,0,0), (0.5,0.5,0), (0,0,1), and (0.5,0.5,1).
Coefficients remain at their original values: removing a term does not double
the surviving term. This tests removal of a component, not an equal-total-weight
comparison or an optimal coefficient search.

Native width 4 and LoRA rank 1 × seeds 31/47/71 × four new arms = 24 new
training banks plus 24 dependent evaluations. Both MLP and attention train.
All eight tasks stay enabled; gradients update only the current task's owner.
Every arm starts from the same ORIGINAL independent step-320 specialist banks
and Adam states, not the last experiment's trained endpoints. Each adds 640
updates (80/task), native LR 3e-5 / LoRA LR 9e-5. Prompt sampling, replay rotation,
RNG seeds, three auxiliary passes (even zero-weight ones), reward KL, update
guards, evaluation masks and scorer are unchanged. Production source is frozen
before launching smoke tests. All raw rollouts, histories, snapshots, final
weights, predictions and likelihoods are retained outside the home quota.

## Predictions and endpoints fixed before launch

Hypothesis: foreign preservation will account for more of the all-eight accuracy
gain, while self preservation will protect owner-alone performance. This is a
hypothesis, not an established result. Conservation is expected to reduce WikiText
drift, but may interact with the specialist terms.

Primary: final-step, unit-strength all-eight test accuracy. Report all cells and
paired seedwise changes, not just a winning condition. Main component contrast:
foreign-only minus self-only separately at conservation=0 and conservation=1.
Also estimate each term's removal effect holding the other two terms fixed.
Secondary: owner-only accuracy, 1/2/4/8 coexistence, owner-removal effects, fixed
996-question retention panel and 16,384 WikiText tokens, per-task and compound
scores, dev-only mixture calibration, time/tokens/memory. Earlier test-bank
inspection makes this exploratory. Three seeds are not enough for strong
population claims; no exact compute match or general safety claim.

## Reusing controls without counting them twice

`controls_manifest.json` pins all 48 old completion files, 24 final parameter
banks and the old plan. `test_components.py` verifies their hashes and identical
initial inputs and unchanged inherited scientific code. It tests bitwise weight
and Adam parity with the archived updater for the four old conditions, plus
inactive-teacher isolation for each new condition. Existing gradient, rollback,
frozen-teacher and evaluation-loading tests also run. Eight real-GPU smoke jobs
exercise every new arm × parameterization, forced interruption/resume and the
96-token memory check. Main jobs are gated on all eight smoke completions.
Old controls contribute their three seeds once; they are not new replications.

Run via `bash launch.sh smoke`, then `bash launch.sh run`. Status:

```sh
bash /mnt/fast1/mn866/casr-preservation-components-20261031/source/launch.sh status
```

Six workers run independently, one per GPU. Launcher/worker locks prevent
duplicate writers. Local atomic checkpoints resume weights, RNG and optimizer
state; the scheduler checks free main-disk space. No Drive sync or home output.
