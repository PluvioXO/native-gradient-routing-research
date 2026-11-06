# Harder branching-navigation capability/refusal budget sweep

4 November 2026. New dataset and output directory; previous experiments stay intact.

## Requested comparison

Shared native widths **1, 2, 4, 8**, plus a full-model baseline. Native updates are in **one MLP block: zero-based 11, block 12 of 24**. They are not repeated across the model. Each channel couples matching up/gate rows and down columns. Attention and all other pretrained weights stay frozen in native arms. Full-model training updates all parameters.

| Arm | Trainable native scalars |
|---|---:|
| Shared 1 | 2,688 |
| Shared 2 | 5,376 |
| Shared 4 | 10,752 |
| Shared 8 | 21,504 |
| Full | Actual model count recorded |

Nested channel masks use the same seeded ordering: width 1 is contained in 2, 2 in 4, and 4 in 8. Each arm independently learns its capability from the same pinned pretrained Qwen2.5-0.5B-Instruct (`7ae557604adf67be50417f59c2c2f167def9a775`). There is no dependence of width 8 on a successful width-4 teacher. This tests budget-dependent learnability, not disjoint routing. One seed, 31, for this harder-task pilot.

## What makes the task harder

The previous task followed one successor in a ten-node cycle for one or two hops. The new task has **16 nodes with two distinct labeled outgoing edges per node (L/R)**, varied directed graph topology, and an explicit sequence of left/right commands. The answer is the full visited-node trace after the start. Distractor branches can lead to marked nodes that are not on the commanded path.

Each graph has two self-avoiding six-command plans from the same start, with opposite first commands. Shorter queries use their prefixes. Counterfactual commands therefore select different paths in the same graph. Train depths **1, 2, 3, 4** together; evaluate unseen lengths **5, 6** separately as out-of-distribution probes. Two prompt wordings are used during training and development; a third wording is held out and tested on paired final-test graphs. No claim of arbitrary-depth generalization follows from the primary test.

The refusal rule remains: refuse iff the marked node occurs **inside the requested path, excluding start and endpoint**. Train and evaluate every intermediate position, including the third visited node of a four-step path. Negative cases include no marker, marked start, marked endpoint, an off-route branch destination, and an unrelated outside node. Thus a one-hop endpoint must be answered even though that same node is an intermediate blocked node for a longer request. The rule is learned from targets and is not stated in the model's input prompt.

All graphs, starts, commands, paths, marker roles, prompt templates and labels are saved. Split graphs are disjoint; the wording probe intentionally reuses test graphs for a paired format comparison. The 5/6-step probe uses a separate graph bank. Data seed 41106; exclude all 2,784 known previous world hashes. Maximum training prefix is checked against 512 tokens before launch; prompts are never silently truncated.

## Fixed acquisition budget and qualification

Every arm receives the same two-stage supervised curriculum, with a fresh optimizer at each stage:

1. **512 basic updates**, depths 1 and 2, two examples of each per update.
2. **2,048 hard updates**, depths 1 through 4, one example of each per update.

All 16 marker placements and the marker-free version share the same capability answer during acquisition. Fixed horizons replace the previous tiny-perfect gate: an error in a tiny marker group cannot prevent a budget arm from attempting the harder curriculum. Keep loss and per-depth learning curves even when a narrow arm stagnates.

At each stage save final and development-selected best weights. Start the hard stage from the best basic checkpoint. Rank acquisition checkpoints by feasibility, worst-depth exact path accuracy, overall path accuracy, collateral retention and earlier step. A hard-stage capability parent qualifies for policy training only with at least **80% accuracy at every depth** and at least **70% in every marker/depth case**. If it fails, retain and evaluate its capability endpoints, skip its policy fits explicitly, and continue all other arms. Failure is a measured budget/recipe outcome, not a hardware error or a proof of absolute incapacity.

Use 256 train graphs, each with two plans. Development has 24 graphs / 48 plans; primary test has 128 graphs / 256 plans. The paired held-out-wording probe uses the first 64 test graphs. OOD has 64 separate graphs; observations have 16. Data generation is independent of performance.

## Policy calibration and preservation

Qualified arms each get the same three-factor LR search: **1, 0.3, 0.1**, multiplying sparse 3e-4 or full 1e-5. Each policy candidate receives **1,024 updates**, with no behavior-based early stop. Two allowed depths are sampled per update: (1,3), then (2,4), alternating. Each is paired with a blocked example from the same graph/command plan; blocked intermediate positions rotate. This balances easy-depth rehearsal, hard tasks and late-position decisions.

Losses match the earlier shared method: allowed trace NLL; balanced positive/negative refusal-prefix decision loss; refusal continuation and EOS grammar (0.1); full-response capability-teacher KL (0.1); frozen-base arithmetic/copy conservation KL (0.05). For shared native arms every component updates the same selected channels. Conservation is also enabled in all acquisition arms. AdamW, zero decay, clip norm 1, 32-update warmup. All arms use FP32.

To bound RAM and disk use with longer answers, preserve the full-response teacher distribution on a fixed, performance-independent bank: eight training graphs, both plans, five allowed marker cases at every depth (**320 rows**). Cache every completion-token logit including EOS in FP16; compute KL in FP32. Each policy update includes a teacher-KL example at each of its two sampled depths. The gold-answer NLL continues sampling the complete policy training bank. This bank reduction is a declared change from v3, not a claim of an identical training recipe.

Evaluate development every **64 updates**, saving raw predictions, losses, gradients and LR. Select best policy checkpoints using only primary development: at least 80% allowed path accuracy in **every allowed case**, at most 10% false refusal in each, at least 90% correct refusal in **every intermediate-position case**, arithmetic/copy exact accuracy no more than 10 points below base, and each development benchmark within 10 points with at least 80% of baseline valid-option probability mass. Retain best infeasible fallbacks explicitly if none passes. Rank feasible first, worst task-case quality, balanced allowed/refusal score, collateral retention, earlier step; lower LR breaks exact ties across candidates.

No all-width pass gate blocks the study. There is no automatic additional-seed stage in this pilot. Test both best and final capability endpoints, and best/final endpoints for each selected policy recipe, only after all enabled training finishes. Test data, held-out wording, longer paths and refusal-direction metrics never select rates or checkpoints.

## Measurements and persistence

Record exact traces, endpoint accuracy, refusal, false refusal, malformed/truncated output, and per-depth / marker / intermediate-position scores. Preserve arithmetic/copy exact scores as the original retention metric, and also record conservative answer-parsed arithmetic as a separate diagnostic. ARC-Easy and WinoGrande include both forced-choice accuracy and valid-option probability mass. Formatting changes must not be described as erased arithmetic knowledge.

Directions remain **observation only**: no penalty, projection, ablation, steering or direction-based selection. Collect all 24 prompt-end block outputs before final norm, original XSTest responses and matched four-hop endpoint-versus-late-intermediate responses. Parameter hashes must match before/after each observation. Compare to the corresponding selected capability parent. These are geometric associations, not causal circuit isolation or attack-resistance results.

Use Weatherwax's six RTX 2080s: four independent native acquisition workers use one GPU each; full-model training uses two. Models, caches, raw outputs, optimizer checkpoints, best/final weights and activation tensors stay under `/mnt/fast1/mn866/casr-colocalization-branching-20261104`, outside the home quota. Reuse the existing model cache. Atomic writes and a 40 GiB free-space guard apply. Checkpoint ownership includes run/source/data identity, mask, seed, stage, selected parent step and teacher-cache identity. Do not modify old runs or Ogg.

Validation covers graph solutions, branching counterfactuals, all refusal positions, held-out splits, nested masks and frozen originals, real optimizer-state resume, fixed acquisition budgets despite early success, all-depth policy rehearsal, late-position gradients and best-checkpoint selection. A relaxed, small-data hardware smoke exercises the complete five-arm pipeline before research launch; its metrics are not scientific results.

This is a harder synthetic reasoning pilot at one chosen block in a small model. Capacity failures, sensitivity to optimization and absent OOD generalization remain possible outcomes. No adversarial SFT/RL refusal-removal experiment is included.
