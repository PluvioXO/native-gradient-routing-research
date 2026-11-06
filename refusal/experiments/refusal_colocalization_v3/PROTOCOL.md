# Mixed-hop shared versus disjoint native updates

4 November 2026. This is a new run; v1/v2 weights and results remain intact.

## Question and conditions

Can native updates learn a two-hop pointer capability plus selective refusal while preserving one-hop instructions, unmarked inputs and collateral behavior? Does sharing update locations help once initialization and parameter budgets are controlled?

Qwen2.5-0.5B-Instruct, pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP32, block 12 (zero-based 11). Attention stays frozen in native/scattered fits. All experiments are supervised, not RL or an adversarial-removal evaluation.

| Policy condition | Capability/policy support | Unique trainable scalars |
|---|---|---:|
| Shared 4 | Same four native MLP channels | 10,752 |
| Disjoint 4+4 | Four capability plus four separate policy channels | 21,504 |
| Shared 8 | Same eight channels for both losses | 21,504 |
| Scattered 4-equivalent | Shared scattered mask across MLPs | 10,752 |
| Full | All pretrained parameters | Report actual count |

Native channels couple up/gate rows to down columns. Shared 8 uses **exactly the union** of the disjoint fit's capability and policy channels. All three native policy conditions start from identical effective pretrained-plus-capability weights: embed the native4 teacher's four trained channels into Shared 8, leaving its other channels zero. Validate the resulting materialized MLP weight hash. This controls initialization more tightly than acquiring an independent native8 capability teacher. Shared4 versus disjoint4+4 matches per-loss width; Shared8 versus disjoint4+4 matches total trainable scalars and channel locations. These are not claims of matched compute or identical per-loss degrees of freedom.

The full and scattered controls must acquire the same task independently. A failed acquisition is recorded, and no policy is trained on an unqualified capability parent.

## Data and curriculum

Fresh ten-node cycle worlds, data seed 41105; explicitly exclude all 1,856 worlds from v1 and v2. Banks: tiny 16 worlds; train 512; development 96; final test 256; observational 48. World banks are disjoint. Preserve visited-node trace supervision and the frozen prompt wording from v2.

Acquisition stages, each with a fresh optimizer:

1. Tiny one-hop fit: at most **512** updates (extended from 256 for the nearly passing scattered control), gate 98%.
2. Fresh one-hop fit: at most 1,536 updates, gate 90%.
3. **Mixed one-/two-hop fit:** at most 2,048 updates, two examples of each depth in every four-example update, gate 90% in every depth/marker case.

Every gate needs two consecutive passing evaluations. Evaluate every 32 updates. For acquisition training, every world/depth appears without a marker and with all ten marker placements, all with the correct path answer. Conservation is enabled consistently in all acquisition conditions. Widths and learning rates are not selected from final test results.

Policy refuses iff the marked node is an intermediate visited node. A one-hop path has no intermediate node, so **all one-hop examples must be answered**, including a marked endpoint. There are nine evaluation/training cases per world:

- One hop: unmarked, marked start, marked endpoint, marked outside path (all allowed).
- Two hops: unmarked, marked start, marked endpoint, marked outside path (allowed), marked intermediate node (blocked).

The marked one-hop endpoint and marked two-hop intermediate case share the same graph, start and marker, making hop count necessary to distinguish the target behavior. All cases are equally represented within each depth. Each policy update samples one allowed case from each depth and its same-world blocked two-hop counterpart. Include a full-response capability-teacher KL on **each sampled depth**, rather than traversing only the one-hop prefix of a concatenated teacher cache. Alternate arithmetic/copy conservation examples.

The model is never prompted with the new symbolic refusal rule. No path answer appears in refusal targets.

## Losses, calibration and checkpoint selection

Same weights across conditions: allowed-path NLL to C; positive/negative refusal-prefix decision loss to R; refusal continuation/EOS grammar loss to R (weight 0.1); full-response allowed teacher KL to C (0.1); full-response frozen-base arithmetic/copy KL to C union R (0.05). Both disjoint increments remain active in every forward pass. Shared means C=R. AdamW, zero weight decay, clipping at 1, 32-update warmup. Acquisition LR: sparse 3e-4, full 1e-5.

Seed 31 policy calibration gives **each condition the same three-factor search**: 1, 0.3 and 0.1 times its base LR (sparse: 3e-4, 9e-5, 3e-5; full: 1e-5, 3e-6, 1e-6). Each fit runs **512 policy updates**, with no behavior-based early stop. Numerical/hardware failures remain explicit. This prevents the previous retention guard from giving unstable conditions a shorter recovery horizon. Evaluate task cases, collateral generation and development ARC-Easy/WinoGrande every 32 updates.

Save the final checkpoint and a best development checkpoint. Feasibility requires:

- Allowed exact path accuracy at least 80% **in each of the eight allowed cases**.
- Blocked refusal at least 90%; false refusal at most 10% in each allowed case.
- Arithmetic/copy accuracy no more than 10 percentage points below the frozen base.
- Each development benchmark accuracy no more than 10 points below base, with at least 80% of baseline valid-option probability mass.

Rank checkpoints by feasible first, then worst task-case quality, balanced allowed/refusal score, collateral retention, and earlier update. If none is feasible, retain the best fallback with an explicit failure flag; do not relabel it successful. Across LR candidates use this same rank, with lower LR breaking exact ties. This is development-only selection and never reads directions or final tests.

After all five conditions have a feasible seed-31 development checkpoint, freeze each condition's chosen LR and automatically repeat the same comparison for **seeds 47 and 71**, without further tuning. If this validation gate fails, record the failures, do the available final diagnostic evaluations and stop for a recipe audit. Replicate results must include failures, rather than dropping difficult seeds.

Only after calibration and all enabled replication training have ended, evaluate the frozen base, acquired-capability controls, and **both best and final** endpoints of each selected policy recipe. Retain every LR candidate's training history, rolling resume checkpoint, final weights, best weights, mask, development outputs and selection decision. Intermediate non-best parameter tensors are not all retained; their complete development predictions are.

## Observations, engineering and storage

Directions are **observational only**: zero penalty, no projection, no ablation, no steering, and no direction-based selection or gating. Read-only block-output hooks collect all 24 layers at prompt end before final norm. Reuse independent XSTest fit/held-out prompts; save raw activations, refusal-language flags and both old- and new-prompt generated responses. Compare paired two-hop endpoint-marker versus intermediate-marker contrasts to base and capability parent. Report content/format confounds and low-count fallbacks; cosines cannot establish an independent causal circuit.

Each observation asserts unchanged parameter hashes. Checkpoints identify run, sources, data, seed, geometry, mask, stage and parent. Test actual resumed updates and optimizer moments, terminal acquisition resume, gradient routing, Shared8/disjoint mask equality, identical native warm starts, and best-feasible selection. On a hardware smoke, exercise the complete calibration-to-confirmation-to-evaluation path with small data and deliberately relaxed gates; these relaxed gates cannot support research conclusions.

Six RTX 2080s on `mn866@weatherwax.cs.bath.ac.uk`. Full-model training uses two GPUs; sparse training and inference use one. A dependency scheduler fills freed GPUs as capability parents qualify. Store all output on `/mnt/fast1/mn866/casr-colocalization-mixed-20261104`, outside the home quota, sharing the existing model cache. Atomic writes and a 40 GiB free-space guard apply. Hard-link root final checkpoints to phase finals to avoid duplicate full-model weights. Do not alter old runs or Ogg.

Main limitations: synthetic task, fixed block and small budgets; a three-seed confirmation is conditional on calibration success. This experiment tests learning and coexistence, not adversarial robustness.
