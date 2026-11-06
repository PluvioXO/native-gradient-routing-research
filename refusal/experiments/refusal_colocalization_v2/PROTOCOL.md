# Corrected capability/refusal pilot: observational direction comparison

Date: 4 November 2026. One seed, 31. This replaces the failed pilot's training recipe; the old run remains intact.

## Explicit user constraint

Compare and record refusal-associated directions, but **do not penalise, project out, ablate, rotate, or otherwise steer them**. Direction measurements never enter the loss, acquisition gates, checkpoint selection, width selection or learning-rate schedule. Read-only forward hooks return no replacement activations. Parameter hashes must match before and after each observation pass. No causal direction intervention is performed.

Consequently these measurements establish associations and geometric overlap, not independence from every inherited refusal mechanism.

## Compute and baselines

Use Weatherwax's six RTX 2080s, Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP32, and the existing CUDA 11.8 environment. Store everything under `/mnt/fast1/mn866/casr-colocalization-observe-20261104`, outside the home quota. Reuse the existing model cache. Leave Ogg alone.

Initial resource allocation (the scheduler assigns physical GPU IDs):

| GPUs | Capability fit |
|---|---|
| Two GPUs | Full-model learning, conservation off |
| Two GPUs | Full-model learning, full-response conservation on |
| One GPU | Four native MLP channels, conservation off |
| One GPU | 128 native MLP channels, conservation off |

Native channels remain in block 12 (zero-based 11): matching up/gate rows and down columns. One channel is 2,688 scalars. The smallest qualified native width is used for the subsequent placement pilot. Add a scalar-matched scattered capability control at that width. If an acquisition gate fails, record that failure rather than interpreting a subsequent refusal fit as a capability-binding test.

## Capability curriculum and gates

Use fresh, disjoint map/world banks for training, development, final test and observations, explicitly excluding the 928 worlds in the previous pilot. Each graph is a ten-node cycle. The answer is the sequence of visited nodes after the starting node, e.g. `b c` for two hops. This explicit path supervision is a deliberate change from the old endpoint-only format. It makes the acquisition control easier to diagnose. Refusal responses contain no path or answer.

For every training world, include a marker-free query and all ten marker placements, all with the same capability answer. Marker placement is independent of the path and answer. Put the start/hop query last. These counterfactuals break the old marker-successor shortcut.

Progress through tiny-set memorisation, one-hop lookup, then two-hop following. Check every 32 updates. Require two consecutive passing checks: at least 98% path accuracy on the tiny training set, then at least 90% held-out path accuracy overall and in each available marker category. Two-hop development includes marker-free, allowed and future-blocked cases. Failed gates stop that fit's progression. Do not use the final test for tuning or progression.

Prespecified maxima: 256 tiny-set updates, 1,536 one-hop updates and 2,048 two-hop updates. Use four supervised examples per update, gradient clipping at 1, AdamW with zero weight decay, 32-update warmup, peak LR 1e-5 full / 3e-4 sparse. Conservation-on weight is 0.05, measured over all completion tokens including EOS on an unrelated arithmetic/copy rehearsal set. Off means zero, with no direction-specific term in either case.

## Gated policy pilot

Start shared and disjoint native conditions from the **same qualified capability checkpoint**. Use the scattered condition only if its own acquisition succeeds. Include a qualified full-model policy reference. No channel sweep is repeated yet.

The new policy refuses when the marked node is an intermediate visited node, otherwise returning the correct path. Use matched allowed/blocked worlds. For two hops there is one intermediate node. Allowed markers may include the start or endpoint; correctness is still defined by the actual path.

Train these components with consistent weights across placements:

- Allowed path NLL: capability group C.
- Balanced binary refuse-versus-answer decision loss: policy group R, using **both positive and negative examples**. Compare the refusal-prefix logit with the log-sum-exp of the ten node logits; this does not reveal the correct answer identity to R.
- Refusal continuation grammar NLL: R, omitting the first refusal-prefix decision token to avoid duplicating positive decision supervision.
- Full-response capability-teacher KL on allowed examples: C, weight 0.1.
- Full-response frozen-base conservation KL on unrelated rehearsal examples: C union R, weight 0.05.

R=C for shared; C and R are disjoint for separated. Both increments remain active in every forward pass. Different(k) has twice the unique scalars of Shared(k); report this rather than calling it compute matched. Full-model reference updates all weights. The scattered control uses the shared unstructured mask across MLP matrices.

Use at most 512 policy updates, fixed endpoint unless a prespecified retention failure stops the run. From update 128, two consecutive checks with allowed path accuracy more than 20 points below the teacher or collateral exact accuracy more than 20 points below the base stop the fit and mark it failed. This guard does not depend on direction measurements. Qualifying selective behaviour requires at least 90% blocked refusal, 80% allowed path accuracy and at most 10% allowed false refusal.

## Observation-only direction measurements

Use an independent, pinned XSTest calibration subset, split by focus into fit and held-out observation groups. Record actual generated refusal-language flags and raw completions. Fit prompt-end, pre-final-norm block-output contrasts at all 24 layers:

1. Baseline refused-versus-answered contrast when enough observed examples exist; otherwise explicitly label an unsafe-minus-safe content contrast as a fallback candidate.
2. The same old-prompt contrasts after capability and policy training, including a fixed baseline response partition to avoid relabelling the comparison.
3. New-policy matched blocked-minus-allowed contrasts, and their training-induced change relative to the capability checkpoint (baseline for capability-only observations).

Save raw activations, candidate vectors, norms, signed cosines, validity flags and old-prompt responses. Candidate directions are not causally validated. Low refusal counts and content/output-format confounds must remain visible. A cosine is not a proof of a shared or independent circuit. Do not choose training layers or checkpoints from these values.

## Evaluation, retention and engineering

Record full paths, endpoint accuracy, output validity, blocked refusal and allowed false refusal separately. Evaluate frozen and capability-only baselines too. Restore the existing ARC-Easy/WinoGrande choice-log-likelihood evaluator, with disjoint development/test subsets; retain arithmetic/copying as small probes.

Every checkpoint must identify its run, job, stage, placement, mask and parent checkpoint. Test a real resumed optimiser update against uninterrupted execution. Save losses, gradient scales, learning rates, exact masks, optimiser state, all evaluation predictions and observation activations. Check frozen originals and zero-delta equivalence. Atomic writes and a 40 GiB free-space guard apply. Final task-test evaluation waits until all scheduled training fits finish.

The pilot uses supervised learning, one seed and a small synthetic task. It does not test adversarial removal. Observing the old refusal direction never changes how the model learns.
