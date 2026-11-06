# Refusal loss versus decoder diagnostic

Read-only follow-up to the [Garlick sampling pilot](../refusal_sampler_v5/analysis/FINDINGS.md).
Six fixed final checkpoints: native 4/8 and full-model, original/balanced.
Completed on all six GPUs on **6 November 2026 at 05:14:45 UTC**, without
worker failures. [Findings](analysis/FINDINGS.md) · [Five-page graphs](analysis/decoder_graphs.pdf)
· [Audit](analysis/audit.json) · [Run status](RUN_STATUS.md).

Sum-versus-max disagreement covers 87/5,249 (1.66%) primary-test false
refusals, with 70 correct paths recovered. Threshold calibration shifts
the allowed/refusal tradeoff; no tested rule passes case-wise qualification.
Counterfactual node-first allowed accuracy remains 97.7% for balanced
eight-channel training versus 70.4% ordinary accuracy, showing accessible
path behavior under changed decoding rather than an inseparable circuit.

[Protocol](PROTOCOL.md) freezes score collection, two decoding controls and
development-only threshold selection. No model training is run.

The six logic tests cover the sum/max counterexample, AUC ties, calibration,
case-wise gates and rejection of held-out calibration data. Separate GPU
smokes replay 32 original development responses per model, compare generation
scores to the training forward, and check parameter/chunk identities.

All six development phases finish before any held-out collection. The
coordinator then locks one threshold per model and evaluates primary test,
wording and longer paths. Reports distinguish original greedy from
restricted argmax, fixed-zero sum gating, and calibrated sum gating. The
three constrained rules share the same answer continuation and literal
refusal formatting, so vocabulary/grammar changes remain separate from
the aggregation comparison.

Server root: `mn866@garlick.cs.bath.ac.uk:/mnt/fast1/mn866/casr-refusal-decoder-20261106`.
Atomic chunks resume without changing original V5 data/checkpoints. All
outputs stay on the main disk, outside the home quota; no Drive sync.
GPU allocation skips devices occupied by other processes. No new optimizer,
gradient update, attack, or direction manipulation is performed.

The retained `hardware/status.json`, `results/status.json` and `launcher.log`
record completion. CPU analysis creates `analysis/results.json`, per-case and
disagreement CSVs, threshold candidates, and `decoder_graphs.pdf`. The
post-run saved-record verifier also produces `audit.json` and `formatting.csv`.

This is a one-seed diagnostic on inspected data and a known toy-task
interface. An external decoder does not demonstrate intrinsic refusal
robustness. Original V5 outcomes remain as originally reported. Cached
general retention belongs to the original model; the gate is not applied
to non-navigation tasks.
