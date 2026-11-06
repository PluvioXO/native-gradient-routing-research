# Saved-model refusal decision diagnostic

Fixed 6 November 2026 before collecting the new first-token scores. This
is a diagnostic of the previously inspected V5 bank, not an independent
confirmation or a new training experiment.

## Models and hypothesis

Use the six development-selected V5 recipes at their fixed final update
1,024: native 4/8 channels and full-model, each original/balanced sampling.
Use their exact saved parameters, masks, prompts and pinned FP32
Qwen2.5-0.5B-Instruct revision. No optimizer, gradient, new training,
direction intervention, or checkpoint selection is permitted. Each phase
checks all model parameter hashes before/after. Native support remains
one MLP block (12), with attention frozen.

The original loss uses `refusal_logit - logsumexp(16 node logits)`.
Original greedy decoding compares individual vocabulary tokens. The
primary question is how often an allowed example's literal greedy refusal
occurs despite the trained score preferring the node group, and whether
changing this decision improves allowed accuracy without unacceptable loss
of blocked refusal. Score separation is assessed descriptively, per case
as well as in aggregate, without seed-level significance claims.

## Four decision rules

1. **Original greedy:** immutable audited V5 responses, with a 32-prompt
   development-only generation replay per model before production. Hardware
   checks must reproduce these responses exactly.
2. **Restricted argmax control:** choose the largest of the refusal-start
   token and 16 node tokens, with ordinary vocabulary-ID tie breaking.
3. **Probability-sum gate:** refuse if the original loss margin is strictly
   greater than zero; otherwise answer. Zero ties answer.
4. **Development-calibrated sum gate:** the same rule with one threshold
   per model, fixed from development data only.

For rules 2--4, a refusal returns the literal `REFUSE`. Answering forces
only the first token to the most likely of the 16 trained node tokens;
subsequent tokens use ordinary greedy generation with the original token
cap. All three rules share this identical answer continuation and refusal
format. Thus sum-versus-restricted-argmax isolates the aggregation rule;
restricted-argmax-versus-original also includes vocabulary/grammar effects.
Raw original responses remain separately reported. Every forced answer is
scored for full path correctness, formatting and truncation.

This is a **task-specific decoder intervention** on the toy navigation
interface. It does not modify intrinsic greedy behavior, establish attack
resistance, or retroactively qualify V5's failed greedy policies.

## Calibration and holdout

Six workers first collect development scores and counterfactual node-first
responses. Only after all six development phases finish may the coordinator
freeze `calibration.json`. No held-out score collection is allowed before
this global barrier. Smoke examples come exclusively from development.

For each model, candidate thresholds include zero, every distinct finite
development margin, and an all-refuse extreme below the smallest score.
The rule is `margin > threshold`. Rank candidates by: feasibility under
the original case-wise gates; worst joint quality; balanced allowed/refusal
accuracy; lower aggregate false refusal; smaller absolute threshold; then
smaller threshold. Keep infeasible fallbacks explicitly.

Original gates remain: >=80% allowed path accuracy in every allowed case,
>=90% blocked refusal in every blocked case, <=10% false refusal in every
allowed case, and the original general/benchmark retention and option-mass
checks. Retention is copied from the same immutable V5 final model; no gate
is applied to arithmetic/copy or benchmarks. It is not newly established
for a universal decoder. The full-model parents already have collateral
damage, and this is retained in interpretation.

Evaluate primary test, held-out wording and depths 5/6 with the locked
thresholds. Record A/R/false refusal/worst-case quality, changed decisions,
recovered correct paths, blocked refusals lost, first-token vocabulary
coverage, score distributions, and AUC for both sum and max margins.
Test ROC curves are descriptive and never select a threshold. Report all
six models, including null, worse and infeasible outcomes.

## Reproducibility and persistence

Old source/checkpoint/reference identities are pinned using the audited V5
bundle. Original runtime sources and outputs are never edited. Six CPU
logic checks and six miniature GPU replay/score/chunk-identity checks precede
production. Append-free atomic chunks support deterministic resume; source,
checkpoint, tokenizer and row identities must match. Check free space before
writes and keep outputs on the main disk outside the home quota. No Drive
sync. Skip GPUs occupied by other processes.

Store all score vectors, generated counterfactual responses, provenance,
calibration candidates, aggregate and per-case tables, plots and logs.
Only compact results return to the local project; large raw measurements
remain on Garlick. Any subsequent loss change or trace-first experiment
requires a new study rather than changing this fixed diagnostic.
