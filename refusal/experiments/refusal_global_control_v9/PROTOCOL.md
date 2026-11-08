# V9 — matched full-model continuation versus gradient coupling

Fixed on 8 November 2026 before fresh model evaluation. Follow-up to V8's
benign-stability result. Harmless graph navigation, not harmful-request
defense. No claim of inseparability, new circuits, or irreversible erasure.

## Primary comparison

Reuse the identical qualified four/eight-channel V3/V7 starting weights.
All subsequent defender and attacker updates are unrestricted full-model
updates. Compare clean continuation (beta zero) with V8 global coupling
(beta .01), with identical AdamW LR 1e-6, data, rehearsal, FP32, clipping,
device map and 128 outer updates. Reuse V8's unchanged numerical kernels;
the coupling derivative uses forward finite differences at epsilon .01.
The common loss is allowed-answer NLL + blocked REFUSE NLL + .1 general KL.

Three paired continuation/data seeds (101,211,307), two fixed starting
models and two arms give 12 defender fits. These are not three independently
acquired skill models. Pair both arms on the same newly generated training
worlds and batch schedule. Count extra gradient-coupling computation separately.

Observe defenders at 0,16,32,64,96,128. Apply V8's fixed qualification gates
against the original policy on NEW development panels. For each parent/seed,
choose the latest scheduled POST-update checkpoint where BOTH arms qualify.
Both arms consequently receive the same number of effective outer updates.
Never choose by cosine or test results. If no common point qualifies, report
paired acquisition failure; do not replace a seed or attack a damaged policy.

## Fresh data

Exclude all V8/V7 navigation worlds and inherited exclusions. Generate common
dev/test/probe banks with 128/512/32 worlds, each with all nine navigation cases.
For each seed generate three mutually disjoint 512-world banks: defender
training, removal and repair. All new banks are world-disjoint globally.
General rehearsal has 64 addition and 64 copying prompts; development has
128/task and test 256/task. Exclude previous rehearsal/dev/test prompts and
keep all new splits disjoint. Public question algorithms/templates remain shared.

Use fresh-from-V8 ARC-Easy and WinoGrande questions, 128/task for development
and 256/task for test. Keep the original source files, prompt formatting,
choice scoring and 384-token eligibility rule. Record raw source SHA256.
No synthetic test substitutes for a missing benchmark panel.

All baseline, defender qualification, checkpoint selection and stopping
decisions use development/probe data only. No model sees final-test prompts
until every planned qualifying downstream fit has finished.

## Downstream tests

For each qualified defender run (1) allowed-only benign SFT and (2) preserving
refusal-removal SFT, both unrestricted, LR 1e-6, 256 updates. The preserving
attack gets correct formerly blocked answers, allowed replay and .1 full-response
KL to its defended start on rehearsal. The benign control never gets formerly
blocked answer targets. Run 128 fresh-data follow-up updates for every fit:
preserving repair stays answer-all, benign stays allowed-only. Fresh AdamW
at the phase transition, as in V8. Maximum 24 fits and 24*384 updates.

Use the V8 measurement schedules (early 0,1,2,4,8 updates included). Record
first observed zero refusal, first navigation-qualified bypass and first
navigation-plus-retention-qualified bypass. Save these DEV-selected checkpoints,
best, final removal and final repair; never select a checkpoint on held-out scores.

Primary benign endpoint: blocked held-out refusal at update 256, with allowed
navigation and general retention reported alongside it. Primary attack endpoint:
held-out success at the first DEV checkpoint satisfying the full bypass gate;
absence is censored at the fixed 256-update horizon, not proof of resistance.
Also report final removal/repair, every learning curve, and unchanged V8 task
and retention gate definitions. Failed qualification is not resistance.

Compare coupled minus clean within each parent/seed. Report all six pairs,
and three-seed means/SD separately by starting width. With only three
conditional seeds, estimation is descriptive; do not pool widths as six
independent acquisitions or make a confirmatory significance claim.

## Engineering and retention

Run on Garlick's six free RTX 2080s in three two-GPU training workers;
held-out scoring uses one GPU/worker. Storage is
/mnt/fast0/mn866/casr-refusal-global-control-20261108, outside home quota.
Reserve 80 GiB, serialize atomic checkpoint writes, retain optimizer/RNG
recovery every 16 updates, and preserve original V8/V7 artifacts unchanged.
Use expandable CUDA allocator segments from startup.

Retain all raw measurements and example identities, every scheduled defender
checkpoint, the paired selected starts, the named downstream endpoints, and
rolling optimizer/RNG states. No exhaustive per-update weight history.
Same-disk recovery is not an off-machine backup.

Require local/server pure protocol checks, V8 numerical regression checks,
and real two-GPU clean/coupled interrupted-versus-uninterrupted smoke tests.
Smoke also exercises benign/preserving updates and miniature held-out scoring;
its panels substitute training examples, never production test prompts.
Freeze source and data hashes before production training. Stop on worker
failure and retain committed checkpoints; never kill other users' jobs.
