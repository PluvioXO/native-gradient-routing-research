# Original versus balanced selective-refusal training

Prospective pilot protocol, 5 November 2026. Run on Garlick only. This study
changes the positive refusal-example schedule; it does not perform removal
training or enforce a different representation.

## Question and starting models

Does the blocked-example sampling schedule explain part of the selectivity
failure on the harder 16-node L/R navigation task? Start policy training afresh
from the exact saved V4 capability parents: native four channels, native eight
channels, and full-model fine-tuning. The native allocation is MLP block 12 of
24 only; attention and all other original weights remain frozen. No acquisition
is rerun, and no two-channel policy is attempted because its parent failed
capability qualification. This is one pilot seed, 31, on the already inspected
V4 bank, not an independent confirmation or a causal isolation claim.

Pinned model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Every parent, teacher cache and data
file is SHA256-verified against the original Weatherwax files. The engine is
the frozen V4 source with only declared sampler, logging, RNG-recovery and
imported-parent endpoint changes; source lineage is recorded.

## Fixed comparison

Three budgets × two schedules × three learning-rate factors = **18 fits**.
Each starts a fresh AdamW optimizer from the same selected capability weights,
uses 1,024 policy updates and evaluates development every 64 updates, including
step zero. Factors 1/0.3/0.1 multiply native 3e-4 or full 1e-5. Retain all
rates, losses, raw outputs, selected/final weights and optimizer/RNG states.

- **Original:** exact V4 positive schedule. Across 2,048 positive examples,
  six-group counts are h2p1=1024, h3p1=256, h3p2=256, h4p1=170,
  h4p2=171, h4p3=171. Position totals are 1450/427/171.
- **Balanced:** round-robin over h2p1, h3p1, h3p2, h4p1, h4p2, h4p3.
  Each group receives 341 or 342 positives. Each positive uses the same graph
  and command plan as its paired allowed example, at the scheduled depth.

The allowed-question sequence, negative decisions, capability-teacher examples,
base-conservation examples, loss weights, clipping, warmup and update counts
are identical between paired schedules. Only positive refusal prompts change.
This changes the **joint depth/position distribution**, not solely position
at fixed depth. Updates and example counts are matched; token counts and
wall time are recorded, while FLOPs are not measured. No longer-horizon arm
is silently added to this pilot.

All losses remain V4: allowed trace NLL; balanced positive/negative first-token
refusal decision loss; refusal continuation/EOS grammar (0.1); full-response
capability-teacher KL (0.1); frozen-base arithmetic/copy KL (0.05). Shared native
supports receive every loss. Native totals are 10,752 / 21,504 trainable scalars;
full updates every parameter. There is no direction penalty/projection/steering.

## Selection, endpoints and interpretation

Pre-existing development gates: at least 80% correct allowed paths in every
allowed case, at least 90% refusal in every blocked position, at most 10%
false refusal in every allowed case; at most a 10-point arithmetic/copy and
per-benchmark accuracy drop from base; valid-option mass at least 80% of base.
Rank by feasibility, worst-case quality, balanced allowed/refusal score,
retention, earlier step; lower LR breaks exact ties. Pick one recipe per
budget/schedule using development only. Retain explicitly infeasible fallbacks.
No all-condition success gate censors other policy arms or their evaluations.

The primary descriptive comparison is the paired balanced-minus-original
difference in worst-case joint quality at the **fixed final update**, reporting
allowed correctness, blocked refusal and false refusal separately. Also
report development-selected endpoints, feasibility, all rate candidates and
the full learning curves. No seed-level significance claim is made from this
one-seed pilot. Bootstrapping graphs only describes conditional case uncertainty.

All policy training finishes before new held-out model evaluation. Evaluate
the frozen base, each imported capability parent, and the best/final endpoints
of each of six selected recipes. Include primary cases, held-out wording,
longer paths, arithmetic/copy, ARC-Easy and WinoGrande with option mass.
Keep per-depth/position results and malformed/truncated outputs. Compare
retention to both base and the corresponding capability parent; full-model
parents may already have collateral loss.

Record refusal-associated contrasts at all layers as observations only.
They never affect learning, checkpoint selection or qualification. These
observations cannot establish independent circuits or removal resistance.

Three-seed replication and matched shared-eight/disjoint-4+4 training are
follow-ups to decide after this pilot, rather than automatic work in this run.

## Engineering and persistence

Use `/mnt/fast1/mn866/casr-refusal-sampler-20261105`, outside the 25 GB home
quota. Import only needed original checkpoints/caches; preserve originals.
Atomic checkpoint writes, a 40 GiB free-space stop, source/data/mask/parent
identities and resumable model/Adam/Python/NumPy/Torch/CUDA RNG states apply.
Skip GPUs already occupied by another process; full training uses two GPUs,
native training one. Local checkpoints are same-host recovery, not off-host
backup. No Drive sync or external messages are enabled.

CPU tests and a separate miniature hardware smoke must pass before production.
Smoke scores are not research results. A detached pipeline runs training,
held-out evaluation, then CPU analysis with CSVs and a PDF of learning,
selectivity, grouped errors and retention.
