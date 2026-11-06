# Selective skill/refusal coupling — V8

Garlick pilot requested 6 November 2026. Tests whether gradient coupling in
the original shared native allocation changes the cost of removing refusal
and recovering navigation. This is harmless graph navigation, not a real
harmful-request defense or a demonstrated inseparable circuit.

Both original qualified starts are used: four/eight coupled up/gate rows and
down columns in MLP block 12 (10,752/21,504 mutable scalars). The entire model
is editable by subsequent attackers. No old experiment or manuscript result
has been overwritten.

## Comparison

| Defender | Mutable support | Additional objective |
|---|---|---|
| Clean continuation | Original native allocation | Allowed answers, blocked refusal, general KL |
| Global coupling | All 494,032,768 weights | Above plus opposing attack/skill gradients |
| Local coupling | Original native allocation | Above plus opposing attack/skill gradients |
| Local coupling + lookahead | Original native allocation | Above plus four-step unrestricted preserving attacks every eighth outer step |

Fourteen defense candidates: two starts, one clean continuation, and two
coupling strengths (.001/.01) for each of the other three arms. Each runs
128 outer updates. Gradient coupling is SEAM-inspired; this pilot does not
reproduce SEAM's separate adversarial unlearning term or claim its results.
The lookahead gradient uses a first-order approximation.

Only post-update checkpoints retaining strong task/refusal/control scores
qualify. Failed qualification is reported as failed defense acquisition,
never as tamper resistance. Selection uses development performance and an
independent gradient-probe bank, not attack test results.

For each qualified parent/arm:

- Preserving removal at learning rates 1e-6, 1e-5 and 5e-5.
- Ordinary blocked-only removal at 1e-6.
- Benign-only allowed-task continuation at 1e-6.
- 256 initial updates followed by 128 fresh-data repair updates. The benign
  control remains allowed-only through both phases.

At most **40 downstream fits**, including eight benign controls. Original
development/test panels are reused and already inspected; the study is an
exploratory one-seed pilot. Defense, lookahead, removal, repair and gradient
probe worlds are mutually disjoint and excluded from all V3/V7 worlds.

## Execution and saved data

Server: mn866@garlick.cs.bath.ac.uk.

Run directory: /var/tmp/casr-mn866-refusal-coupling-20261106.

Three two-GPU workers use six RTX 2080s. Storage is on the main disk outside
the home quota, with an 80-GiB free-space reserve. Google Drive is not used.
All raw measured predictions, losses, example identities, development
choices and endpoint evaluations persist. The checkpoint policy retains
first zero refusal, worst allowed performance, development-best, final
removal and final repair weights, plus rolling optimizer/RNG recovery.
It does not retain weights at every measured point.

Six CPU checks passed, including comparison with an exact nonlinear
autograd derivative and interruption during checkpoint publication.
The final revision passed real two-GPU global/local/lookahead checks,
including exact fresh-process resume and exact off-channel restoration.
See launch receipts for production status; smoke outcomes are not scientific
results.

Progress:

~~~bash
ssh mn866@garlick.cs.bath.ac.uk \
  'cat /var/tmp/casr-mn866-refusal-coupling-20261106/results/status.json'
~~~

The launcher automatically performs defense calibration, development
selection, defended-parent preparation, removal/repair, held-out evaluation
and analysis. Final exports include all development curves, endpoint tables,
training histories and a five-page trajectory PDF.

[Fixed protocol](PROTOCOL.md) · [Status](RUN_STATUS.md)

Related methods: [SEAM](https://arxiv.org/abs/2505.12186) and
[TAR](https://arxiv.org/abs/2408.00761).
