# Numerical falsification checks

These checks test specific mathematical claims about gradient routing. They do
not train an LLM, estimate a J-lens, use reinforcement learning, or demonstrate
an advantage of verbalizable representations. The word **semantic** below names
a coordinate whose meaning is assigned by the synthetic data generator.

Run from the repository root:

```sh
python -m pip install -r experiments/requirements.txt
MPLCONFIGDIR=/tmp/jspace-mpl-cache OPENBLAS_NUM_THREADS=1 python experiments/routing_checks.py
```

The recorded run used Python 3.14.2, NumPy 2.5.3, and Matplotlib 3.11.2. It uses
double precision, explicit analytical gradients, and exact population losses for
standard Gaussian inputs. Default parameters are 12,000 gradient steps, learning
rate 0.05, and random-basis seeds 20260924 through 20260943. A complete run takes
under a minute on the development machine. Output paths default to `results/`.

## 1. An exact loss-ascent and leakage counterexample

Let `h=A theta`, `A=[[1,0],[2,1]]`, `P=diag(1,0)`, and use squared loss with
activation gradient `g=(1,-1)` at `theta=0`. A routed gradient step gives

```
K = A A^T = [[1,2],[2,5]]
delta_h = -eta K P g = (-eta,-2 eta)
```

At `eta=0.1`, loss rises from 1 to 1.125 and 80% of the squared activation-update
norm is outside the selected span. Both statements are exact; no training
instability or sampling noise is needed. A parameter nullspace projection
instead gives `delta_h=(-0.02,0)` and lowers loss to 0.9802.

The script independently diagonalizes 40,200 two-dimensional symmetrized kernels
and checks the soft-mask descent condition

```
rho^2 (1+epsilon)^2 <= 4 epsilon.
```

For this counterexample, the minimum universally safe complement weight is
0.381966. Universal means all activation gradients at this local Jacobian;
individual gradients may descend below this threshold. This is a first-order
statement, not a finite-step learning-rate guarantee.

## 2. A controlled teacher--student post-training analogue

The inherited checkpoint is specified directly; it is not obtained by an actual
pretraining run. For `x~N(0,I_4)`, use

```
h = (H0 + A B) x,    B_initial = 0,
prediction = W h.
```

`H0` maps `x1` to coordinate 0, an optional inherited copy `beta*x1` to coordinate
1, `x0` to coordinate 2, `x2` to coordinate 3, and `x3` to coordinate 4. Remaining
coordinates initially vanish. The initial target head is
`w=(0.2,0.2,1,0,0,0,0,0)` and the retain head selects coordinate 3. The initial
target rule is `x0+(0.2+0.2*beta)*x1`; the new target is `x0+2*x1`, while the retain
target remains `x2`. The selected semantic projector is `P=e0 e0^T`.

Factors:

- Upstream kernel: `K=I`, or `K=I` with `K[0,1]=K[1,0]=0.8`; `A` is its symmetric
  positive square root.
- Inherited copy: `beta=0` or `beta=1`.
- Downstream head: frozen, or trained by ordinary gradients.
- Upstream update: ordinary; semantic activation route; random activation route;
  semantic parameter constraint; random parameter constraint.

For a route `P`, activation routing uses `A^T P` on the hidden gradient. The
parameter-constrained arm uses `F A^T`, where
`F=I-N^dagger N` and `N=(I-P) A`. It is a standard nullspace projected gradient,
included as an oracle control with access to the full tiny Jacobian. It is not a
new scalable LLM algorithm. Both heads' losses use the same upstream operator;
there is no separate unmasked retain loss in this toy.

Random rank-one spans are sampled in the initial retain head's nullspace and
rejected if `abs(w^T q)<0.2`. This uses only the initial readout, never evaluation
outcomes, and avoids giving the random arm an infeasible retain constraint or a
vanishing target readout. It is stronger capacity matching than rank matching
alone, but does not make the kernels identical. The seed only changes this
random direction: ordinary and semantic results repeat deterministically across
seeds. Thus the 800 saved condition rows are **not 800 independent statistical
replicates**. Error bars are standard deviations over the 20 random orientations,
not confidence intervals or evidence of robustness across pretrained models.

The retain task is deliberately separated and remains exact. This is a negative
control for bookkeeping, not evidence of protecting unrelated LLM capabilities.

### Endpoints

- Clean sign accuracy is computed exactly as
  `1-arccos(cosine(prediction_coefficients,teacher_coefficients))/pi`.
- MSE is the exact Gaussian population squared error, so coefficient error is
  also exposed even when classification is insensitive to scale.
- Zero ablation deletes `P h`; its drop is measured before and after training.
- Baseline restoration substitutes `h-P(h-h0)`. The reported MSE is between the
  restored **post-training model** and the initial model, not between the
  restored output and the new target. Zero therefore means the learned output
  change is fully erased. It avoids crediting inherited dependence to new
  localization.
- A counterfactual pair flips `x1` while retaining all other input coordinates.
  The swap changes only the selected hidden component; the expected answer is
  computed from the flipped teacher input. A pre-existing copy outside the
  selected component may prevent perfect counterfactual control even when all
  new representation changes are inside it.
- Both the common semantic span and each arm's own route are evaluated. An
  arbitrary constrained route can localize new output in its own span.

### Central results

With a coupled Jacobian and inherited copy (`beta=1`):

| Head | Upstream update | Clean accuracy | MSE surviving semantic restoration | MSE surviving own-route restoration | Semantic swap accuracy |
|---|---|---:|---:|---:|---:|
| Frozen | Ordinary | 1.00000 | 2.24790 | 2.24790 | 0.34517 |
| Frozen | Semantic route | 1.00000 | 0.50568 | 0.50568 | 0.70359 |
| Frozen | Random route | 1.00000 | 2.98250 | 0.19723 | 0.33304 |
| Frozen | Semantic constraint | 0.99998 | approximately 0 | approximately 0 | 0.96975 |
| Frozen | Random constraint | 1.00000 | 2.76683 | approximately 0 | 0.33504 |
| Trainable | Ordinary | 1.00000 | 2.05525 | 2.05525 | 0.52247 |
| Trainable | Semantic route | 1.00000 | 1.92030 | 1.92030 | 0.66521 |
| Trainable | Random route | 1.00000 | 2.51391 | 2.00946 | 0.59426 |
| Trainable | Semantic constraint | 1.00000 | 2.15095 | 2.15095 | 0.70090 |
| Trainable | Random constraint | 1.00000 | 2.52343 | 2.03607 | 0.59636 |

All saved rows reach population loss below `1e-6`. No run diverges. For parameter
constraints, complement representation drift stays below `2.7e-12`; the frozen
suffix restoration inequality has no numerical violations. Unfreezing the suffix
still defeats output erasure, because the suffix itself learns a changed readout
of inherited features. This tests a premise of the guarantee, rather than
contradicting it.

## 3. A pure readout-gain confound

For an output-restricted adapter `h=h0+q a^T x` and frozen scalar readout `w`, the
effective learning rate is proportional to `(w^T q)^2`. The standalone check
uses an assigned semantic coordinate with gain 0.8 and 20 random directions.
Its apparent convergence advantage disappears exactly after dividing each
learning rate by its squared readout gain. The script checks the actual adapter
iterates against the analytical normalized curve, not merely two versions of
the same plot formula. Faster learning under a named semantic route therefore
need not show that semantics matter.

## 4. Capacity and calibration limits

For the family `K=[[1,rho],[rho,1]]`, preserving the complementary activation
changes the target kernel from 1 to `1-rho^2`. The script explicitly constructs
the nullspace projector and verifies the Schur-complement formula on 1,001
coupling values. A plot compares this shrinking capacity with the increasing
minimum stable soft-mask weight.

One final exact example uses a shared two-parameter model whose Jacobian is
`J(x)=[[1,0],[x,1]]`. Constraining complement drift at `x=0` permits update `(1,0)`.
It gives activation update `(1,0)` there but `(1,1)` at unseen `x=1`. Enforcing
both constraints eliminates the entire parameter nullspace. A finite calibration
certificate thus neither certifies unseen inputs nor guarantees useful remaining
capacity.

## Output map

- `results/verification.json`: assertion results, precision errors, versions.
- `results/exact_counterexample.json`: exact counterexample and corrected step.
- `results/teacher_student_runs.csv`: complete condition-level endpoints.
- `results/teacher_student_summary.{csv,json}`: grouped means, standard deviations,
  minima, and maxima.
- `results/teacher_student_trajectories.csv`: saved central-condition loss curves.
- `results/random_projectors.json`: every sampled random basis projector.
- `results/readout_gain_control.csv`: gain-control trajectories.
- `results/coupling_capacity_tradeoff.csv`: capacity and stability calculations.
- `results/calibration_generalization_counterexample.json`: calibration failure.
- Four figure families are saved as both PNG and vector PDF.

The checks support rejecting unconditional localization and optimization claims.
They do not establish a J-space advantage, language-model feasibility, novelty,
publication readiness, or a sufficient experimental result for the main track.
