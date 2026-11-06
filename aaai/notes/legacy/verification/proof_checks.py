#!/usr/bin/env python3
"""Independent symbolic and numerical checks; no language-model experiment."""
import json
from pathlib import Path
import numpy as np
import sympy as sp

eta, eps = sp.symbols('eta eps', real=True)
J = sp.Matrix([[1, 0], [2, 1]])
P = sp.diag(1, 0)
U = sp.eye(2)-P
g = sp.Matrix([1, -1])
K = J*J.T
N = U*J
F = sp.eye(2)-N.pinv()*N
routed = g-eta*K*P*g
assert sp.expand((routed.T*routed)[0]/2-1) == eta+sp.Rational(5,2)*eta**2
assert (g.T*K*P*g)[0] == -1
assert J*F*J.T == sp.diag(sp.Rational(1,5), 0)
M = P+eps*U
S = (K*M+M*K)/2
assert sp.expand(S.det()) == -eps**2+3*eps-1
threshold = (3-sp.sqrt(5))/2
assert sp.simplify(S.det().subs(eps, threshold)) == 0

rng = np.random.default_rng(20260924)
max_constraint_residual = 0.
max_capacity_error = 0.
max_restoration_excess = 0.
max_taylor_excess = 0.
min_projected_descent = float('inf')
rank_cases = 0
for trial in range(100):
    d, p, r = 6, (4 if trial % 2 else 10), 2
    jac = rng.normal(size=(d, p))
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    projector = basis[:, :r]@basis[:, :r].T
    comp = np.eye(d)-projector
    constraint = comp@jac
    feasible = np.eye(p)-np.linalg.pinv(constraint)@constraint
    constrained_kernel = jac@feasible@jac.T
    max_constraint_residual = max(max_constraint_residual, float(np.linalg.norm(comp@constrained_kernel)))
    expected_rank = np.linalg.matrix_rank(jac, tol=1e-9)-np.linalg.matrix_rank(constraint, tol=1e-9)
    assert np.linalg.matrix_rank(jac@feasible, tol=1e-9) == expected_rank
    cotangent = rng.normal(size=d)
    descent = float(cotangent@constrained_kernel@cotangent)
    min_projected_descent = min(min_projected_descent, descent)
    assert descent >= -1e-10
    if p > d:
        kernel = basis.T@jac@jac.T@basis
        a, b, c = kernel[:r,:r], kernel[:r,r:], kernel[r:,r:]
        schur = a-b@np.linalg.solve(c,b.T)
        max_capacity_error = max(max_capacity_error, float(np.linalg.norm((basis.T@constrained_kernel@basis)[:r,:r]-schur)))
    rank_cases += 1
    # Frozen nonlinear suffix: tanh is 1-Lipschitz in l2.
    h0, delta = rng.normal(size=d), rng.normal(size=d)
    w = rng.normal(size=(3,d))
    restored = h0+comp@delta
    lhs = np.linalg.norm(np.tanh(w@restored)-np.tanh(w@h0))
    rhs = np.linalg.norm(w,2)*np.linalg.norm(comp@delta)
    max_restoration_excess=max(max_restoration_excess,float(lhs-rhs))
    assert lhs <= rhs+1e-12
    # Nonlinear upstream: h(theta)=tanh(B theta), ||D²h||<=2||B||².
    upstream = rng.normal(size=(d,p))
    theta = rng.normal(size=p)
    h = np.tanh(upstream@theta)
    local_jac = (1-h*h)[:,None]*upstream
    step = -1e-3*local_jac.T@projector@cotangent
    rem = np.linalg.norm(np.tanh(upstream@(theta+step))-h-local_jac@step)
    bound = np.linalg.norm(upstream,2)**2*np.linalg.norm(step)**2
    max_taylor_excess=max(max_taylor_excess,float(rem-bound))
    assert rem <= bound+1e-12
assert max_constraint_residual<1e-10
assert max_capacity_error<1e-10
output={
 'symbolic_counterexample_loss_increase':'eta + 5*eta**2/2',
 'symbolic_soft_determinant':'-epsilon**2 + 3*epsilon - 1',
 'symbolic_threshold':'(3 - sqrt(5))/2',
 'symbolic_constrained_kernel':[['1/5','0'],['0','0']],
 'random_rank_and_descent_cases':rank_cases,
 'max_constraint_kernel_residual':max_constraint_residual,
 'max_conditional_capacity_error':max_capacity_error,
 'minimum_projected_descent_score':min_projected_descent,
 'nonlinear_restoration_bound_violations':0,
 'nonlinear_taylor_bound_violations':0,
 'scope':'Finite-dimensional symbolic algebra and synthetic numerical checks only.'
}
path=Path(__file__).resolve().parent/'proof_checks.json'
path.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
