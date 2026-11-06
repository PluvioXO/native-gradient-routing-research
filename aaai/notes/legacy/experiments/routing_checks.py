#!/usr/bin/env python3
"""Reproducible, deliberately small falsification checks for gradient routing.

No autodiff, downloaded data, LLMs, or empirical claims about J-space are used.
The teacher--student experiment uses exact population gradients for x~N(0,I).
Run: python experiments/routing_checks.py --output results
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def csv_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sign_accuracy(prediction, target):
    """Gaussian sign agreement; degenerate zero scores use random tie-breaking."""
    denom = np.linalg.norm(prediction) * np.linalg.norm(target)
    if denom < 1e-15:
        return 0.5
    cosine = float(np.clip(np.dot(prediction, target) / denom, -1, 1))
    return float(1 - np.arccos(cosine) / np.pi)


def exact_checks(output):
    # K=A A^T; h=A theta, target=(-1,1), theta=0 gives g=(1,-1).
    A = np.array([[1., 0.], [2., 1.]])
    K = A @ A.T
    P = np.diag([1., 0.])
    g = np.array([1., -1.])
    eta = 0.1
    delta_theta = -eta * A.T @ P @ g
    delta_h = A @ delta_theta
    loss_before = float(g @ g / 2)
    loss_after = float((g + delta_h) @ (g + delta_h) / 2)
    complement = np.eye(2) - P
    # Parameter projection is an ordinary constrained-gradient construction.
    constraint = complement @ A
    F = np.eye(2) - np.linalg.pinv(constraint) @ constraint
    corrected_delta_h = -eta * A @ F @ A.T @ g
    corrected_after = float((g + corrected_delta_h) @ (g + corrected_delta_h) / 2)
    rho = 2 / np.sqrt(5)
    epsilon_min = (1 - np.sqrt(1 - rho**2)) / (1 + np.sqrt(1 - rho**2))
    epsilons = np.linspace(0, 1, 1001)
    mineigs = []
    for epsilon in epsilons:
        M = P + epsilon * complement
        mineigs.append(float(np.linalg.eigvalsh((K @ M + M @ K) / 2)[0]))
    assert loss_after > loss_before
    assert np.allclose(delta_h, [-0.1, -0.2])
    assert np.isclose(np.linalg.norm(complement @ delta_h)**2 / np.linalg.norm(delta_h)**2, .8)
    assert np.linalg.norm(complement @ corrected_delta_h) < 1e-14
    assert corrected_after < loss_before
    assert np.linalg.norm(F @ F - F) < 1e-14
    result = {
        "A": A.tolist(), "K": K.tolist(), "P": P.tolist(), "g": g.tolist(),
        "learning_rate": eta,
        "loss_before": loss_before, "routed_loss_after": loss_after,
        "routed_activation_change": delta_h.tolist(),
        "routed_complement_energy_fraction": .8,
        "routed_loss_directional_derivative": float(-g @ K @ P @ g),
        "corrected_activation_change": corrected_delta_h.tolist(),
        "corrected_loss_after": corrected_after,
        "rho": float(rho), "soft_mask_minimum_epsilon": float(epsilon_min),
    }
    write_json(output / "exact_counterexample.json", result)

    # Universal descent is positivity of sym(KM), not positivity of M alone.
    rhos = np.linspace(0, .995, 200)
    eps_grid = np.linspace(0, 1, 201)
    rho_grid, eps_mesh = np.meshgrid(rhos, eps_grid)
    closed_form_minima = .5 * (1 + eps_mesh - np.sqrt((1 - eps_mesh)**2 + rho_grid**2 * (1 + eps_mesh)**2))
    symmetrized_kernels = np.zeros(rho_grid.shape + (2, 2))
    symmetrized_kernels[..., 0, 0] = 1
    symmetrized_kernels[..., 1, 1] = eps_mesh
    symmetrized_kernels[..., 0, 1] = symmetrized_kernels[..., 1, 0] = rho_grid * (1 + eps_mesh) / 2
    minima = np.linalg.eigvalsh(symmetrized_kernels)[..., 0]
    eigenvalue_error = float(np.max(np.abs(minima - closed_form_minima)))
    assert eigenvalue_error < 1e-12
    theorem_stable = rho_grid**2 * (1 + eps_mesh)**2 <= 4 * eps_mesh + 1e-12
    numerical_stable = minima >= -1e-12
    assert np.array_equal(theorem_stable, numerical_stable)
    checks = {
        "soft_stability_grid_points": int(minima.size),
        "soft_stability_formula_disagreements": int(np.sum(theorem_stable != numerical_stable)),
        "soft_stability_closed_form_eigenvalue_error": eigenvalue_error,
        "nullspace_projector_idempotence_error": float(np.linalg.norm(F @ F - F)),
        "nullspace_complement_update_norm": float(np.linalg.norm(complement @ corrected_delta_h)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    im = axes[0].pcolormesh(rhos, eps_grid, minima, cmap="RdBu", shading="auto", vmin=-.2, vmax=.2)
    boundary = (1 - np.sqrt(1 - rhos**2)) / (1 + np.sqrt(1 - rhos**2))
    axes[0].plot(rhos, boundary, color="black", lw=1.8, label="Exact stability threshold")
    axes[0].set(xlabel="Cross-block coupling $\\rho$", ylabel="Complement weight $\\epsilon$", title="Minimum eigenvalue of sym($KM$)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=axes[0], label="Negative values permit loss ascent")
    axes[1].plot(epsilons, mineigs, color="#254c79", lw=2)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].axvline(epsilon_min, color="#b0433d", ls="--", label=f"$\\epsilon_{{min}}={epsilon_min:.4f}$")
    axes[1].set(xlabel="Complement weight $\\epsilon$", ylabel="Minimum eigenvalue", title="Counterexample: $K=[[1,2],[2,5]]$")
    axes[1].legend(fontsize=9)
    fig.savefig(output / "soft_routing_stability.png", dpi=180)
    fig.savefig(output / "soft_routing_stability.pdf")
    plt.close(fig)
    return checks


def teacher_student(output, seed_count, steps, learning_rate):
    d, input_dim = 8, 4
    identity = np.eye(d)
    P_semantic = np.zeros((d, d))
    P_semantic[0, 0] = 1
    W0 = np.zeros((2, d))
    W0[0, :3] = [.2, .2, 1.]
    W0[1, 3] = 1.
    teacher = np.array([[1., 2., 0., 0.], [0., 0., 1., 0.]])
    methods = ["ordinary", "semantic_route", "random_route", "semantic_constraint", "random_constraint"]
    specs, As, H0s, operators, Ps = [], [], [], [], []
    random_projectors = {}
    for seed in range(seed_count):
        rng = np.random.default_rng(20260924 + seed)
        # Match capacity: the random span lies in the frozen retain head's null
        # space and has non-negligible coupling to the frozen target head.
        # This depends on the INITIAL head, not on test outcomes or task labels.
        for attempt in range(10000):
            q = rng.normal(size=d)
            q[3] = 0
            q /= np.linalg.norm(q)
            if abs(W0[0] @ q) >= .2:
                break
        else:
            raise RuntimeError("Random capacity calibration failed")
        random_projectors[seed] = np.outer(q, q)
        for geometry in ["isotropic", "coupled"]:
            K = identity.copy()
            if geometry == "coupled":
                K[0, 1] = K[1, 0] = .8
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            A = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
            for bypass in [0., 1.]:
                H0 = np.zeros((d, input_dim))
                H0[0, 1] = 1
                H0[1, 1] = bypass
                H0[2, 0] = H0[3, 2] = H0[4, 3] = 1
                for head_mode in ["frozen", "trainable"]:
                    for method in methods:
                        P = random_projectors[seed] if method.startswith("random") else P_semantic
                        if method == "ordinary":
                            operator = A.T
                        elif method.endswith("route"):
                            operator = A.T @ P
                        else:
                            constraint = (identity - P) @ A
                            F = identity - np.linalg.pinv(constraint) @ constraint
                            operator = F @ A.T
                            assert np.linalg.norm((identity - P) @ A @ F) < 1e-12
                        specs.append(dict(seed=seed, geometry=geometry, inherited_bypass=int(bypass), head=head_mode, method=method))
                        As.append(A)
                        H0s.append(H0)
                        operators.append(operator)
                        Ps.append(P)

    A = np.array(As)
    H0 = np.array(H0s)
    T = np.array(operators)
    P = np.array(Ps)
    B = np.zeros_like(H0)
    W = np.broadcast_to(W0, (len(specs), 2, d)).copy()
    train_head = np.array([spec["head"] == "trainable" for spec in specs])
    frozen = ~train_head
    active = np.ones(len(specs), dtype=bool)
    ascent_steps = np.zeros(len(specs), dtype=int)
    loss_increases = np.zeros(len(specs), dtype=int)
    snapshots = []
    initial_loss = np.sum((W @ H0 - teacher)**2, axis=(1, 2)) / 2
    previous_loss = initial_loss.copy()
    max_gradient_error = 0.
    first_step_leak = None
    for step in range(steps + 1):
        H = H0 + A @ B
        residual = W @ H - teacher
        loss = np.sum(residual**2, axis=(1, 2)) / 2
        previously_active = active.copy()
        if step > 0:
            loss_increases += (loss > previous_loss + 1e-12) & previously_active
        previous_loss = loss.copy()
        broken = ~np.isfinite(loss) | (loss > 1e10)
        active &= ~broken
        if step % max(1, steps // 100) == 0 or step == steps:
            for idx, spec in enumerate(specs):
                if spec["inherited_bypass"] == 1 and spec["geometry"] == "coupled":
                    snapshots.append({**spec, "step": step, "loss": float(loss[idx]) if active[idx] else None})
        if step == steps:
            break
        # Population loss is 1/2 ||W(H0+AB)-teacher||_F^2.
        hidden_grad = W.transpose(0, 2, 1) @ residual
        true_grad_B = A.transpose(0, 2, 1) @ hidden_grad
        grad_B = T @ hidden_grad
        grad_W = residual @ H.transpose(0, 2, 1)
        grad_W[frozen] = 0
        if step == 0:
            actual_hidden_step = A @ grad_B
            complement_step = actual_hidden_step - P @ actual_hidden_step
            first_step_leak = np.sum(complement_step**2, axis=(1, 2)) / np.maximum(np.sum(actual_hidden_step**2, axis=(1, 2)), 1e-30)
            # Check ordinary analytical derivatives against central differences.
            idx = next(i for i, s in enumerate(specs) if s["method"] == "ordinary")
            rng = np.random.default_rng(123)
            direction = rng.normal(size=B[idx].shape)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-6
            def ordinary_loss(b):
                residual_local = W[idx] @ (H0[idx] + A[idx] @ b) - teacher
                return np.sum(residual_local**2) / 2
            numerical = (ordinary_loss(B[idx] + epsilon * direction) - ordinary_loss(B[idx] - epsilon * direction)) / (2 * epsilon)
            analytical = np.sum(true_grad_B[idx] * direction)
            max_gradient_error = float(abs(numerical - analytical))
            assert max_gradient_error < 1e-8
        directional_descent = np.sum(true_grad_B * grad_B, axis=(1, 2)) + np.sum(grad_W**2, axis=(1, 2))
        ascent_steps += (directional_descent < -1e-12) & active
        grad_B[~active] = 0
        grad_W[~active] = 0
        B -= learning_rate * grad_B
        W -= learning_rate * grad_W

    H = H0 + A @ B
    baseline = np.broadcast_to(W0, W.shape) @ H0
    all_rows = []
    constraint_errors = []
    erasure_inequality_errors = []
    for idx, spec in enumerate(specs):
        row = {**spec, "converged": bool(active[idx] and loss[idx] < 1e-6), "diverged": bool(not active[idx]),
               "final_loss": float(loss[idx]) if active[idx] else None,
               "loss_increase_steps": int(loss_increases[idx]), "ascent_direction_steps": int(ascent_steps[idx]),
               "first_step_route_complement_energy": float(first_step_leak[idx])}
        if not active[idx]:
            # Keep schema identical for machine-readable analysis.
            for key in ["new_accuracy", "retain_accuracy", "new_mse", "retain_mse",
                        "semantic_ablation_new_accuracy", "semantic_ablation_retain_accuracy", "semantic_ablation_drop", "semantic_selectivity", "semantic_baseline_restoration_mse", "semantic_swap_accuracy", "semantic_swap_mse",
                        "route_baseline_restoration_mse", "route_complement_drift_norm", "route_swap_accuracy", "baseline_new_accuracy", "baseline_semantic_ablation_drop", "new_semantic_ablation_effect"]:
                row[key] = None
            all_rows.append(row)
            continue
        prediction = W[idx] @ H[idx]
        delta_H = H[idx] - H0[idx]
        semantic_ablated = W[idx] @ (identity - P_semantic) @ H[idx]
        base_semantic_ablated = W0 @ (identity - P_semantic) @ H0[idx]
        semantic_restored = W[idx] @ (H[idx] - P_semantic @ delta_H)
        route_restored = W[idx] @ (H[idx] - P[idx] @ delta_H)
        # A counterfactual flips x1 while preserving all other input features.
        flip = np.eye(input_dim)
        flip[1, 1] = -1
        target_cf = teacher @ flip
        semantic_swap = W[idx] @ (H[idx] + P_semantic @ H[idx] @ (flip - np.eye(input_dim)))
        route_swap = W[idx] @ (H[idx] + P[idx] @ H[idx] @ (flip - np.eye(input_dim)))
        clean_acc = sign_accuracy(prediction[0], teacher[0])
        ablated_acc = sign_accuracy(semantic_ablated[0], teacher[0])
        base_acc = sign_accuracy(baseline[idx, 0], teacher[0])
        base_ablation_drop = base_acc - sign_accuracy(base_semantic_ablated[0], teacher[0])
        retain_acc = sign_accuracy(prediction[1], teacher[1])
        retain_ablated_acc = sign_accuracy(semantic_ablated[1], teacher[1])
        row.update({
            "new_accuracy": clean_acc, "retain_accuracy": retain_acc,
            "new_mse": float(np.sum((prediction[0] - teacher[0])**2)),
            "retain_mse": float(np.sum((prediction[1] - teacher[1])**2)),
            "semantic_ablation_new_accuracy": ablated_acc,
            "semantic_ablation_retain_accuracy": retain_ablated_acc,
            "semantic_ablation_drop": clean_acc - ablated_acc,
            "semantic_selectivity": (clean_acc - ablated_acc) - (retain_acc - retain_ablated_acc),
            "semantic_baseline_restoration_mse": float(np.sum((semantic_restored[0] - baseline[idx, 0])**2)),
            "semantic_swap_accuracy": sign_accuracy(semantic_swap[0], target_cf[0]),
            "semantic_swap_mse": float(np.sum((semantic_swap[0] - target_cf[0])**2)),
            "route_baseline_restoration_mse": float(np.sum((route_restored[0] - baseline[idx, 0])**2)),
            "route_complement_drift_norm": float(np.linalg.norm((identity - P[idx]) @ delta_H)),
            "route_swap_accuracy": sign_accuracy(route_swap[0], target_cf[0]),
            "baseline_new_accuracy": base_acc,
            "baseline_semantic_ablation_drop": base_ablation_drop,
            "new_semantic_ablation_effect": clean_acc - ablated_acc - base_ablation_drop,
        })
        if spec["head"] == "frozen":
            restored_difference = np.linalg.norm(route_restored - baseline[idx])
            rhs = np.linalg.norm(W[idx], ord=2) * np.linalg.norm((identity - P[idx]) @ delta_H)
            erasure_inequality_errors.append(float(max(0, restored_difference - rhs)))
        if spec["method"].endswith("constraint"):
            constraint_errors.append(row["route_complement_drift_norm"])
        all_rows.append(row)
    assert max(constraint_errors) < 1e-9
    assert max(erasure_inequality_errors) < 1e-9
    csv_rows(output / "teacher_student_runs.csv", all_rows)
    csv_rows(output / "teacher_student_trajectories.csv", snapshots)
    write_json(output / "random_projectors.json", {str(k): v.tolist() for k, v in random_projectors.items()})

    summaries = []
    grouping = ["geometry", "inherited_bypass", "head", "method"]
    metrics = [k for k in all_rows[0] if k not in grouping + ["seed", "converged", "diverged"]]
    for key in sorted({tuple(row[k] for k in grouping) for row in all_rows}):
        selected = [row for row in all_rows if tuple(row[k] for k in grouping) == key]
        summary = dict(zip(grouping, key))
        summary["runs"] = len(selected)
        summary["converged_runs"] = sum(row["converged"] for row in selected)
        summary["diverged_runs"] = sum(row["diverged"] for row in selected)
        for metric in metrics:
            vals = [row[metric] for row in selected if row[metric] is not None]
            summary[metric + "_mean"] = float(np.mean(vals)) if vals else None
            summary[metric + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.
            summary[metric + "_min"] = float(np.min(vals)) if vals else None
            summary[metric + "_max"] = float(np.max(vals)) if vals else None
        summaries.append(summary)
    csv_rows(output / "teacher_student_summary.csv", summaries)
    write_json(output / "teacher_student_summary.json", summaries)

    # Central comparison: pre-existing copy + correlated upstream Jacobian.
    central = [row for row in summaries if row["geometry"] == "coupled" and row["inherited_bypass"] == 1]
    labels = ["Ordinary", "Semantic\nroute", "Random\nroute", "Semantic\nconstraint", "Random\nconstraint"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    plot_metrics = [
        ("new_accuracy", "New-rule sign accuracy", (0.5, 1.02)),
        ("semantic_baseline_restoration_mse", "New output surviving semantic restoration (MSE)", None),
        ("semantic_swap_accuracy", "Correct semantic counterfactual (sign accuracy)", (0., 1.02)),
    ]
    for head_idx, head in enumerate(["frozen", "trainable"]):
        for col, (metric, title, limits) in enumerate(plot_metrics):
            ax = axes[head_idx, col]
            rows = [next(r for r in central if r["method"] == method and r["head"] == head) for method in methods]
            values = [r[metric + "_mean"] for r in rows]
            errors = [r[metric + "_sd"] for r in rows]
            ax.bar(np.arange(len(methods)), values, yerr=errors, capsize=3,
                   color=["#777777", "#4477aa", "#66aabb", "#cc6677", "#ddaa55"])
            ax.set_xticks(np.arange(len(methods)), labels, fontsize=8)
            ax.set_title(title, fontsize=10)
            if limits:
                ax.set_ylim(*limits)
            ax.set_ylabel(f"{head.capitalize()} downstream head")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    fig.savefig(output / "teacher_student_comparison.png", dpi=180)
    fig.savefig(output / "teacher_student_comparison.pdf")
    plt.close(fig)

    print("Central comparison: coupled Jacobian, inherited bypass=1")
    print("head method converged accuracy semantic_restore_MSE route_restore_MSE semantic_swap")
    for row in central:
        print(row["head"], row["method"], f"{row['converged_runs']}/{row['runs']}",
              f"{row['new_accuracy_mean']:.6g}", f"{row['semantic_baseline_restoration_mse_mean']:.6g}",
              f"{row['route_baseline_restoration_mse_mean']:.6g}", f"{row['semantic_swap_accuracy_mean']:.6g}")
    return {
        "total_runs": len(specs), "seeds": seed_count, "steps": steps,
        "learning_rate": learning_rate,
        "finite_difference_gradient_error": max_gradient_error,
        "maximum_constraint_complement_drift": float(max(constraint_errors)),
        "maximum_frozen_head_erasure_bound_violation": float(max(erasure_inequality_errors)),
        "diverged_runs": int(sum(row["diverged"] for row in all_rows)),
        "converged_runs": int(sum(row["converged"] for row in all_rows)),
    }


def capacity_and_gain_checks(output):
    """A route can look semantically special solely through readout gain."""
    rng = np.random.default_rng(20260924)
    d = 8
    w = np.zeros(d)
    w[:2] = [.8, .6]
    semantic = np.eye(d)[0]
    qs = [semantic]
    while len(qs) < 21:
        q = rng.normal(size=d)
        q /= np.linalg.norm(q)
        if abs(w @ q) >= .05:
            qs.append(q)
    gains_squared = (np.array(qs) @ w)**2
    target_change = np.array([1., -2., .5, 0.])
    eta = .2
    steps = np.arange(151)
    # Exact discrete GD on 1/2 ||gain*a-target_change||^2.
    ordinary_residual = (1 - eta * gains_squared[:, None])**steps[None, :]
    normalized_residual = np.broadcast_to((1 - eta)**steps, ordinary_residual.shape)
    # Check actual adapter updates rather than merely plotting the formula.
    adapter = np.zeros((len(qs), len(target_change)))
    max_error = 0.
    gains = np.array(qs) @ w
    rows = []
    for step in steps:
        output_residual = adapter * gains[:, None] - target_change
        expected = -target_change * (1 - eta)**step
        max_error = max(max_error, float(np.max(np.abs(output_residual - expected))))
        for idx in range(len(qs)):
            rows.append({"route": "semantic" if idx == 0 else f"random_{idx - 1}",
                         "step": int(step), "readout_gain_squared": float(gains_squared[idx]),
                         "ordinary_loss": float(.5 * target_change @ target_change * ordinary_residual[idx, step]**2),
                         "gain_normalized_loss": float(.5 * np.sum(output_residual[idx]**2))})
        adapter -= (eta / gains_squared)[:, None] * gains[:, None] * output_residual
    assert max_error < 1e-12
    csv_rows(output / "readout_gain_control.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for idx in range(len(qs)):
        for ax, residual, title in [(axes[0], ordinary_residual, "Same learning rate"),
                                     (axes[1], normalized_residual, "Learning rate divided by readout gain²")]:
            ax.semilogy(steps, .5 * (target_change @ target_change) * residual[idx]**2,
                        color="#b0433d" if idx == 0 else "#4477aa", alpha=1 if idx == 0 else .3,
                        lw=2 if idx == 0 else 1,
                        label="Semantic coordinate" if idx == 0 else ("Random directions" if idx == 1 else None))
            ax.set(xlabel="Gradient steps", ylabel="Population loss", title=title)
            ax.set_ylim(1e-12, 5)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "readout_gain_control.png", dpi=180)
    fig.savefig(output / "readout_gain_control.pdf")
    plt.close(fig)

    rhos = np.linspace(0, .999, 1001)
    capacity_error = 0.
    capacity_rows = []
    P = np.diag([1., 0.])
    for rho in rhos:
        K = np.array([[1., rho], [rho, 1.]])
        vals, vecs = np.linalg.eigh(K)
        J = (vecs * np.sqrt(vals)) @ vecs.T
        N = (np.eye(2) - P) @ J
        F = np.eye(2) - np.linalg.pinv(N) @ N
        kernel = J @ F @ J.T
        expected_kernel = np.diag([1 - rho**2, 0.])
        capacity_error = max(capacity_error, float(np.max(np.abs(kernel - expected_kernel))))
        epsilon_min = (1 - np.sqrt(1 - rho**2)) / (1 + np.sqrt(1 - rho**2))
        capacity_rows.append({"rho": float(rho), "conditional_capacity": float(kernel[0, 0]),
                              "theoretical_capacity": float(1 - rho**2), "soft_epsilon_min": float(epsilon_min)})
    assert capacity_error < 1e-12
    csv_rows(output / "coupling_capacity_tradeoff.csv", capacity_rows)
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    ax.plot(rhos, 1 - rhos**2, lw=2, label="Preserve-complement learning capacity")
    ax.plot(rhos, (1 - np.sqrt(1 - rhos**2)) / (1 + np.sqrt(1 - rhos**2)), lw=2,
            label="Minimum stable complement weight")
    ax.set(xlabel="Cross-block coupling $\\rho$", ylabel="Normalized capacity / mask weight",
           title="Higher coupling limits isolation and strict routing")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(output / "coupling_capacity_tradeoff.png", dpi=180)
    fig.savefig(output / "coupling_capacity_tradeoff.pdf")
    plt.close(fig)

    # Calibrating a parameter constraint on one input does not certify others.
    J_calibration = np.eye(2)
    J_heldout = np.array([[1., 0.], [1., 1.]])
    N_calibration = (np.eye(2) - P) @ J_calibration
    F_calibration = np.eye(2) - np.linalg.pinv(N_calibration) @ N_calibration
    update = F_calibration @ np.array([1., 0.])
    N_both = np.vstack([(np.eye(2) - P) @ J_calibration, (np.eye(2) - P) @ J_heldout])
    F_both = np.eye(2) - np.linalg.pinv(N_both) @ N_both
    calibration_failure = {
        "J_calibration": J_calibration.tolist(), "J_heldout": J_heldout.tolist(),
        "parameter_update": update.tolist(),
        "calibration_activation_change": (J_calibration @ update).tolist(),
        "heldout_activation_change": (J_heldout @ update).tolist(),
        "heldout_complement_energy_fraction": .5,
        "nullspace_dimension_one_input": int(2 - np.linalg.matrix_rank(N_calibration)),
        "nullspace_dimension_both_inputs": int(2 - np.linalg.matrix_rank(N_both)),
        "both_input_projector_norm": float(np.linalg.norm(F_both)),
    }
    write_json(output / "calibration_generalization_counterexample.json", calibration_failure)
    return {"readout_gain_normalization_max_error": max_error,
            "conditional_capacity_kernel_max_error": capacity_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--learning-rate", type=float, default=.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = exact_checks(args.output)
    checks.update(capacity_and_gain_checks(args.output))
    checks.update(teacher_student(args.output, args.seeds, args.steps, args.learning_rate))
    checks["python_version"] = platform.python_version()
    checks["numpy_version"] = np.__version__
    checks["matplotlib_version"] = matplotlib.__version__
    write_json(args.output / "verification.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
