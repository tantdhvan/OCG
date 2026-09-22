"""Pilot and batch numerical references for the concave benchmark utilities.

The reference is deliberately kept separate from ``run_experiments.py`` so
that the original experiment pipeline and its stored outputs are unchanged.
For the two benchmark utilities used in the draft, the occupancy problem is a
convex optimization problem in the standard sense: maximize a concave
objective over linear occupancy-flow constraints.
"""

import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_experiments import (  # noqa: E402
    AGGREGATE_K_VALUES,
    BETA,
    HORIZON,
    RESULTS_DIR,
    SIGMA,
    build_features,
    build_gridworld,
    build_allocation_mdp,
    exposure_from_occupancy,
    finite_horizon_dp,
    iterative_marginal_replanning,
    nearest_target_greedy_occupancy,
    occupancy_continuous_greedy_generic,
    occupancy_from_policy,
    random_policy_occupancy,
    allocation_marginal_replanning,
    utility_value,
)


# Fixed before the multi-seed run; these are acceptance thresholds for a
# numerical reference, not mathematical guarantees about the solver.
MAX_INITIAL_RESIDUAL = 1e-7
MAX_FLOW_RESIDUAL = 1e-7
MAX_LAYER_MASS_ERROR = 1e-6
MIN_OCCUPANCY = -1e-8
MAX_OBJECTIVE_DISCREPANCY = 1e-8


def occupancy_flow_residual(d, transitions, start_dist):
    """Return maximum absolute occupancy-flow equality residual."""
    H, S, A = d.shape
    state_mass = np.sum(d, axis=2)
    initial_residual = float(np.max(np.abs(state_mass[0] - start_dist)))
    flow_residual = 0.0
    for h in range(1, H):
        predicted = np.einsum("sa,saj->j", d[h - 1], transitions)
        flow_residual = max(flow_residual, float(np.max(np.abs(state_mass[h] - predicted))))
    return {
        "initial_distribution_residual": initial_residual,
        "flow_residual": flow_residual,
        "max_flow_residual": max(initial_residual, flow_residual),
        "layer_mass_error": float(np.max(np.abs(np.sum(state_mass, axis=1) - 1.0))),
        "min_occupancy": float(np.min(d)),
    }


def direct_cvx_occupancy(
    transitions,
    phi,
    start_dist,
    utility="soft",
    beta=BETA,
    weights=None,
    solver="CLARABEL",
    tol=1e-8,
    max_iter=500,
    enforce_layer_mass=False,
):
    """Solve the benchmark utility directly over occupancy variables.

    The returned objective is a numerical reference, not a formal certificate
    of the global optimum. Solver status, residuals, and runtime must be
    reported alongside it.
    """
    H, S, A, m = phi.shape
    if weights is None:
        weights = np.ones(m, dtype=float)
    weights = np.asarray(weights, dtype=float)
    start_dist = np.asarray(start_dist, dtype=float)

    d = cp.Variable((H, S, A), nonneg=True)
    constraints = [cp.sum(d[0, :, :], axis=1) == start_dist]
    if enforce_layer_mass:
        constraints.append(cp.sum(d[0, :, :]) == 1.0)
    for h in range(1, H):
        incoming = cp.sum(cp.multiply(d[h - 1, :, :][:, :, None], transitions), axis=(0, 1))
        constraints.append(cp.sum(d[h, :, :], axis=1) == incoming)
        if enforce_layer_mass:
            constraints.append(cp.sum(d[h, :, :]) == 1.0)

    d_expanded = cp.reshape(d, (H, S, A, 1), order="C")
    exposure = cp.sum(cp.multiply(d_expanded, phi), axis=(0, 1, 2))
    if utility == "soft":
        objective_expr = cp.sum(cp.multiply(weights, 1.0 - cp.exp(-beta * exposure)))
    elif utility == "log":
        objective_expr = cp.sum(cp.multiply(weights, cp.log1p(exposure)))
    else:
        raise ValueError(f"Unknown utility: {utility}")

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    requested_solver = solver
    solver_used = solver
    solver_fallback_from = ""
    solve_kwargs = {"solver": solver}
    if solver.upper() == "CLARABEL":
        solve_kwargs.update(
            {
                "max_iter": max_iter,
                "tol_gap_abs": tol,
                "tol_gap_rel": tol,
                "tol_feas": tol,
            }
        )
    elif solver.upper() == "SCS":
        solve_kwargs.update({"eps": tol, "max_iters": max_iter})
    started = time.perf_counter()
    try:
        problem.solve(**solve_kwargs)
    except cp.error.SolverError:
        if solver.upper() != "CLARABEL":
            raise
        # Clarabel can fail on otherwise small, well-posed soft-utility
        # instances. SCS is an independent conic fallback; its tighter
        # eps=1e-7 setting is required by the fixed residual rule below.
        solver_used = "SCS"
        solver_fallback_from = "CLARABEL"
        problem.solve(solver="SCS", eps=max(tol, 1e-7), max_iters=max(max_iter * 10, 100000))
    wall_time = time.perf_counter() - started

    if d.value is None:
        raise RuntimeError(f"Direct-CVX returned no solution; status={problem.status}")
    d_value = np.asarray(d.value, dtype=float)
    diagnostics = occupancy_flow_residual(d_value, transitions, start_dist)
    x_value = exposure_from_occupancy(d_value, phi)
    diagnostics.update(
        {
            "utility": utility,
            "solver": solver_used,
            "requested_solver": requested_solver,
            "solver_fallback_from": solver_fallback_from,
            "solver_status": str(problem.status),
            "solver_objective": float(problem.value),
            "evaluated_objective": utility_value(x_value, utility=utility, beta=beta, weights=weights),
            "wall_time_sec": wall_time,
            "solver_time_sec": float(getattr(problem.solver_stats, "solve_time", np.nan)),
            "solver_num_iters": int(getattr(problem.solver_stats, "num_iters", -1) or -1),
            "variable_count": int(H * S * A),
            "constraint_count": int(len(constraints)),
            "horizon": H,
            "state_count": S,
            "action_count": A,
            "target_count": m,
            "enforce_layer_mass": bool(enforce_layer_mass),
        }
    )
    objective_discrepancy = abs(diagnostics["solver_objective"] - diagnostics["evaluated_objective"])
    diagnostics["objective_discrepancy"] = objective_discrepancy
    diagnostics["accepted_status"] = diagnostics["solver_status"] == "optimal"
    diagnostics["accepted_residuals"] = (
        diagnostics["initial_distribution_residual"] <= MAX_INITIAL_RESIDUAL
        and diagnostics["flow_residual"] <= MAX_FLOW_RESIDUAL
        and diagnostics["layer_mass_error"] <= MAX_LAYER_MASS_ERROR
        and diagnostics["min_occupancy"] >= MIN_OCCUPANCY
        and objective_discrepancy <= MAX_OBJECTIVE_DISCREPANCY * max(1.0, abs(diagnostics["evaluated_objective"]))
    )
    diagnostics["accepted"] = diagnostics["accepted_status"] and diagnostics["accepted_residuals"]
    return d_value, diagnostics


def make_gridworld_instance(seed=1000):
    env = build_gridworld(n=10, noise_stay=0.05, seed=seed)
    phi = build_features(env, horizon=HORIZON, sigma=SIGMA)
    start_dist = np.zeros(len(env.states), dtype=float)
    start_dist[env.state_to_idx[env.start_state]] = 1.0
    return env, phi, start_dist


def run_pilot(seed=1000):
    """Run both direct references and existing baselines on one map."""
    env, phi, start_dist = make_gridworld_instance(seed)
    rows = []
    for utility in ["soft", "log"]:
        weights = np.ones(phi.shape[-1], dtype=float)
        d_ref, ref_diag = direct_cvx_occupancy(
            env.transitions,
            phi,
            start_dist,
            utility=utility,
            weights=weights,
        )
        ref_diag.update({"map_seed": seed, "method": "Direct-CVX"})
        rows.append(ref_diag)

        reward_add = np.sum(phi, axis=-1)
        policy_add, _ = finite_horizon_dp(env.transitions, reward_add, start_dist)
        d_add = occupancy_from_policy(env.transitions, policy_add, start_dist)
        d_ocg, _ = occupancy_continuous_greedy_generic(
            env.transitions,
            phi,
            start_dist,
            K=40,
            utility=utility,
            weights=weights,
        )
        d_random = random_policy_occupancy(env.transitions, HORIZON, start_dist)
        for method, d_method, calls in [
            ("Random", d_random, 0),
            ("Additive", d_add, 1),
            ("OCG K=40", d_ocg, 40),
        ]:
            x_method = exposure_from_occupancy(d_method, phi)
            row = {
                "map_seed": seed,
                "method": method,
                "utility": utility,
                "solver_status": "not_applicable",
                "evaluated_objective": utility_value(x_method, utility=utility, weights=weights),
                "reference_objective": ref_diag["evaluated_objective"],
                "reference_ratio": utility_value(x_method, utility=utility, weights=weights)
                / ref_diag["evaluated_objective"],
                "relative_gap": (ref_diag["evaluated_objective"] - utility_value(x_method, utility=utility, weights=weights))
                / ref_diag["evaluated_objective"],
                "planning_calls": calls,
            }
            row.update(occupancy_flow_residual(d_method, env.transitions, start_dist))
            rows.append(row)

    result = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_DIR / "direct_cvx_pilot.csv", index=False)
    with (RESULTS_DIR / "direct_cvx_pilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(orient="records"), handle, indent=2)
    return result


def _method_record(method, d, transitions, phi, start_dist, utility, planning_calls, reference_value):
    x_value = exposure_from_occupancy(d, phi)
    value = utility_value(x_value, utility=utility, weights=np.ones(phi.shape[-1]))
    residuals = occupancy_flow_residual(d, transitions, start_dist)
    return {
        "method": method,
        "utility": utility,
        "solver_status": "not_applicable",
        "evaluated_objective": value,
        "reference_objective": reference_value,
        "reference_ratio": value / reference_value,
        "relative_gap": (reference_value - value) / reference_value,
        "planning_calls": planning_calls,
        **residuals,
    }


def _reference_record(diag, instance_type, instance_seed):
    row = dict(diag)
    row.update(
        {
            "instance_type": instance_type,
            "instance_seed": instance_seed,
            "method": "Direct-CVX",
            "reference_objective": row["evaluated_objective"],
            "reference_ratio": 1.0,
            "relative_gap": 0.0,
            "planning_calls": np.nan,
        }
    )
    return row


def run_direct_cvx_gridworld(seeds=range(1000, 1010)):
    rows = []
    for seed in seeds:
        env = build_gridworld(n=10, noise_stay=0.05, seed=seed, obstacle_count=8, target_count=12)
        phi = build_features(env, horizon=HORIZON, sigma=SIGMA)
        start_dist = np.zeros(len(env.states), dtype=float)
        start_dist[env.state_to_idx[env.start_state]] = 1.0
        weights = np.ones(phi.shape[-1], dtype=float)

        for utility in ["soft", "log"]:
            d_ref, ref_diag = direct_cvx_occupancy(
                env.transitions, phi, start_dist, utility=utility, weights=weights
            )
            ref_value = ref_diag["evaluated_objective"]
            ref_row = _reference_record(ref_diag, "gridworld", seed)
            ref_row["utility"] = utility
            rows.append(ref_row)

            reward_add = np.sum(phi, axis=-1)
            policy_add, _ = finite_horizon_dp(env.transitions, reward_add, start_dist)
            d_add = occupancy_from_policy(env.transitions, policy_add, start_dist)
            d_random = random_policy_occupancy(env.transitions, HORIZON, start_dist)
            feasible = [("Random", d_random, 0), ("Additive", d_add, 1)]
            d_ocg, _ = occupancy_continuous_greedy_generic(
                env.transitions,
                phi,
                start_dist,
                K=40,
                utility=utility,
                weights=weights,
            )
            feasible.append(("OCG K=40", d_ocg, 40))
            if utility == "soft":
                feasible.extend(
                    [
                        ("Nearest-target", nearest_target_greedy_occupancy(env, phi, start_dist), 0),
                        ("Marginal", iterative_marginal_replanning(env.transitions, phi, start_dist, rounds=5), 6),
                    ]
                )
            for method, d_method, calls in feasible:
                row = _method_record(
                    method,
                    d_method,
                    env.transitions,
                    phi,
                    start_dist,
                    utility,
                    calls,
                    ref_value,
                )
                row.update({"instance_type": "gridworld", "instance_seed": seed})
                rows.append(row)
        print(f"GridWorld Direct-CVX completed seed={seed}")
    return pd.DataFrame(rows)


def run_direct_cvx_allocation(seeds=range(2000, 2010), q=5, m=8, horizon=30):
    rows = []
    for seed in seeds:
        transitions, start_dist, phi, eta = build_allocation_mdp(seed, q=q, m=m, horizon=horizon)
        weights = np.ones(phi.shape[-1], dtype=float)
        for utility in ["soft", "log"]:
            d_ref, ref_diag = direct_cvx_occupancy(
                transitions, phi, start_dist, utility=utility, weights=weights
            )
            ref_value = ref_diag["evaluated_objective"]
            ref_row = _reference_record(ref_diag, "allocation", seed)
            ref_row.update({"utility": utility, "q": q, "m": m, "horizon": horizon})
            rows.append(ref_row)

            reward_add = np.sum(phi, axis=-1)
            policy_add, _ = finite_horizon_dp(transitions, reward_add, start_dist)
            d_add = occupancy_from_policy(transitions, policy_add, start_dist)
            d_random = random_policy_occupancy(transitions, horizon, start_dist)
            myopic_policy = np.zeros((horizon, q), dtype=int)
            for h in range(horizon):
                for s in range(q):
                    myopic_policy[h, s] = int(np.argmax(eta[s, :]))
            d_myopic = occupancy_from_policy(transitions, myopic_policy, start_dist)
            d_marginal = allocation_marginal_replanning(
                transitions, phi, start_dist, utility=utility, rounds=5
            )
            d_ocg, _ = occupancy_continuous_greedy_generic(
                transitions,
                phi,
                start_dist,
                K=40,
                utility=utility,
                weights=weights,
            )
            feasible = [
                ("Random", d_random, 0),
                ("Myopic-effectiveness", d_myopic, 0),
                ("Additive", d_add, 1),
                ("Marginal", d_marginal, 6),
                ("OCG K=40", d_ocg, 40),
            ]
            for method, d_method, calls in feasible:
                row = _method_record(
                    method,
                    d_method,
                    transitions,
                    phi,
                    start_dist,
                    utility,
                    calls,
                    ref_value,
                )
                row.update(
                    {
                        "instance_type": "allocation",
                        "instance_seed": seed,
                        "q": q,
                        "m": m,
                        "horizon": horizon,
                    }
                )
                rows.append(row)
        print(f"Allocation Direct-CVX completed seed={seed}")
    return pd.DataFrame(rows)


def validate_and_summarize_direct_cvx(df):
    """Apply fixed acceptance rules before writing aggregate summaries."""
    reference = df[df["method"] == "Direct-CVX"].copy()
    if not bool(reference["accepted"].all()):
        bad = reference.loc[~reference["accepted"], ["instance_type", "instance_seed", "utility", "solver_status", "max_flow_residual", "layer_mass_error"]]
        raise RuntimeError(f"Direct-CVX acceptance failure; no aggregate written:\n{bad.to_string(index=False)}")

    check_cols = ["instance_type", "instance_seed", "utility"]
    merged = df[df["method"] != "Direct-CVX"].merge(
        reference[check_cols + ["reference_objective"]], on=check_cols, how="left", suffixes=("", "_refcheck")
    )
    if merged["reference_objective_refcheck"].isna().any():
        raise RuntimeError("Missing reference objective for at least one method row")
    violations = merged[merged["evaluated_objective"] > merged["reference_objective_refcheck"] + MAX_OBJECTIVE_DISCREPANCY]
    if not violations.empty:
        raise RuntimeError(
            "Known feasible method exceeds Direct-CVX beyond tolerance; no aggregate written:\n"
            + violations[["instance_type", "instance_seed", "utility", "method", "evaluated_objective", "reference_objective_refcheck"]].to_string(index=False)
        )

    summaries = []
    for keys, sub in df.groupby(["instance_type", "utility", "method"]):
        summaries.append(
            {
                "instance_type": keys[0],
                "utility": keys[1],
                "method": keys[2],
                "n_instances": len(sub),
                "objective_mean": sub["evaluated_objective"].mean(),
                "objective_std": sub["evaluated_objective"].std(),
                "reference_ratio_mean": sub["reference_ratio"].mean(),
                "reference_ratio_std": sub["reference_ratio"].std(),
                "relative_gap_mean": sub["relative_gap"].mean(),
                "relative_gap_std": sub["relative_gap"].std(),
                "runtime_mean_sec": sub["wall_time_sec"].mean() if "wall_time_sec" in sub else np.nan,
            }
        )
    return pd.DataFrame(summaries)


def run_direct_cvx_full():
    grid = run_direct_cvx_gridworld()
    allocation = run_direct_cvx_allocation()
    full = pd.concat([grid, allocation], ignore_index=True, sort=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULTS_DIR / "direct_cvx_full_results.csv", index=False)
    summary = validate_and_summarize_direct_cvx(full)
    summary.to_csv(RESULTS_DIR / "direct_cvx_full_summary.csv", index=False)
    print("\n=== Direct-CVX full summary ===")
    print(summary.to_string(index=False))
    return full, summary


if __name__ == "__main__":
    pilot = run_pilot()
    print(pilot.to_string(index=False))
