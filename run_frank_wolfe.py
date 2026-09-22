"""Frank--Wolfe baseline over the occupancy polytope.

This file keeps the additional baseline separate from the original experiment
pipeline.  It uses the same finite-horizon dynamic-programming linear oracle as
OCG, the same utilities, seeds, and values of K, but chooses a numerical
line-search step on each FW segment.
"""

import json
import sys
import time
from pathlib import Path

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
    allocation_marginal_replanning,
    build_allocation_mdp,
    build_features,
    build_gridworld,
    exposure_from_occupancy,
    finite_horizon_dp,
    occupancy_from_policy,
    utility_grad,
    utility_value,
)
from run_direct_cvx import occupancy_flow_residual  # noqa: E402


LINE_SEARCH_TOL = 1e-8
LINE_SEARCH_MAX_ITER = 100


def _golden_section_maximize(objective, tol=LINE_SEARCH_TOL, max_iter=LINE_SEARCH_MAX_ITER):
    """Maximize a smooth concave function on [0, 1] reproducibly.

    The objective is evaluated through a small cache so that repeated golden
    section points are counted once.  Endpoint evaluations are included in the
    returned objective-evaluation count.
    """
    cache = {}

    def evaluate(gamma):
        gamma = float(gamma)
        key = gamma.hex()
        if key not in cache:
            cache[key] = float(objective(gamma))
        return cache[key]

    left, right = 0.0, 1.0
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = evaluate(c)
    fd = evaluate(d)
    while right - left > tol and len(cache) < max_iter + 2:
        if fc < fd:
            left = c
            c, fc = d, fd
            d = left + ratio * (right - left)
            fd = evaluate(d)
        else:
            right = d
            d, fd = c, fc
            c = right - ratio * (right - left)
            fc = evaluate(c)

    candidates = [(0.0, evaluate(0.0)), (1.0, evaluate(1.0))]
    candidates.extend([(c, fc), (d, fd), ((left + right) / 2.0, evaluate((left + right) / 2.0))])
    gamma, value = max(candidates, key=lambda pair: pair[1])
    return float(gamma), float(value), len(cache)


def frank_wolfe_occupancy(
    transitions,
    phi,
    start_dist,
    K,
    utility="soft",
    beta=BETA,
    weights=None,
    line_search_tol=LINE_SEARCH_TOL,
):
    """Run FW with K DP-oracle calls and numerical line search.

    The first oracle is evaluated at zero.  Since both experimental utilities
    have unit gradient at zero (for unit weights), the first FW vertex is the
    existing Additive solution.  Subsequent iterations use a line search on
    the segment from the current occupancy to the new oracle vertex.
    """
    H, S, A, _ = phi.shape
    if weights is None:
        weights = np.ones(phi.shape[-1], dtype=float)
    weights = np.asarray(weights, dtype=float)

    started = time.perf_counter()
    z = np.zeros((H, S, A), dtype=float)
    history = []
    objective_evaluations = 0
    oracle_calls = 0
    line_search_gammas = []

    for iteration in range(K):
        exposure = exposure_from_occupancy(z, phi)
        grad = utility_grad(exposure, utility=utility, beta=beta, weights=weights)
        reward = np.einsum("u,hsau->hsa", grad, phi)
        policy, _ = finite_horizon_dp(transitions, reward, start_dist)
        vertex = occupancy_from_policy(transitions, policy, start_dist)
        oracle_calls += 1

        if iteration == 0:
            gamma = 1.0
            z = vertex
            objective_evaluations += 1
        else:
            def segment_value(step):
                candidate = (1.0 - step) * z + step * vertex
                return utility_value(
                    exposure_from_occupancy(candidate, phi),
                    utility=utility,
                    beta=beta,
                    weights=weights,
                )

            gamma, _, evaluations = _golden_section_maximize(segment_value, tol=line_search_tol)
            z = (1.0 - gamma) * z + gamma * vertex
            objective_evaluations += evaluations

        line_search_gammas.append(gamma)
        history.append(
            utility_value(
                exposure_from_occupancy(z, phi),
                utility=utility,
                beta=beta,
                weights=weights,
            )
        )
        objective_evaluations += 1

    runtime = time.perf_counter() - started
    diagnostics = occupancy_flow_residual(z, transitions, start_dist)
    return z, np.asarray(history), {
        "runtime_sec": runtime,
        "planning_calls": oracle_calls,
        "objective_evaluations": objective_evaluations,
        "line_search_tolerance": line_search_tol,
        "line_search_gammas": line_search_gammas,
        **diagnostics,
    }


def make_gridworld_instance(seed=1000):
    env = build_gridworld(n=10, noise_stay=0.05, seed=seed)
    phi = build_features(env, horizon=HORIZON, sigma=SIGMA)
    start_dist = np.zeros(len(env.states), dtype=float)
    start_dist[env.state_to_idx[env.start_state]] = 1.0
    return env, phi, start_dist


def run_pilot(seed=1000):
    rows = []
    env, phi, start_dist = make_gridworld_instance(seed)
    for utility in ["soft", "log"]:
        for K in AGGREGATE_K_VALUES:
            z, history, diagnostics = frank_wolfe_occupancy(
                env.transitions, phi, start_dist, K=K, utility=utility
            )
            rows.append(
                {
                    "instance_type": "gridworld",
                    "instance_seed": seed,
                    "utility": utility,
                    "method": f"FW K={K}",
                    "K": K,
                    "evaluated_objective": float(history[-1]),
                    "history_last": float(history[-1]),
                    "history_min_delta": float(np.min(np.diff(history))) if len(history) > 1 else np.nan,
                    **{key: value for key, value in diagnostics.items() if key != "line_search_gammas"},
                }
            )
    result = pd.DataFrame(rows)
    output = RESULTS_DIR / "frank_wolfe_pilot.csv"
    result.to_csv(output, index=False)
    (RESULTS_DIR / "frank_wolfe_pilot_summary.json").write_text(
        json.dumps({"rows": rows, "output": str(output)}, indent=2), encoding="utf-8"
    )
    return result


def run_gridworld(seeds=range(1000, 1010)):
    rows = []
    for seed in seeds:
        env, phi, start_dist = make_gridworld_instance(seed)
        for utility in ["soft", "log"]:
            for K in AGGREGATE_K_VALUES:
                z, history, diagnostics = frank_wolfe_occupancy(
                    env.transitions, phi, start_dist, K=K, utility=utility
                )
                rows.append(
                    {
                        "instance_type": "gridworld",
                        "instance_seed": seed,
                        "utility": utility,
                        "method": f"FW K={K}",
                        "K": K,
                        "evaluated_objective": float(history[-1]),
                        "history_min_delta": float(np.min(np.diff(history))) if len(history) > 1 else np.nan,
                        **{key: value for key, value in diagnostics.items() if key != "line_search_gammas"},
                    }
                )
    return pd.DataFrame(rows)


def run_allocation(seeds=range(2000, 2010), q=5, m=8, horizon=30):
    rows = []
    for seed in seeds:
        transitions, start_dist, phi, _ = build_allocation_mdp(seed, q=q, m=m, horizon=horizon)
        for utility in ["soft", "log"]:
            for K in AGGREGATE_K_VALUES:
                z, history, diagnostics = frank_wolfe_occupancy(
                    transitions, phi, start_dist, K=K, utility=utility
                )
                rows.append(
                    {
                        "instance_type": "allocation",
                        "instance_seed": seed,
                        "utility": utility,
                        "method": f"FW K={K}",
                        "K": K,
                        "evaluated_objective": float(history[-1]),
                        "history_min_delta": float(np.min(np.diff(history))) if len(history) > 1 else np.nan,
                        **{key: value for key, value in diagnostics.items() if key != "line_search_gammas"},
                    }
                )
    return pd.DataFrame(rows)


def run_full():
    result = pd.concat([run_gridworld(), run_allocation()], ignore_index=True)
    result.to_csv(RESULTS_DIR / "frank_wolfe_full_results.csv", index=False)
    summary = (
        result.groupby(["instance_type", "utility", "method"])
        .agg(
            objective_mean=("evaluated_objective", "mean"),
            objective_std=("evaluated_objective", "std"),
            runtime_mean_sec=("runtime_sec", "mean"),
            oracle_calls=("planning_calls", "mean"),
            objective_evaluations=("objective_evaluations", "mean"),
            max_flow_residual=("max_flow_residual", "max"),
            layer_mass_error=("layer_mass_error", "max"),
            history_min_delta=("history_min_delta", "min"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "frank_wolfe_full_summary.csv", index=False)
    return result, summary


if __name__ == "__main__":
    run_pilot()
