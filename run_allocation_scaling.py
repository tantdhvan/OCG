"""Coarse joint scaling check for the stochastic allocation MDP."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_experiments import (  # noqa: E402
    RESULTS_DIR,
    allocation_marginal_replanning,
    build_allocation_mdp,
    exposure_from_occupancy,
    finite_horizon_dp,
    occupancy_continuous_greedy_generic,
    occupancy_from_policy,
    utility_value,
)
from run_direct_cvx import occupancy_flow_residual  # noqa: E402
from run_frank_wolfe import frank_wolfe_occupancy  # noqa: E402


SEEDS = range(3000, 3005)
SETTINGS = [(8, 30), (16, 60), (32, 120)]
UTILITIES = ["soft", "log"]


def summarize_occupancy(d, phi, utility):
    x = exposure_from_occupancy(d, phi)
    value = utility_value(x, utility=utility)
    probs = x / max(np.sum(x), 1e-12)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return value, entropy, int(np.sum(x >= 1.0))


def run_setting(seed, m, horizon, utility):
    transitions, start_dist, phi, _ = build_allocation_mdp(seed, q=5, m=m, horizon=horizon)
    rows = []

    def record(method, d, started, planning_calls, objective_evaluations):
        value, entropy, covered = summarize_occupancy(d, phi, utility)
        rows.append(
            {
                "instance_type": "allocation_scaling",
                "instance_seed": seed,
                "m": m,
                "q": 5,
                "horizon": horizon,
                "utility": utility,
                "method": method,
                "evaluated_objective": value,
                "exposure_entropy": entropy,
                "covered_groups_tau_1.0": covered,
                "runtime_sec": time.perf_counter() - started,
                "planning_calls": planning_calls,
                "objective_evaluations": objective_evaluations,
                **occupancy_flow_residual(d, transitions, start_dist),
            }
        )

    started = time.perf_counter()
    reward = np.sum(phi, axis=-1)
    policy, _ = finite_horizon_dp(transitions, reward, start_dist)
    d_add = occupancy_from_policy(transitions, policy, start_dist)
    record("Additive", d_add, started, planning_calls=1, objective_evaluations=1)

    for K in [20, 40]:
        started = time.perf_counter()
        d_ocg, history = occupancy_continuous_greedy_generic(
            transitions, phi, start_dist, K=K, utility=utility
        )
        record(f"OCG K={K}", d_ocg, started, planning_calls=K, objective_evaluations=K)

    started = time.perf_counter()
    d_fw, history_fw, diagnostics_fw = frank_wolfe_occupancy(
        transitions, phi, start_dist, K=40, utility=utility
    )
    record(
        "FW K=40",
        d_fw,
        started,
        planning_calls=diagnostics_fw["planning_calls"],
        objective_evaluations=diagnostics_fw["objective_evaluations"],
    )
    return rows


def run_full():
    rows = []
    for m, horizon in SETTINGS:
        for seed in SEEDS:
            for utility in UTILITIES:
                rows.extend(run_setting(seed, m, horizon, utility))
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "allocation_scaling_results.csv", index=False)
    summary = (
        result.groupby(["m", "q", "horizon", "utility", "method"])
        .agg(
            objective_mean=("evaluated_objective", "mean"),
            objective_std=("evaluated_objective", "std"),
            entropy_mean=("exposure_entropy", "mean"),
            runtime_mean_sec=("runtime_sec", "mean"),
            planning_calls=("planning_calls", "mean"),
            objective_evaluations=("objective_evaluations", "mean"),
            max_flow_residual=("max_flow_residual", "max"),
            max_layer_mass_error=("layer_mass_error", "max"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "allocation_scaling_summary.csv", index=False)
    return result, summary


if __name__ == "__main__":
    _, summary = run_full()
    print(summary.to_string(index=False))
