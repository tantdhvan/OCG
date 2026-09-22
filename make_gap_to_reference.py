"""Create per-instance gap-to-reference data and a compact comparison figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_experiments import (  # noqa: E402
    AGGREGATE_K_VALUES,
    FIGURES_DIR,
    HORIZON,
    RESULTS_DIR,
    SIGMA,
    build_allocation_mdp,
    build_features,
    build_gridworld,
    occupancy_continuous_greedy_generic,
)
from run_frank_wolfe import make_gridworld_instance  # noqa: E402


def gridworld_instance(seed):
    env = build_gridworld(n=10, noise_stay=0.05, seed=seed)
    phi = build_features(env, horizon=HORIZON, sigma=SIGMA)
    start = np.zeros(len(env.states), dtype=float)
    start[env.state_to_idx[env.start_state]] = 1.0
    return env.transitions, phi, start


def build_gap_table():
    reference = pd.read_csv(RESULTS_DIR / "direct_cvx_full_results.csv")
    reference = reference[reference.method == "Direct-CVX"]
    reference = reference[
        ["instance_type", "instance_seed", "utility", "evaluated_objective"]
    ].rename(columns={"evaluated_objective": "reference_objective"})
    fw = pd.read_csv(RESULTS_DIR / "frank_wolfe_full_results.csv")
    fw = fw[["instance_type", "instance_seed", "utility", "K", "evaluated_objective"]]
    fw = fw.rename(columns={"evaluated_objective": "objective"})
    fw["method"] = "FW"

    rows = []
    for seed in range(1000, 1010):
        transitions, phi, start = gridworld_instance(seed)
        for utility in ["soft", "log"]:
            for K in AGGREGATE_K_VALUES:
                z, _ = occupancy_continuous_greedy_generic(
                    transitions, phi, start, K=K, utility=utility
                )
                x = np.einsum("hsau,hsa->u", phi, z)
                if utility == "soft":
                    value = float(np.sum(1.0 - np.exp(-x)))
                else:
                    value = float(np.sum(np.log1p(x)))
                rows.append(
                    {
                        "instance_type": "gridworld",
                        "instance_seed": seed,
                        "utility": utility,
                        "K": K,
                        "method": "OCG",
                        "objective": value,
                    }
                )

    for seed in range(2000, 2010):
        transitions, start, phi, _ = build_allocation_mdp(seed, q=5, m=8, horizon=30)
        for utility in ["soft", "log"]:
            for K in AGGREGATE_K_VALUES:
                z, _ = occupancy_continuous_greedy_generic(
                    transitions, phi, start, K=K, utility=utility
                )
                x = np.einsum("hsau,hsa->u", phi, z)
                if utility == "soft":
                    value = float(np.sum(1.0 - np.exp(-x)))
                else:
                    value = float(np.sum(np.log1p(x)))
                rows.append(
                    {
                        "instance_type": "allocation",
                        "instance_seed": seed,
                        "utility": utility,
                        "K": K,
                        "method": "OCG",
                        "objective": value,
                    }
                )

    ocg = pd.DataFrame(rows)
    data = pd.concat([ocg, fw], ignore_index=True)
    data = data.merge(reference, on=["instance_type", "instance_seed", "utility"], how="left")
    data["relative_gap"] = (data["reference_objective"] - data["objective"]) / data[
        "reference_objective"
    ]
    expected_rows = 40 * len(AGGREGATE_K_VALUES) * 2
    expected_keys = {
        (kind, seed, utility, K, method)
        for kind, seeds in [("gridworld", range(1000, 1010)), ("allocation", range(2000, 2010))]
        for seed in seeds
        for utility in ["soft", "log"]
        for K in AGGREGATE_K_VALUES
        for method in ["OCG", "FW"]
    }
    actual_keys = set(
        zip(data.instance_type, data.instance_seed, data.utility, data.K, data.method)
    )
    assert len(data) == expected_rows
    assert actual_keys == expected_keys
    assert data.reference_objective.notna().all()
    return data


def make_plot(data):
    data.to_csv(RESULTS_DIR / "gap_to_reference_results.csv", index=False)
    summary = (
        data.groupby(["instance_type", "utility", "method", "K"])
        .agg(
            mean_relative_gap=("relative_gap", "mean"),
            std_relative_gap=("relative_gap", "std"),
            min_relative_gap=("relative_gap", "min"),
            max_relative_gap=("relative_gap", "max"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "gap_to_reference_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    colors = {"OCG": "#1f77b4", "FW": "#d62728"}
    titles = {
        ("gridworld", "soft"): "GridWorld · soft",
        ("gridworld", "log"): "GridWorld · log",
        ("allocation", "soft"): "Allocation · soft",
        ("allocation", "log"): "Allocation · log",
    }
    for ax, key in zip(axes.flat, titles):
        subset = summary[(summary.instance_type == key[0]) & (summary.utility == key[1])]
        for method in ["OCG", "FW"]:
            row = subset[subset.method == method].sort_values("K")
            ax.errorbar(
                row.K,
                100.0 * row.mean_relative_gap,
                yerr=100.0 * row.std_relative_gap.fillna(0.0),
                marker="o",
                linewidth=1.7,
                capsize=3,
                label=method,
                color=colors[method],
            )
        ax.set_title(titles[key], fontsize=10)
        ax.set_xticks(list(AGGREGATE_K_VALUES))
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("Relative gap (%)")
        ax.set_xlabel("DP-oracle calls K")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Gap to Direct-CVX numerical reference", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "gap_to_reference.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "gap_to_reference.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


if __name__ == "__main__":
    summary = make_plot(build_gap_table())
    print(summary.to_string(index=False))
