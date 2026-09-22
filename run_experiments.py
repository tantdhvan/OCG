"""Reproduce the tabular experiments for the expected-exposure OCG paper.

The script evaluates all policies through exact finite-horizon occupancy
measures. It does not use Monte Carlo rollouts.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

SEED = 7
GRID_N = 10
HORIZON = 24
NOISE_STAY = 0.05
SIGMA = 1.35
BETA = 1.0
OCG_K_VALUES = [1, 2, 5, 10, 20, 40]
AGGREGATE_K_VALUES = [5, 10, 20, 40]
N_RANDOM_MAPS = 10
N_SCALING_MAPS = 5
SCALING_GRID_SETTINGS = [
    {"grid_n": 10, "horizon": 24, "target_count": 12, "obstacle_count": 8},
    {"grid_n": 15, "horizon": 32, "target_count": 20, "obstacle_count": 20},
    {"grid_n": 20, "horizon": 40, "target_count": 30, "obstacle_count": 40},
]
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_DIR = str(RESULTS_DIR)

np.random.seed(SEED)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GridWorld:
    n: int
    states: list
    state_to_idx: dict
    start_state: tuple
    targets: list
    obstacles: set
    transitions: np.ndarray  # shape: S x A x S
    action_names: list


def build_gridworld(n=10, noise_stay=0.05, seed=None, obstacle_count=8, target_count=12):
    """Build a tabular gridworld.

    seed=None returns the fixed representative map used for visual figures.
    seed=int returns a random map with the requested size, obstacle count, and
    target count, used for aggregate statistics and scaling experiments.
    """
    start_state = (0, 0)
    if seed is None:
        obstacles = {
            (3, 3), (3, 4), (3, 5),
            (6, 4), (6, 5), (6, 6),
            (4, 7), (5, 7),
        }
        targets = [
            (1, 7), (1, 8), (2, 8),
            (6, 1), (7, 1), (8, 2),
            (8, 8), (7, 8), (8, 7),
            (4, 1), (5, 2), (2, 5),
        ]
    else:
        rng = np.random.default_rng(seed)
        cells = [(i, j) for i in range(n) for j in range(n) if (i, j) != start_state]
        obstacle_idx = rng.choice(len(cells), size=obstacle_count, replace=False)
        obstacles = {cells[i] for i in obstacle_idx}
        free_cells = [c for c in cells if c not in obstacles]

        # Encourage separated targets by sampling from quadrants when possible.
        quadrants = [
            [c for c in free_cells if c[0] < n // 2 and c[1] < n // 2],
            [c for c in free_cells if c[0] < n // 2 and c[1] >= n // 2],
            [c for c in free_cells if c[0] >= n // 2 and c[1] < n // 2],
            [c for c in free_cells if c[0] >= n // 2 and c[1] >= n // 2],
        ]
        targets = []
        max_per_quadrant = max(1, int(np.ceil(target_count / 4)))
        for q in quadrants:
            take = min(max_per_quadrant, len(q), target_count - len(targets))
            if take:
                idx = rng.choice(len(q), size=take, replace=False)
                targets.extend([q[i] for i in idx])
        if len(targets) < target_count:
            remaining = [c for c in free_cells if c not in set(targets)]
            idx = rng.choice(len(remaining), size=target_count - len(targets), replace=False)
            targets.extend([remaining[i] for i in idx])

    states = [(i, j) for i in range(n) for j in range(n) if (i, j) not in obstacles]
    state_to_idx = {s: idx for idx, s in enumerate(states)}

    action_names = ["up", "down", "left", "right", "stay"]
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    S = len(states)
    A = len(action_names)
    transitions = np.zeros((S, A, S), dtype=float)

    def move(state, action_id):
        di, dj = deltas[action_id]
        ni, nj = state[0] + di, state[1] + dj
        if ni < 0 or ni >= n or nj < 0 or nj >= n or (ni, nj) in obstacles:
            return state
        return (ni, nj)

    for s_idx, state in enumerate(states):
        stay_idx = state_to_idx[state]
        for a in range(A):
            next_state = move(state, a)
            next_idx = state_to_idx[next_state]
            transitions[s_idx, a, next_idx] += 1.0 - noise_stay
            transitions[s_idx, a, stay_idx] += noise_stay

    return GridWorld(
        n=n,
        states=states,
        state_to_idx=state_to_idx,
        start_state=start_state,
        targets=targets,
        obstacles=obstacles,
        transitions=transitions,
        action_names=action_names,
    )


def build_features(env: GridWorld, horizon: int, sigma: float):
    """Feature phi[h, s, a, u]: soft exposure of current cell to target u."""
    S = len(env.states)
    A = len(env.action_names)
    m = len(env.targets)
    phi_state = np.zeros((S, m), dtype=float)

    for s_idx, (i, j) in enumerate(env.states):
        for u, (ti, tj) in enumerate(env.targets):
            dist2 = (i - ti) ** 2 + (j - tj) ** 2
            phi_state[s_idx, u] = np.exp(-dist2 / (sigma ** 2))

    # Same feature for all actions and all time steps.
    phi = np.repeat(phi_state[:, None, :], A, axis=1)
    phi = np.repeat(phi[None, :, :, :], horizon, axis=0)
    return phi


def G_value(x, beta=BETA, weights=None):
    if weights is None:
        weights = np.ones_like(x)
    return float(np.sum(weights * (1.0 - np.exp(-beta * x))))


def G_grad(x, beta=BETA, weights=None):
    if weights is None:
        weights = np.ones_like(x)
    return weights * beta * np.exp(-beta * x)


def utility_value(x, utility="soft", beta=BETA, weights=None):
    if weights is None:
        weights = np.ones_like(x)
    if utility == "soft":
        return float(np.sum(weights * (1.0 - np.exp(-beta * x))))
    if utility == "log":
        return float(np.sum(weights * np.log1p(x)))
    raise ValueError(f"Unknown utility: {utility}")


def utility_grad(x, utility="soft", beta=BETA, weights=None):
    if weights is None:
        weights = np.ones_like(x)
    if utility == "soft":
        return weights * beta * np.exp(-beta * x)
    if utility == "log":
        return weights / (1.0 + x)
    raise ValueError(f"Unknown utility: {utility}")


def exposure_from_occupancy(d, phi):
    """Compute Phi d."""
    return np.einsum("hsa,hsau->u", d, phi)


def finite_horizon_dp(transitions, reward, start_dist):
    """
    Solve max expected additive reward finite-horizon MDP.

    transitions: S x A x S
    reward: H x S x A
    start_dist: S

    Returns deterministic Markov policy: H x S action ids.
    """
    H, S, A = reward.shape
    V = np.zeros((H + 1, S), dtype=float)
    policy = np.zeros((H, S), dtype=int)

    for h in reversed(range(H)):
        Q = reward[h] + transitions @ V[h + 1]
        policy[h] = np.argmax(Q, axis=1)
        V[h] = np.max(Q, axis=1)

    return policy, float(start_dist @ V[0])


def occupancy_from_policy(transitions, policy, start_dist):
    """Forward-propagate occupancy for a deterministic Markov policy."""
    H, S = policy.shape
    A = transitions.shape[1]
    d = np.zeros((H, S, A), dtype=float)
    state_dist = start_dist.copy()

    for h in range(H):
        for s in range(S):
            a = policy[h, s]
            d[h, s, a] = state_dist[s]
        next_dist = np.zeros(S, dtype=float)
        for s in range(S):
            a = policy[h, s]
            next_dist += state_dist[s] * transitions[s, a]
        state_dist = next_dist

    return d


def random_policy_occupancy(transitions, horizon, start_dist):
    """Occupancy for a uniform-random Markov policy."""
    S, A, _ = transitions.shape
    d = np.zeros((horizon, S, A), dtype=float)
    state_dist = start_dist.copy()

    for h in range(horizon):
        for a in range(A):
            d[h, :, a] = state_dist / A
        next_dist = np.zeros(S, dtype=float)
        for s in range(S):
            for a in range(A):
                next_dist += state_dist[s] * (1.0 / A) * transitions[s, a]
        state_dist = next_dist
    return d


def occupancy_continuous_greedy(transitions, phi, start_dist, K, beta=BETA, weights=None):
    """Run Occupancy Continuous Greedy and return the averaged occupancy z_K."""
    H, S, A, m = phi.shape
    z = np.zeros((H, S, A), dtype=float)
    history = []

    for k in range(K):
        x = exposure_from_occupancy(z, phi)
        grad = G_grad(x, beta=beta, weights=weights)
        reward = np.einsum("u,hsau->hsa", grad, phi)
        policy, _ = finite_horizon_dp(transitions, reward, start_dist)
        d_k = occupancy_from_policy(transitions, policy, start_dist)
        z += d_k / K
        history.append(G_value(exposure_from_occupancy(z, phi), beta=beta, weights=weights))

    return z, np.array(history)


def occupancy_continuous_greedy_generic(transitions, phi, start_dist, K, utility="soft", beta=BETA, weights=None):
    """Run OCG for a selectable separable DR-submodular utility."""
    H, S, A, m = phi.shape
    z = np.zeros((H, S, A), dtype=float)
    history = []

    for _ in range(K):
        x = exposure_from_occupancy(z, phi)
        grad = utility_grad(x, utility=utility, beta=beta, weights=weights)
        reward = np.einsum("u,hsau->hsa", grad, phi)
        policy, _ = finite_horizon_dp(transitions, reward, start_dist)
        d_k = occupancy_from_policy(transitions, policy, start_dist)
        z += d_k / K
        history.append(utility_value(exposure_from_occupancy(z, phi), utility=utility, beta=beta, weights=weights))

    return z, np.array(history)


def nearest_target_greedy_occupancy(env: GridWorld, phi, start_dist):
    """Greedy baseline: at each step move toward the currently least exposed target."""
    H, S, A, m = phi.shape
    policy = np.zeros((H, S), dtype=int)
    target_coords = np.array(env.targets)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    # Build a deterministic policy state-wise from a simple geometric rule.
    exposure = np.zeros(m)
    for h in range(H):
        least_target = int(np.argmin(exposure))
        ti, tj = target_coords[least_target]
        for s_idx, (i, j) in enumerate(env.states):
            best_a = 4
            best_dist = (i - ti) ** 2 + (j - tj) ** 2
            for a, (di, dj) in enumerate(actions):
                ni, nj = i + di, j + dj
                if (ni, nj) not in env.state_to_idx:
                    continue
                dist = (ni - ti) ** 2 + (nj - tj) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_a = a
            policy[h, s_idx] = best_a

        d_prefix = occupancy_from_policy(env.transitions, policy[: h + 1], start_dist)
        exposure = exposure_from_occupancy(d_prefix, phi[: h + 1])
    return occupancy_from_policy(env.transitions, policy, start_dist)


def iterative_marginal_replanning(transitions, phi, start_dist, rounds=5, beta=BETA, weights=None):
    """
    Simple baseline: repeatedly plan against the marginal reward at the current
    policy occupancy, then replace the policy. This is a local fixed-point style
    heuristic, not the theorem algorithm.
    """
    H, S, A, m = phi.shape
    reward0 = np.sum(phi, axis=-1)
    policy, _ = finite_horizon_dp(transitions, reward0, start_dist)
    d = occupancy_from_policy(transitions, policy, start_dist)

    for _ in range(rounds):
        x = exposure_from_occupancy(d, phi)
        grad = G_grad(x, beta=beta, weights=weights)
        reward = np.einsum("u,hsau->hsa", grad, phi)
        policy, _ = finite_horizon_dp(transitions, reward, start_dist)
        d = occupancy_from_policy(transitions, policy, start_dist)
    return d


def evaluate_method(name, d, phi, beta=BETA, weights=None, tau=0.5, planning_calls=0):
    x = exposure_from_occupancy(d, phi)
    value = G_value(x, beta=beta, weights=weights)
    covered = int(np.sum(x >= tau))
    probs = x / max(np.sum(x), 1e-12)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return {
        "method": name,
        "G(Phi d)": value,
        "covered_targets_tau_0.5": covered,
        "exposure_entropy": entropy,
        "planning_calls": planning_calls,
    }


def evaluate_method_generic(name, d, phi, utility="soft", beta=BETA, weights=None, tau=1.0, planning_calls=0):
    x = exposure_from_occupancy(d, phi)
    value = utility_value(x, utility=utility, beta=beta, weights=weights)
    covered = int(np.sum(x >= tau))
    probs = x / max(np.sum(x), 1e-12)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return {
        "method": name,
        "G(Phi d)": value,
        "covered_groups_tau_1.0": covered,
        "exposure_entropy": entropy,
        "planning_calls": planning_calls,
    }


def timed_eval(name, fn, phi, planning_calls):
    start = time.perf_counter()
    d = fn()
    elapsed = time.perf_counter() - start
    row = evaluate_method(name, d, phi, planning_calls=planning_calls)
    row["runtime_sec"] = elapsed
    return d, row


def state_visitation(d):
    """Sum occupancy over time and actions to get expected state visits."""
    return np.sum(d, axis=(0, 2))


def grid_values(env: GridWorld, values_by_state, fill=np.nan):
    grid = np.full((env.n, env.n), fill, dtype=float)
    for s_idx, (i, j) in enumerate(env.states):
        grid[i, j] = values_by_state[s_idx]
    return grid


def plot_grid_heatmap(env: GridWorld, d, title, filename):
    visits = state_visitation(d)
    grid = grid_values(env, visits, fill=np.nan)

    plt.figure(figsize=(6.2, 5.4))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")
    plt.imshow(grid, cmap=cmap)
    plt.colorbar(label="Expected visits")

    target_y = [p[0] for p in env.targets]
    target_x = [p[1] for p in env.targets]
    plt.scatter(target_x, target_y, marker="*", s=130, c="white", edgecolors="red", linewidths=1.1, label="Targets")
    plt.scatter([env.start_state[1]], [env.start_state[0]], marker="o", s=90, c="cyan", edgecolors="black", label="Start")

    plt.xticks(range(env.n))
    plt.yticks(range(env.n))
    plt.grid(color="white", linewidth=0.35, alpha=0.45)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def plot_exposure_bars(exposures, filename):
    labels = list(exposures.keys())
    m = len(next(iter(exposures.values())))
    x = np.arange(m)
    width = 0.8 / len(labels)

    plt.figure(figsize=(8.2, 4.4))
    for idx, label in enumerate(labels):
        plt.bar(x + idx * width, exposures[label], width=width, label=label)
    plt.xticks(x + width * (len(labels) - 1) / 2, [str(i + 1) for i in range(m)])
    plt.xlabel("Target")
    plt.ylabel("Expected exposure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def plot_ocg_value_curve(curves, filename):
    plt.figure(figsize=(6.5, 4.2))
    for K, hist in curves.items():
        plt.plot(np.arange(1, len(hist) + 1), hist, marker="o", label=f"K={K}")
    plt.xlabel("OCG iteration")
    plt.ylabel("G(Phi z_k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def plot_summary_table(df, filename):
    show_cols = ["method", "G(Phi d)", "covered_targets_tau_0.5", "exposure_entropy", "planning_calls"]
    display_df = df[show_cols].copy()
    display_df["G(Phi d)"] = display_df["G(Phi d)"].map(lambda v: f"{v:.3f}")
    display_df["exposure_entropy"] = display_df["exposure_entropy"].map(lambda v: f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(9.8, 0.55 * len(display_df) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def plot_aggregate_bars(agg, filename):
    methods = agg["method"].tolist()
    means = agg["G_mean"].to_numpy()
    stds = agg["G_std"].to_numpy()
    x = np.arange(len(methods))

    plt.figure(figsize=(8.0, 4.2))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("G(Phi d), mean +/- std.")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def plot_sensitivity_k(multiseed_df, filename):
    rows = []
    for method, sub in multiseed_df.groupby("method"):
        if method.startswith("OCG K="):
            K = int(method.split("=")[1])
            rows.append((K, sub["G(Phi d)"].mean(), sub["G(Phi d)"].std()))
    rows = sorted(rows)
    K = np.array([r[0] for r in rows])
    mean = np.array([r[1] for r in rows])
    std = np.array([r[2] for r in rows])

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(K, mean, marker="o")
    plt.fill_between(K, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Number of OCG iterations K")
    plt.ylabel("G(Phi d)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def plot_allocation_sensitivity_k(allocation_df, filename):
    """Plot objective and runtime sensitivity to K for allocation OCG runs."""
    rows = []
    for (utility, method), sub in allocation_df.groupby(["utility", "method"]):
        if method.startswith("OCG K="):
            K = int(method.split("=")[1])
            rows.append(
                {
                    "utility": utility,
                    "K": K,
                    "G_mean": sub["G(Phi d)"].mean(),
                    "G_std": sub["G(Phi d)"].std(),
                    "runtime_mean": sub["runtime_sec"].mean(),
                    "runtime_std": sub["runtime_sec"].std(),
                }
            )
    sens = pd.DataFrame(rows).sort_values(["utility", "K"])
    sens.to_csv(RESULTS_DIR / "allocation_k_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    labels = {"soft": "Soft coverage", "log": "Log utility"}
    for utility, sub in sens.groupby("utility"):
        sub = sub.sort_values("K")
        label = labels.get(utility, utility)
        axes[0].plot(sub["K"], sub["G_mean"], marker="o", label=label)
        axes[0].fill_between(
            sub["K"],
            sub["G_mean"] - sub["G_std"],
            sub["G_mean"] + sub["G_std"],
            alpha=0.15,
        )
        axes[1].plot(sub["K"], sub["runtime_mean"], marker="o", label=label)
        axes[1].fill_between(
            sub["K"],
            sub["runtime_mean"] - sub["runtime_std"],
            sub["runtime_mean"] + sub["runtime_std"],
            alpha=0.15,
        )

    axes[0].set_xlabel("Number of OCG iterations K")
    axes[0].set_ylabel("G(Phi d)")
    axes[1].set_xlabel("Number of OCG iterations K")
    axes[1].set_ylabel("Runtime (s)")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def plot_allocation_bars(agg, filename, title):
    methods = agg["method"].tolist()
    means = agg["G_mean"].to_numpy()
    stds = agg["G_std"].to_numpy()
    x = np.arange(len(methods))
    short_labels = {
        "Random": "Rand",
        "Myopic-effectiveness": "Myopic",
        "Additive": "Add",
        "Marginal": "Marg",
        "OCG K=5": "K=5",
        "OCG K=10": "K=10",
        "OCG K=20": "K=20",
        "OCG K=40": "K=40",
    }
    tick_labels = [short_labels.get(method, method) for method in methods]

    plt.figure(figsize=(6.2, 4.0))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, tick_labels, rotation=0, fontsize=8)
    plt.ylabel("G(Phi d), mean +/- std.")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=220)
    plt.savefig(FIGURES_DIR / f"{filename}.pdf")
    plt.close()


def summarize_grid_scaling(rows):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "gridworld_scaling_results.csv", index=False)

    summary = (
        df.groupby(["grid_n", "horizon", "target_count", "method"])
        .agg(
            state_count_mean=("state_count", "mean"),
            G_mean=("G(Phi d)", "mean"),
            G_std=("G(Phi d)", "std"),
            covered_mean=("covered_targets_tau_0.5", "mean"),
            entropy_mean=("exposure_entropy", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            planning_calls=("planning_calls", "mean"),
        )
        .reset_index()
    )
    order = {"Additive": 0, "OCG K=20": 1, "OCG K=40": 2}
    summary["order"] = summary["method"].map(order)
    summary = summary.sort_values(["grid_n", "order"]).drop(columns=["order"])
    summary["best_found_ratio"] = summary.groupby(["grid_n", "horizon", "target_count"])["G_mean"].transform(
        lambda values: values / values.max()
    )
    summary.to_csv(RESULTS_DIR / "gridworld_scaling_summary.csv", index=False)
    return summary


def build_allocation_mdp(seed, q=5, m=8, horizon=30):
    """Known stochastic allocation MDP with demand modes and group actions."""
    rng = np.random.default_rng(seed)
    start_dist = rng.dirichlet(np.ones(q))
    transitions = np.zeros((q, m, q), dtype=float)
    for s in range(q):
        for a in range(m):
            base = rng.dirichlet(np.ones(q))
            stay = np.zeros(q)
            stay[s] = 1.0
            action_bias = np.zeros(q)
            action_bias[a % q] = 1.0
            transitions[s, a, :] = 0.60 * base + 0.25 * stay + 0.15 * action_bias
            transitions[s, a, :] /= transitions[s, a, :].sum()
    eta = rng.uniform(0.25, 1.50, size=(q, m))
    phi = np.zeros((horizon, q, m, m), dtype=float)
    for h in range(horizon):
        for s in range(q):
            for a in range(m):
                phi[h, s, a, a] = eta[s, a]
    return transitions, start_dist, phi, eta


def deterministic_policy_occupancy(transitions, policy, start_dist):
    return occupancy_from_policy(transitions, policy, start_dist)


def allocation_marginal_replanning(transitions, phi, start_dist, utility="soft", rounds=5):
    reward0 = np.sum(phi, axis=-1)
    policy, _ = finite_horizon_dp(transitions, reward0, start_dist)
    d = occupancy_from_policy(transitions, policy, start_dist)
    for _ in range(rounds):
        x = exposure_from_occupancy(d, phi)
        grad = utility_grad(x, utility=utility)
        reward = np.einsum("u,hsau->hsa", grad, phi)
        policy, _ = finite_horizon_dp(transitions, reward, start_dist)
        d = occupancy_from_policy(transitions, policy, start_dist)
    return d


def run_one_allocation(seed, utility="soft", horizon=30, q=5, m=8):
    transitions, start_dist, phi, eta = build_allocation_mdp(seed, q=q, m=m, horizon=horizon)
    results = []

    def timed(name, fn, calls):
        start = time.perf_counter()
        d = fn()
        elapsed = time.perf_counter() - start
        row = evaluate_method_generic(name, d, phi, utility=utility, planning_calls=calls)
        row["runtime_sec"] = elapsed
        results.append(row)
        return d

    timed("Random", lambda: random_policy_occupancy(transitions, horizon, start_dist), 0)

    reward_add = np.sum(phi, axis=-1)
    timed("Additive", lambda: occupancy_from_policy(transitions, finite_horizon_dp(transitions, reward_add, start_dist)[0], start_dist), 1)

    myopic_policy = np.zeros((horizon, q), dtype=int)
    for h in range(horizon):
        for s in range(q):
            myopic_policy[h, s] = int(np.argmax(eta[s, :]))
    timed("Myopic-effectiveness", lambda: deterministic_policy_occupancy(transitions, myopic_policy, start_dist), 0)

    timed("Marginal", lambda: allocation_marginal_replanning(transitions, phi, start_dist, utility=utility, rounds=5), 6)

    for K in AGGREGATE_K_VALUES:
        timed(
            f"OCG K={K}",
            lambda K=K: occupancy_continuous_greedy_generic(transitions, phi, start_dist, K=K, utility=utility)[0],
            K,
        )
    return pd.DataFrame(results)


def run_allocation_experiment():
    order = ["Random", "Myopic-effectiveness", "Additive", "Marginal", "OCG K=5", "OCG K=10", "OCG K=20", "OCG K=40"]
    combined_rows = []
    for utility in ["soft", "log"]:
        all_rows = []
        for i in range(N_RANDOM_MAPS):
            df_i = run_one_allocation(seed=2000 + i, utility=utility)
            df_i["instance_seed"] = 2000 + i
            df_i["utility"] = utility
            all_rows.append(df_i)
            combined_rows.append(df_i)
        multi = pd.concat(all_rows, ignore_index=True)
        multi.to_csv(os.path.join(OUTPUT_DIR, f"allocation_{utility}_results.csv"), index=False)
        agg = (
            multi.groupby("method")
            .agg(
                G_mean=("G(Phi d)", "mean"),
                G_std=("G(Phi d)", "std"),
                covered_mean=("covered_groups_tau_1.0", "mean"),
                entropy_mean=("exposure_entropy", "mean"),
                runtime_mean=("runtime_sec", "mean"),
                planning_calls=("planning_calls", "mean"),
            )
            .reset_index()
        )
        agg["order"] = agg["method"].map({m: i for i, m in enumerate(order)})
        agg = agg.sort_values("order").drop(columns=["order"])
        agg["best_found_ratio"] = agg["G_mean"] / agg["G_mean"].max()
        agg.to_csv(os.path.join(OUTPUT_DIR, f"allocation_{utility}_aggregate_table.csv"), index=False)
        plot_allocation_bars(agg, f"allocation_{utility}_aggregate_bars", f"Stochastic Allocation MDP ({utility} utility)")
        print(f"\n=== Stochastic Allocation MDP ({utility}) ===")
        print(agg.to_string(index=False))
    plot_allocation_sensitivity_k(pd.concat(combined_rows, ignore_index=True), "allocation_sensitivity_k")


def run_one_map(env, horizon=HORIZON, sigma=SIGMA, include_all_k=True):
    phi = build_features(env, horizon=horizon, sigma=sigma)
    S = len(env.states)
    start_dist = np.zeros(S, dtype=float)
    start_dist[env.state_to_idx[env.start_state]] = 1.0
    weights = np.ones(len(env.targets), dtype=float)

    results = []
    occupancies = {}
    ocg_curves = {}

    d_random, row = timed_eval(
        "Random",
        lambda: random_policy_occupancy(env.transitions, horizon, start_dist),
        phi,
        planning_calls=0,
    )
    occupancies["Random"] = d_random
    results.append(row)

    def additive():
        reward_add = np.sum(phi, axis=-1)
        policy_add, _ = finite_horizon_dp(env.transitions, reward_add, start_dist)
        return occupancy_from_policy(env.transitions, policy_add, start_dist)

    d_add, row = timed_eval("Additive", additive, phi, planning_calls=1)
    occupancies["Additive"] = d_add
    results.append(row)

    d_nearest, row = timed_eval(
        "Nearest-target",
        lambda: nearest_target_greedy_occupancy(env, phi, start_dist),
        phi,
        planning_calls=0,
    )
    occupancies["Nearest-target"] = d_nearest
    results.append(row)

    d_replan, row = timed_eval(
        "Marginal",
        lambda: iterative_marginal_replanning(env.transitions, phi, start_dist, rounds=5),
        phi,
        planning_calls=6,
    )
    occupancies["Marginal"] = d_replan
    results.append(row)

    k_values = OCG_K_VALUES if include_all_k else AGGREGATE_K_VALUES
    for K in k_values:
        start = time.perf_counter()
        d_ocg, hist = occupancy_continuous_greedy(env.transitions, phi, start_dist, K=K, weights=weights)
        elapsed = time.perf_counter() - start
        label = f"OCG K={K}"
        occupancies[label] = d_ocg
        ocg_curves[K] = hist
        row = evaluate_method(label, d_ocg, phi, weights=weights, planning_calls=K)
        row["runtime_sec"] = elapsed
        results.append(row)

    return phi, pd.DataFrame(results), occupancies, ocg_curves


def run_gridworld_scaling_experiment():
    rows = []
    for setting_id, setting in enumerate(SCALING_GRID_SETTINGS):
        for i in range(N_SCALING_MAPS):
            seed = 3000 + 100 * setting_id + i
            env = build_gridworld(
                n=setting["grid_n"],
                noise_stay=NOISE_STAY,
                seed=seed,
                obstacle_count=setting["obstacle_count"],
                target_count=setting["target_count"],
            )
            horizon = setting["horizon"]
            phi = build_features(env, horizon=horizon, sigma=SIGMA)
            S = len(env.states)
            start_dist = np.zeros(S, dtype=float)
            start_dist[env.state_to_idx[env.start_state]] = 1.0
            weights = np.ones(len(env.targets), dtype=float)

            def record(method, d, elapsed, calls):
                row = evaluate_method(method, d, phi, weights=weights, planning_calls=calls)
                row["runtime_sec"] = elapsed
                row["grid_n"] = setting["grid_n"]
                row["horizon"] = horizon
                row["target_count"] = setting["target_count"]
                row["obstacle_count"] = setting["obstacle_count"]
                row["state_count"] = S
                row["map_seed"] = seed
                rows.append(row)

            start = time.perf_counter()
            reward_add = np.sum(phi, axis=-1)
            policy_add, _ = finite_horizon_dp(env.transitions, reward_add, start_dist)
            d_add = occupancy_from_policy(env.transitions, policy_add, start_dist)
            record("Additive", d_add, time.perf_counter() - start, 1)

            for K in [20, 40]:
                start = time.perf_counter()
                d_ocg, _ = occupancy_continuous_greedy(env.transitions, phi, start_dist, K=K, weights=weights)
                record(f"OCG K={K}", d_ocg, time.perf_counter() - start, K)

    summary = summarize_grid_scaling(rows)
    print("\n=== GridWorld scaling summary ===")
    print(summary.to_string(index=False))


def run_experiment():
    # Representative fixed map for qualitative figures.
    env = build_gridworld(n=GRID_N, noise_stay=NOISE_STAY, seed=None)
    phi, df, occupancies, ocg_curves = run_one_map(env, include_all_k=True)
    df = df.sort_values("G(Phi d)", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_DIR, "results_single_map.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")

    # Choose the best OCG run for figures.
    best_ocg_label = max(
        [k for k in occupancies if k.startswith("OCG")],
        key=lambda label: G_value(exposure_from_occupancy(occupancies[label], phi)),
    )

    plot_grid_heatmap(env, occupancies["Additive"], "State visitation: additive baseline", "heatmap_additive")
    plot_grid_heatmap(env, occupancies[best_ocg_label], f"State visitation: {best_ocg_label}", "heatmap_ocg_best")
    plot_grid_heatmap(env, occupancies["Marginal"], "State visitation: marginal replanning", "heatmap_marginal_replanning")

    exposures = {
        "Additive": exposure_from_occupancy(occupancies["Additive"], phi),
        "Marginal": exposure_from_occupancy(occupancies["Marginal"], phi),
        best_ocg_label: exposure_from_occupancy(occupancies[best_ocg_label], phi),
    }
    plot_exposure_bars(exposures, "target_exposure_bars")
    plot_ocg_value_curve(ocg_curves, "ocg_progress")
    plot_summary_table(df, "summary_table")

    # Aggregate random-map evaluation.
    all_rows = []
    for i in range(N_RANDOM_MAPS):
        env_i = build_gridworld(n=GRID_N, noise_stay=NOISE_STAY, seed=1000 + i)
        _, df_i, _, _ = run_one_map(env_i, include_all_k=False)
        df_i["map_seed"] = 1000 + i
        all_rows.append(df_i)
    multi = pd.concat(all_rows, ignore_index=True)
    multi.to_csv(os.path.join(OUTPUT_DIR, "results_multiseed.csv"), index=False)

    agg = (
        multi.groupby("method")
        .agg(
            G_mean=("G(Phi d)", "mean"),
            G_std=("G(Phi d)", "std"),
            covered_mean=("covered_targets_tau_0.5", "mean"),
            entropy_mean=("exposure_entropy", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            planning_calls=("planning_calls", "mean"),
        )
        .reset_index()
    )
    order = ["Random", "Nearest-target", "Additive", "Marginal", "OCG K=5", "OCG K=10", "OCG K=20", "OCG K=40"]
    agg["order"] = agg["method"].map({m: i for i, m in enumerate(order)})
    agg = agg.sort_values("order").drop(columns=["order"])
    agg["best_found_ratio"] = agg["G_mean"] / agg["G_mean"].max()
    agg.to_csv(os.path.join(OUTPUT_DIR, "aggregate_table.csv"), index=False)
    plot_aggregate_bars(agg, "aggregate_value_bars")
    plot_sensitivity_k(multi, "sensitivity_k")

    print(f"\nBest OCG run: {best_ocg_label}")
    print(f"Result files saved under: {RESULTS_DIR}")
    print(f"Figures saved under: {FIGURES_DIR}")
    print("\n=== Aggregate over random maps ===")
    print(agg.to_string(index=False))
    print("\nGenerated result files:")
    for name in sorted(os.listdir(RESULTS_DIR)):
        print(" -", name)
    print("\nGenerated figure files:")
    for name in sorted(os.listdir(FIGURES_DIR)):
        print(" -", name)


if __name__ == "__main__":
    run_experiment()
    run_gridworld_scaling_experiment()
    run_allocation_experiment()
