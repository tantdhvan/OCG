"""Validate the committed or regenerated reproducibility artifacts."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def require_file(name):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return path


def main():
    checks = []

    multi = pd.read_csv(require_file("results_multiseed.csv"))
    checks.append(("GridWorld multiseed rows", len(multi) == 80))
    checks.append(("GridWorld multiseed seeds", sorted(multi.map_seed.unique()) == list(range(1000, 1010))))

    allocation = pd.read_csv(require_file("allocation_soft_results.csv"))
    allocation_log = pd.read_csv(require_file("allocation_log_results.csv"))
    checks.append(("Allocation soft rows", len(allocation) == 80))
    checks.append(("Allocation log rows", len(allocation_log) == 80))

    grid_scaling = pd.read_csv(require_file("gridworld_scaling_results.csv"))
    checks.append(("GridWorld scaling rows", len(grid_scaling) == 45))

    allocation_scaling = pd.read_csv(require_file("allocation_scaling_results.csv"))
    checks.append(("Allocation scaling rows", len(allocation_scaling) == 120))

    direct = pd.read_csv(require_file("direct_cvx_full_results.csv"))
    references = direct[direct.method == "Direct-CVX"]
    checks.append(("Direct-CVX rows", len(references) == 40))
    checks.append(("Direct-CVX accepted", bool(references.accepted.all())))
    checks.append(("Direct-CVX residual threshold", references.max_flow_residual.max() <= 1e-7))
    checks.append(("Direct-CVX layer-mass threshold", references.layer_mass_error.max() <= 1e-6))

    fw = pd.read_csv(require_file("frank_wolfe_full_results.csv"))
    checks.append(("Frank-Wolfe rows", len(fw) == 160))
    checks.append(("Frank-Wolfe oracle-call accounting", bool((fw.planning_calls == fw.K).all())))
    checks.append(("Frank-Wolfe flow threshold", fw.max_flow_residual.max() <= 1e-7))

    gap = pd.read_csv(require_file("gap_to_reference_results.csv"))
    checks.append(("Gap table rows", len(gap) == 320))
    checks.append(("Gap table references present", bool(gap.reference_objective.notna().all())))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if failed:
        raise SystemExit("Artifact validation failed: " + "; ".join(failed))
    print(f"All {len(checks)} artifact checks passed.")


if __name__ == "__main__":
    main()
