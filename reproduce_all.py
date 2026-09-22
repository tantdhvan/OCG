"""Run every experiment stage used by the manuscript.

The implementation of each experiment remains in its own module so that
individual stages can also be rerun or inspected independently.
"""

from make_gap_to_reference import build_gap_table, make_plot
from run_allocation_scaling import run_full as run_allocation_scaling
from run_direct_cvx import run_direct_cvx_full
from run_experiments import (
    run_allocation_experiment,
    run_experiment,
    run_gridworld_scaling_experiment,
)
from run_frank_wolfe import run_full as run_frank_wolfe


def main():
    print("[1/6] Main GridWorld and allocation experiments")
    run_experiment()
    run_gridworld_scaling_experiment()
    run_allocation_experiment()

    print("[2/6] Direct-CVX numerical references")
    run_direct_cvx_full()

    print("[3/6] Frank-Wolfe baseline")
    run_frank_wolfe()

    print("[4/6] Allocation scaling check")
    run_allocation_scaling()

    print("[5/6] Gap-to-reference table and figure")
    make_plot(build_gap_table())

    print("[6/6] Complete")
    print("Run validate_outputs.py to check the generated artifact counts and diagnostics.")


if __name__ == "__main__":
    main()
