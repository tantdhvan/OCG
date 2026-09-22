# Reproducibility package

This repository contains the tabular experiment code and the stored outputs
for the manuscript:

> Expected-Exposure DR-Submodular Rewards for Model-Based Reinforcement
> Learning via Occupancy Continuous Greedy

The package is self-contained. It evaluates policies through exact finite-
horizon occupancy measures under known transition models; it does not use
Monte Carlo rollouts for the reported objectives.

## What is included

```text
run_experiments.py          Main GridWorld and stochastic-allocation experiments
run_direct_cvx.py           Direct-CVX numerical reference for the two utilities
run_frank_wolfe.py          Frank-Wolfe occupancy baseline
run_allocation_scaling.py   Coarse allocation scaling check
make_gap_to_reference.py    Per-instance OCG/FW gap table and figure
reproduce_all.py            Ordered driver for the complete experiment suite
validate_outputs.py         Artifact-count and numerical-diagnostic checks
requirements.txt            Python dependencies
MANIFEST.md                 File-level package inventory
results/                    CSV/JSON experiment outputs from the frozen run
figures/                    PNG/PDF figures from the frozen run
```

The committed `results/` and `figures/` directories are a snapshot of the
manuscript experiment run. They can be inspected without rerunning the full
suite. Re-running the suite refreshes these files in place.

## Installation

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Direct-CVX stage uses CVXPY with Clarabel as the requested solver and SCS
as a recorded fallback. The package does not require a commercial solver.

The reference environment used for the frozen local snapshot was:

```text
Python 3.14.5
numpy 2.4.6
pandas 3.0.3
matplotlib 3.11.0
cvxpy 1.9.3
solvers: CLARABEL, SCS, SCIPY, HIGHS, OSQP
```

Exact runtimes can vary with hardware, Python, BLAS, and solver versions.

## One-command reproduction

After installation, run:

```bash
python reproduce_all.py
python validate_outputs.py
```

`reproduce_all.py` runs the stages in this order:

1. main GridWorld and allocation experiments;
2. GridWorld scaling;
3. Direct-CVX numerical references;
4. Frank-Wolfe baseline;
5. allocation scaling;
6. per-instance gap-to-reference table and figure.

The complete run overwrites the CSV and figure files under `results/` and
`figures/`. The run is deterministic for the fixed seeds, apart from small
floating-point and solver-version differences.

## Stage-by-stage commands

The stages can also be run separately from the repository root:

```bash
# Main OCG and baseline experiments, scaling outputs, and standard figures
python run_experiments.py

# Pilot only, useful for checking the CVXPY installation
python run_direct_cvx.py

# Pilot only, useful for checking the Frank-Wolfe implementation
python run_frank_wolfe.py

# Full numerical reference and full Frank-Wolfe runs
python -c "from run_direct_cvx import run_direct_cvx_full; run_direct_cvx_full()"
python -c "from run_frank_wolfe import run_full; run_full()"

# Allocation scaling and gap figure
python run_allocation_scaling.py
python make_gap_to_reference.py

# Stored-output audit
python validate_outputs.py
```

The module entry points of `run_direct_cvx.py` and `run_frank_wolfe.py` are
deliberately pilot runs. Use `reproduce_all.py` or the explicit full-run
commands above for the complete reference and baseline tables.

## Experiment protocol

### GridWorld

- Base map: `10 x 10`, horizon `H=24`, four actions, and start state `(0,0)`.
- Intended moves have probability `0.95`; the remaining probability is a
  stay-put transition.
- The base maps have 8 obstacles and 12 target cells.
- Aggregate seeds are `1000,...,1009` (10 maps).
- The representative qualitative map uses the deterministic fixed-map branch
  in `build_gridworld`.
- Exposure features use Gaussian target exposure with `sigma=1.35`.
- The soft-coverage utility is `sum_u (1 - exp(-x_u))`; the logarithmic utility
  is `sum_u log(1 + x_u)`.
- OCG uses `K in {5,10,20,40}` for aggregate comparisons and additionally
  records `K=1,2` in the representative/sensitivity outputs.
- Baselines include Random, Nearest-target where applicable, Additive, and
  Marginal replanning.

The GridWorld scaling check uses five seeds for each setting:

| grid | horizon | targets | obstacles | seeds |
| --- | ---: | ---: | ---: | --- |
| `10 x 10` | 24 | 12 | 8 | `3000,...,3004` |
| `15 x 15` | 32 | 20 | 20 | `3100,...,3104` |
| `20 x 20` | 40 | 30 | 40 | `3200,...,3204` |

### Stochastic allocation MDP

- Base setting: `m=8` groups, `q=5` demand modes, and horizon `H=30`.
- Aggregate seeds are `2000,...,2009` (10 random instances).
- Initial distributions and transition rows are generated from seeded
  Dirichlet draws.
- Each action allocates attention to one group; exposure features use the
  generated state-dependent effectiveness coefficients.
- Both soft-coverage and logarithmic utilities are evaluated.
- Baselines include Random, Myopic-effectiveness, Additive, and Marginal
  replanning.

The allocation scaling check uses `q=5`, five seeds per setting, and
`(m,H) in {(8,30),(16,60),(32,120)}`. It compares Additive, OCG with
`K=20`, OCG with `K=40`, and Frank-Wolfe with `K=40` for both utilities.

## Direct-CVX reference and Frank-Wolfe accounting

For the two concave benchmark utilities only, `run_direct_cvx.py` maximizes
the utility directly over the linear occupancy-flow constraints. This is a
numerical reference for the experiments, not a certificate for the full
monotone DR-submodular class. The code records solver status, requested and
used solver, fallback information, residuals, objective discrepancy, and
acceptance status.

The fixed acceptance thresholds are:

```text
initial-distribution residual <= 1e-7
flow residual                <= 1e-7
layer-mass error             <= 1e-6
minimum occupancy            >= -1e-8
objective discrepancy        <= 1e-8 * max(1, |evaluated objective|)
```

Frank-Wolfe uses the same additive-MDP dynamic-programming oracle as OCG. For
each `K`, both methods use exactly `K` DP-oracle calls. Frank-Wolfe chooses
segment steps by golden-section line search with tolerance `1e-8`; its extra
scalar objective evaluations and runtime are recorded separately. Equal DP
oracle counts therefore do not claim equal total wall-clock cost.

## Output inventory

The main experiment produces:

- `results_single_map.csv` and `results_multiseed.csv`;
- `aggregate_table.csv`;
- `gridworld_scaling_results.csv` and `gridworld_scaling_summary.csv`;
- `allocation_soft_results.csv`, `allocation_soft_aggregate_table.csv`;
- `allocation_log_results.csv`, `allocation_log_aggregate_table.csv`;
- `allocation_k_sensitivity.csv`;
- standard PNG/PDF figures for aggregate values, sensitivity to `K`, heatmaps,
  target exposure, OCG progress, and allocation comparisons.

The additional audit stages produce:

- `direct_cvx_pilot.csv` and `direct_cvx_pilot_summary.json`;
- `direct_cvx_full_results.csv` and `direct_cvx_full_summary.csv`;
- `frank_wolfe_pilot.csv` and `frank_wolfe_pilot_summary.json`;
- `frank_wolfe_full_results.csv` and `frank_wolfe_full_summary.csv`;
- `allocation_scaling_results.csv` and `allocation_scaling_summary.csv`;
- `gap_to_reference_results.csv` and `gap_to_reference_summary.csv`;
- `figures/gap_to_reference.png` and `figures/gap_to_reference.pdf`.

The gap table covers the 40 small GridWorld/allocation instances with direct
references, and reports per-instance relative gaps for OCG and Frank-Wolfe at
`K in {5,10,20,40}`. Scaling settings without a direct reference are not
included in that figure.

## Interpretation and limitations

- All objective values are computed from exact finite-horizon occupancy
  propagation under the generated known transition model.
- The stored Direct-CVX values are numerical references for the two utilities
  tested here; they do not extend the theoretical guarantee to arbitrary
  monotone DR-submodular utilities.
- Runtime columns are machine-dependent. Planning-call columns count DP oracle
  calls, not all arithmetic or line-search evaluations.
- The scaling study is a coarse tabular check over the listed sizes, not a
  general asymptotic runtime benchmark.
- The committed CSV and figure snapshot is the authoritative record for the
  manuscript revision; regenerated results may differ slightly in last digits
  across dependency versions.

## License and citation

This package is distributed with the manuscript source as an experiment
artifact. Add the repository license and citation metadata required by the
hosting repository before public release.
