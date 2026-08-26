# Competitive Graph Recoloring: Reproducibility Package

This repository contains the simulation, deterministic replay, analysis, and
figure-generation code for the research article *Degree-Corrected Energy
Relaxation Bounds for Competitive Recoloring on Complex Networks*. It simulates
strict best-response recoloring on sparse graph families and records the
Lyapunov potential

```text
Phi = number of monochromatic edges.
```

The bound `recolor_steps <= |E|` applies to accepted recoloring moves. The
theorem-aligned scheduler also records all independent uniform vertex
activations, including rejected moves.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The wrapper also works without installation:

```powershell
python run_experiments.py demo
```

## Quick Demo

```powershell
python run_experiments.py demo --family er --n 500 --avg-degree 8 --k 3 --seed 2026 --verify
```

This prints a JSON summary and writes a potential plot to:

```text
outputs/figures/potential_er_n500_seed2026.png
```

## Scaling Experiment

For a paper-style run over Erdos-Renyi and Barabasi-Albert sparse graphs:

```powershell
python run_experiments.py experiment --families er ba --n-values 100 200 400 800 --trials 1000 --avg-degree 8 --k 3 --seed 2026
```

Outputs:

```text
outputs/data/scaling_results.csv
outputs/figures/steps_vs_edges.png
```

## Parameter Study

The systematic study used in the manuscript is reproduced by:

```powershell
python run_experiments.py grid --families er ba ws --n-values 100 200 400 800 --k-values 2 3 4 5 8 --avg-degrees 4 8 12 16 --trials 40 --seed 2026
```

It writes the trial-level CSV and the three-panel publication figure to
`outputs/data/parameter_results.csv` and
`outputs/figures/parameter_summary.png`.

## Degree-Corrected Relaxation Replay

The frozen 9,600-row grid can be replayed without changing the raw CSV:

```powershell
python replay_relaxation.py --input outputs/data/parameter_results.csv --output outputs/data/relaxation_replay.csv
python plot_relaxation.py --input outputs/data/relaxation_replay.csv --output outputs/figures/relaxation_validation.png
```

The replay fails on the first discrepancy in graph statistics, initial or final
potential, accepted moves, or activations. The separate output adds `H`, `W`,
`X0`, `tau_H`, `B_H`, maximum degree, and the `k > Delta` indicator.

The frozen inputs and outputs used for the manuscript are stored at:

```text
outputs/data/parameter_results.csv
outputs/data/relaxation_replay.csv
```

All graph, initialization, activation, and tie-breaking randomness is
deterministic from the recorded `seed` field. The main reproducibility outputs
are the replay CSV and `outputs/figures/relaxation_validation.png`.

## Scientific Reports Analysis

Figure 2 and its descriptive summaries are generated directly from the frozen
replay table; this command does not rerun any stochastic trajectory:

```powershell
python scientific_reports_analysis.py `
  --input outputs/data/relaxation_replay.csv `
  --output-dir outputs/scientific_reports
```

It produces:

```text
outputs/scientific_reports/figure_2_structural_determinants.png
outputs/scientific_reports/figure_2_structural_determinants.eps
outputs/scientific_reports/scientific_reports_group_summary.csv
outputs/scientific_reports/scientific_reports_controlled_cells.csv
outputs/scientific_reports/scientific_reports_statistics.json
```

The script validates the expected 9,600-row dataset before computing balanced
marginal means, controlled-cell summaries, degree-heterogeneity measures, and
two-sided 95% normal confidence intervals.

## Final Statistical and Exhaustive Checks

The submission-stage analysis uses the frozen replay table only. It adds
10,000-resample percentile-bootstrap confidence intervals, publication Figure
2, and the two supplementary tables:

```powershell
python scientific_reports_final_analysis.py `
  --input outputs/data/relaxation_replay.csv `
  --output-dir outputs/scientific_reports_final
```

The bootstrap uses seed `20260827`. It does not rerun, append, or alter any
stochastic trajectory. The output directory contains `Figure_2.png`,
`Figure_2.eps`, `Supplementary_Table_S1.csv`,
`Supplementary_Table_S2.csv`, and `bootstrap_summary.json`.

The theorem inequality and equilibrium-energy ceiling can also be checked by
exhaustive enumeration of all colorings with two or three colors on every
connected non-isomorphic simple graph with two to five vertices:

```powershell
python exhaustive_small_graph_check.py `
  --output outputs/scientific_reports_final/exhaustive_check.json
```

The archived release associated with the manuscript is available at
https://doi.org/10.5281/zenodo.22116965.

## Novelty and Acceptance Upgrade

The upgraded manuscript uses one bootstrap convention for both main figures.
The following command regenerates Figure 1, Figure 2, and Supplementary Tables
S1-S2 from the unchanged frozen replay data:

```powershell
python scientific_reports_upgrade_analysis.py `
  --input outputs/data/relaxation_replay.csv `
  --output-dir outputs/scientific_reports_upgrade
```

The bootstrap uses 10,000 percentile resamples and seed `20260827`. Figure 1a
is unchanged apart from being re-rendered alongside the new Figure 1b bootstrap
intervals.

The heterogeneous-activation and finite-confidence corollaries are checked on
small graphs with:

```powershell
python exhaustive_scheduler_check.py
```

The checker uses every coloring with `k = 2, 3` on all connected
non-isomorphic simple graphs with two to five vertices. It tests uniform,
degree-biased, inverse-degree-biased, and index-weighted positive activation
vectors, as well as three geometric-tail allocations.

## Tests

```powershell
python -m unittest discover -s tests
```

The tests check deterministic seeding, strict Lyapunov decrease, fixed points,
the accepted-move edge bound, and the degree-corrected threshold diagnostics.
