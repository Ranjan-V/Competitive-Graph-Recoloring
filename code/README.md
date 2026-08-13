# Degree-Corrected Energy Relaxation Bounds for Competitive Recoloring on Complex Networks

This folder contains the reproducible Python package for the computational part of the CS&F Short Communication. It simulates strict best-response recoloring on sparse graph families and records the Lyapunov potential

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

## Tests

```powershell
python -m unittest discover -s tests
```

The tests check deterministic seeding, strict Lyapunov decrease, fixed points,
the accepted-move edge bound, and the degree-corrected threshold diagnostics.
