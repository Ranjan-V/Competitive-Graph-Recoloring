# Competitive Graph Recoloring Code

This folder contains a reproducible Python package for the computational part of the AML letter. It simulates strict best-response recoloring on sparse graph families and records the Lyapunov potential

```text
Phi = number of monochromatic edges.
```

The bound `recolor_steps <= |E|` applies to accepted recoloring moves. The code also records sweeps and node evaluations separately, because those are scheduler costs rather than Lyapunov-decreasing moves.

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

The plot overlays the deterministic edge bound `y = |E|`, trial scatter points, family means, and a scaled `n log n` visual guide for sparse graphs.

## Tests

```powershell
python -m unittest discover -s tests
```

The tests check deterministic seeding, strict Lyapunov decrease, the fixed-point condition, and the accepted-move edge bound.

