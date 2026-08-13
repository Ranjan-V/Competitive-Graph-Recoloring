"""Command-line interface for reproducible recoloring experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import run_parameter_experiment, run_scaling_experiment, write_records_csv
from .graphs import SUPPORTED_FAMILIES, make_graph
from .plotting import plot_parameter_summary, plot_potential_history, plot_scaling_summary
from .simulation import is_fixed_point, run_competitive_recoloring


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="competitive-recoloring",
        description="Simulate competitive graph recoloring and export AML-ready figures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run one graph and plot Phi(t).")
    demo.add_argument("--family", choices=SUPPORTED_FAMILIES, default="er")
    demo.add_argument("--n", type=int, default=500)
    demo.add_argument("--avg-degree", type=float, default=8.0)
    demo.add_argument("--k", type=int, default=3, dest="k_colors")
    demo.add_argument("--seed", type=int, default=2026)
    demo.add_argument("--outdir", type=Path, default=Path("outputs"))
    demo.add_argument("--verify", action="store_true", help="Recompute Phi after every accepted move.")

    experiment = subparsers.add_parser(
        "experiment",
        help="Run scaling experiments and plot steps versus |E|.",
    )
    experiment.add_argument("--families", nargs="+", choices=SUPPORTED_FAMILIES, default=["er", "ba"])
    experiment.add_argument("--n-values", nargs="+", type=int, default=[100, 200, 400, 800])
    experiment.add_argument("--trials", type=int, default=100)
    experiment.add_argument("--avg-degree", type=float, default=8.0)
    experiment.add_argument("--k", type=int, default=3, dest="k_colors")
    experiment.add_argument("--seed", type=int, default=2026)
    experiment.add_argument("--outdir", type=Path, default=Path("outputs"))
    experiment.add_argument("--verify", action="store_true", help="Recompute Phi after every accepted move.")

    grid = subparsers.add_parser("grid", help="Run the theorem-aligned parameter grid.")
    grid.add_argument("--families", nargs="+", choices=SUPPORTED_FAMILIES, default=list(SUPPORTED_FAMILIES))
    grid.add_argument("--n-values", nargs="+", type=int, default=[100, 200, 400, 800])
    grid.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5, 8])
    grid.add_argument("--avg-degrees", nargs="+", type=float, default=[4, 8, 12, 16])
    grid.add_argument("--trials", type=int, default=50)
    grid.add_argument("--seed", type=int, default=2026)
    grid.add_argument("--outdir", type=Path, default=Path("outputs"))
    grid.add_argument("--verify", action="store_true")

    return parser


def _run_demo(args: argparse.Namespace) -> int:
    graph = make_graph(
        args.family,
        n=args.n,
        average_degree=args.avg_degree,
        seed=args.seed,
    )
    result, colors = run_competitive_recoloring(
        graph,
        args.k_colors,
        seed=args.seed + 1,
        verify=args.verify,
    )
    figure_path = plot_potential_history(
        result.potential_history,
        args.outdir / "figures" / f"potential_{args.family}_n{args.n}_seed{args.seed}.png",
    )
    summary = {
        "family": args.family,
        "n": result.n,
        "m": result.m,
        "k_colors": result.k_colors,
        "seed": args.seed,
        "initial_phi": result.initial_potential,
        "final_phi": result.final_potential,
        "accepted_recolor_steps": result.recolor_steps,
        "edge_bound": result.m,
        "sweeps": result.sweeps,
        "node_evaluations": result.node_evaluations,
        "converged": result.converged,
        "fixed_point_verified": is_fixed_point(graph, colors, args.k_colors),
        "figure": str(figure_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    records = run_scaling_experiment(
        families=args.families,
        n_values=args.n_values,
        trials=args.trials,
        k_colors=args.k_colors,
        average_degree=args.avg_degree,
        seed=args.seed,
        verify=args.verify,
    )
    csv_path = write_records_csv(
        records,
        args.outdir / "data" / "scaling_results.csv",
    )
    figure_path = plot_scaling_summary(
        records,
        args.outdir / "figures" / "steps_vs_edges.png",
    )
    failures = [record for record in records if not record.converged]
    max_ratio = max(record.steps_per_edge for record in records)
    summary = {
        "runs": len(records),
        "families": args.families,
        "n_values": args.n_values,
        "trials": args.trials,
        "k_colors": args.k_colors,
        "average_degree": args.avg_degree,
        "seed": args.seed,
        "nonconverged_runs": len(failures),
        "max_steps_over_edges": max_ratio,
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


def _run_grid(args: argparse.Namespace) -> int:
    records = run_parameter_experiment(
        families=args.families,
        n_values=args.n_values,
        k_values=args.k_values,
        average_degrees=args.avg_degrees,
        trials=args.trials,
        seed=args.seed,
        verify=args.verify,
    )
    csv_path = write_records_csv(records, args.outdir / "data" / "parameter_results.csv")
    figure_path = plot_parameter_summary(records, args.outdir / "figures" / "parameter_summary.png")
    summary = {
        "runs": len(records),
        "nonconverged_runs": sum(not record.converged for record in records),
        "mean_T_over_m": sum(record.steps_per_edge for record in records) / len(records),
        "mean_T_over_phi0": sum(record.steps_per_initial_potential for record in records) / len(records),
        "mean_phi0_over_m": sum(record.initial_potential_per_edge for record in records) / len(records),
        "mean_A_over_nm": sum(record.activations_per_nm for record in records) / len(records),
        "max_T_over_m": max(record.steps_per_edge for record in records),
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    print(json.dumps(summary, indent=2))
    return 1 if summary["nonconverged_runs"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        return _run_demo(args)
    if args.command == "experiment":
        return _run_experiment(args)
    if args.command == "grid":
        return _run_grid(args)

    parser.error(f"Unknown command {args.command!r}.")
    return 2
