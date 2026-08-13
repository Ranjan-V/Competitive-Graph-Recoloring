"""Replay the frozen parameter grid and record Theorem 1 diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from competitive_recoloring.graphs import make_graph
from competitive_recoloring.simulation import run_uniform_activation_recoloring


NEW_FIELDS = (
    "max_degree",
    "threshold_h",
    "weight_w",
    "excess_x0",
    "parity_epsilon",
    "threshold_activation_count",
    "relaxation_bound",
    "high_palette",
)


def _integer(row: dict[str, str], name: str) -> int:
    return int(row[name])


def replay(input_path: Path, output_path: Path) -> dict[str, int]:
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Baseline CSV has no header.")
        fieldnames = reader.fieldnames
        rows = list(reader)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    high_palette = 0

    with temporary_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=[*fieldnames, *NEW_FIELDS])
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            graph = make_graph(
                row["family"],
                n=_integer(row, "n"),
                average_degree=float(row["average_degree"]),
                seed=_integer(row, "seed"),
            )
            result, _ = run_uniform_activation_recoloring(
                graph,
                _integer(row, "k_colors"),
                seed=_integer(row, "seed") + 1,
            )

            reproduced = {
                "n": result.n,
                "m": result.m,
                "k_colors": result.k_colors,
                "recolor_steps": result.recolor_steps,
                "node_evaluations": result.node_evaluations,
                "activation_count": result.activation_count,
                "initial_potential": result.initial_potential,
                "final_potential": result.final_potential,
                "converged": str(result.converged),
                "edge_bound": result.m,
                "degree_square_sum": sum(degree * degree for _, degree in graph.degree()),
            }
            mismatches = {
                name: (row[name], str(value))
                for name, value in reproduced.items()
                if row[name] != str(value)
            }
            if mismatches:
                raise RuntimeError(f"Replay mismatch at row {index}: {mismatches}")
            if result.threshold_activation_count is None:
                raise RuntimeError(f"Threshold was not reached at row {index}.")

            is_high_palette = result.k_colors > result.max_degree
            high_palette += int(is_high_palette)
            writer.writerow(
                {
                    **row,
                    "max_degree": result.max_degree,
                    "threshold_h": result.threshold_h,
                    "weight_w": result.weight_w,
                    "excess_x0": result.excess_x0,
                    "parity_epsilon": result.parity_epsilon,
                    "threshold_activation_count": result.threshold_activation_count,
                    "relaxation_bound": f"{result.relaxation_bound:.12g}",
                    "high_palette": is_high_palette,
                }
            )

            if index % 400 == 0:
                print(f"Verified {index}/{len(rows)} rows", flush=True)

    temporary_path.replace(output_path)
    return {"rows": len(rows), "high_palette": high_palette}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(replay(args.input, args.output))


if __name__ == "__main__":
    main()
