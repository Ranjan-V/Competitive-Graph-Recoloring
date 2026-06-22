"""Experiment orchestration and CSV export."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .graphs import make_graph
from .simulation import run_competitive_recoloring


@dataclass(frozen=True)
class ExperimentRecord:
    family: str
    n: int
    m: int
    k_colors: int
    average_degree: float
    trial: int
    seed: int
    recolor_steps: int
    sweeps: int
    node_evaluations: int
    initial_potential: int
    final_potential: int
    converged: bool
    edge_bound: int
    n_log_n: float

    @property
    def steps_per_edge(self) -> float:
        return self.recolor_steps / self.edge_bound if self.edge_bound else 0.0


def _trial_seed(master_seed: int, family: str, n: int, trial: int) -> int:
    family_code = sum((i + 1) * ord(char) for i, char in enumerate(family))
    sequence = np.random.SeedSequence([master_seed, family_code, n, trial])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def run_scaling_experiment(
    *,
    families: Sequence[str],
    n_values: Sequence[int],
    trials: int,
    k_colors: int,
    average_degree: float,
    seed: int,
    verify: bool = False,
) -> list[ExperimentRecord]:
    """Run deterministic scaling experiments over graph families and sizes."""

    if trials < 1:
        raise ValueError("trials must be at least 1.")

    records: list[ExperimentRecord] = []
    for family in families:
        for n in n_values:
            for trial in range(trials):
                trial_seed = _trial_seed(seed, family, n, trial)
                graph = make_graph(
                    family,
                    n=n,
                    average_degree=average_degree,
                    seed=trial_seed,
                )
                result, _ = run_competitive_recoloring(
                    graph,
                    k_colors,
                    seed=trial_seed + 1,
                    verify=verify,
                )
                records.append(
                    ExperimentRecord(
                        family=family,
                        n=n,
                        m=result.m,
                        k_colors=k_colors,
                        average_degree=average_degree,
                        trial=trial,
                        seed=trial_seed,
                        recolor_steps=result.recolor_steps,
                        sweeps=result.sweeps,
                        node_evaluations=result.node_evaluations,
                        initial_potential=result.initial_potential,
                        final_potential=result.final_potential,
                        converged=result.converged,
                        edge_bound=result.m,
                        n_log_n=n * math.log(max(n, 2)),
                    )
                )

    return records


def write_records_csv(records: Iterable[ExperimentRecord], path: str | Path) -> Path:
    """Write experiment records to CSV and return the resolved output path."""

    records = list(records)
    if not records:
        raise ValueError("No experiment records to write.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return output_path

