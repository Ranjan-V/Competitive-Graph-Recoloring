"""Exhaustive implementation-level check of the frozen structural inequalities."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np


def scheduler_vectors(nodes: list[int], degrees: dict[int, int]) -> dict[str, np.ndarray]:
    raw = {
        "uniform": np.ones(len(nodes), dtype=float),
        "degree_plus_one": np.array([degrees[i] + 1 for i in nodes], dtype=float),
        "inverse_degree_plus_one": np.array(
            [1.0 / (degrees[i] + 1) for i in nodes], dtype=float
        ),
        "index_plus_one": np.arange(1, len(nodes) + 1, dtype=float),
    }
    return {name: values / values.sum() for name, values in raw.items()}


def verify_graph(graph: nx.Graph, k: int) -> tuple[int, int, int, int]:
    nodes = list(graph.nodes())
    degrees = dict(graph.degree())
    b = {i: degrees[i] // k for i in nodes}
    w = {i: degrees[i] - b[i] for i in nodes}
    s = sum(b.values())
    h = s // 2
    configurations = 0
    equilibria = 0
    scheduler_checks = 0
    tail_checks = 0
    probabilities = scheduler_vectors(nodes, degrees)
    for assignment in itertools.product(range(k), repeat=len(nodes)):
        colors = dict(zip(nodes, assignment))
        phi = sum(colors[u] == colors[v] for u, v in graph.edges())
        improving = set()
        for i in nodes:
            counts = [0] * k
            for j in graph.neighbors(i):
                counts[colors[j]] += 1
            if min(counts) < counts[colors[i]]:
                improving.add(i)
        lhs = sum(w[i] for i in improving)
        rhs = 2 * phi - s
        if lhs < rhs:
            raise AssertionError(
                f"Instability violation: n={len(nodes)}, k={k}, phi={phi}, "
                f"lhs={lhs}, rhs={rhs}, edges={sorted(graph.edges())}, colors={assignment}"
            )
        if phi > h:
            r = phi - h
            epsilon = s - 2 * h
            level_excess = 2 * r - epsilon
            if level_excess != rhs:
                raise AssertionError("Excess-level identity failed")
            for name, probability in probabilities.items():
                p = dict(zip(nodes, probability))
                nonisolated = [i for i in nodes if degrees[i] > 0]
                lam = min(p[i] / w[i] for i in nonisolated)
                improving_probability = sum(p[i] for i in improving)
                q = lam * level_excess
                if improving_probability + 1e-14 < q:
                    raise AssertionError(
                        f"Scheduler bound violation: scheduler={name}, n={len(nodes)}, "
                        f"k={k}, improve_p={improving_probability}, q={q}"
                    )
                if not (0.0 < q <= 1.0 + 1e-14):
                    raise AssertionError(f"Invalid success lower bound q={q}")
                scheduler_checks += 1
                x0 = phi - h
                for delta in (0.1, 0.05, 0.01):
                    t = math.ceil(math.log(x0 / delta) / q)
                    if math.exp(-q * t) > delta / x0 + 1e-14:
                        raise AssertionError("Geometric tail allocation failed")
                    tail_checks += 1
        if not improving:
            equilibria += 1
            if phi > h:
                raise AssertionError(
                    f"Equilibrium ceiling violation: n={len(nodes)}, k={k}, "
                    f"phi={phi}, H={h}, edges={sorted(graph.edges())}, colors={assignment}"
                )
        configurations += 1
    return configurations, equilibria, scheduler_checks, tail_checks


def main() -> None:
    graphs = [
        graph
        for graph in nx.graph_atlas_g()
        if 2 <= graph.number_of_nodes() <= 5
        and graph.number_of_edges() > 0
        and nx.is_connected(graph)
    ]
    configurations = 0
    equilibria = 0
    cases = 0
    scheduler_checks = 0
    tail_checks = 0
    by_n: dict[str, int] = {}
    for graph in graphs:
        by_n[str(graph.number_of_nodes())] = by_n.get(str(graph.number_of_nodes()), 0) + 1
        for k in (2, 3):
            checked, eq, scheduler_count, tail_count = verify_graph(graph, k)
            configurations += checked
            equilibria += eq
            scheduler_checks += scheduler_count
            tail_checks += tail_count
            cases += 1
    result = {
        "graph_source": "NetworkX graph_atlas_g (non-isomorphic simple graphs)",
        "connected_graphs": len(graphs),
        "graphs_by_n": by_n,
        "palette_sizes": [2, 3],
        "graph_palette_cases": cases,
        "configurations_checked": configurations,
        "equilibrium_configurations_checked": equilibria,
        "scheduler_distributions": [
            "uniform", "degree_plus_one", "inverse_degree_plus_one", "index_plus_one"
        ],
        "scheduler_probability_checks": scheduler_checks,
        "geometric_tail_allocation_checks": tail_checks,
        "violations": 0,
    }
    output = Path("outputs/scientific_reports_upgrade/exhaustive_scheduler_check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
