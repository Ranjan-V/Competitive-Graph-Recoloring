"""Exhaustive implementation-level check of the frozen structural inequalities."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import networkx as nx


def verify_graph(graph: nx.Graph, k: int) -> tuple[int, int]:
    nodes = list(graph.nodes())
    degrees = dict(graph.degree())
    b = {i: degrees[i] // k for i in nodes}
    w = {i: degrees[i] - b[i] for i in nodes}
    s = sum(b.values())
    h = s // 2
    configurations = 0
    equilibria = 0
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
        if not improving:
            equilibria += 1
            if phi > h:
                raise AssertionError(
                    f"Equilibrium ceiling violation: n={len(nodes)}, k={k}, "
                    f"phi={phi}, H={h}, edges={sorted(graph.edges())}, colors={assignment}"
                )
        configurations += 1
    return configurations, equilibria


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
    by_n: dict[str, int] = {}
    for graph in graphs:
        by_n[str(graph.number_of_nodes())] = by_n.get(str(graph.number_of_nodes()), 0) + 1
        for k in (2, 3):
            checked, eq = verify_graph(graph, k)
            configurations += checked
            equilibria += eq
            cases += 1
    result = {
        "graph_source": "NetworkX graph_atlas_g (non-isomorphic simple graphs)",
        "connected_graphs": len(graphs),
        "graphs_by_n": by_n,
        "palette_sizes": [2, 3],
        "graph_palette_cases": cases,
        "configurations_checked": configurations,
        "equilibrium_configurations_checked": equilibria,
        "violations": 0,
    }
    output = Path("outputs/scientific_reports_final/exhaustive_check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
