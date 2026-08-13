"""Deterministic graph generators used in the experiments."""

from __future__ import annotations

import networkx as nx


SUPPORTED_FAMILIES = ("er", "ba", "ws")


def make_graph(
    family: str,
    *,
    n: int,
    average_degree: float,
    seed: int,
) -> nx.Graph:
    """Create a standard sparse graph family with reproducible seeds."""

    if n < 1:
        raise ValueError("n must be positive.")
    if average_degree <= 0:
        raise ValueError("average_degree must be positive.")

    family = family.lower()
    if family == "er":
        p = min(1.0, average_degree / max(n - 1, 1))
        return nx.erdos_renyi_graph(n=n, p=p, seed=seed)

    if family == "ba":
        if n < 2:
            return nx.empty_graph(n)
        attachment_edges = max(1, min(n - 1, round(average_degree / 2)))
        return nx.barabasi_albert_graph(n=n, m=attachment_edges, seed=seed)

    if family == "ws":
        if n < 3:
            return nx.empty_graph(n)
        neighbor_degree = int(2 * round(average_degree / 2))
        neighbor_degree = max(2, min(neighbor_degree, n - 1))
        if neighbor_degree % 2:
            neighbor_degree -= 1
        return nx.watts_strogatz_graph(
            n=n, k=neighbor_degree, p=0.1, seed=seed
        )

    supported = ", ".join(SUPPORTED_FAMILIES)
    raise ValueError(f"Unsupported graph family {family!r}. Use one of: {supported}.")
