"""Core asynchronous competitive recoloring dynamics.

The implementation records accepted recoloring moves. This is the quantity
controlled by the Lyapunov argument: every accepted move strictly decreases the
number of monochromatic edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Hashable, Mapping

import networkx as nx
import numpy as np

Coloring = dict[Hashable, int]


@dataclass(frozen=True)
class RecoloringResult:
    """Summary of one competitive recoloring run."""

    n: int
    m: int
    k_colors: int
    seed: int | None
    recolor_steps: int
    sweeps: int
    node_evaluations: int
    activation_count: int
    initial_potential: int
    final_potential: int
    potential_history: tuple[int, ...]
    converged: bool
    max_degree: int = 0
    threshold_h: int = 0
    weight_w: int = 0
    excess_x0: int = 0
    parity_epsilon: int = 0
    threshold_activation_count: int | None = None
    relaxation_bound: float = 0.0

    @property
    def edge_bound_gap(self) -> int:
        """Slack in the deterministic bound recolor_steps <= |E|."""

        return self.m - self.recolor_steps


def _validate_color_count(k_colors: int) -> None:
    if k_colors < 1:
        raise ValueError("k_colors must be at least 1.")


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_coloring(
    graph: nx.Graph,
    k_colors: int,
    seed: int | None = None,
) -> Coloring:
    """Return a deterministic random coloring for a fixed graph and seed."""

    _validate_color_count(k_colors)
    generator = _rng(seed)
    return {
        node: int(generator.integers(0, k_colors))
        for node in graph.nodes()
    }


def global_potential(graph: nx.Graph, colors: Mapping[Hashable, int]) -> int:
    """Count monochromatic/conflicting edges."""

    return sum(1 for u, v in graph.edges() if colors[u] == colors[v])


def relaxation_parameters(
    graph: nx.Graph,
    k_colors: int,
    initial_potential: int,
) -> tuple[int, int, int, int, int, float]:
    """Return ``(Delta, H, W, X0, epsilon, B_H)`` for Theorem 1."""

    _validate_color_count(k_colors)
    degrees = [degree for _, degree in graph.degree()]
    max_degree = max(degrees, default=0)
    floors = [degree // k_colors for degree in degrees]
    floor_sum = sum(floors)
    threshold_h = floor_sum // 2
    weight_w = max(
        (degree - floor for degree, floor in zip(degrees, floors)),
        default=0,
    )
    excess_x0 = max(initial_potential - threshold_h, 0)
    parity_epsilon = floor_sum - 2 * threshold_h
    harmonic_factor = fsum(
        1.0 / (2 * level - parity_epsilon)
        for level in range(1, excess_x0 + 1)
    )
    relaxation_bound = graph.number_of_nodes() * weight_w * harmonic_factor
    return (
        max_degree,
        threshold_h,
        weight_w,
        excess_x0,
        parity_epsilon,
        relaxation_bound,
    )


def count_conflicts(
    graph: nx.Graph,
    colors: Mapping[Hashable, int],
    node: Hashable,
    color: int,
) -> int:
    """Count neighbors of ``node`` currently carrying ``color``."""

    return sum(1 for neighbor in graph.neighbors(node) if colors[neighbor] == color)


def _neighbor_color_counts(
    graph: nx.Graph,
    colors: Mapping[Hashable, int],
    node: Hashable,
    k_colors: int,
) -> np.ndarray:
    counts = np.zeros(k_colors, dtype=np.int64)
    for neighbor in graph.neighbors(node):
        color = colors[neighbor]
        if color < 0 or color >= k_colors:
            raise ValueError(f"Color {color} is outside [0, {k_colors - 1}].")
        counts[color] += 1
    return counts


def best_response(
    graph: nx.Graph,
    colors: Mapping[Hashable, int],
    node: Hashable,
    k_colors: int,
) -> tuple[int, int, int]:
    """Return ``(best_color, current_conflicts, best_conflicts)``.

    The current color is retained unless another color is strictly better. If
    several colors are strictly better, the lowest color index is selected; this
    tie-break keeps runs reproducible without changing the Lyapunov argument.
    """

    _validate_color_count(k_colors)
    current_color = colors[node]
    if current_color < 0 or current_color >= k_colors:
        raise ValueError(f"Color {current_color} is outside [0, {k_colors - 1}].")

    counts = _neighbor_color_counts(graph, colors, node, k_colors)
    current_conflicts = int(counts[current_color])
    best_color = current_color
    best_conflicts = current_conflicts

    for color, conflicts in enumerate(counts):
        conflicts = int(conflicts)
        if color != current_color and conflicts < best_conflicts:
            best_color = color
            best_conflicts = conflicts

    return best_color, current_conflicts, best_conflicts


def is_fixed_point(
    graph: nx.Graph,
    colors: Mapping[Hashable, int],
    k_colors: int,
) -> bool:
    """Return True when no node has a strictly improving recoloring move."""

    return all(
        best_response(graph, colors, node, k_colors)[0] == colors[node]
        for node in graph.nodes()
    )


def run_competitive_recoloring(
    graph: nx.Graph,
    k_colors: int,
    *,
    seed: int | None = None,
    initial_colors: Mapping[Hashable, int] | None = None,
    max_sweeps: int | None = None,
    verify: bool = False,
) -> tuple[RecoloringResult, Coloring]:
    """Run random-sweep asynchronous competitive recoloring.

    A sweep is one random permutation of the nodes. The process stops after a
    complete sweep with no accepted moves, which is equivalent to reaching a
    fixed point for this strict best-response rule.
    """

    _validate_color_count(k_colors)
    generator = _rng(seed)
    nodes = list(graph.nodes())
    colors = (
        dict(initial_colors)
        if initial_colors is not None
        else {node: int(generator.integers(0, k_colors)) for node in nodes}
    )

    if set(colors) != set(nodes):
        raise ValueError("initial_colors must define exactly one color per node.")

    phi = global_potential(graph, colors)
    initial_phi = phi
    potential_history = [phi]
    recolor_steps = 0
    node_evaluations = 0
    sweeps = 0
    converged = False

    while max_sweeps is None or sweeps < max_sweeps:
        sweeps += 1
        sweep_changes = 0

        for index in generator.permutation(len(nodes)):
            node = nodes[int(index)]
            best_color, old_conflicts, new_conflicts = best_response(
                graph,
                colors,
                node,
                k_colors,
            )
            node_evaluations += 1

            old_color = colors[node]
            if best_color == old_color:
                continue

            delta_phi = new_conflicts - old_conflicts
            if delta_phi >= 0:
                raise RuntimeError("Accepted recoloring did not reduce potential.")

            colors[node] = best_color
            phi += delta_phi
            recolor_steps += 1
            sweep_changes += 1
            potential_history.append(phi)

            if verify:
                exact_phi = global_potential(graph, colors)
                if exact_phi != phi:
                    raise RuntimeError(
                        f"Incremental potential {phi} disagrees with exact {exact_phi}."
                    )

        if sweep_changes == 0:
            converged = True
            break

    result = RecoloringResult(
        n=graph.number_of_nodes(),
        m=graph.number_of_edges(),
        k_colors=k_colors,
        seed=seed,
        recolor_steps=recolor_steps,
        sweeps=sweeps,
        node_evaluations=node_evaluations,
        activation_count=node_evaluations,
        initial_potential=initial_phi,
        final_potential=phi,
        potential_history=tuple(potential_history),
        converged=converged,
    )
    return result, colors


def run_uniform_activation_recoloring(
    graph: nx.Graph,
    k_colors: int,
    *,
    seed: int | None = None,
    initial_colors: Mapping[Hashable, int] | None = None,
    verify: bool = False,
) -> tuple[RecoloringResult, Coloring]:
    """Run independent uniform vertex activations until the first fixed point.

    Rejected activations are counted, while ``recolor_steps`` counts only
    accepted strict improvements. An incrementally maintained set of improving
    vertices permits exact fixed-point detection without an extra audit sweep.
    """

    _validate_color_count(k_colors)
    generator = _rng(seed)
    nodes = list(graph.nodes())
    colors = (
        dict(initial_colors)
        if initial_colors is not None
        else {node: int(generator.integers(0, k_colors)) for node in nodes}
    )
    if set(colors) != set(nodes):
        raise ValueError("initial_colors must define exactly one color per node.")

    phi = global_potential(graph, colors)
    initial_phi = phi
    (
        max_degree,
        threshold_h,
        weight_w,
        excess_x0,
        parity_epsilon,
        relaxation_bound,
    ) = relaxation_parameters(graph, k_colors, initial_phi)
    history = [phi]
    improving = {
        node
        for node in nodes
        if best_response(graph, colors, node, k_colors)[0] != colors[node]
    }
    steps = 0
    activations = 0
    threshold_activations = 0 if phi <= threshold_h else None

    while improving:
        node = nodes[int(generator.integers(0, len(nodes)))]
        activations += 1
        if node not in improving:
            continue

        best_color, old_conflicts, new_conflicts = best_response(
            graph, colors, node, k_colors
        )
        if best_color == colors[node]:
            improving.discard(node)
            continue

        delta_phi = new_conflicts - old_conflicts
        if delta_phi >= 0:
            raise RuntimeError("Accepted recoloring did not reduce potential.")
        colors[node] = best_color
        phi += delta_phi
        steps += 1
        history.append(phi)
        if threshold_activations is None and phi <= threshold_h:
            threshold_activations = activations

        affected = set(graph.neighbors(node))
        affected.add(node)
        for affected_node in affected:
            if best_response(graph, colors, affected_node, k_colors)[0] == colors[affected_node]:
                improving.discard(affected_node)
            else:
                improving.add(affected_node)

        if verify and global_potential(graph, colors) != phi:
            raise RuntimeError("Incremental potential disagrees with exact potential.")

    result = RecoloringResult(
        n=graph.number_of_nodes(),
        m=graph.number_of_edges(),
        k_colors=k_colors,
        seed=seed,
        recolor_steps=steps,
        sweeps=0,
        node_evaluations=activations,
        activation_count=activations,
        initial_potential=initial_phi,
        final_potential=phi,
        potential_history=tuple(history),
        converged=True,
        max_degree=max_degree,
        threshold_h=threshold_h,
        weight_w=weight_w,
        excess_x0=excess_x0,
        parity_epsilon=parity_epsilon,
        threshold_activation_count=threshold_activations,
        relaxation_bound=relaxation_bound,
    )
    return result, colors
