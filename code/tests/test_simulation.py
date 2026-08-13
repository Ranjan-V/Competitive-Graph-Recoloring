from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from competitive_recoloring import (  # noqa: E402
    global_potential,
    is_fixed_point,
    make_graph,
    run_competitive_recoloring,
    run_uniform_activation_recoloring,
)
from competitive_recoloring.simulation import relaxation_parameters  # noqa: E402


class CompetitiveRecoloringTests(unittest.TestCase):
    def test_potential_decreases_and_edge_bound_holds(self) -> None:
        graph = make_graph("er", n=80, average_degree=6, seed=11)
        result, colors = run_competitive_recoloring(
            graph,
            3,
            seed=12,
            verify=True,
        )

        history = result.potential_history
        self.assertTrue(result.converged)
        self.assertTrue(is_fixed_point(graph, colors, 3))
        self.assertLessEqual(result.recolor_steps, result.initial_potential)
        self.assertLessEqual(result.recolor_steps, result.m)
        self.assertEqual(result.final_potential, global_potential(graph, colors))
        self.assertTrue(all(later < earlier for earlier, later in zip(history, history[1:])))

    def test_same_seed_is_reproducible(self) -> None:
        graph = make_graph("ba", n=60, average_degree=4, seed=21)
        first, _ = run_competitive_recoloring(graph, 4, seed=22)
        second, _ = run_competitive_recoloring(graph, 4, seed=22)

        self.assertEqual(first.recolor_steps, second.recolor_steps)
        self.assertEqual(first.potential_history, second.potential_history)

    def test_edgeless_graph_converges_immediately(self) -> None:
        graph = nx.empty_graph(10)
        result, colors = run_competitive_recoloring(graph, 3, seed=5, verify=True)

        self.assertTrue(result.converged)
        self.assertEqual(result.recolor_steps, 0)
        self.assertEqual(result.initial_potential, 0)
        self.assertTrue(is_fixed_point(graph, colors, 3))

    def test_uniform_activation_scheduler(self) -> None:
        graph = make_graph("ws", n=60, average_degree=8, seed=31)
        result, colors = run_uniform_activation_recoloring(
            graph, 3, seed=32, verify=True
        )

        self.assertTrue(is_fixed_point(graph, colors, 3))
        self.assertLessEqual(result.recolor_steps, result.initial_potential)
        self.assertGreaterEqual(result.activation_count, result.recolor_steps)
        self.assertEqual(result.node_evaluations, result.activation_count)
        self.assertIsNotNone(result.threshold_activation_count)
        self.assertLessEqual(result.threshold_activation_count, result.activation_count)

    def test_degree_corrected_relaxation_parameters(self) -> None:
        graph = nx.path_graph(4)
        delta, threshold, weight, excess, parity, bound = relaxation_parameters(
            graph, 2, initial_potential=3
        )

        self.assertEqual((delta, threshold, weight, excess, parity), (2, 1, 1, 2, 0))
        self.assertAlmostEqual(bound, 3.0)

    def test_high_palette_threshold_is_absorption(self) -> None:
        graph = nx.path_graph(8)
        result, colors = run_uniform_activation_recoloring(
            graph,
            3,
            seed=77,
            initial_colors={node: 0 for node in graph.nodes()},
            verify=True,
        )

        self.assertEqual(result.threshold_h, 0)
        self.assertEqual(result.weight_w, 2)
        self.assertEqual(result.threshold_activation_count, result.activation_count)
        self.assertEqual(result.final_potential, 0)
        self.assertTrue(is_fixed_point(graph, colors, 3))

    def test_all_graph_families_are_reproducible(self) -> None:
        for family in ("er", "ba", "ws"):
            first = make_graph(family, n=50, average_degree=6, seed=41)
            second = make_graph(family, n=50, average_degree=6, seed=41)
            self.assertEqual(set(first.edges()), set(second.edges()))


if __name__ == "__main__":
    unittest.main()
