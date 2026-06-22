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
)


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


if __name__ == "__main__":
    unittest.main()

