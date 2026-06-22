"""Competitive graph recoloring simulations for AML-style experiments."""

from .graphs import make_graph
from .simulation import (
    RecoloringResult,
    best_response,
    count_conflicts,
    global_potential,
    is_fixed_point,
    random_coloring,
    run_competitive_recoloring,
)

__all__ = [
    "RecoloringResult",
    "best_response",
    "count_conflicts",
    "global_potential",
    "is_fixed_point",
    "make_graph",
    "random_coloring",
    "run_competitive_recoloring",
]

