"""Competitive graph recoloring simulations for network-dynamics experiments."""

from .graphs import make_graph
from .simulation import (
    RecoloringResult,
    best_response,
    count_conflicts,
    global_potential,
    is_fixed_point,
    random_coloring,
    relaxation_parameters,
    run_competitive_recoloring,
    run_uniform_activation_recoloring,
)

__all__ = [
    "RecoloringResult",
    "best_response",
    "count_conflicts",
    "global_potential",
    "is_fixed_point",
    "make_graph",
    "random_coloring",
    "relaxation_parameters",
    "run_competitive_recoloring",
    "run_uniform_activation_recoloring",
]
