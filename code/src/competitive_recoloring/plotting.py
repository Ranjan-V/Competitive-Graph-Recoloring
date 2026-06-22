"""Plot utilities for the recoloring experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiments import ExperimentRecord


def plot_potential_history(
    potential_history: Iterable[int],
    path: str | Path,
    *,
    title: str = "Lyapunov Potential vs. Accepted Recoloring Steps",
) -> Path:
    """Plot the monotone Lyapunov potential history for a single run."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = list(potential_history)
    x_values = list(range(len(history)))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.step(x_values, history, where="post", color="#1f77b4", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Accepted recoloring steps")
    ax.set_ylabel("Conflicting edges $\\Phi$")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_scaling_summary(
    records: Iterable[ExperimentRecord],
    path: str | Path,
    *,
    show_n_log_n_guide: bool = True,
) -> Path:
    """Plot accepted recolorings against the deterministic |E| bound."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        raise ValueError("No experiment records to plot.")

    grouped: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)

    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    palette = {
        "er": "#1f77b4",
        "ba": "#d62728",
    }

    max_axis = max(max(record.m, record.recolor_steps) for record in records)
    bound_x = [0, max_axis * 1.05]
    ax.plot(bound_x, bound_x, color="#222222", linestyle="--", linewidth=1.6, label="$y=|E|$")

    for family, family_records in sorted(grouped.items()):
        color = palette.get(family, None)
        ax.scatter(
            [record.m for record in family_records],
            [record.recolor_steps for record in family_records],
            s=18,
            alpha=0.25,
            color=color,
            edgecolors="none",
            label=f"{family.upper()} trials",
        )

        by_n: dict[int, list[ExperimentRecord]] = defaultdict(list)
        for record in family_records:
            by_n[record.n].append(record)
        summary = [
            (
                n,
                mean(record.m for record in group),
                mean(record.recolor_steps for record in group),
                mean(record.n_log_n for record in group),
            )
            for n, group in sorted(by_n.items())
        ]
        ax.plot(
            [item[1] for item in summary],
            [item[2] for item in summary],
            marker="o",
            linewidth=2,
            color=color,
            label=f"{family.upper()} mean",
        )

        if show_n_log_n_guide and len(summary) >= 2:
            numerator = sum(item[2] * item[3] for item in summary)
            denominator = sum(item[3] ** 2 for item in summary)
            scale = numerator / denominator if denominator else 0.0
            ax.plot(
                [item[1] for item in summary],
                [scale * item[3] for item in summary],
                linestyle=":",
                linewidth=1.8,
                color=color,
                label=f"{family.upper()} scaled $n\\log n$ guide",
            )

    ax.set_title("Competitive Recoloring Convergence")
    ax.set_xlabel("Number of edges $|E|$")
    ax.set_ylabel("Accepted recoloring steps")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
