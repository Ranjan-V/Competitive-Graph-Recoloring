"""Plot utilities for the recoloring experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    show_n_log_n_guide: bool = False,
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

    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
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
                0.0,
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
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_parameter_summary(
    records: Iterable[ExperimentRecord], path: str | Path
) -> Path:
    """Create a compact three-panel summary with 95% confidence intervals."""

    records = list(records)
    if not records:
        raise ValueError("No experiment records to plot.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = {"er": "#1769aa", "ba": "#c43c39", "ws": "#238b57"}

    def summarize(groups, value):
        output = []
        for x, group in sorted(groups.items()):
            values = np.asarray([value(item) for item in group], dtype=float)
            ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            output.append((x, float(values.mean()), float(ci)))
        return output

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35), constrained_layout=True)
    for family in ("er", "ba", "ws"):
        family_records = [r for r in records if r.family == family]

        by_k = defaultdict(list)
        for record in family_records:
            by_k[record.k_colors].append(record)
        values = summarize(by_k, lambda r: r.steps_per_edge)
        axes[0].errorbar(
            [x for x, _, _ in values], [y for _, y, _ in values],
            yerr=[ci for _, _, ci in values], marker="o", ms=3,
            lw=1.1, capsize=2, color=palette[family], label=family.upper(),
        )

        baseline = [r for r in family_records if r.k_colors == 3 and r.average_degree == 8]
        by_n = defaultdict(list)
        for record in baseline:
            by_n[record.n].append(record)
        values = summarize(by_n, lambda r: r.steps_per_edge)
        axes[1].errorbar(
            [x for x, _, _ in values], [y for _, y, _ in values],
            yerr=[ci for _, _, ci in values], marker="o", ms=3,
            lw=1.1, capsize=2, color=palette[family], label=family.upper(),
        )

        density = [r for r in family_records if r.k_colors == 3]
        by_degree = defaultdict(list)
        for record in density:
            by_degree[record.average_degree].append(record)
        values = summarize(by_degree, lambda r: r.steps_per_edge)
        axes[2].errorbar(
            [x for x, _, _ in values], [y for _, y, _ in values],
            yerr=[ci for _, _, ci in values], marker="o", ms=3,
            lw=1.1, capsize=2, color=palette[family], label=family.upper(),
        )

    k_values = sorted({record.k_colors for record in records})
    axes[0].plot(
        k_values, [1 / k for k in k_values], color="#333333", ls="--",
        lw=1.0, label="bound $1/k$",
    )

    axes[0].set_xlabel("Number of colors $k$")
    axes[1].set_xlabel(r"Vertices $n$ ($k=3$, $\bar d=8$)")
    axes[2].set_xlabel(r"Target mean degree $\bar d$ ($k=3$)")
    axes[0].set_ylabel("Mean accepted moves $T/|E|$")
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(0.03, 0.94, label, transform=ax.transAxes, va="top", fontweight="bold")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
