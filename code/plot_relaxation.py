"""Create the composite relaxation figure from deterministic replay data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"er": "#176B87", "ba": "#C44E52", "ws": "#4C8C4A"}
LABELS = {"er": "ER", "ba": "BA", "ws": "WS"}


def ci95(values: pd.Series) -> float:
    return 1.96 * values.std(ddof=1) / np.sqrt(values.count())


def create_figure(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path)
    data["normalized_relaxation"] = np.where(
        data["relaxation_bound"] > 0,
        data["threshold_activation_count"] / data["relaxation_bound"],
        0.0,
    )
    cells = (
        data.groupby(["family", "n", "k_colors", "average_degree"], as_index=False)
        .agg(
            mean_tau=("threshold_activation_count", "mean"),
            mean_bound=("relaxation_bound", "mean"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    ax = axes[0]
    for family in ("er", "ba", "ws"):
        group = cells[cells["family"] == family]
        ax.scatter(
            group["mean_bound"],
            group["mean_tau"],
            s=20,
            alpha=0.72,
            color=COLORS[family],
            edgecolors="white",
            linewidths=0.35,
            label=LABELS[family],
        )
    positive = cells[(cells["mean_bound"] > 0) & (cells["mean_tau"] > 0)]
    lower = min(positive["mean_bound"].min(), positive["mean_tau"].min())
    upper = max(positive["mean_bound"].max(), positive["mean_tau"].max())
    ax.plot([lower, upper], [lower, upper], color="#333333", lw=1, ls="--", label="Equality")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mean certificate $\overline{B}_H$")
    ax.set_ylabel(r"Mean hitting time $\overline{\tau}_H$")
    ax.set_title("(a) Controlled 40-run cells")
    ax.grid(True, which="both", color="#D9D9D9", lw=0.45)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    summary = (
        data.groupby(["family", "k_colors"])["normalized_relaxation"]
        .agg(["mean", ci95])
        .reset_index()
    )
    for family in ("er", "ba", "ws"):
        group = summary[summary["family"] == family]
        ax.errorbar(
            group["k_colors"],
            group["mean"],
            yerr=group["ci95"],
            marker="o",
            markersize=4,
            lw=1.25,
            capsize=2,
            color=COLORS[family],
            label=LABELS[family],
        )
    ax.axhline(1.0, color="#333333", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.2)
    ax.set_xticks(sorted(data["k_colors"].unique()))
    ax.set_xlabel("Number of colors, $k$")
    ax.set_ylabel(r"Mean normalized time $\tau_H/B_H$")
    ax.set_title("(b) Palette-size dependence")
    ax.grid(True, color="#D9D9D9", lw=0.45)
    ax.legend(frameon=False, fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_figure(args.input, args.output)


if __name__ == "__main__":
    main()
