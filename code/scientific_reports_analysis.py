"""Create Scientific Reports summaries and Figure 2 from frozen replay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"er": "#176B87", "ba": "#C44E52", "ws": "#4C8C4A"}
LABELS = {"er": "ER", "ba": "BA", "ws": "WS"}
FAMILIES = ("er", "ba", "ws")


def ci95(values: pd.Series) -> float:
    """Return the two-sided normal 95% confidence-interval half-width."""

    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / np.sqrt(len(clean)))


def load_replay(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "family",
        "n",
        "m",
        "k_colors",
        "average_degree",
        "degree_square_sum",
        "max_degree",
        "threshold_h",
        "weight_w",
        "excess_x0",
        "threshold_activation_count",
        "relaxation_bound",
        "activation_count",
        "high_palette",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Replay data are missing columns: {sorted(missing)}")
    if len(data) != 9600:
        raise ValueError(f"Expected 9,600 frozen trajectories, found {len(data)}.")

    data = data.copy()
    data["realized_degree"] = 2.0 * data["m"] / data["n"]
    data["degree_variance"] = (
        data["degree_square_sum"] / data["n"] - data["realized_degree"] ** 2
    ).clip(lower=0.0)
    data["degree_cv2"] = np.where(
        data["realized_degree"] > 0,
        data["degree_variance"] / data["realized_degree"] ** 2,
        np.nan,
    )
    data["normalized_time"] = np.where(
        data["relaxation_bound"] > 0,
        data["threshold_activation_count"] / data["relaxation_bound"],
        0.0,
    )
    return data


def grouped_summary(data: pd.DataFrame, dimension: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (family, level), group in data.groupby(["family", dimension], sort=True):
        tau = group["threshold_activation_count"]
        bound = group["relaxation_bound"]
        ratio = group["normalized_time"]
        rows.append(
            {
                "dimension": dimension,
                "family": family,
                "level": level,
                "runs": len(group),
                "mean_tau_h": tau.mean(),
                "sd_tau_h": tau.std(ddof=1),
                "ci95_tau_h": ci95(tau),
                "mean_b_h": bound.mean(),
                "sd_b_h": bound.std(ddof=1),
                "ci95_b_h": ci95(bound),
                "mean_tau_over_b": ratio.mean(),
                "sd_tau_over_b": ratio.std(ddof=1),
                "ci95_tau_over_b": ci95(ratio),
                "mean_h": group["threshold_h"].mean(),
                "mean_x0": group["excess_x0"].mean(),
                "mean_w": group["weight_w"].mean(),
                "mean_delta": group["max_degree"].mean(),
                "mean_degree_variance": group["degree_variance"].mean(),
                "mean_degree_cv2": group["degree_cv2"].mean(),
            }
        )
    return pd.DataFrame(rows)


def controlled_cells(data: pd.DataFrame) -> pd.DataFrame:
    cells = (
        data.groupby(
            ["family", "n", "k_colors", "average_degree"], as_index=False
        )
        .agg(
            runs=("threshold_activation_count", "size"),
            mean_tau_h=("threshold_activation_count", "mean"),
            mean_b_h=("relaxation_bound", "mean"),
            mean_w=("weight_w", "mean"),
            mean_delta=("max_degree", "mean"),
            mean_degree_variance=("degree_variance", "mean"),
            mean_degree_cv2=("degree_cv2", "mean"),
        )
    )
    cells["certificate_looseness"] = np.where(
        cells["mean_tau_h"] > 0,
        cells["mean_b_h"] / cells["mean_tau_h"],
        np.nan,
    )
    cells["cell_tau_over_b"] = np.where(
        cells["mean_b_h"] > 0,
        cells["mean_tau_h"] / cells["mean_b_h"],
        0.0,
    )
    return cells


def headline_statistics(data: pd.DataFrame, cells: pd.DataFrame) -> dict[str, object]:
    high_palette = data[data["high_palette"].astype(bool)]
    return {
        "trajectories": int(len(data)),
        "controlled_cells": int(len(cells)),
        "largest_controlled_cell_ratio": float(cells["cell_tau_over_b"].max()),
        "mean_tau_h": float(data["threshold_activation_count"].mean()),
        "ci95_tau_h": ci95(data["threshold_activation_count"]),
        "mean_b_h": float(data["relaxation_bound"].mean()),
        "mean_tau_over_b": float(data["normalized_time"].mean()),
        "ci95_tau_over_b": ci95(data["normalized_time"]),
        "family_mean_tau_h": {
            key: float(value)
            for key, value in data.groupby("family")[
                "threshold_activation_count"
            ].mean().items()
        },
        "family_mean_b_h": {
            key: float(value)
            for key, value in data.groupby("family")["relaxation_bound"].mean().items()
        },
        "family_mean_w": {
            key: float(value)
            for key, value in data.groupby("family")["weight_w"].mean().items()
        },
        "family_mean_degree_variance": {
            key: float(value)
            for key, value in data.groupby("family")["degree_variance"].mean().items()
        },
        "high_palette_runs": int(len(high_palette)),
        "high_palette_mean_absorption_time": float(
            high_palette["activation_count"].mean()
        ),
        "high_palette_mean_certificate": float(
            high_palette["relaxation_bound"].mean()
        ),
    }


def plot_marginal(
    ax: plt.Axes,
    summary: pd.DataFrame,
    dimension: str,
    xlabel: str,
    panel: str,
) -> None:
    subset = summary[summary["dimension"] == dimension]
    for family in FAMILIES:
        group = subset[subset["family"] == family].sort_values("level")
        ax.errorbar(
            group["level"],
            group["mean_tau_h"],
            yerr=group["ci95_tau_h"],
            marker="o",
            markersize=4,
            linewidth=1.25,
            capsize=2,
            color=COLORS[family],
            label=LABELS[family],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Mean threshold time $\tau_H$")
    ax.set_title(panel, loc="left", fontweight="bold")
    ax.grid(True, color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, fontsize=7)


def create_figure(data: pd.DataFrame, summary: pd.DataFrame, cells: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    plot_marginal(axes[0, 0], summary, "n", "Number of vertices, $n$", "a")
    plot_marginal(axes[0, 1], summary, "k_colors", "Number of colors, $k$", "b")
    plot_marginal(
        axes[1, 0],
        summary,
        "average_degree",
        r"Target mean degree, $\bar d$",
        "c",
    )

    ax = axes[1, 1]
    for family in FAMILIES:
        group = cells[cells["family"] == family]
        ax.scatter(
            group["mean_w"],
            group["certificate_looseness"],
            s=18,
            alpha=0.68,
            color=COLORS[family],
            edgecolors="white",
            linewidths=0.3,
            label=LABELS[family],
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Controlled-cell mean structural weight $W$")
    ax.set_ylabel(r"Certificate-to-time ratio $\overline{B}_H/\overline{\tau}_H$")
    ax.set_title("d", loc="left", fontweight="bold")
    ax.grid(True, which="both", color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, fontsize=7)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    fig.savefig(path.with_suffix(".eps"), format="eps", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = load_replay(args.input)
    summaries = pd.concat(
        [
            grouped_summary(data, "n"),
            grouped_summary(data, "k_colors"),
            grouped_summary(data, "average_degree"),
        ],
        ignore_index=True,
    )
    cells = controlled_cells(data)
    statistics = headline_statistics(data, cells)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output_dir / "scientific_reports_group_summary.csv", index=False)
    cells.to_csv(args.output_dir / "scientific_reports_controlled_cells.csv", index=False)
    (args.output_dir / "scientific_reports_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True), encoding="utf-8"
    )
    create_figure(
        data,
        summaries,
        cells,
        args.output_dir / "figure_2_structural_determinants.png",
    )
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
