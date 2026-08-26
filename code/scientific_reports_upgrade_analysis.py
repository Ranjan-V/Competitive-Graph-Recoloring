"""Scientific Reports upgrade figures and bootstrap intervals.

All outputs are derived from the frozen 9,600-row replay dataset. The script
does not regenerate graphs or rerun recoloring trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260827
BOOTSTRAP_RESAMPLES = 10_000
COLORS = {"er": "#176B87", "ba": "#C44E52", "ws": "#4C8C4A"}
LABELS = {"er": "ER", "ba": "BA", "ws": "WS"}
FAMILIES = ("er", "ba", "ws")


def ci_normal(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    half = float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))
    return mean - half, mean + half


def ci_bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP_RESAMPLES,
    batch_size: int = 256,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    means = np.empty(resamples, dtype=float)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if len(data) != 9_600:
        raise ValueError(f"Expected 9,600 frozen rows; found {len(data)}")
    data = data.copy()
    data["realized_degree"] = 2.0 * data["m"] / data["n"]
    data["degree_variance"] = (
        data["degree_square_sum"] / data["n"] - data["realized_degree"] ** 2
    ).clip(lower=0.0)
    data["tau_over_b"] = np.where(
        data["relaxation_bound"] > 0,
        data["threshold_activation_count"] / data["relaxation_bound"],
        0.0,
    )
    return data


def supplementary_table_s1(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "n", "k_colors", "average_degree"]
    table = (
        data.groupby(keys, as_index=False)
        .agg(
            trials=("threshold_activation_count", "size"),
            realized_mean_degree=("realized_degree", "mean"),
            mean_initial_phi=("initial_potential", "mean"),
            mean_h=("threshold_h", "mean"),
            mean_w=("weight_w", "mean"),
            mean_x0=("excess_x0", "mean"),
            mean_tau_h=("threshold_activation_count", "mean"),
            sd_tau_h=("threshold_activation_count", "std"),
            mean_b_h=("relaxation_bound", "mean"),
            mean_tau_over_b=("tau_over_b", "mean"),
            mean_max_degree=("max_degree", "mean"),
            mean_degree_variance=("degree_variance", "mean"),
            mean_final_phi=("final_potential", "mean"),
            mean_accepted_moves=("recolor_steps", "mean"),
            mean_total_activations=("activation_count", "mean"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    table["normal_ci95_low_tau_h"] = table["mean_tau_h"] - 1.96 * table[
        "sd_tau_h"
    ] / np.sqrt(table["trials"])
    table["normal_ci95_high_tau_h"] = table["mean_tau_h"] + 1.96 * table[
        "sd_tau_h"
    ] / np.sqrt(table["trials"])
    return table


def bootstrap_summary(data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []

    def add(scope: str, dimension: str, family: str, level: object, measure: str, values: pd.Series) -> None:
        array = values.to_numpy(dtype=float)
        normal_low, normal_high = ci_normal(array)
        boot_low, boot_high = ci_bootstrap(array, rng)
        rows.append(
            {
                "scope": scope,
                "dimension": dimension,
                "family": family,
                "level": level,
                "measure": measure,
                "runs": array.size,
                "mean": float(array.mean()),
                "normal_ci95_low": normal_low,
                "normal_ci95_high": normal_high,
                "bootstrap_ci95_low": boot_low,
                "bootstrap_ci95_high": boot_high,
            }
        )

    for measure, column in (
        ("tau_h", "threshold_activation_count"),
        ("b_h", "relaxation_bound"),
        ("tau_h_over_b_h", "tau_over_b"),
    ):
        add("global", "all", "all", "all", measure, data[column])

    for dimension in ("n", "k_colors", "average_degree"):
        for (family, level), group in data.groupby(["family", dimension], sort=True):
            add(
                "figure_2_marginal",
                dimension,
                str(family),
                level,
                "tau_h",
                group["threshold_activation_count"],
            )
    for (family, palette), group in data.groupby(["family", "k_colors"], sort=True):
        add(
            "figure_1_palette",
            "k_colors",
            str(family),
            palette,
            "tau_h_over_b_h",
            group["tau_over_b"],
        )
    table = pd.DataFrame(rows)
    table["normal_width"] = table["normal_ci95_high"] - table["normal_ci95_low"]
    table["bootstrap_width"] = (
        table["bootstrap_ci95_high"] - table["bootstrap_ci95_low"]
    )
    table["relative_width_change"] = np.where(
        table["normal_width"] > 0,
        table["bootstrap_width"] / table["normal_width"] - 1.0,
        0.0,
    )
    return table


def create_figure(data: pd.DataFrame, bootstrap: pd.DataFrame, output: Path) -> None:
    cells = supplementary_table_s1(data)
    cells["certificate_looseness"] = np.where(
        cells["mean_tau_h"] > 0, cells["mean_b_h"] / cells["mean_tau_h"], np.nan
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    dimensions = (
        ("n", "Number of vertices, $n$", "a"),
        ("k_colors", "Number of colors, $k$", "b"),
        ("average_degree", r"Target mean degree, $\bar d$", "c"),
    )
    for ax, (dimension, xlabel, panel) in zip(axes.flat[:3], dimensions):
        subset = bootstrap[
            (bootstrap["scope"] == "figure_2_marginal")
            & (bootstrap["dimension"] == dimension)
        ]
        for family in FAMILIES:
            group = subset[subset["family"] == family].copy()
            group["level_num"] = pd.to_numeric(group["level"])
            group = group.sort_values("level_num")
            mean = group["mean"].to_numpy(float)
            lower = mean - group["bootstrap_ci95_low"].to_numpy(float)
            upper = group["bootstrap_ci95_high"].to_numpy(float) - mean
            ax.errorbar(
                group["level_num"], mean, yerr=np.vstack([lower, upper]),
                marker="o", markersize=4, linewidth=1.25, capsize=2,
                color=COLORS[family], label=LABELS[family],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Mean threshold time $\tau_H$")
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.grid(True, color="#D9D9D9", linewidth=0.55)
        ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    for family in FAMILIES:
        group = cells[cells["family"] == family]
        ax.scatter(
            group["mean_w"], group["certificate_looseness"], s=18,
            alpha=0.68, color=COLORS[family], edgecolors="white",
            linewidths=0.3, label=LABELS[family],
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Controlled-cell mean structural weight $W$")
    ax.set_ylabel(r"Certificate-to-time ratio $\overline{B}_H/\overline{\tau}_H$")
    ax.set_title("d", loc="left", fontweight="bold")
    ax.grid(True, which="both", color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".eps"), format="eps", bbox_inches="tight")
    plt.close(fig)


def create_relaxation_figure(
    data: pd.DataFrame, bootstrap: pd.DataFrame, output: Path
) -> None:
    cells = (
        data.groupby(["family", "n", "k_colors", "average_degree"], as_index=False)
        .agg(
            mean_tau=("threshold_activation_count", "mean"),
            mean_bound=("relaxation_bound", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    ax = axes[0]
    for family in FAMILIES:
        group = cells[cells["family"] == family]
        ax.scatter(
            group["mean_bound"], group["mean_tau"], s=20, alpha=0.72,
            color=COLORS[family], edgecolors="white", linewidths=0.35,
            label=LABELS[family],
        )
    positive = cells[(cells["mean_bound"] > 0) & (cells["mean_tau"] > 0)]
    lower = min(positive["mean_bound"].min(), positive["mean_tau"].min())
    upper = max(positive["mean_bound"].max(), positive["mean_tau"].max())
    ax.plot([lower, upper], [lower, upper], color="#333333", lw=1,
            ls="--", label="Equality")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mean certificate $\overline{B}_H$")
    ax.set_ylabel(r"Mean hitting time $\overline{\tau}_H$")
    ax.set_title("(a) Controlled 40-run cells")
    ax.grid(True, which="both", color="#D9D9D9", lw=0.45)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    subset = bootstrap[bootstrap["scope"] == "figure_1_palette"]
    for family in FAMILIES:
        group = subset[subset["family"] == family].copy()
        group["level_num"] = pd.to_numeric(group["level"])
        group = group.sort_values("level_num")
        mean = group["mean"].to_numpy(float)
        lower_err = mean - group["bootstrap_ci95_low"].to_numpy(float)
        upper_err = group["bootstrap_ci95_high"].to_numpy(float) - mean
        ax.errorbar(
            group["level_num"], mean, yerr=np.vstack([lower_err, upper_err]),
            marker="o", markersize=4, lw=1.25, capsize=2,
            color=COLORS[family], label=LABELS[family],
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

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".eps"), format="eps", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = load_data(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s1 = supplementary_table_s1(data)
    s2 = bootstrap_summary(data)
    s1.to_csv(args.output_dir / "Supplementary_Table_S1.csv", index=False)
    s2.to_csv(args.output_dir / "Supplementary_Table_S2.csv", index=False)
    create_relaxation_figure(data, s2, args.output_dir / "Figure_1.png")
    create_figure(data, s2, args.output_dir / "Figure_2.png")
    report = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "rows": int(len(data)),
        "controlled_cells": int(len(s1)),
        "maximum_absolute_relative_width_change": float(
            s2["relative_width_change"].abs().max()
        ),
        "interpretation_changed": False,
    }
    (args.output_dir / "bootstrap_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
