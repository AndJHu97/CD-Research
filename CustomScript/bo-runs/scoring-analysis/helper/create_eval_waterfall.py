#!/usr/bin/env python3
"""Create a two-panel figure from an eval_frank summary CSV.

Panel A: Search-space reduction waterfall (absolute + relative per filter).
Panel B: Success-rate bar chart vs literature covalent docking benchmarks (207 complexes).

Usage:
    python create_eval_waterfall.py summary_eval_frank.csv [--output figure.png]
    python create_eval_waterfall.py summary_eval_frank.csv --skip-filter0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DATASET_SIZE = 207

# Canonical filter names → short display labels (order in CSV may vary).
FILTER_SHORT_NAMES: dict[str, str] = {
    "Nucleophilic residue selection (Filter 0)": "Nucleophilic\nselection",
    "Rel_Side_SASA gte 12.0": "Side-chain\nSASA",
    "Rel_Side_SASA gte 13.0": "Side-chain\nSASA",
    "deprotonation_prob gte 0.14": "Deprotonation\nprob.",
    "LGBM ranker top-1": "LGBM\ntop-1",
    "LGBM ranker top-3": "LGBM\ntop-3",
}

FILTER0_PATTERN = re.compile(r"nucleophilic.*residue.*selection", re.I)

FILTER_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (FILTER0_PATTERN, "Nucleophilic\nselection"),
    (re.compile(r"rel_side_sasa", re.I), "Side-chain\nSASA"),
    (re.compile(r"deprotonation_prob", re.I), "Deprotonation\nprob."),
    (re.compile(r"lgbm.*reactivity ranker.*top-?1", re.I), "LGBM\ntop-1"),
    (re.compile(r"lgbm.*reactivity ranker.*top-?3", re.I), "LGBM\ntop-3"),
    (re.compile(r"lgbm.*ranker.*top-?1", re.I), "LGBM\ntop-1"),
    (re.compile(r"lgbm.*ranker.*top-?3", re.I), "LGBM\ntop-3"),
]

# Literature covalent docking benchmarks: (method label, % error, dataset size).
DOCKING_BENCHMARKS: list[tuple[str, float, int]] = [
    ("AutoDock4", 45.0, 207),
    ("CovDock", 41.0, 207),
    ("FITTED", 44.0, 175),
    ("GOLD", 47.0, 207),
    ("ICM-Pro", 38.0, 207),
    ("MOE", 63.0, 207),
]

METHOD_LABEL = "Frankenstein"
FRANKENSTEIN_COLOR = "#55A868"
BENCHMARK_COLORS = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3", "#937860", "#64B5CD"]


def short_filter_name(raw_name: str) -> str:
    if raw_name in FILTER_SHORT_NAMES:
        return FILTER_SHORT_NAMES[raw_name]
    for pattern, label in FILTER_NAME_PATTERNS:
        if pattern.search(raw_name):
            return label
    return raw_name.replace("_", " ")


def load_summary(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def is_filter0(category: str) -> bool:
    return bool(FILTER0_PATTERN.search(str(category)))


def get_filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    filters = df[df["Group"] == "Filter"].copy()
    if filters.empty:
        raise ValueError("No Filter rows found in summary CSV.")
    filters["Short_Name"] = filters["Category"].map(short_filter_name)
    return filters


def prepare_waterfall_filters(
    filters: pd.DataFrame,
    skip_filter0: bool,
) -> tuple[pd.DataFrame, str | None]:
    """
    Optionally drop Filter 0 and rebase abs-reduction % onto Avg_N_After from Filter 0.

    eval_frank reports Avg_Abs_Reduction_Pct against the full PDB denominator.
    With --skip-filter0, abs reduction uses the post-nucleophilic pool as 100%.
    Relative reduction per step is unchanged.
    """
    if not skip_filter0:
        return filters.copy(), None

    f0_mask = filters["Category"].map(is_filter0)
    if not f0_mask.any():
        raise ValueError(
            "--skip-filter0 requires a nucleophilic residue selection (Filter 0) row."
        )

    f0_row = filters[f0_mask].iloc[0]
    baseline_n = float(f0_row["Avg_N_After"])
    if baseline_n <= 0:
        raise ValueError("--skip-filter0: Filter 0 Avg_N_After is missing or zero.")

    remaining = filters[~f0_mask].copy()
    remaining["Avg_Abs_Reduction_Pct"] = (
        (remaining["Avg_N_Before"] - remaining["Avg_N_After"]) / baseline_n * 100.0
    )

    baseline_note = (
        f"abs. reduction rebased to post-nucleophilic pool "
        f"(avg N={baseline_n:.1f})"
    )
    return remaining, baseline_note


def compute_f0_to_last_reduction(filters: pd.DataFrame) -> float:
    """Pool reduction from Filter 0 output to last-filter output."""
    f0_mask = filters["Category"].map(is_filter0)
    if not f0_mask.any():
        raise ValueError("Cannot compute Filter 0→last reduction without Filter 0 row.")

    f0_row = filters[f0_mask].iloc[0]
    last_row = filters.iloc[-1]
    n_start = float(f0_row["Avg_N_After"])
    n_end = float(last_row["Avg_N_After"])
    if n_start <= 0:
        raise ValueError("Filter 0 Avg_N_After is missing or zero.")

    return (n_start - n_end) / n_start * 100.0


def compute_overall_reduction(filters: pd.DataFrame) -> float:
    """Cumulative abs. reduction from Filter 0 through the last filter."""
    return float(filters["Avg_Abs_Reduction_Pct"].fillna(0).sum())


def get_overall_hits(df: pd.DataFrame) -> int:
    overall = df[df["Group"] == "Overall"]
    if not overall.empty:
        return int(overall.iloc[0]["N_Hits"])

    cys = df[(df["Group"] == "Residue_Type") & (df["Category"] == "CYS")]
    if not cys.empty:
        return int(cys.iloc[0]["N_Hits"])

    raise ValueError("No Overall or CYS Residue_Type row with N_Hits found in summary CSV.")


def compute_panel_b_stats(df: pd.DataFrame) -> dict[str, float | int]:
    hits = get_overall_hits(df)
    our_pct = hits / EVAL_DATASET_SIZE * 100.0

    benchmark_pcts = {
        label: 100.0 - error_pct
        for label, error_pct, _ in DOCKING_BENCHMARKS
    }

    return {
        "hits": hits,
        "our_pct": our_pct,
        "benchmark_pcts": benchmark_pcts,
    }


def plot_waterfall(
    ax: plt.Axes,
    filters: pd.DataFrame,
    f0_to_last_reduction: float,
    overall_reduction: float,
    baseline_note: str | None = None,
) -> None:
    """Upward waterfall of cumulative search-space eliminated per filter."""
    names = filters["Short_Name"].tolist()
    abs_pct = filters["Avg_Abs_Reduction_Pct"].fillna(0).to_numpy(dtype=float)
    rel_pct = filters["Avg_Rel_Reduction_Pct"].fillna(0).to_numpy(dtype=float)

    n = len(names)
    x = np.arange(n)
    width = 0.55

    cumulative = 0.0
    abs_bottoms: list[float] = []
    abs_heights: list[float] = []
    for drop in abs_pct:
        abs_bottoms.append(cumulative)
        abs_heights.append(drop)
        cumulative += drop

    ax.bar(
        x,
        abs_heights,
        width,
        bottom=abs_bottoms,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
        label="Abs. reduction",
        zorder=3,
    )

    for i in range(n - 1):
        step_top = abs_bottoms[i] + abs_heights[i]
        ax.plot(
            [x[i] + width / 2, x[i + 1] - width / 2],
            [step_top, step_top],
            color="#4C72B0",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            zorder=2,
        )

    total_reduction = cumulative
    ax.axhline(total_reduction, color="#888888", linewidth=0.8, linestyle=":", zorder=1)
    total_label = (
        f"Total\n{f0_to_last_reduction:.1f}%\n"
        f"(Overall reduction: {overall_reduction:.1f}%)"
    )
    label_x = x[-1] + width / 2 + 0.05
    ax.text(
        label_x,
        total_reduction - 1.0,
        total_label,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )

    for i, (a_drop, r_drop) in enumerate(zip(abs_pct, rel_pct)):
        bar_bottom = abs_bottoms[i]
        bar_height = abs_heights[i]
        if bar_height >= 4.0:
            ax.text(
                x[i],
                bar_bottom + bar_height / 2,
                f"+{a_drop:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                x[i],
                bar_bottom + bar_height + 1.0,
                f"+{a_drop:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#2F4A72",
                fontweight="bold",
            )
        step_top = bar_bottom + bar_height
        ax.text(
            x[i],
            step_top + 1.5,
            f"rel {r_drop:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#DD8452",
            fontweight="bold",
        )

    ax.legend(
        handles=[
            mpatches.Patch(color="#4C72B0", label="Abs. reduction"),
            mpatches.Patch(color="#DD8452", label="Rel. reduction (per step)"),
        ],
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )

    ax.set_xlim(-0.55, label_x + 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    if baseline_note:
        ax.set_ylabel("Cumulative reduction (% of post-nucleophilic pool)")
        ax.set_title(
            "A  Search-space reduction (post-nucleophilic selection)",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )
    else:
        ax.set_ylabel("Cumulative search-space reduction (%)")
        ax.set_title("A  Search-space reduction", loc="left", fontweight="bold", fontsize=11)
    ax.set_ylim(0, total_reduction + 7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_benchmark_bars(
    ax: plt.Axes,
    stats: dict[str, float | int],
    overall_reduction: float,
) -> str:
    benchmark_pcts = stats["benchmark_pcts"]

    benchmark_labels = [label for label, _, _ in DOCKING_BENCHMARKS]
    categories = [METHOD_LABEL, *benchmark_labels]
    values = [float(stats["our_pct"]), *(
        float(benchmark_pcts[label]) for label in benchmark_labels
    )]
    colors = [FRANKENSTEIN_COLOR, *BENCHMARK_COLORS[: len(benchmark_labels)]]

    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.72)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, max(values) + 9)
    ax.set_title(
        "B  Success rate vs covalent docking benchmarks",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"Overall reduction: {overall_reduction:.1f}%. "
        f"Docking bars: success = 100% − reported % error (≤2 Å RMSD, top-1 pose; "
        f"most n={EVAL_DATASET_SIZE}, FITTED n=175). "
        f"{METHOD_LABEL}: {stats['hits']}/{EVAL_DATASET_SIZE} complexes (residue recovery)."
    )


def create_figure(
    csv_path: Path,
    output_path: Path | None = None,
    skip_filter0: bool = False,
) -> Path:
    df = load_summary(csv_path)
    filters = get_filter_rows(df)
    waterfall_filters, baseline_note = prepare_waterfall_filters(filters, skip_filter0)
    f0_to_last_reduction = compute_f0_to_last_reduction(filters)
    overall_reduction = compute_overall_reduction(filters)
    stats = compute_panel_b_stats(df)

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.28}
    )

    plot_waterfall(
        ax_a,
        waterfall_filters,
        f0_to_last_reduction,
        overall_reduction,
        baseline_note=baseline_note,
    )
    footnote = plot_benchmark_bars(ax_b, stats, overall_reduction)
    fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.18)

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create waterfall + benchmark figure from an eval_frank summary CSV."
    )
    parser.add_argument(
        "summary_csv", type=Path, help="Path to summary CSV (e.g. summary_eval_frank.csv)."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path (default: same name as CSV with .png extension).",
    )
    parser.add_argument(
        "--skip-filter0",
        action="store_true",
        help=(
            "Omit nucleophilic selection (Filter 0) from the waterfall and rebase "
            "abs. reduction % onto Avg_N_After from Filter 0."
        ),
    )
    args = parser.parse_args()

    if not args.summary_csv.exists():
        raise SystemExit(f"File not found: {args.summary_csv}")

    out = create_figure(args.summary_csv, args.output, skip_filter0=args.skip_filter0)
    print(f"[INFO] Figure written to: {out}")


if __name__ == "__main__":
    main()
