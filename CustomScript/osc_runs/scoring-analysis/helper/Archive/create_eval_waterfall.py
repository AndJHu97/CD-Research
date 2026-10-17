#!/usr/bin/env python3
"""Create a four-panel 2×2 figure from an eval_frank summary CSV.

Panel A: Search-space reduction Sankey (post-nucleophilic pool through filters).
Panel B: Per-filter target retention vs random baseline (If_Match totals; Filter_Residue_Type when present).
Panel C: Failure attribution bar chart (% of N_Failures per filter).
Panel D: Success-rate bar chart vs literature covalent docking benchmarks (207 complexes).

Usage:
    python create_eval_waterfall.py summary_eval_frank.csv [--output figure.png]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HELPER_DIR = Path(__file__).resolve().parent
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))

from create_waterfall import (  # noqa: E402
    FIG_HEIGHT,
    FIG_WIDTH,
    compute_failure_attribution,
    compute_filter_accuracy_stats,
    plot_failure_attribution_bars,
    plot_filter_accuracy_lines,
    plot_search_space_sankey,
)

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
FILTER_HIT_COLOR = "#4C72B0"
RANDOM_BASELINE_COLOR = "#C0C0C0"
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


def prepare_filters_for_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows for Panel B: prefer Filter_Residue_Type (eval is per-residue)."""
    residue_filters = df[df["Group"] == "Filter_Residue_Type"].copy()
    if not residue_filters.empty:
        cys = residue_filters[residue_filters["Category"] == "CYS"]
        filters = cys.copy() if not cys.empty else residue_filters.copy()
    else:
        filters = df[df["Group"] == "Filter"].copy()

    if filters.empty:
        raise ValueError("No Filter or Filter_Residue_Type rows found for Panel B.")

    if "Filter" in filters.columns and filters["Filter"].notna().any():
        filters["_Filter_Name"] = filters["Filter"]
    else:
        filters["_Filter_Name"] = filters["Category"]
    filters["Short_Name"] = filters["_Filter_Name"].map(short_filter_name)
    return _ensure_if_match_totals(filters)


def _ensure_if_match_totals(filters: pd.DataFrame) -> pd.DataFrame:
    """Ensure Total_N_*_If_Match exist (from CSV, Avg×N_Count, or Avg fallback)."""
    out = filters.copy()

    def _fill_total(total_col: str, avg_col: str, count_col: str) -> None:
        if total_col in out.columns and out[total_col].notna().any():
            return
        if avg_col in out.columns and out[avg_col].notna().any():
            if count_col in out.columns and out[count_col].notna().any():
                out[total_col] = out[avg_col] * out[count_col]
            else:
                out[total_col] = pd.to_numeric(out[avg_col], errors="coerce")
            return
        base_total = total_col.replace("_If_Match", "")
        base_avg = avg_col.replace("_If_Match", "")
        if base_total in out.columns and out[base_total].notna().any():
            out[total_col] = out[base_total]
        elif base_avg in out.columns:
            out[total_col] = pd.to_numeric(out[base_avg], errors="coerce")

    _fill_total(
        "Total_N_Before_If_Match", "Avg_N_Before_If_Match", "N_Count_If_Match"
    )
    _fill_total(
        "Total_N_After_If_Match", "Avg_N_After_If_Match", "N_Count_If_Match"
    )
    return out


def prepare_filters_for_sankey(filters: pd.DataFrame) -> pd.DataFrame:
    """Normalize eval filter rows for create_waterfall Sankey (Total_N from Avg_N if needed)."""
    out = filters.copy()
    out["_Filter_Name"] = out["Category"]
    if "Total_N_Before" not in out.columns or out["Total_N_Before"].isna().all():
        out["Total_N_Before"] = pd.to_numeric(out["Avg_N_Before"], errors="coerce")
        out["Total_N_After"] = pd.to_numeric(out["Avg_N_After"], errors="coerce")
    else:
        out["Total_N_Before"] = out["Total_N_Before"].fillna(out["Avg_N_Before"])
        out["Total_N_After"] = out["Total_N_After"].fillna(out["Avg_N_After"])
    return out


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


def get_overall_matchable(df: pd.DataFrame) -> int:
    overall = df[df["Group"] == "Overall"]
    if not overall.empty:
        return int(overall.iloc[0]["N_Matchable"])

    cys = df[(df["Group"] == "Residue_Type") & (df["Category"] == "CYS")]
    if not cys.empty:
        return int(cys.iloc[0]["N_Matchable"])

    raise ValueError(
        "No Overall or CYS Residue_Type row with N_Matchable found in summary CSV."
    )


def get_overall_hits(df: pd.DataFrame) -> int:
    overall = df[df["Group"] == "Overall"]
    if not overall.empty:
        return int(overall.iloc[0]["N_Hits"])

    cys = df[(df["Group"] == "Residue_Type") & (df["Category"] == "CYS")]
    if not cys.empty:
        return int(cys.iloc[0]["N_Hits"])

    raise ValueError("No Overall or CYS Residue_Type row with N_Hits found in summary CSV.")


def compute_cumulative_random_baseline(accuracy: pd.DataFrame) -> float:
    """Product of per-filter random retention rates (post Filter 0), as %."""
    cumulative = 1.0
    for pct in accuracy["Random_Hit_Rate_Pct"]:
        cumulative *= float(pct) / 100.0
    return cumulative * 100.0


def compute_panel_d_stats(
    df: pd.DataFrame,
    accuracy: pd.DataFrame,
) -> dict[str, float | int]:
    hits = get_overall_hits(df)
    our_pct = hits / EVAL_DATASET_SIZE * 100.0
    cumulative_random_pct = compute_cumulative_random_baseline(accuracy)

    benchmark_pcts = {
        label: 100.0 - error_pct
        for label, error_pct, _ in DOCKING_BENCHMARKS
    }

    return {
        "hits": hits,
        "our_pct": our_pct,
        "cumulative_random_pct": cumulative_random_pct,
        "benchmark_pcts": benchmark_pcts,
    }


def plot_benchmark_bars(
    ax: plt.Axes,
    stats: dict[str, float | int],
    overall_reduction: float,
) -> str:
    benchmark_pcts = stats["benchmark_pcts"]

    benchmark_labels = [label for label, _, _ in DOCKING_BENCHMARKS]
    random_label = "Random baseline"
    categories = [METHOD_LABEL, random_label, *benchmark_labels]
    values = [
        float(stats["our_pct"]),
        float(stats["cumulative_random_pct"]),
        *(float(benchmark_pcts[label]) for label in benchmark_labels),
    ]
    colors = [
        FRANKENSTEIN_COLOR,
        RANDOM_BASELINE_COLOR,
        *BENCHMARK_COLORS[: len(benchmark_labels)],
    ]

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
        "D  Success rate vs covalent screening benchmarks",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"Overall reduction: {overall_reduction:.1f}%. "
        f"Random baseline: product of per-filter random retention "
        f"(1 − avg N after / avg N before) across post-nucleophilic filters "
        f"({stats['cumulative_random_pct']:.1f}%). "
        f"Docking bars: success = 100% − reported % error (≤2 Å RMSD, top-1 pose; "
        f"most n={EVAL_DATASET_SIZE}, FITTED n=175). "
        f"{METHOD_LABEL}: {stats['hits']}/{EVAL_DATASET_SIZE} complexes (residue recovery)."
    )


def create_figure(
    csv_path: Path,
    output_path: Path | None = None,
    skip_filter0: bool = False,
) -> Path:
    del skip_filter0  # Sankey always shows post–Filter 0 flow (deprecated flag)

    df = load_summary(csv_path)
    filters = get_filter_rows(df)
    sankey_filters = prepare_filters_for_sankey(filters)
    accuracy_filters = prepare_filters_for_accuracy(df)
    f0_to_last_reduction = compute_f0_to_last_reduction(filters)
    overall_reduction = compute_overall_reduction(filters)
    accuracy = compute_filter_accuracy_stats(accuracy_filters)
    failure_attribution = compute_failure_attribution(sankey_filters)
    total_failures = int(failure_attribution["N_Failures"].sum())
    stats = compute_panel_d_stats(df, accuracy)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        hspace=0.40,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_search_space_sankey(ax_a, sankey_filters, f0_to_last_reduction, overall_reduction)
    plot_filter_accuracy_lines(
        ax_b,
        accuracy,
        title="Filter accuracy vs random baseline",
    )
    plot_failure_attribution_bars(ax_c, failure_attribution, total_failures)
    footnote = plot_benchmark_bars(ax_d, stats, overall_reduction)
    fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.12)

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create eval summary figure (Sankey + accuracy + failure + benchmarks)."
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
        help="Deprecated: Panel A Sankey always shows post-nucleophilic filters only.",
    )
    args = parser.parse_args()

    if not args.summary_csv.exists():
        raise SystemExit(f"File not found: {args.summary_csv}")

    out = create_figure(args.summary_csv, args.output, skip_filter0=args.skip_filter0)
    print(f"[INFO] Figure written to: {out}")


if __name__ == "__main__":
    main()
