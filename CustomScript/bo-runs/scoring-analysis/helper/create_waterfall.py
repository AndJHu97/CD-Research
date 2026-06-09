#!/usr/bin/env python3
"""Create a two-panel figure from a baby_frank summary CSV.

Panel A: Search-space reduction waterfall (absolute + relative per filter).
Panel B: Per-residue hit-rate bar chart with aggregate statistics.

Usage:
    python create_waterfall.py summary_baby_frank.csv [--output figure.png]
    python create_waterfall.py summary_baby_frank.csv --skip-filter0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Canonical filter names → short display labels (order in CSV may vary).
FILTER_SHORT_NAMES: dict[str, str] = {
    "Nucleophilic residue selection (Filter 0)": "Nucleophilic\nselection",
    "Rel_Side_SASA gte 13.0": "Side-chain\nSASA",
    "deprotonation_prob gte 0.14": "Deprotonation\nprob.",
    "LGBM ranker top-1": "LGBM\ntop-1",
}

FILTER0_PATTERN = re.compile(r"nucleophilic.*residue.*selection", re.I)

FILTER_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (FILTER0_PATTERN, "Nucleophilic\nselection"),
    (re.compile(r"rel_side_sasa", re.I), "Side-chain\nSASA"),
    (re.compile(r"deprotonation_prob", re.I), "Deprotonation\nprob."),
    (re.compile(r"lgbm.*reactivity ranker.*top-?1", re.I), "LGBM\ntop-1"),
]

PANEL_B_RESIDUES = ("CYS", "LYS", "HIS", "TYR", "THR", "SER")
RESIDUE_LABELS = {
    "CYS": "Cys",
    "LYS": "Lys",
    "HIS": "His",
    "TYR": "Tyr",
    "THR": "Thr",
    "SER": "Ser",
}

CYS_ONLY_BENCHMARK_PCT = 53.7


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

    baby_frank reports Avg_Abs_Reduction_Pct against the full PDB denominator.
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


def get_residue_rows(df: pd.DataFrame) -> pd.DataFrame:
    residues = df[df["Group"] == "Residue_Type"].copy()
    if residues.empty:
        raise ValueError("No Residue_Type rows found in summary CSV.")
    return residues


def compute_panel_b_stats(residues: pd.DataFrame) -> dict[str, float | int]:
    panel_res = residues[residues["Category"].isin(PANEL_B_RESIDUES)].copy()

    hits = int(panel_res["N_Hits"].sum())
    matchable = int(panel_res["N_Matchable"].sum())
    overall_six_pct = (hits / matchable * 100) if matchable else 0.0

    all_hits = int(residues["N_Hits"].sum())
    all_matchable = int(residues["N_Matchable"].sum())
    cys_row = residues[residues["Category"] == "CYS"]
    cys_hits = int(cys_row["N_Hits"].sum()) if not cys_row.empty else 0
    cys_matchable = int(cys_row["N_Matchable"].sum()) if not cys_row.empty else 0

    non_cys_hits = all_hits - cys_hits
    non_cys_matchable = all_matchable - cys_matchable
    non_cys_pct = (non_cys_hits / non_cys_matchable * 100) if non_cys_matchable else 0.0

    per_residue_pct = {
        row["Category"]: float(row["Hit_Rate_Pct"])
        for _, row in panel_res.iterrows()
    }

    return {
        "per_residue_pct": per_residue_pct,
        "overall_six_hits": hits,
        "overall_six_matchable": matchable,
        "overall_six_pct": overall_six_pct,
        "non_cys_hits": non_cys_hits,
        "non_cys_matchable": non_cys_matchable,
        "non_cys_pct": non_cys_pct,
    }


def plot_waterfall(
    ax: plt.Axes,
    filters: pd.DataFrame,
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
    ax.text(
        n - 0.45,
        total_reduction,
        f"Total\n({total_reduction:.1f}%)",
        ha="left",
        va="center",
        fontsize=8,
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

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    if baseline_note:
        ax.set_ylabel("Cumulative reduction (% of post-nucleophilic pool)")
        ax.set_title(
            "A  Search-space reduction (post-Filter 0)",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )
    else:
        ax.set_ylabel("Cumulative search-space reduction (%)")
        ax.set_title("A  Search-space reduction", loc="left", fontweight="bold", fontsize=11)
    ax.set_ylim(0, min(total_reduction + 14, 108))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_accuracy_bars(ax: plt.Axes, stats: dict[str, float | int]) -> None:
    per_res = stats["per_residue_pct"]

    categories: list[str] = [RESIDUE_LABELS[r] for r in PANEL_B_RESIDUES]
    values: list[float] = [per_res.get(r, 0.0) for r in PANEL_B_RESIDUES]

    categories.extend(["Overall\n(6 res.)", "Non-Cys\n(all res.)", "Cys-only\nscreeners*"])
    values.extend([
        float(stats["overall_six_pct"]),
        float(stats["non_cys_pct"]),
        CYS_ONLY_BENCHMARK_PCT,
    ])

    colors = ["#4C72B0"] * len(PANEL_B_RESIDUES)
    colors.extend(["#55A868", "#C44E52", "#AAAAAA"])

    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.72)

    ax.axhline(CYS_ONLY_BENCHMARK_PCT, color="#888888", linewidth=1.0, linestyle="--", zorder=0)

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
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("B  Accuracy by residue", loc="left", fontweight="bold", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    footnote = (
        f"*Literature benchmark for cysteine-only covalent screeners ({CYS_ONLY_BENCHMARK_PCT}%). "
        f"Overall (6 res.): {stats['overall_six_hits']}/{stats['overall_six_matchable']} hits. "
        f"Non-Cys: {stats['non_cys_hits']}/{stats['non_cys_matchable']} hits."
    )
    ax.text(0.0, -0.22, footnote, transform=ax.transAxes, fontsize=7, color="#555555", va="top")


def create_figure(
    csv_path: Path,
    output_path: Path | None = None,
    skip_filter0: bool = False,
) -> Path:
    df = load_summary(csv_path)
    filters = get_filter_rows(df)
    waterfall_filters, baseline_note = prepare_waterfall_filters(filters, skip_filter0)
    residues = get_residue_rows(df)
    stats = compute_panel_b_stats(residues)

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.38}
    )

    plot_waterfall(ax_a, waterfall_filters, baseline_note=baseline_note)
    plot_accuracy_bars(ax_b, stats)
    fig.subplots_adjust(bottom=0.10)

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create waterfall + accuracy figure from a baby_frank summary CSV."
    )
    parser.add_argument("summary_csv", type=Path, help="Path to summary CSV (e.g. summary_baby_frank.csv).")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
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
