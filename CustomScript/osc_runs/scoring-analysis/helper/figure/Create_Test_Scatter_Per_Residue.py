#!/usr/bin/env python3
"""DIDN'T USE THIS ONE (I USED THE CREATED_TEST_FILTER_RECOVERY_CURVE.PY): One row per query: where each true site ranked among all of its candidates.

Each query group (unique ``Name``) is ranked by ``pred_score`` (highest score =
rank 1). The top ``--top-pct`` percent of each group (rounded up, at least one
residue) is the predicted set.

Every query is a horizontal grey bar spanning all of its candidate residues on
a linear rank axis; rows are sorted by query size so the negatives form a
wedge. A light band marks each query's top X%, and a dot marks the rank of
each true site (green = inside the top X%, magenta = outside).

Usage:
    python helper/figure/Create_Test_Scatter_Per_Residue.py \\
        --input path/to/test_predictions.csv \\
        --output rank_by_query.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def _configure_figure_fonts() -> None:
    """Use bold Arial for all figure text (register Windows fonts when needed)."""
    from matplotlib import font_manager

    for path in (
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
        Path("/mnt/c/Windows/Fonts/arialbd.ttf"),
        Path("/mnt/c/Windows/Fonts/ARIAL.TTF"),
        Path("/mnt/c/Windows/Fonts/ARIALBD.TTF"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
    ):
        if path.is_file():
            try:
                font_manager.fontManager.addfont(str(path))
            except (OSError, RuntimeError, ValueError):
                pass

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "legend.fontsize": 12,
            "axes.unicode_minus": False,
        }
    )


GROUP_COL = "Name"
RESIDUE_COL = "Residue"
SCORE_COL = "pred_score"
RELEVANCE_COL = "relevance"

CAT_TP = "tp"
CAT_MISSED = "missed"
CAT_NEG = "neg"

TP_COLOR = "#1E9E3A"
MISSED_COLOR = "#D81B9C"
CANDIDATE_COLOR = "#CFCFCF"
TOP_BAND_COLOR = "#A8DDB5"


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in (GROUP_COL, RESIDUE_COL, SCORE_COL, RELEVANCE_COL) if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df[RELEVANCE_COL] = pd.to_numeric(df[RELEVANCE_COL], errors="coerce").fillna(0).astype(int)
    df[RESIDUE_COL] = df[RESIDUE_COL].astype(str).str.strip().str.upper()
    n_before = len(df)
    df = df.dropna(subset=[SCORE_COL])
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} row(s) with non-numeric {SCORE_COL}.")
    return df


def assign_ranks_and_categories(df: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = (
        df.groupby(GROUP_COL)[SCORE_COL].rank(method="min", ascending=False).astype(int)
    )
    df["group_size"] = df.groupby(GROUP_COL)[SCORE_COL].transform("size")
    df["top_n"] = np.maximum(1, np.ceil(df["group_size"] * top_pct / 100.0)).astype(int)
    df["in_top"] = df["rank"] <= df["top_n"]

    positive = df[RELEVANCE_COL] == 1
    df["category"] = np.where(
        positive & df["in_top"], CAT_TP, np.where(positive, CAT_MISSED, CAT_NEG)
    )
    return df


def plot(df: pd.DataFrame, top_pct: float, output: Path) -> None:
    _configure_figure_fonts()

    queries = (
        df.groupby(GROUP_COL)
        .agg(size=("group_size", "first"), top_n=("top_n", "first"), best=("rank", "min"))
        .reset_index()
    )
    first_site = (
        df[df[RELEVANCE_COL] == 1].groupby(GROUP_COL)["rank"].min().rename("site_rank")
    )
    queries = queries.join(first_site, on=GROUP_COL)
    queries = queries.sort_values(["size", "site_rank"], ascending=[True, True], kind="stable")
    queries["row"] = np.arange(len(queries))
    row_of = dict(zip(queries[GROUP_COL], queries["row"]))

    n_rows = len(queries)
    sizes = np.sort(queries["size"].to_numpy())
    # A lone oversized query would stretch the axis but is only one row tall.
    if len(sizes) > 1 and sizes[-1] > 1.5 * sizes[-2]:
        x_max = float(sizes[-2]) * 1.04
    else:
        x_max = float(sizes[-1]) * 1.01

    fig, ax = plt.subplots(figsize=(11.0, 8.5))

    ax.barh(
        queries["row"], queries["size"], left=0.5, height=1.0,
        color=CANDIDATE_COLOR, linewidth=0, zorder=1,
    )
    ax.barh(
        queries["row"], queries["top_n"], left=0.5, height=1.0,
        color=TOP_BAND_COLOR, linewidth=0, zorder=2,
    )

    pos = df[df[RELEVANCE_COL] == 1].copy()
    pos["row"] = pos[GROUP_COL].map(row_of)
    found = pos[pos["category"] == CAT_TP]
    missed = pos[pos["category"] == CAT_MISSED]
    ax.scatter(
        np.minimum(missed["rank"], x_max), missed["row"], s=14, c=MISSED_COLOR,
        edgecolors="none", zorder=4,
    )
    ax.scatter(found["rank"], found["row"], s=14, c=TP_COLOR, edgecolors="none", zorder=5)

    for _, q in queries[queries["size"] > x_max].iterrows():
        ax.annotate(
            f"{int(q['size']):,} candidates →",
            xy=(x_max, q["row"]), xytext=(-4, 6), textcoords="offset points",
            ha="right", va="bottom", fontsize=10, color="#555555",
        )

    n_candidates = len(df)
    n_kept = int(df["in_top"].sum())
    n_sites = len(pos)
    n_found = len(found)
    n_rank1 = int((pos["rank"] == 1).sum())
    pct_text = f"{top_pct:g}%"
    ax.text(
        0.97, 0.97,
        f"{n_rows:,} queries  •  {n_candidates:,} candidate residues\n"
        f"Top {pct_text} keeps {n_kept:,} ({n_kept / n_candidates * 100:.1f}%)\n"
        f"True site found: {n_found}/{n_sites} ({n_found / n_sites * 100:.0f}%)\n"
        f"Ranked #1: {n_rank1}/{n_sites} ({n_rank1 / n_sites * 100:.0f}%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=13, linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#999999"},
    )

    ax.set_xlim(0.5 - x_max * 0.005, x_max)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("Rank within query (1 = highest pred_score)", fontsize=14)
    ax.set_ylabel(f"Queries (n = {n_rows:,}, sorted by number of candidates)", fontsize=14)
    ax.set_title("Each true site vs. every candidate residue in its query", fontsize=15, loc="left")
    ax.tick_params(labelsize=12)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    handles = [
        Patch(facecolor=CANDIDATE_COLOR, label="Candidate residues"),
        Patch(facecolor=TOP_BAND_COLOR, label=f"Top {pct_text} of query"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=TP_COLOR,
               markeredgecolor=TP_COLOR, markersize=8, label=f"True site found (n={n_found})"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MISSED_COLOR,
               markeredgecolor=MISSED_COLOR, markersize=8, label=f"True site missed (n={len(missed)})"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    svg_path = output.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", type=Path, required=True, help="Per-residue prediction CSV/TSV.")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (PNG + SVG are written). Default: <input stem>_rank_by_query.png",
    )
    p.add_argument("--top-pct", type=float, default=1.0, help="Top percent per query group (default 1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_rank_by_query.png")

    df = load_predictions(args.input)
    df = assign_ranks_and_categories(df, args.top_pct)

    counts = df["category"].value_counts()
    print(
        f"{df[GROUP_COL].nunique()} query groups, {len(df)} residues | "
        f"TP={int(counts.get(CAT_TP, 0))}, missed={int(counts.get(CAT_MISSED, 0))}, "
        f"negatives={int(counts.get(CAT_NEG, 0))}"
    )

    plot(df, args.top_pct, output)


if __name__ == "__main__":
    main()
