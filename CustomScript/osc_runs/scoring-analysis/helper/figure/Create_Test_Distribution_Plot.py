#!/usr/bin/env python3
"""Score distributions of true sites vs. negatives around the top-X% cutoff.

Each query group (unique ``Name``) is ranked by ``pred_score``; the top
``--top-pct`` percent (rounded up, at least one residue) is the predicted set.
Because that cutoff differs per query, scores are centered on it:

    margin = pred_score - (score of the last residue inside the query's top X%)

so ``margin >= 0`` is exactly "inside the top X%" for every query and a single
decision threshold sits at 0.

Two densities are drawn, each normalized to unit area (positives and negatives
are scaled independently, so the shapes are comparable despite the imbalance):

* True positive  — relevance = 1, margin >= 0 (solid blue)
* False negative — relevance = 1, margin <  0 (blue dots on white)
* True negative  — relevance = 0, margin <  0 (solid orange)
* False positive — relevance = 0, margin >= 0 (orange lines on white)

Usage:
    python helper/figure/Create_Test_Distribution_Plot.py \\
        --input path/to/test_predictions.csv \\
        --output score_distribution.png
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
            "legend.fontsize": 13,
            "axes.unicode_minus": False,
            "hatch.linewidth": 1.6,
        }
    )


GROUP_COL = "Name"
RESIDUE_COL = "Residue"
SCORE_COL = "pred_score"
RELEVANCE_COL = "relevance"

POS_COLOR = "#3B5A8C"
NEG_COLOR = "#F07040"
THRESHOLD_COLOR = "#F2C14E"
FN_HATCH = "o"
FP_HATCH = "---"

GRID_POINTS = 2000


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in (GROUP_COL, SCORE_COL, RELEVANCE_COL) if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df[RELEVANCE_COL] = pd.to_numeric(df[RELEVANCE_COL], errors="coerce").fillna(0).astype(int)
    n_before = len(df)
    df = df.dropna(subset=[SCORE_COL])
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} row(s) with non-numeric {SCORE_COL}.")
    return df


def add_cutoff_margin(df: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """Center each query's scores on the score of its last top-X% residue."""
    df = df.sort_values([GROUP_COL, SCORE_COL], ascending=[True, False], kind="stable").copy()
    group_size = df.groupby(GROUP_COL)[SCORE_COL].transform("size")
    top_n = np.maximum(1, np.ceil(group_size * top_pct / 100.0)).astype(int)
    position = df.groupby(GROUP_COL).cumcount() + 1

    cutoff = df[SCORE_COL].where(position == top_n).groupby(df[GROUP_COL]).transform("max")
    df["margin"] = df[SCORE_COL] - cutoff
    df["in_top"] = df["margin"] >= 0
    return df


def smoothed_density(values: np.ndarray, grid: np.ndarray, bw_scale: float) -> np.ndarray:
    """Gaussian KDE (Scott bandwidth) via binned convolution; area-normalized."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.zeros_like(grid)
    std = float(np.std(values)) or 1.0
    bandwidth = 1.06 * std * n ** (-1 / 5) * bw_scale

    step = grid[1] - grid[0]
    edges = np.concatenate([grid - step / 2, [grid[-1] + step / 2]])
    counts, _ = np.histogram(values, bins=edges)

    sigma_bins = max(bandwidth / step, 0.5)
    half = int(np.ceil(4 * sigma_bins))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts.astype(float), kernel, mode="same")
    return smoothed / (n * step)


def _split_at_zero(grid: np.ndarray, density: np.ndarray) -> tuple[tuple, tuple]:
    zero = float(np.interp(0.0, grid, density))
    left = grid < 0
    right = grid > 0
    x_left = np.append(grid[left], 0.0)
    y_left = np.append(density[left], zero)
    x_right = np.insert(grid[right], 0, 0.0)
    y_right = np.insert(density[right], 0, zero)
    return (x_left, y_left), (x_right, y_right)


def plot(df: pd.DataFrame, top_pct: float, output: Path, bw_scale: float, x_range: tuple | None) -> None:
    _configure_figure_fonts()

    pos = df.loc[df[RELEVANCE_COL] == 1, "margin"].to_numpy()
    neg = df.loc[df[RELEVANCE_COL] == 0, "margin"].to_numpy()

    if x_range is None:
        lo = min(np.percentile(neg, 0.5), np.percentile(pos, 0.5))
        hi = max(np.percentile(neg, 99.95), np.percentile(pos, 99.5))
        pad = 0.04 * (hi - lo)
        x_range = (lo - pad, hi + pad)
    grid = np.linspace(x_range[0], x_range[1], GRID_POINTS)

    pos_density = smoothed_density(pos, grid, bw_scale)
    neg_density = smoothed_density(neg, grid, bw_scale)
    (pos_xl, pos_yl), (pos_xr, pos_yr) = _split_at_zero(grid, pos_density)
    (neg_xl, neg_yl), (neg_xr, neg_yr) = _split_at_zero(grid, neg_density)

    fig, ax = plt.subplots(figsize=(12.0, 6.0))

    ax.fill_between(neg_xl, neg_yl, color=NEG_COLOR, linewidth=0, zorder=1)
    ax.fill_between(pos_xr, pos_yr, color=POS_COLOR, linewidth=0, zorder=1)
    ax.fill_between(
        pos_xl, pos_yl, facecolor="white", edgecolor=POS_COLOR, hatch=FN_HATCH, linewidth=0, zorder=2,
    )
    ax.fill_between(
        neg_xr, neg_yr, facecolor="white", edgecolor=NEG_COLOR, hatch=FP_HATCH, linewidth=0, zorder=2,
    )
    ax.plot(grid, neg_density, color="black", linewidth=1.2, zorder=3)
    ax.plot(grid, pos_density, color="black", linewidth=1.2, zorder=3)
    ax.axvline(0.0, color=THRESHOLD_COLOR, linewidth=2.2, zorder=4)

    n_tp = int((pos >= 0).sum())
    n_fn = int((pos < 0).sum())
    n_fp = int((neg >= 0).sum())
    n_tn = int((neg < 0).sum())
    pct_text = f"{top_pct:g}%"

    handles = [
        Patch(facecolor=POS_COLOR, edgecolor="black", label=f"True positive\n(n={n_tp:,})"),
        Patch(facecolor=NEG_COLOR, edgecolor="black", label=f"True negative\n(n={n_tn:,})"),
        Patch(facecolor="white", edgecolor=NEG_COLOR, hatch=FP_HATCH, label=f"False positive\n(n={n_fp:,})"),
        Patch(facecolor="white", edgecolor=POS_COLOR, hatch=FN_HATCH, label=f"False negative\n(n={n_fn:,})"),
        Line2D([0], [0], color=THRESHOLD_COLOR, linewidth=2.2, label=f"Top {pct_text} cutoff"),
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        handlelength=3.2, handleheight=2.4, labelspacing=1.1,
    )

    ax.set_xlim(*x_range)
    ax.set_ylim(0, max(pos_density.max(), neg_density.max()) * 1.05)
    ax.set_yticks([])
    ax.set_xlabel(f"pred_score relative to each query's top-{pct_text} cutoff", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

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
        help="Output path (PNG + SVG are written). Default: <input stem>_score_distribution.png",
    )
    p.add_argument("--top-pct", type=float, default=1.0, help="Top percent per query group (default 1).")
    p.add_argument(
        "--bw-scale",
        type=float,
        default=1.0,
        help="Multiplier on the Scott KDE bandwidth; >1 smoother, <1 sharper (default 1).",
    )
    p.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Fix the x-axis range instead of using data percentiles.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_score_distribution.png")

    df = load_predictions(args.input)
    df = add_cutoff_margin(df, args.top_pct)

    positive = df[RELEVANCE_COL] == 1
    print(
        f"{df[GROUP_COL].nunique()} query groups, {len(df)} residues | "
        f"TP={int((positive & df['in_top']).sum())}, FN={int((positive & ~df['in_top']).sum())}, "
        f"FP={int((~positive & df['in_top']).sum())}, TN={int((~positive & ~df['in_top']).sum())}"
    )

    plot(df, args.top_pct, output, args.bw_scale, tuple(args.x_range) if args.x_range else None)


if __name__ == "__main__":
    main()
