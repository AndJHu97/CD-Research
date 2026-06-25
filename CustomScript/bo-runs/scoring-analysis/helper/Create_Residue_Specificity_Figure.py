#!/usr/bin/env python3
"""Create a three-panel residue-type specificity figure (Panels A–C).

Panel A: Top-10% shortlist composition (true target × residue type in shortlist).
Panel B: Top-10% enrichment vs candidate pool (top_pct_frac / base_frac per residue).
Panel C: Rank-1 type prediction (true target × rank-1 predicted type).

Usage:
    python helper/Create_Residue_Specificity_Figure.py \
        --residue-composition residue_composition_detail.csv \
        [--output residue_specificity.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESIDUE_TYPES = ("CYS", "HIS", "LYS", "SER", "THR", "TYR")
RESIDUE_LABELS = {
    "CYS": "Cys",
    "HIS": "His",
    "LYS": "Lys",
    "SER": "Ser",
    "THR": "Thr",
    "TYR": "Tyr",
}
# Square heatmap cells: side length in inches (6×6 grid per panel).
CELL_SIZE_IN = 0.58
COLORBAR_WIDTH_IN = 0.22
PANEL_GAP_IN = 0.55
LEFT_MARGIN_IN = 1.05
BOTTOM_MARGIN_IN = 1.35
TOP_MARGIN_IN = 0.55


def _panel_title(ax: plt.Axes, panel_label: str, title: str) -> None:
    ax.set_title(f"{panel_label}  {title}", loc="left", fontweight="bold", fontsize=11)


def load_residue_composition(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "query_group" in df.columns:
        df = df[~df["query_group"].astype(str).str.startswith("__")].copy()
    required = {"query_group", "label_type", "rank1_type", "n_candidates"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}")

    count_cols = [f"n_{res}" for res in RESIDUE_TYPES]
    missing_counts = [c for c in count_cols if c not in df.columns]
    if missing_counts:
        raise ValueError(
            f"{csv_path}: missing candidate-pool count columns {missing_counts}. "
            "Re-run Training_Cov_Screen.py to regenerate residue_composition_detail.csv."
        )

    frac_cols = [f"top_pct_frac_{res}" for res in RESIDUE_TYPES]
    missing_frac = [c for c in frac_cols if c not in df.columns]
    if missing_frac:
        raise ValueError(f"{csv_path}: missing top-10% fraction columns {missing_frac}")

    df = df.copy()
    df["label_type"] = df["label_type"].astype(str).str.strip().str.upper()
    df["rank1_type"] = df["rank1_type"].astype(str).str.strip().str.upper()
    df["n_candidates"] = df["n_candidates"].astype(float)

    for res in RESIDUE_TYPES:
        base_frac = df[f"n_{res}"].astype(float) / df["n_candidates"]
        df[f"base_frac_{res}"] = base_frac
        top_frac = df[f"top_pct_frac_{res}"].astype(float)
        df[f"enrichment_{res}"] = np.where(base_frac > 0, top_frac / base_frac, np.nan)

    return df


def _mean_matrix_by_label(
    df: pd.DataFrame,
    value_cols: list[str],
) -> np.ndarray:
    matrix = np.full((len(RESIDUE_TYPES), len(RESIDUE_TYPES)), np.nan, dtype=float)
    for i, label in enumerate(RESIDUE_TYPES):
        subset = df.loc[df["label_type"] == label]
        if subset.empty:
            continue
        for j, col in enumerate(value_cols):
            matrix[i, j] = float(subset[col].mean())
    return matrix


def compute_top_pct_composition_matrix(df: pd.DataFrame) -> np.ndarray:
    """Mean top-10% shortlist composition per true target type."""
    cols = [f"top_pct_frac_{res}" for res in RESIDUE_TYPES]
    return _mean_matrix_by_label(df, cols)


def compute_enrichment_matrix(df: pd.DataFrame) -> np.ndarray:
    """Mean top-10% / candidate-pool enrichment per true target type."""
    cols = [f"enrichment_{res}" for res in RESIDUE_TYPES]
    return _mean_matrix_by_label(df, cols)


def compute_rank1_confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    """Row-normalized rank-1 prediction rates per true target type."""
    matrix = np.zeros((len(RESIDUE_TYPES), len(RESIDUE_TYPES)), dtype=float)
    for i, label in enumerate(RESIDUE_TYPES):
        subset = df.loc[df["label_type"] == label]
        if subset.empty:
            continue
        counts = subset["rank1_type"].value_counts()
        n = len(subset)
        for j, pred in enumerate(RESIDUE_TYPES):
            matrix[i, j] = float(counts.get(pred, 0) / n)
    return matrix


def _annotate_fraction_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    vmax: float,
    as_percent: bool,
) -> None:
    n = len(RESIDUE_TYPES)
    display = matrix * 100.0 if as_percent else matrix
    for i in range(n):
        for j in range(n):
            val = display[i, j]
            if not np.isfinite(val):
                continue
            text_color = "white" if val > (vmax * 0.55) else "#222222"
            weight = "bold" if i == j else "normal"
            if as_percent:
                label = "0" if val < 0.5 else f"{val:.0f}"
            else:
                label = f"{val:.2f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                fontweight=weight,
            )


def _annotate_enrichment_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    norm: mcolors.Normalize,
) -> None:
    n = len(RESIDUE_TYPES)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            mapped = norm(val)
            text_color = "white" if mapped > 0.62 or mapped < 0.18 else "#222222"
            weight = "bold" if i == j else "normal"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                fontweight=weight,
            )


def _style_heatmap_axes(
    ax: plt.Axes,
    *,
    row_labels: list[str],
    col_labels: list[str],
    xlabel: str,
    panel_label: str,
    title: str,
    subtitle: str,
) -> None:
    n = len(RESIDUE_TYPES)
    ax.set_xticks(range(n))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylabel("True target type")
    ax.set_xlabel(xlabel)
    _panel_title(ax, panel_label, title)
    ax.text(
        0.0,
        -0.18,
        subtitle,
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
    )


def plot_residue_specificity_panels(
    fig: plt.Figure,
    ax_a: plt.Axes,
    ax_b: plt.Axes,
    ax_c: plt.Axes,
    composition_df: pd.DataFrame,
) -> None:
    top_pct_matrix = compute_top_pct_composition_matrix(composition_df)
    enrichment_matrix = compute_enrichment_matrix(composition_df)
    rank1_matrix = compute_rank1_confusion_matrix(composition_df)

    row_counts = composition_df.groupby("label_type").size()
    row_labels = [
        f"{RESIDUE_LABELS[res]}\n(n={int(row_counts.get(res, 0))})"
        for res in RESIDUE_TYPES
    ]
    col_labels = [RESIDUE_LABELS[res] for res in RESIDUE_TYPES]

    frac_vmax = max(top_pct_matrix.max(), rank1_matrix.max()) * 100.0
    frac_vmax = max(frac_vmax, 1.0)
    frac_cmap = plt.cm.Blues
    frac_norm = mcolors.Normalize(vmin=0.0, vmax=frac_vmax)

    finite_enrich = enrichment_matrix[np.isfinite(enrichment_matrix)]
    if finite_enrich.size:
        enrich_lim = float(np.nanpercentile(finite_enrich, 95))
        enrich_lim = max(enrich_lim, 1.25)
    else:
        enrich_lim = 2.0
    enrich_norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=enrich_lim)
    enrich_cmap = plt.cm.RdYlBu_r

    panels = (
        (
            ax_a,
            top_pct_matrix,
            frac_cmap,
            frac_norm,
            "A",
            "Top-10% shortlist composition",
            "Mean per-protein top-10% type fractions, averaged within true-target group",
            "Residue type in shortlist",
            "fraction",
        ),
        (
            ax_b,
            enrichment_matrix,
            enrich_cmap,
            enrich_norm,
            "B",
            "Top-10% enrichment",
            "Mean per-protein enrichment (top-10% fraction / candidate-pool fraction)",
            "Residue type",
            "enrichment",
        ),
        (
            ax_c,
            rank1_matrix,
            frac_cmap,
            frac_norm,
            "C",
            "Rank-1 type prediction",
            "Fraction of query groups per true-target type with each rank-1 type",
            "Rank-1 predicted type",
            "fraction",
        ),
    )

    n_types = len(RESIDUE_TYPES)
    heatmap_extent = (-0.5, n_types - 0.5, n_types - 0.5, -0.5)

    for ax, matrix, cmap, norm, panel_label, title, subtitle, xlabel, mode in panels:
        display = matrix * 100.0 if mode == "fraction" else matrix
        ax.imshow(
            display,
            cmap=cmap,
            norm=norm,
            aspect="equal",
            interpolation="nearest",
            origin="upper",
            extent=heatmap_extent,
        )
        ax.set_xlim(heatmap_extent[0], heatmap_extent[1])
        ax.set_ylim(heatmap_extent[2], heatmap_extent[3])
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        if mode == "enrichment":
            _annotate_enrichment_heatmap(ax, matrix, norm=norm)
        else:
            _annotate_fraction_heatmap(ax, matrix, vmax=frac_vmax, as_percent=True)
        _style_heatmap_axes(
            ax,
            row_labels=row_labels,
            col_labels=col_labels,
            xlabel=xlabel,
            panel_label=panel_label,
            title=title,
            subtitle=subtitle,
        )

    colorbars = (
        (ax_a, frac_cmap, frac_norm, "Row fraction (%)"),
        (ax_b, enrich_cmap, enrich_norm, "Enrichment (×)"),
        (ax_c, frac_cmap, frac_norm, "Row fraction (%)"),
    )
    for ax, cmap, norm, label in colorbars:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        heat_pos = ax.get_position()
        cax = fig.add_axes(
            [
                heat_pos.x1 + 0.012,
                heat_pos.y0 + heat_pos.height * 0.07,
                COLORBAR_WIDTH_IN / fig.get_figwidth(),
                heat_pos.height * 0.86,
            ]
        )
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)


def _add_square_heatmap_axes(fig: plt.Figure, panel_index: int) -> plt.Axes:
    """Place a square 6×6 heatmap axes; width matches height (square cells)."""
    n_types = len(RESIDUE_TYPES)
    heatmap_in = n_types * CELL_SIZE_IN
    panel_stride_in = heatmap_in + COLORBAR_WIDTH_IN + PANEL_GAP_IN
    left_in = LEFT_MARGIN_IN + panel_index * panel_stride_in
    fig_w, fig_h = fig.get_size_inches()
    ax = fig.add_axes(
        [
            left_in / fig_w,
            BOTTOM_MARGIN_IN / fig_h,
            heatmap_in / fig_w,
            heatmap_in / fig_h,
        ]
    )
    return ax


def create_figure(
    *,
    residue_composition: Path,
    output_path: Path,
) -> None:
    composition_df = load_residue_composition(residue_composition)

    n_types = len(RESIDUE_TYPES)
    heatmap_in = n_types * CELL_SIZE_IN
    panel_stride_in = heatmap_in + COLORBAR_WIDTH_IN + PANEL_GAP_IN
    fig_w = LEFT_MARGIN_IN + 3 * panel_stride_in + 0.25
    fig_h = BOTTOM_MARGIN_IN + heatmap_in + TOP_MARGIN_IN

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = _add_square_heatmap_axes(fig, 0)
    ax_b = _add_square_heatmap_axes(fig, 1)
    ax_c = _add_square_heatmap_axes(fig, 2)

    plot_residue_specificity_panels(fig, ax_a, ax_b, ax_c, composition_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved figure -> {output_path}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    analysis_dir = here.parent

    p = argparse.ArgumentParser(
        description="Create residue-type specificity figure (composition, enrichment, rank-1)."
    )
    p.add_argument(
        "--residue-composition",
        type=Path,
        default=analysis_dir / "residue_composition_detail.csv",
        help="Per-query-group residue composition CSV from Training_Cov_Screen.py.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=analysis_dir / "residue_specificity.png",
        help="Output figure path (default: scoring-analysis/residue_specificity.png).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.residue_composition.is_file():
        raise FileNotFoundError(f"Required input not found: {args.residue_composition}")

    create_figure(
        residue_composition=args.residue_composition,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
