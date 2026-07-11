#!/usr/bin/env python3
"""Create a four-panel residue-type specificity figure (Panels A–D).

Panel A: Top-10% shortlist composition (true target × residue type in shortlist).
Panel B: Top-10% enrichment vs candidate pool (top_pct_frac / base_frac per residue).
Panel C: Rank-1 type prediction (true target × rank-1 predicted type).
Panel D: Per-residue test accuracy table from summary JSON.

Usage:
    python helper/Create_Residue_Specificity_Figure.py \
        --residue-composition residue_composition_detail.csv \
        --summary normal_summary.json \
        [--output residue_specificity.png]
"""

from __future__ import annotations

import argparse
import json
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
COLORBAR_PAD_IN = 0.12
COLORBAR_WIDTH_IN = 0.22
COLORBAR_LABEL_IN = 0.55
# Gap between left-column colorbar and right-column panels (room for B y-labels).
COLUMN_GAP_IN = 2.15
ROW_GAP_IN = 0.75
ROW_XLABEL_CLEARANCE_IN = 0.55
LEFT_MARGIN_IN = 1.15
BOTTOM_MARGIN_IN = 1.45
TOP_MARGIN_IN = 0.65
RIGHT_MARGIN_IN = 0.25

TABLE_HEADERS = (
    "Residue",
    "Groups",
    "Labels",
    "Hit @\ntop 10%",
    "Median\nrank",
    "Median SS\nred.",
    "SS red.\n(hit)",
    "SS red.\n(miss)",
)
# Relative column widths (must sum to 1.0).
TABLE_COL_WIDTHS = (0.09, 0.09, 0.09, 0.12, 0.10, 0.14, 0.15, 0.15)


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


def _accuracy_table_row(
    *,
    residue: str,
    n_query_groups: int,
    n_labels: int,
    hit_rate_at_top_pct: float,
    median_rank: float,
    median_ss_reduction: float,
    median_ss_reduction_when_hit: float,
    median_ss_reduction_when_miss: float | None,
) -> dict[str, object]:
    return {
        "Residue": residue,
        "query groups": n_query_groups,
        "number of labels": n_labels,
        "Top 10% Hit Rate": hit_rate_at_top_pct,
        "Median Rank": median_rank,
        "Median SS Reduction": median_ss_reduction,
        "Median SS Reduction (hit)": median_ss_reduction_when_hit,
        "Median SS Reduction (miss)": median_ss_reduction_when_miss,
    }


def load_per_residue_accuracy(summary_path: Path) -> pd.DataFrame:
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("test_per_residue_accuracy")
    if not records:
        raise ValueError(f"{summary_path}: missing test_per_residue_accuracy")

    overall = payload.get("test_overall")
    if not overall:
        raise ValueError(f"{summary_path}: missing test_overall")

    by_type = {
        str(row["target_residue_type"]).strip().upper(): row for row in records
    }
    rows: list[dict[str, object]] = []
    for res in RESIDUE_TYPES:
        if res not in by_type:
            continue
        row = by_type[res]
        if "label_hit_rate_at_top_pct" not in row:
            raise ValueError(
                f"{summary_path}: missing label_hit_rate_at_top_pct for {res}. "
                "Re-run Training_Cov_Screen.py to regenerate the summary JSON."
            )
        miss_ss = row.get("median_ss_reduction_when_miss")
        rows.append(
            _accuracy_table_row(
                residue=RESIDUE_LABELS[res],
                n_query_groups=int(row["n_query_groups"]),
                n_labels=int(row["n_labels"]),
                hit_rate_at_top_pct=float(row["label_hit_rate_at_top_pct"]),
                median_rank=float(row["median_rank"]),
                median_ss_reduction=float(row["median_ss_reduction"]),
                median_ss_reduction_when_hit=float(row["median_ss_reduction_when_hit"]),
                median_ss_reduction_when_miss=(
                    float(miss_ss)
                    if miss_ss is not None and np.isfinite(float(miss_ss))
                    else None
                ),
            )
        )
    if not rows:
        raise ValueError(f"{summary_path}: no recognized residue types in test_per_residue_accuracy")

    residue_df = pd.DataFrame(rows).sort_values(
        "Top 10% Hit Rate",
        ascending=False,
        kind="stable",
    )

    miss_ss = overall.get("median_ss_reduction_when_miss")
    overall_row = _accuracy_table_row(
        residue="Overall",
        n_query_groups=int(overall["n_query_groups"]),
        n_labels=int(overall["n_labels"]),
        hit_rate_at_top_pct=float(overall["hit_at_top_pct"]),
        median_rank=float(overall["median_rank"]),
        median_ss_reduction=float(overall["median_search_space_reduction"]),
        median_ss_reduction_when_hit=float(
            overall["median_ss_reduction_when_hit_at_top_pct"]
        ),
        median_ss_reduction_when_miss=(
            float(miss_ss)
            if miss_ss is not None and np.isfinite(float(miss_ss))
            else None
        ),
    )
    return pd.concat(
        [residue_df, pd.DataFrame([overall_row])],
        ignore_index=True,
    )


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
    show_row_labels: bool,
) -> None:
    n = len(RESIDUE_TYPES)
    ax.set_xticks(range(n))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n))
    if show_row_labels:
        ax.set_yticklabels(row_labels, fontsize=8)
        if panel_label in ("A", "C"):
            ax.set_ylabel("True target type")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    _panel_title(ax, panel_label, title)


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
            "Rank-1 predicted type",
            "fraction",
        ),
    )

    n_types = len(RESIDUE_TYPES)
    heatmap_extent = (-0.5, n_types - 0.5, n_types - 0.5, -0.5)

    for ax, matrix, cmap, norm, panel_label, title, xlabel, mode in panels:
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
            show_row_labels=panel_label in ("A", "B", "C"),
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
                heat_pos.x1 + COLORBAR_PAD_IN / fig.get_figwidth(),
                heat_pos.y0 + heat_pos.height * 0.07,
                COLORBAR_WIDTH_IN / fig.get_figwidth(),
                heat_pos.height * 0.86,
            ]
        )
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)


def _format_table_cell(column: str, value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    if column == "Residue":
        return str(value)
    if column == "Top 10% Hit Rate":
        return f"{float(value) * 100:.1f}%"
    if column == "Median Rank":
        number = float(value)
        return f"{number:.0f}" if number.is_integer() else f"{number:.1f}"
    if column.startswith("Median SS"):
        return f"{float(value):.3f}"
    if column in ("query groups", "number of labels"):
        return str(int(value))
    return str(value)


def plot_residue_accuracy_table(
    ax: plt.Axes,
    accuracy_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel_title(ax, "D", "Per-residue test accuracy")

    display_cols = list(accuracy_df.columns)
    cell_text = [
        [_format_table_cell(col, row[col]) for col in display_cols]
        for _, row in accuracy_df.iterrows()
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=TABLE_HEADERS,
        colWidths=TABLE_COL_WIDTHS,
        loc="upper center",
        bbox=[0.0, 0.0, 1.0, 0.84],
        cellLoc="center",
    )
    table.auto_set_font_size(False)

    n_data_rows = len(accuracy_df)
    body_height = min(0.11, 0.78 / max(n_data_rows, 1))
    header_height = min(0.26, 0.22 + 0.01 * max(0, 7 - n_data_rows))

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_height(header_height)
            cell.set_text_props(fontweight="bold", fontsize=6.0, ha="center", va="center")
            continue

        is_overall = str(accuracy_df.iloc[row_idx - 1]["Residue"]) == "Overall"
        cell.set_height(body_height)
        if is_overall:
            cell.set_facecolor("#eef1f5")
            weight = "bold"
        else:
            cell.set_facecolor("white")
            weight = "bold" if col_idx == 0 else "normal"
        cell.set_text_props(fontsize=7, fontweight=weight, ha="center", va="center")


def _layout_metrics() -> tuple[float, float, float]:
    """Return heatmap side, column stride, and right-column content width."""
    n_types = len(RESIDUE_TYPES)
    heatmap_in = n_types * CELL_SIZE_IN
    panel_content_in = (
        heatmap_in + COLORBAR_PAD_IN + COLORBAR_WIDTH_IN + COLORBAR_LABEL_IN
    )
    column_stride_in = panel_content_in + COLUMN_GAP_IN
    return heatmap_in, column_stride_in, panel_content_in


def _row_bottom_in(row: int, heatmap_in: float) -> float:
    if row == 0:
        return BOTTOM_MARGIN_IN + heatmap_in + ROW_GAP_IN + ROW_XLABEL_CLEARANCE_IN
    return BOTTOM_MARGIN_IN


def _add_square_heatmap_axes(fig: plt.Figure, row: int, col: int) -> plt.Axes:
    """Place a square 6×6 heatmap axes; width matches height (square cells)."""
    heatmap_in, column_stride_in, _ = _layout_metrics()
    left_in = LEFT_MARGIN_IN + col * column_stride_in
    bottom_in = _row_bottom_in(row, heatmap_in)
    fig_w, fig_h = fig.get_size_inches()
    return fig.add_axes(
        [
            left_in / fig_w,
            bottom_in / fig_h,
            heatmap_in / fig_w,
            heatmap_in / fig_h,
        ]
    )


def _add_table_axes(fig: plt.Figure) -> plt.Axes:
    """Place panel D table in the bottom-right grid cell."""
    heatmap_in, column_stride_in, panel_content_in = _layout_metrics()
    left_in = LEFT_MARGIN_IN + column_stride_in
    bottom_in = _row_bottom_in(1, heatmap_in)
    fig_w, fig_h = fig.get_size_inches()
    return fig.add_axes(
        [
            left_in / fig_w,
            bottom_in / fig_h,
            panel_content_in / fig_w,
            heatmap_in / fig_h,
        ]
    )


def create_figure(
    *,
    residue_composition: Path,
    summary_path: Path,
    output_path: Path,
) -> None:
    composition_df = load_residue_composition(residue_composition)
    accuracy_df = load_per_residue_accuracy(summary_path)

    heatmap_in, column_stride_in, panel_content_in = _layout_metrics()
    fig_w = LEFT_MARGIN_IN + column_stride_in + panel_content_in + RIGHT_MARGIN_IN
    fig_h = (
        BOTTOM_MARGIN_IN
        + 2 * heatmap_in
        + ROW_GAP_IN
        + ROW_XLABEL_CLEARANCE_IN
        + TOP_MARGIN_IN
    )

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = _add_square_heatmap_axes(fig, row=0, col=0)
    ax_b = _add_square_heatmap_axes(fig, row=0, col=1)
    ax_c = _add_square_heatmap_axes(fig, row=1, col=0)
    ax_d = _add_table_axes(fig)

    plot_residue_specificity_panels(fig, ax_a, ax_b, ax_c, composition_df)
    plot_residue_accuracy_table(ax_d, accuracy_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved figure -> {output_path}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    analysis_dir = here.parent

    p = argparse.ArgumentParser(
        description=(
            "Create residue-type specificity figure "
            "(composition, enrichment, rank-1, per-residue accuracy table)."
        )
    )
    p.add_argument(
        "--residue-composition",
        type=Path,
        default=analysis_dir / "residue_composition_detail.csv",
        help="Per-query-group residue composition CSV from Training_Cov_Screen.py.",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=analysis_dir / "normal_summary.json",
        help="Model summary JSON with test_per_residue_accuracy (default: normal_summary.json).",
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
    if not args.summary.is_file():
        raise FileNotFoundError(f"Required input not found: {args.summary}")

    create_figure(
        residue_composition=args.residue_composition,
        summary_path=args.summary,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
