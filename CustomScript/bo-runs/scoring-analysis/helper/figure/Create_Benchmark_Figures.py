#!/usr/bin/env python3
"""Create a C207 benchmark comparison figure (Panels A–B + companion table).

Panel A: CovSite blind site recovery vs perfect-match random baseline (Hit @ top-X% and Hit @ K).
Panel B: Literature pose-prediction success rates for covalent docking tools.

With --direct-comparison: single portrait panel comparing ranked detection, random baseline,
and mean structural screening performance (methodology comparison).

Usage:
    python Create_Benchmark_Figures.py \\
        --summary screen_summary.json \\
        [--scores screen_scores.csv] \\
        [--output benchmark_comparison.png]

    python Create_Benchmark_Figures.py \\
        --summary screen_summary.json \\
        --scores screen_scores.csv \\
        --direct-comparison \\
        --noncovalent-summary noncovalent_only_screen_summary.json \\
        --output benchmark_direct_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
        }
    )

BENCHMARK_SET_LABEL = "C207"

COVSITE_COLOR = "#2C7A4B"
RANDOM_COLOR = "#B8B8B8"
DOCKING_COLORS = ["#8172B3", "#DD8452", "#C44E52", "#4C72B0", "#937860", "#64B5CD"]

# Direct-comparison panel palette.
DIRECT_COVSITE_COLOR = "#2a78d6"
DIRECT_NONCOVALENT_COLOR = "#2E8B6E"  # non-covalent-only ranked model
DIRECT_RANDOM_COLOR = "#73726c"
DIRECT_DOCKING_COLOR = "#e34948"
MUTED_TEXT_COLOR = "#666666"

# Literature covalent docking: (tool, % success at <2 Å RMSD, benchmark N).
DEFAULT_DOCKING_BENCHMARKS: list[tuple[str, float, int]] = [
    ("AutoDock4", 55.0, 207),
    ("CovDock", 59.0, 207),
    ("FITTED", 56.0, 175),
    ("GOLD", 53.0, 207),
    ("ICM-Pro", 62.0, 207),
    ("MOE", 37.0, 207),
]


Y_AXIS_MAX = 100.0


def _panel_title(ax: plt.Axes, panel_label: str, title: str = "", *, pad: float = 10) -> None:
    _ = title
    ax.set_title(
        panel_label,
        loc="left",
        fontweight="bold",
        fontsize=11,
        pad=pad,
    )


def random_baseline_hit_at_k(n_residues: int | float, k: int) -> float:
    """Chance of Hit@K under a uniformly random rank over 1..n."""
    n = int(n_residues)
    if n <= 0:
        return float("nan")
    return float(min(k, n) / n)


def random_baseline_hit_at_top_pct(n_residues: int | float, p_frac: float) -> float:
    """Chance of Hit@top-X% under uniform random rank (floor-based qualifying ranks)."""
    n = int(n_residues)
    if n <= 1:
        return 1.0
    qualifying_ranks = np.floor(p_frac * (n - 1) + 1)
    return float(qualifying_ranks / n)


LABEL_KEY_COLS = ("Name", "Residue", "ResNum", "Chain")


def _normalize_resnum(val: object) -> str:
    """Coerce ResNum so 45 and 45.0 compare equal (matches Training_Cov_Screen)."""
    if pd.isna(val):
        return ""
    try:
        f = float(val)  # type: ignore[arg-type]
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def make_label_key(name: object, residue: object, resnum: object, chain: object) -> tuple[str, str, str, str]:
    return (
        str(name).strip().upper(),
        str(residue).strip().upper(),
        _normalize_resnum(resnum),
        str(chain).strip().upper(),
    )


def _group_query_groups_by_label(df: pd.DataFrame) -> dict[tuple[str, str, str, str], list[int]]:
    label_to_ns: dict[tuple[str, str, str, str], list[int]] = {}
    for _, row in df.iterrows():
        key = make_label_key(row["Name"], row["Residue"], row["ResNum"], row["Chain"])
        label_to_ns.setdefault(key, []).append(int(row["n_residues"]))
    return label_to_ns


def load_label_to_qg_pool_sizes(
    *,
    scores_path: Path | None,
    query_groups_path: Path | None,
) -> dict[tuple[str, str, str, str], list[int]]:
    """
    Map each label site -> list of n_residues (one per matched warhead query group).

    Requires label identity columns plus per-query-group pool size.
    """
    if query_groups_path is not None:
        df = pd.read_csv(query_groups_path)
        missing = {"n_residues", *LABEL_KEY_COLS} - set(df.columns)
        if missing:
            raise ValueError(
                f"{query_groups_path}: missing column(s) {missing}. "
                "Re-export with Cov_Screen.py --export-query-groups, or pass --scores."
            )
        return _group_query_groups_by_label(df)

    if scores_path is None or not scores_path.is_file():
        raise FileNotFoundError(
            "Need label-to-query-group mapping: pass --scores (candidate CSV) "
            "or --query-groups (with Name, Residue, ResNum, Chain, n_residues)."
        )

    scores = pd.read_csv(scores_path)
    for col in ("query_group", "relevance", *LABEL_KEY_COLS):
        if col not in scores.columns:
            raise ValueError(f"{scores_path}: missing '{col}' column")

    rows: list[dict[str, object]] = []
    for qg, grp in scores.groupby("query_group", sort=True):
        targets = grp.loc[grp["relevance"] == 1]
        if targets.empty:
            continue
        tgt = targets.iloc[0]
        rows.append({
            "query_group": qg,
            "n_residues": int(len(grp)),
            "Name": tgt["Name"],
            "Residue": tgt["Residue"],
            "ResNum": tgt["ResNum"],
            "Chain": tgt["Chain"],
        })

    if not rows:
        raise ValueError(f"{scores_path}: no query groups with relevance==1 targets found.")

    return _group_query_groups_by_label(pd.DataFrame(rows))


def label_random_baseline(
    query_groups_n_residues: list[int],
    *,
    mode: str,
    k: int | None = None,
    p_frac: float | None = None,
) -> float:
    """
    Perfect-match label baseline: product of per-query-group random hit probs
    (all matched warheads must hit by chance).
    """
    prob = 1.0
    for n in query_groups_n_residues:
        if mode == "top_pct":
            if p_frac is None:
                raise ValueError("p_frac required for mode='top_pct'")
            prob *= random_baseline_hit_at_top_pct(n, p_frac)
        elif mode == "top_k":
            if k is None:
                raise ValueError("k required for mode='top_k'")
            prob *= random_baseline_hit_at_k(n, k)
        else:
            raise ValueError(f"mode must be 'top_pct' or 'top_k', got {mode!r}")
    return float(prob)


def compute_mean_label_random_baseline(
    label_to_qg_ns: dict[tuple[str, str, str, str], list[int]],
    *,
    mode: str,
    k: int | None = None,
    p_frac: float | None = None,
) -> float:
    if not label_to_qg_ns:
        raise ValueError("No labels found for random-baseline computation.")
    baselines = [
        label_random_baseline(ns, mode=mode, k=k, p_frac=p_frac)
        for ns in label_to_qg_ns.values()
    ]
    return float(np.mean(baselines))


def load_covsite_cys_metrics(summary_path: Path) -> dict[str, float | int]:
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("per_residue_accuracy")
    if not records:
        raise ValueError(f"{summary_path}: missing per_residue_accuracy")

    cys_rows = [
        row for row in records
        if str(row.get("target_residue_type", "")).strip().upper() == "CYS"
    ]
    if not cys_rows:
        raise ValueError(f"{summary_path}: no CYS row in per_residue_accuracy")

    row = cys_rows[0]
    for key in (
        "label_hit_rate_at_k",
        "label_hit_rate_at_top_pct",
        "median_rank",
        "median_ss_reduction",
        "n_query_groups",
        "n_labels",
    ):
        if key not in row:
            raise ValueError(f"{summary_path}: missing per_residue_accuracy CYS.{key}")

    k = payload.get("k")
    if k is None:
        raise ValueError(f"{summary_path}: missing top-level 'k'")

    top_pct = payload.get("top_pct_threshold", 10.0)
    perfect_match = payload.get("perfect_match", True)
    return {
        "label_hit_rate_at_k": float(row["label_hit_rate_at_k"]),
        "label_hit_rate_at_top_pct": float(row["label_hit_rate_at_top_pct"]),
        "median_rank": float(row["median_rank"]),
        "median_ss_reduction": float(row["median_ss_reduction"]),
        "n_query_groups": int(row["n_query_groups"]),
        "n_labels": int(row["n_labels"]),
        "k": int(k),
        "top_pct_threshold": float(top_pct),
        "perfect_match": bool(perfect_match),
    }


def sort_docking_benchmarks_desc(
    benchmarks: list[tuple[str, float, int]],
) -> list[tuple[str, float, int]]:
    return sorted(benchmarks, key=lambda item: item[1], reverse=True)


def _annotate_bar(
    ax: plt.Axes,
    bar: mpatches.Rectangle,
    value_pct: float,
    n: int | None,
    *,
    fontsize: float = 7.5,
    y_offset: float = 1.5,
) -> None:
    label = f"{value_pct:.1f}%"
    if n is not None:
        label += f"\n(N={n:,})"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        linespacing=1.15,
    )


def plot_panel_a_blind_recovery(
    ax: plt.Axes,
    *,
    covsite_top_pct: float,
    random_top_pct: float,
    covsite_top_k: float,
    random_top_k: float,
    n_labels: int,
    k: int,
    top_pct_threshold: float,
) -> None:
    # Pair gap between top-% and top-K metric groups.
    x = np.array([0.0, 1.0, 2.35, 3.35])
    values = [covsite_top_pct, random_top_pct, covsite_top_k, random_top_k]
    colors = [COVSITE_COLOR, RANDOM_COLOR, COVSITE_COLOR, RANDOM_COLOR]
    n_ann = [n_labels, n_labels, n_labels, n_labels]
    tick_labels = [
        f"CovSite\n@ top {top_pct_threshold:g}%",
        f"Random\n@ top {top_pct_threshold:g}%",
        f"CovSite\n@ top {k}",
        f"Random\n@ top {k}",
    ]

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.72)
    for bar, val, n in zip(bars, values, n_ann):
        if val >= 85.0:
            # High bars: label inside to avoid overlapping the panel title.
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 2.5,
                f"{val:.1f}%\n(N={n:,})" if n is not None else f"{val:.1f}%",
                ha="center",
                va="top",
                fontsize=7.0,
                fontweight="bold",
                color="white",
                linespacing=1.15,
            )
        else:
            _annotate_bar(ax, bar, val, n, fontsize=7.0)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, Y_AXIS_MAX)
    _panel_title(ax, "A", pad=14)
   
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_group_bracket(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    y: float,
    label: str,
    *,
    color: str,
    tick: float = 2.5,
) -> None:
    """Draw a flat bracket above a span of bars with a centered label."""
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y - tick, y, y, y - tick],
        color=color,
        linewidth=1.3,
        solid_capstyle="butt",
        zorder=6,
        clip_on=False,
    )
    ax.text(
        0.5 * (x_left + x_right),
        y + 1.2,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=color,
        zorder=6,
        clip_on=False,
    )


def plot_direct_comparison(
    ax: plt.Axes,
    *,
    covsite_top_pct: float,
    random_top_pct: float,
    covsite_top_k: float,
    random_top_k: float,
    docking_benchmarks: list[tuple[str, float, int]],
    k: int,
    top_pct_threshold: float,
    noncovalent_top_pct: float | None = None,
    noncovalent_top_k: float | None = None,
) -> None:
    """Panel A: ranked models + random, then each structural tool as its own bar.

    Bar order when non-covalent metrics are provided:
      full @ top% → non-covalent @ top% → random @ top% →
      full @ top K → non-covalent @ top K → random @ top K →
      each docking tool (all red)
    """
    tools_sorted = sorted(docking_benchmarks, key=lambda item: item[1], reverse=True)
    tool_names = [name for name, _, _ in tools_sorted]
    tool_values = [float(pct) for _, pct, _ in tools_sorted]
    include_noncovalent = (
        noncovalent_top_pct is not None and noncovalent_top_k is not None
    )

    ranking_values: list[float] = []
    ranking_colors: list[str] = []
    ranking_labels: list[str] = []
    if include_noncovalent:
        ranking_values = [
            covsite_top_pct,
            float(noncovalent_top_pct),
            random_top_pct,
            covsite_top_k,
            float(noncovalent_top_k),
            random_top_k,
        ]
        ranking_colors = [
            DIRECT_COVSITE_COLOR,
            DIRECT_NONCOVALENT_COLOR,
            DIRECT_RANDOM_COLOR,
            DIRECT_COVSITE_COLOR,
            DIRECT_NONCOVALENT_COLOR,
            DIRECT_RANDOM_COLOR,
        ]
        ranking_labels = [
            f"CovSite\n@ top {top_pct_threshold:g}%",
            f"Non-covalent model\n@ top {top_pct_threshold:g}%",
            f"Random\n@ top {top_pct_threshold:g}%",
            f"CovSite\n@ top {k}",
            f"Non-covalent model\n@ top {k}",
            f"Random\n@ top {k}",
        ]
        # Gaps between metric triplets; larger gap before structural tools.
        ranking_x = np.array([0.0, 1.1, 2.2, 3.8, 4.9, 6.0])
    else:
        ranking_values = [
            covsite_top_pct,
            random_top_pct,
            covsite_top_k,
            random_top_k,
        ]
        ranking_colors = [
            DIRECT_COVSITE_COLOR,
            DIRECT_RANDOM_COLOR,
            DIRECT_COVSITE_COLOR,
            DIRECT_RANDOM_COLOR,
        ]
        ranking_labels = [
            f"CovSite\n@ top {top_pct_threshold:g}%",
            f"Random\n@ top {top_pct_threshold:g}%",
            f"CovSite\n@ top {k}",
            f"Random\n@ top {k}",
        ]
        ranking_x = np.array([0.0, 1.15, 2.6, 3.75])

    n_rank = len(ranking_x)
    n_tools = len(tool_values)
    tool_start = float(ranking_x[-1] + 1.7)
    tool_x = tool_start + np.arange(n_tools, dtype=float) * 1.05
    x = np.concatenate([ranking_x, tool_x])
    values = ranking_values + tool_values
    colors = ranking_colors + [DIRECT_DOCKING_COLOR] * n_tools
    tick_labels = ranking_labels + [name.replace("-", "-\n") if len(name) > 8 else name for name in tool_names]
    bar_width = 0.78

    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        width=bar_width,
        zorder=2,
    )

    for bar, val in zip(bars, values):
        cx = bar.get_x() + bar.get_width() / 2
        if val >= 85.0:
            ax.text(
                cx,
                bar.get_height() - 2.2,
                f"{val:.1f}%",
                ha="center",
                va="top",
                fontsize=9.0,
                fontweight="bold",
                color="white",
                zorder=5,
            )
        else:
            ax.text(
                cx,
                bar.get_height() + 1.6,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.0,
                fontweight="bold",
                zorder=5,
            )

    # Bracket spanning all structural-tool bars.
    if n_tools:
        bracket_y = max(tool_values) + 10.0
        _draw_group_bracket(
            ax,
            float(tool_x[0] - bar_width * 0.45),
            float(tool_x[-1] + bar_width * 0.45),
            bracket_y,
            "Structural tools",
            color=DIRECT_DOCKING_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    for label in ax.get_yticklabels():
        label.set_fontsize(9.5)
        label.set_fontweight("bold")
    y_top = max(105.0, (max(tool_values) if tool_values else 100.0) + 18.0)
    ax.set_ylim(0, y_top)
    ax.set_xlim(float(x.min()) - 0.7, float(x.max()) + 0.7)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.4)
    ax.tick_params(width=1.2)

    legend_handles = [
        mpatches.Patch(
            facecolor=DIRECT_COVSITE_COLOR,
            edgecolor="white",
            label=f"Hit rate @ top {top_pct_threshold:g}% (full model)",
        ),
    ]
    if include_noncovalent:
        legend_handles.append(
            mpatches.Patch(
                facecolor=DIRECT_NONCOVALENT_COLOR,
                edgecolor="white",
                label=f"Non-covalent model @ top {top_pct_threshold:g}% / top {k}",
            )
        )
    legend_handles.extend(
        [
            mpatches.Patch(
                facecolor=DIRECT_RANDOM_COLOR,
                edgecolor="white",
                label=(
                    f"Random site detection @ top {top_pct_threshold:g}% / top {k}"
                ),
            ),
            mpatches.Patch(
                facecolor=DIRECT_DOCKING_COLOR,
                edgecolor="white",
                label="% success <2Å RMSD (structural tools)",
            ),
        ]
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2 if not include_noncovalent else 3,
        frameon=False,
        fontsize=8.5,
        prop={"weight": "bold", "size": 8.5},
        handlelength=1.4,
        columnspacing=1.0,
    )
    _panel_title(ax, "A", pad=8)


def plot_panel_b_pose_accuracy(
    ax: plt.Axes,
    *,
    docking_benchmarks: list[tuple[str, float, int]],
) -> None:
    labels = [name for name, _, _ in docking_benchmarks]
    values = [pct for _, pct, _ in docking_benchmarks]
    n_values = [n for _, _, n in docking_benchmarks]
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        color=DOCKING_COLORS[: len(labels)],
        edgecolor="white",
        linewidth=0.8,
        width=0.72,
    )

    for bar, val, n in zip(bars, values, n_values):
        _annotate_bar(ax, bar, val, n, fontsize=7.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=22, ha="right")
    ax.set_ylabel("% success at <2 Å RMSD")
    ax.set_ylim(0, Y_AXIS_MAX)
    _panel_title(ax, "B")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_table_value(label: str, value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    label_l = label.lower()
    if (
        "hit rate" in label_l
        or "random baseline" in label_l
        or "non-covalent" in label_l
        or "search space" in label_l
        or "search-space" in label_l
    ):
        return f"{float(value):.1f}%"
    if label_l == "median rank":
        number = float(value)
        return f"{number:.0f}" if number.is_integer() else f"{number:.1f}"
    return str(value)


def build_companion_table_rows(
    *,
    covsite_top_pct: float,
    random_top_pct: float,
    covsite_top_k: float,
    random_top_k: float,
    median_rank: float,
    median_ss_reduction_pct: float,
    n_labels: int,
    n_query_groups: int,
    k: int,
    top_pct_threshold: float,
    noncovalent_top_pct: float | None = None,
    noncovalent_top_k: float | None = None,
) -> list[tuple[str, object]]:
    _ = (n_labels, n_query_groups)  # kept for call-site compatibility
    rows: list[tuple[str, object]] = [
        (f"Hit rate top {top_pct_threshold:g}%", covsite_top_pct),
    ]
    if noncovalent_top_pct is not None:
        rows.append(
            (f"Non-covalent only top {top_pct_threshold:g}%", noncovalent_top_pct)
        )
    rows.append((f"Random baseline top {top_pct_threshold:g}%", random_top_pct))
    rows.append((f"Hit rate top {k} rank", covsite_top_k))
    if noncovalent_top_k is not None:
        rows.append((f"Non-covalent only top {k} rank", noncovalent_top_k))
    rows.extend(
        [
            (f"Random baseline top {k} rank", random_top_k),
            ("Median rank", median_rank),
            ("Median search space reduction", median_ss_reduction_pct),
        ]
    )
    return rows


def plot_companion_table(
    ax: plt.Axes,
    rows: list[tuple[str, object]],
    *,
    panel_label: str = "B",
) -> None:
    """Two-column summary table styled like the reference benchmark figure."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if panel_label:
        _panel_title(ax, panel_label, pad=8)

    cell_text = [
        [label, _format_table_value(label, value)] for label, value in rows
    ]

    table = ax.table(
        cellText=cell_text,
        colWidths=[0.55, 0.25],
        loc="center",
        bbox=[0.18, 0.05, 0.64, 0.85],
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.scale(1.0, 1.0)

    n_rows = len(rows)
    row_height = min(0.12, 0.85 / max(n_rows, 1))
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.8)
        cell.set_height(row_height)
        cell.PAD = 0.12
        # Alternating soft blue / white like the reference table.
        if row_idx % 2 == 0:
            cell.set_facecolor("#dbe4f0" if col_idx == 0 else "#eef2f7")
        else:
            cell.set_facecolor("#cfd9e8" if col_idx == 0 else "white")
        if col_idx == 0:
            cell.set_text_props(
                fontsize=9,
                ha="left",
                va="center",
                fontfamily="Arial",
                fontweight="bold",
            )
        else:
            cell.set_text_props(
                fontsize=9,
                ha="left",
                va="center",
                fontfamily="Arial",
                fontweight="bold",
            )


def _load_benchmark_metrics(
    *,
    summary_path: Path,
    scores_path: Path | None,
    query_groups_path: Path | None,
    k: int | None = None,
) -> dict[str, object]:
    """Shared data loading for standard and direct-comparison figures."""
    metrics = load_covsite_cys_metrics(summary_path)
    k_eff = int(k if k is not None else metrics["k"])
    top_pct = float(metrics["top_pct_threshold"])
    p_frac = top_pct / 100.0

    label_to_qg_ns = load_label_to_qg_pool_sizes(
        scores_path=scores_path,
        query_groups_path=query_groups_path,
    )
    n_labels_found = len(label_to_qg_ns)
    n_qg_found = sum(len(ns) for ns in label_to_qg_ns.values())

    if not metrics.get("perfect_match", True):
        print(
            "[WARN] summary perfect_match=false; random baseline still uses "
            "product-over-warheads (perfect-match) logic."
        )
    if n_labels_found != int(metrics["n_labels"]):
        print(
            f"[WARN] Found {n_labels_found} labels in scores/query-groups CSV "
            f"but summary reports {metrics['n_labels']}."
        )

    random_top_k_pct = 100.0 * compute_mean_label_random_baseline(
        label_to_qg_ns, mode="top_k", k=k_eff,
    )
    random_top_pct_pct = 100.0 * compute_mean_label_random_baseline(
        label_to_qg_ns, mode="top_pct", p_frac=p_frac,
    )
    covsite_top_k_pct = 100.0 * float(metrics["label_hit_rate_at_k"])
    covsite_top_pct_pct = 100.0 * float(metrics["label_hit_rate_at_top_pct"])
    median_ss_pct = 100.0 * float(metrics["median_ss_reduction"])
    n_labels = int(metrics["n_labels"])
    n_query_groups = int(metrics["n_query_groups"])

    return {
        "metrics": metrics,
        "k_eff": k_eff,
        "top_pct": top_pct,
        "random_top_k_pct": random_top_k_pct,
        "random_top_pct_pct": random_top_pct_pct,
        "covsite_top_k_pct": covsite_top_k_pct,
        "covsite_top_pct_pct": covsite_top_pct_pct,
        "median_ss_pct": median_ss_pct,
        "n_labels": n_labels,
        "n_query_groups": n_query_groups,
        "n_labels_found": n_labels_found,
        "n_qg_found": n_qg_found,
    }


def create_direct_comparison_figure(
    *,
    summary_path: Path,
    scores_path: Path | None,
    query_groups_path: Path | None,
    output_path: Path,
    docking_benchmarks: list[tuple[str, float, int]],
    k: int | None = None,
    noncovalent_summary_path: Path | None = None,
) -> None:
    _configure_figure_fonts()
    data = _load_benchmark_metrics(
        summary_path=summary_path,
        scores_path=scores_path,
        query_groups_path=query_groups_path,
        k=k,
    )
    metrics = data["metrics"]
    k_eff = int(data["k_eff"])
    top_pct = float(data["top_pct"])
    covsite_top_pct_pct = float(data["covsite_top_pct_pct"])
    random_top_pct_pct = float(data["random_top_pct_pct"])
    covsite_top_k_pct = float(data["covsite_top_k_pct"])
    random_top_k_pct = float(data["random_top_k_pct"])
    median_ss_pct = float(data["median_ss_pct"])
    n_labels = int(data["n_labels"])
    n_query_groups = int(data["n_query_groups"])

    noncovalent_top_pct_pct: float | None = None
    noncovalent_top_k_pct: float | None = None
    if noncovalent_summary_path is not None:
        noncov = load_covsite_cys_metrics(noncovalent_summary_path)
        noncovalent_top_pct_pct = 100.0 * float(noncov["label_hit_rate_at_top_pct"])
        noncovalent_top_k_pct = 100.0 * float(noncov["label_hit_rate_at_k"])
        if int(noncov["k"]) != k_eff:
            print(
                f"[WARN] Non-covalent summary k={noncov['k']} differs from "
                f"main summary k={k_eff}; plotting Hit@K from each file as stored."
            )
        if float(noncov["top_pct_threshold"]) != top_pct:
            print(
                f"[WARN] Non-covalent top_pct_threshold="
                f"{noncov['top_pct_threshold']} differs from main "
                f"top_pct_threshold={top_pct}."
            )

    tool_values = [pct for _, pct, _ in docking_benchmarks]
    structural_mean = float(np.mean(tool_values))

    fig = plt.figure(figsize=(16.5, 10.5), facecolor="white")
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.55, 0.72],
        hspace=0.28,
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    plot_direct_comparison(
        ax,
        covsite_top_pct=covsite_top_pct_pct,
        random_top_pct=random_top_pct_pct,
        covsite_top_k=covsite_top_k_pct,
        random_top_k=random_top_k_pct,
        docking_benchmarks=docking_benchmarks,
        k=k_eff,
        top_pct_threshold=top_pct,
        noncovalent_top_pct=noncovalent_top_pct_pct,
        noncovalent_top_k=noncovalent_top_k_pct,
    )

    ax_table = fig.add_subplot(gs[1, 0])
    table_rows = build_companion_table_rows(
        covsite_top_pct=covsite_top_pct_pct,
        random_top_pct=random_top_pct_pct,
        covsite_top_k=covsite_top_k_pct,
        random_top_k=random_top_k_pct,
        median_rank=float(metrics["median_rank"]),
        median_ss_reduction_pct=median_ss_pct,
        n_labels=n_labels,
        n_query_groups=n_query_groups,
        k=k_eff,
        top_pct_threshold=top_pct,
        noncovalent_top_pct=noncovalent_top_pct_pct,
        noncovalent_top_k=noncovalent_top_k_pct,
    )
    plot_companion_table(ax_table, table_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved direct-comparison figure -> {output_path}")
    print(f"  CovSite @ top-{top_pct:g}%:     {covsite_top_pct_pct:.1f}% (N={n_labels:,})")
    if noncovalent_top_pct_pct is not None:
        print(
            f"  Non-covalent @ top-{top_pct:g}%: "
            f"{noncovalent_top_pct_pct:.1f}%"
        )
    print(f"  Random @ top-{top_pct:g}%:      {random_top_pct_pct:.1f}% (N={n_labels:,})")
    print(f"  CovSite @ top-{k_eff}:          {covsite_top_k_pct:.1f}% (N={n_labels:,})")
    if noncovalent_top_k_pct is not None:
        print(f"  Non-covalent @ top-{k_eff}:     {noncovalent_top_k_pct:.1f}%")
    print(f"  Random @ top-{k_eff}:           {random_top_k_pct:.1f}% (N={n_labels:,})")
    print(f"  Structural site detection:    {structural_mean:.1f}% (mean of 6 tools)")


def create_figure(
    *,
    summary_path: Path,
    scores_path: Path | None,
    query_groups_path: Path | None,
    output_path: Path,
    docking_benchmarks: list[tuple[str, float, int]],
    k: int | None = None,
) -> None:
    _configure_figure_fonts()
    data = _load_benchmark_metrics(
        summary_path=summary_path,
        scores_path=scores_path,
        query_groups_path=query_groups_path,
        k=k,
    )
    metrics = data["metrics"]
    k_eff = int(data["k_eff"])
    top_pct = float(data["top_pct"])
    random_top_k_pct = float(data["random_top_k_pct"])
    random_top_pct_pct = float(data["random_top_pct_pct"])
    covsite_top_k_pct = float(data["covsite_top_k_pct"])
    covsite_top_pct_pct = float(data["covsite_top_pct_pct"])
    median_ss_pct = float(data["median_ss_pct"])
    n_labels = int(data["n_labels"])
    n_query_groups = int(data["n_query_groups"])
    n_labels_found = int(data["n_labels_found"])
    n_qg_found = int(data["n_qg_found"])

    docking_sorted = sort_docking_benchmarks_desc(docking_benchmarks)

    fig = plt.figure(figsize=(14.5, 7.5))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.35, 1.0],
        hspace=0.35,
        wspace=0.22,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.10,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[:, 1])

    plot_panel_a_blind_recovery(
        ax_a,
        covsite_top_pct=covsite_top_pct_pct,
        random_top_pct=random_top_pct_pct,
        covsite_top_k=covsite_top_k_pct,
        random_top_k=random_top_k_pct,
        n_labels=n_labels,
        k=k_eff,
        top_pct_threshold=top_pct,
    )
    plot_panel_b_pose_accuracy(
        ax_b,
        docking_benchmarks=docking_sorted,
    )

    table_rows = build_companion_table_rows(
        covsite_top_pct=covsite_top_pct_pct,
        random_top_pct=random_top_pct_pct,
        covsite_top_k=covsite_top_k_pct,
        random_top_k=random_top_k_pct,
        median_rank=float(metrics["median_rank"]),
        median_ss_reduction_pct=median_ss_pct,
        n_labels=n_labels,
        n_query_groups=n_query_groups,
        k=k_eff,
        top_pct_threshold=top_pct,
    )
    plot_companion_table(ax_table, table_rows, panel_label="")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved figure -> {output_path}")
    print(f"  CovSite Hit@top-{top_pct:g}%: {covsite_top_pct_pct:.1f}% (N={n_labels:,} labels)")
    print(
        f"  Random @ top-{top_pct:g}%:     {random_top_pct_pct:.1f}% "
        f"(N={n_labels_found:,} labels, perfect-match product over {n_qg_found:,} QGs)"
    )
    print(f"  CovSite Hit@top-{k_eff}:       {covsite_top_k_pct:.1f}% (N={n_labels:,} labels)")
    print(
        f"  Random @ top-{k_eff}:            {random_top_k_pct:.1f}% "
        f"(N={n_labels_found:,} labels, perfect-match product over {n_qg_found:,} QGs)"
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Create C207 benchmark comparison figure from screen_summary.json.",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=here / "screen_summary.json",
        help="Cov_Screen summary JSON (default: screen_summary.json).",
    )
    p.add_argument(
        "--scores",
        type=Path,
        default=here / "screen_scores.csv",
        help="Candidate-level scores CSV (label + query_group mapping for random baseline).",
    )
    p.add_argument(
        "--query-groups",
        type=Path,
        default=None,
        help="Optional per-query-group CSV (Name, Residue, ResNum, Chain, n_residues).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=here / "benchmark_comparison.png",
        help="Output figure path.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override K for Hit@K metric and random baseline (default: from summary JSON).",
    )
    p.add_argument(
        "--noncovalent-summary",
        type=Path,
        default=None,
        help=(
            "Optional Cov_Screen summary JSON for a non-covalent-only model. "
            "With --direct-comparison, inserts two bars (top-%% and top-K) "
            "after each full-model bar and before random."
        ),
    )
    p.add_argument(
        "--direct-comparison",
        action="store_true",
        help=(
            "Single portrait panel comparing ranked site detection, random baseline, "
            "and mean structural screening performance (combines Panels A/B methodology)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary.is_file():
        raise FileNotFoundError(f"Required input not found: {args.summary}")

    scores_path = args.scores if args.scores.is_file() else None
    if args.query_groups is None and scores_path is None:
        raise FileNotFoundError(
            f"Need --scores ({args.scores}) or --query-groups for random baseline pool sizes."
        )

    noncovalent_summary = args.noncovalent_summary
    if noncovalent_summary is not None and not noncovalent_summary.is_file():
        raise FileNotFoundError(
            f"Non-covalent summary not found: {noncovalent_summary}"
        )

    if args.direct_comparison:
        create_direct_comparison_figure(
            summary_path=args.summary,
            scores_path=scores_path,
            query_groups_path=args.query_groups,
            output_path=args.output,
            docking_benchmarks=DEFAULT_DOCKING_BENCHMARKS,
            k=args.k,
            noncovalent_summary_path=noncovalent_summary,
        )
        return

    if noncovalent_summary is not None:
        print(
            "[WARN] --noncovalent-summary is only used with --direct-comparison; "
            "ignoring for the standard figure."
        )

    create_figure(
        summary_path=args.summary,
        scores_path=scores_path,
        query_groups_path=args.query_groups,
        output_path=args.output,
        docking_benchmarks=DEFAULT_DOCKING_BENCHMARKS,
        k=args.k,
    )


if __name__ == "__main__":
    main()
