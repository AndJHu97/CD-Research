#!/usr/bin/env python3
"""Create a C207 benchmark comparison figure (Panels A–B + companion table).

Panel A: CovSite blind site recovery vs perfect-match random baseline (Hit @ top-X% and Hit @ K).
Panel B: Literature pose-prediction success rates for covalent docking tools.

With --direct-comparison: single portrait panel comparing ranked detection, random baseline,
and mean structural screening performance (methodology comparison).

Usage:
    python Create_Benchmark_FIgures.py \\
        --summary screen_summary.json \\
        [--scores screen_scores.csv] \\
        [--output benchmark_comparison.png]

    python Create_Benchmark_FIgures.py \\
        --summary screen_summary.json \\
        --scores screen_scores.csv \\
        --direct-comparison \\
        --output benchmark_direct_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BENCHMARK_SET_LABEL = "C207"

COVSITE_COLOR = "#2C7A4B"
RANDOM_COLOR = "#B8B8B8"
DOCKING_COLORS = ["#8172B3", "#DD8452", "#C44E52", "#4C72B0", "#937860", "#64B5CD"]

# Direct-comparison panel palette.
DIRECT_COVSITE_COLOR = "#2a78d6"
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


def _panel_title(ax: plt.Axes, panel_label: str, title: str, *, pad: float = 10) -> None:
    ax.set_title(
        f"{panel_label}  {title}",
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
    _panel_title(ax, "A", "Blind site recovery", pad=14)
   
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
) -> None:
    """Single-panel methodology comparison: blind recovery + structural mean."""
    tool_values = sorted(
        (pct for _, pct, _ in docking_benchmarks),
        reverse=True,
    )
    structural_mean = float(np.mean(tool_values))

    # Pair gap between metric groups (matches Panel A spacing).
    x = np.array([0.0, 1.0, 2.35, 3.35, 4.85])
    values = [
        covsite_top_pct,
        random_top_pct,
        covsite_top_k,
        random_top_k,
        structural_mean,
    ]
    colors = [
        DIRECT_COVSITE_COLOR,
        DIRECT_RANDOM_COLOR,
        DIRECT_COVSITE_COLOR,
        DIRECT_RANDOM_COLOR,
        DIRECT_DOCKING_COLOR,
    ]
    tick_labels = [
        f"Ranked site detection\n(CovSite @ top {top_pct_threshold:g}%)",
        "Random site detection",
        f"Ranked site detection\n(CovSite @ top {k})",
        "Random site detection",
        "Structural site detection\n(6 tools)",
    ]

    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        width=0.62,
        zorder=2,
    )

    structural_center = x[4]
    dot_spread = np.linspace(-0.20, 0.20, len(tool_values))
    for val, dx in zip(tool_values, dot_spread):
        ax.scatter(
            structural_center + dx,
            val,
            s=52,
            color=DIRECT_DOCKING_COLOR,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
        )

    ax.plot(
        [structural_center, structural_center],
        [min(tool_values), max(tool_values)],
        color=DIRECT_DOCKING_COLOR,
        alpha=0.4,
        linewidth=1.2,
        zorder=3,
    )

    structural_label_y = max(tool_values) + 7.0
    for idx, (bar, val) in enumerate(zip(bars, values)):
        cx = bar.get_x() + bar.get_width() / 2
        if idx == 4:
            ax.text(
                cx,
                structural_label_y,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                zorder=5,
            )
            ax.text(
                cx,
                structural_label_y - 3.2,
                "(mean)",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=MUTED_TEXT_COLOR,
                zorder=5,
            )
        elif val >= 85.0:
            ax.text(
                cx,
                bar.get_height() - 2.5,
                f"{val:.1f}%",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="white",
                zorder=5,
            )
        else:
            ax.text(
                cx,
                bar.get_height() + 2.0,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Performance (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        mpatches.Patch(
            facecolor=DIRECT_COVSITE_COLOR,
            edgecolor="white",
            label="Hit rate @ top 10% (blind, ranked)",
        ),
        mpatches.Patch(
            facecolor=DIRECT_RANDOM_COLOR,
            edgecolor="white",
            label="Random baseline",
        ),
        mpatches.Patch(
            facecolor=DIRECT_DOCKING_COLOR,
            edgecolor="white",
            label="% success <2Å RMSD (given known site)",
        ),
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=DIRECT_DOCKING_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.2,
            markersize=8,
            label="Individual tool",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        columnspacing=1.2,
    )


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
    _panel_title(ax, "B", "Pose accuracy given known site")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_table_value(label: str, value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    if (
        "Hit rate" in label
        or "Random baseline" in label
        or "search-space" in label.lower()
    ):
        return f"{float(value):.1f}%"
    if label == "Median rank":
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
) -> list[tuple[str, str]]:
    n_display = (
        f"{n_labels:,} labels / {n_query_groups:,} query groups"
        if n_labels != n_query_groups
        else f"{n_query_groups:,}"
    )
    return [
        ("Benchmark set", BENCHMARK_SET_LABEL),
        ("N proteins / query groups", n_display),
        ("Residue type", "Cys only"),
        (f"Hit rate @ top {top_pct_threshold:g}% (CovSite)", covsite_top_pct),
        (f"Random baseline @ top {top_pct_threshold:g}% (perfect-match)", random_top_pct),
        (f"Hit rate @ top {k} (CovSite)", covsite_top_k),
        (f"Random baseline @ top {k} (perfect-match)", random_top_k),
        ("Median rank", median_rank),
        ("Median search-space reduction", median_ss_reduction_pct),
    ]


def plot_companion_table(ax: plt.Axes, rows: list[tuple[str, str]]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.995,
        "Benchmark summary",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    cell_text = [
        [label, _format_table_value(label, value)] for label, value in rows
    ]

    table = ax.table(
        cellText=cell_text,
        colWidths=[0.54, 0.26],
        loc="upper center",
        bbox=[0.14, 0.0, 0.72, 0.92],
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.scale(1.0, 4.3)

    n_rows = len(rows)
    row_height = min(0.236, 1.75 / max(n_rows, 1))
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.6)
        cell.set_height(row_height)
        cell.PAD = 0.18
        if col_idx == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(fontweight="bold", fontsize=8.5, ha="left", va="center")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(fontsize=8.5, ha="left", va="center")


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
) -> None:
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

    tool_values = [pct for _, pct, _ in docking_benchmarks]
    structural_mean = float(np.mean(tool_values))

    fig = plt.figure(figsize=(10.5, 12.0), facecolor="white")
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.15, 0.55],
        hspace=0.38,
        left=0.10,
        right=0.97,
        top=0.88,
        bottom=0.06,
    )

    fig.suptitle(
        "Site detection benchmark on C207",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.935,
        "Blind ranked discovery vs. structural docking given known site",
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED_TEXT_COLOR,
    )

    ax = fig.add_subplot(gs[0])
    plot_direct_comparison(
        ax,
        covsite_top_pct=covsite_top_pct_pct,
        random_top_pct=random_top_pct_pct,
        covsite_top_k=covsite_top_k_pct,
        random_top_k=random_top_k_pct,
        docking_benchmarks=docking_benchmarks,
        k=k_eff,
        top_pct_threshold=top_pct,
    )

    ax_table = fig.add_subplot(gs[1])
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
    plot_companion_table(ax_table, table_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved direct-comparison figure -> {output_path}")
    print(f"  CovSite @ top-{top_pct:g}%:     {covsite_top_pct_pct:.1f}% (N={n_labels:,})")
    print(f"  Random @ top-{top_pct:g}%:      {random_top_pct_pct:.1f}% (N={n_labels:,})")
    print(f"  CovSite @ top-{k_eff}:          {covsite_top_k_pct:.1f}% (N={n_labels:,})")
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

    fig = plt.figure(figsize=(11.5, 13.5))
    fig.suptitle(
        "Benchmark comparison on C207",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 0.95],
        width_ratios=[1.05, 1.12],
        hspace=0.40,
        wspace=0.32,
        left=0.10,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

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
    plot_companion_table(ax_table, table_rows)

    docking_n_note = ", ".join(
        f"n={n}" for n in sorted({n for _, _, n in docking_sorted})
    )
    
    

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

    if args.direct_comparison:
        create_direct_comparison_figure(
            summary_path=args.summary,
            scores_path=scores_path,
            query_groups_path=args.query_groups,
            output_path=args.output,
            docking_benchmarks=DEFAULT_DOCKING_BENCHMARKS,
            k=args.k,
        )
        return

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
