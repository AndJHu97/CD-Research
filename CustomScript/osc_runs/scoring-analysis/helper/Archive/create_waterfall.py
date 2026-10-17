#!/usr/bin/env python3
"""Create a four-panel 2×2 figure from a baby_frank summary CSV.

Panel A: Search-space reduction Sankey (post-nucleophilic pool through filters;
         full Total_N_Before/After, not If_Match).
Panel B: Overall per-filter target retention vs random baseline (If_Match totals).
Panel C: Failure attribution bar chart (% of N_Failures per filter).
Panel D: Per-residue hit-rate bar chart with aggregate statistics, or with
         --radar a hit-rate vs random-baseline radar (6 residues + overall).

Usage:
    python create_waterfall.py summary_baby_frank.csv [--output figure.png]
    python create_waterfall.py summary_baby_frank.csv --skip-filter0
    python create_waterfall.py summary_baby_frank.csv --radar
"""

from __future__ import annotations

import argparse
import re
from io import BytesIO
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.image import imread
from plotly.io import to_image

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

PANEL_D_RESIDUES = ("CYS", "LYS", "HIS", "TYR", "THR", "SER")
FILTER_HIT_COLOR = "#4C72B0"
SANKEY_PASS_COLOR = "#4C72B0"
SANKEY_FAIL_COLOR = "#DD8452"
SANKEY_IMG_WIDTH = 1300
SANKEY_IMG_HEIGHT = 880
SANKEY_LEFT_LABEL_RESERVE = 0.02  # room for Post-Nucleophilic Pool label
SANKEY_POOL_LABEL_X = -0.08  # axes-fraction x for pool label (lower = further left)
SANKEY_RIGHT_INSET = 0.02
SANKEY_FONT_SIZE = 15
SANKEY_LABEL_FONT_SIZE = 11  # matplotlib overlay (drawn above the Sankey image)
FIG_WIDTH = 14.85   # 13.5 × 1.10
FIG_HEIGHT = 11.025  # 10.5 × 1.05
FIG_HEIGHT_RADAR = 12.075  # 11.5 × 1.05
FAILURE_ATTRIBUTION_COLOR = "#C44E52"
RANDOM_BASELINE_COLOR = "#C0C0C0"
HIT_RATE_COLOR = "#55A868"
RADAR_GRID_STEP = 10  # grid ring spacing (%); rings are unlabeled
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


def filter_name_column(df: pd.DataFrame) -> str:
    if "Filter" in df.columns and df["Filter"].notna().any():
        return "Filter"
    return "Category"


def normalize_filter_name(filters: pd.DataFrame) -> pd.DataFrame:
    out = filters.copy()
    name_col = filter_name_column(out)
    out["_Filter_Name"] = out[name_col]
    out["Short_Name"] = out["_Filter_Name"].map(short_filter_name)
    return out


def get_filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    filters = df[df["Group"] == "Filter"].copy()
    if filters.empty:
        raise ValueError("No Filter rows found in summary CSV.")
    return normalize_filter_name(filters)


def prepare_waterfall_filters(
    filters: pd.DataFrame,
    skip_filter0: bool,
) -> tuple[pd.DataFrame, str | None]:
    """
    Optionally drop Filter 0 and rebase abs-reduction % onto Total_N_After from Filter 0.

    Abs/rel reduction % are computed from Total_N_Before and Total_N_After per filter.
    Abs uses the Filter 0 Total_N_Before (PDB pool) as denominator; with --skip-filter0
    it uses Filter 0 Total_N_After (post-nucleophilic pool) instead.
    """
    f0_row = _filter0_row(filters)
    pdb_total = _get_total_n(f0_row, "Total_N_Before")
    f0_after_total = _get_total_n(f0_row, "Total_N_After")

    if not skip_filter0:
        return add_waterfall_reduction_columns(filters, abs_denominator=pdb_total), None

    f0_mask = filters["_Filter_Name"].map(is_filter0)
    remaining = filters[~f0_mask].copy()
    enriched = add_waterfall_reduction_columns(remaining, abs_denominator=f0_after_total)
    baseline_note = (
        f"abs. reduction rebased to post-nucleophilic pool "
        f"(N={f0_after_total:,.0f})"
    )
    return enriched, baseline_note


def compute_f0_to_last_reduction(filters: pd.DataFrame) -> float:
    """Pool reduction from Filter 0 output to last-filter output (full Total_N)."""
    f0_row = _filter0_row(filters)
    last_row = filters.iloc[-1]
    n_start = _get_full_total_n(f0_row, "Total_N_After")
    n_end = _get_full_total_n(last_row, "Total_N_After")
    if n_start <= 0:
        raise ValueError("Filter 0 Total_N_After is missing or zero.")

    return (n_start - n_end) / n_start * 100.0


def compute_overall_reduction(filters: pd.DataFrame) -> float:
    """Cumulative abs. reduction from Filter 0 through the last filter (full Total_N)."""
    f0_row = _filter0_row(filters)
    pdb_total = _get_full_total_n(f0_row, "Total_N_Before")
    enriched = add_waterfall_reduction_columns(filters, abs_denominator=pdb_total)
    return float(enriched["Abs_Reduction_Pct"].fillna(0).sum())


def get_residue_rows(df: pd.DataFrame) -> pd.DataFrame:
    residues = df[df["Group"] == "Residue_Type"].copy()
    if residues.empty:
        raise ValueError("No Residue_Type rows found in summary CSV.")
    return residues


def get_overall_matchable(df: pd.DataFrame) -> int:
    overall = df[df["Group"] == "Overall"]
    if not overall.empty:
        return int(overall.iloc[0]["N_Matchable"])

    residues = get_residue_rows(df)
    return int(residues["N_Matchable"].sum())


def _get_total_n(row: pd.Series, column: str) -> float:
    """Read Total_N_Before/After from a summary row, or derive from Avg × N_Count."""
    if column in row.index and pd.notna(row[column]):
        return float(row[column])
    if "_If_Match" in column:
        avg_col = (
            "Avg_N_Before_If_Match" if "Before" in column else "Avg_N_After_If_Match"
        )
        count_col = "N_Count_If_Match"
    else:
        avg_col = "Avg_N_Before" if "Before" in column else "Avg_N_After"
        count_col = "N_Count"
    if (
        count_col in row.index
        and pd.notna(row[count_col])
        and avg_col in row.index
        and pd.notna(row[avg_col])
    ):
        return float(row[avg_col]) * float(row[count_col])
    raise ValueError(
        f"Summary row missing {column} (and cannot derive from Avg × N_Count). "
        "Re-run baby_frank.py with --output-summary."
    )


def _get_full_total_n(row: pd.Series, column: str) -> float:
    """Read full-pool Total_N_Before/After for Panel A (never If_Match)."""
    if column not in ("Total_N_Before", "Total_N_After"):
        raise ValueError(f"Panel A expects Total_N_Before or Total_N_After, got {column!r}")
    return _get_total_n(row, column)


def _has_if_match_n(row: pd.Series) -> bool:
    return (
        "Total_N_Before_If_Match" in row.index
        and pd.notna(row.get("Total_N_Before_If_Match"))
        and "Total_N_After_If_Match" in row.index
        and pd.notna(row.get("Total_N_After_If_Match"))
    )


def _pool_removed(row: pd.Series, *, at_risk_only: bool = False) -> float:
    if at_risk_only and _has_if_match_n(row):
        return _get_total_n(row, "Total_N_Before_If_Match") - _get_total_n(
            row, "Total_N_After_If_Match"
        )
    return _get_total_n(row, "Total_N_Before") - _get_total_n(row, "Total_N_After")


def _filter0_row(
    filters: pd.DataFrame,
    filter_name_col: str = "_Filter_Name",
) -> pd.Series:
    f0_mask = filters[filter_name_col].map(is_filter0)
    if not f0_mask.any():
        raise ValueError("No nucleophilic residue selection (Filter 0) row found.")
    return filters[f0_mask].iloc[0]


def _filter_rel_reduction_pct(row: pd.Series, *, at_risk_only: bool = False) -> float:
    if at_risk_only and _has_if_match_n(row):
        total_before = _get_total_n(row, "Total_N_Before_If_Match")
    else:
        total_before = _get_total_n(row, "Total_N_Before")
    if total_before <= 0:
        return 0.0
    return _pool_removed(row, at_risk_only=at_risk_only) / total_before * 100.0


def _filter_abs_reduction_pct(row: pd.Series, abs_denominator: float) -> float:
    if abs_denominator <= 0:
        return 0.0
    return _pool_removed(row) / abs_denominator * 100.0


def add_waterfall_reduction_columns(
    filters: pd.DataFrame,
    *,
    abs_denominator: float,
) -> pd.DataFrame:
    """Add Abs_Reduction_Pct / Rel_Reduction_Pct from Total_N_Before/After."""
    out = filters.copy()
    out["Abs_Reduction_Pct"] = [
        _filter_abs_reduction_pct(row, abs_denominator) for _, row in out.iterrows()
    ]
    out["Rel_Reduction_Pct"] = [_filter_rel_reduction_pct(row) for _, row in out.iterrows()]
    return out


def compute_filter_accuracy_stats(
    filters: pd.DataFrame,
    *,
    filter_name_col: str = "_Filter_Name",
    require_if_match: bool = True,
) -> pd.DataFrame:
    """
    Per-filter target retention vs random baseline for filters after Nucleophilic selection.

    Uses Total_N_*_If_Match (target still alive entering the filter; excludes
    dead-target pool inflation). Filter retention per step:
      pool_removed = Total_N_Before_If_Match - Total_N_After_If_Match
      retention = (1 - N_Failures / pool_removed) × 100
      random retention = (1 - Total_N_After_If_Match / Total_N_Before_If_Match) × 100
    """
    post_f0 = filters[~filters[filter_name_col].map(is_filter0)].copy()
    if post_f0.empty:
        raise ValueError("No filters after nucleophilic selection (Filter 0) found.")

    rows: list[dict[str, float | int | str]] = []
    for _, row in post_f0.iterrows():
        n_failures = int(row.get("N_Failures", 0) or 0)
        if require_if_match and not _has_if_match_n(row):
            raise ValueError(
                "Summary CSV missing Total_N_*_If_Match columns required for Panel B. "
                "Re-run baby_frank.py with --output-summary."
            )

        if _has_if_match_n(row):
            total_before = _get_total_n(row, "Total_N_Before_If_Match")
            total_after = _get_total_n(row, "Total_N_After_If_Match")
        else:
            total_before = _get_total_n(row, "Total_N_Before")
            total_after = _get_total_n(row, "Total_N_After")

        pool_removed = total_before - total_after

        if pool_removed <= 0:
            filter_hit_rate = 0.0
        else:
            filter_hit_rate = (1.0 - n_failures / pool_removed) * 100.0

        if total_before > 0:
            random_hit_rate = (1.0 - total_after / total_before) * 100.0
        else:
            random_hit_rate = 0.0

        rows.append(
            {
                "Short_Name": row["Short_Name"],
                "Filter_Hit_Rate_Pct": filter_hit_rate,
                "Random_Hit_Rate_Pct": random_hit_rate,
                "N_Failures": n_failures,
                "Denominator": pool_removed,
                "Pool_Removed": pool_removed,
            }
        )

    return pd.DataFrame(rows)


def compute_failure_attribution(filters: pd.DataFrame) -> pd.DataFrame:
    """Share of total N_Failures attributed to each post-Filter-0 step."""
    post_f0 = filters[~filters["_Filter_Name"].map(is_filter0)].copy()
    if post_f0.empty:
        raise ValueError("No filters after nucleophilic selection (Filter 0) found.")

    failure_counts = [
        int(row.get("N_Failures", 0) or 0) for _, row in post_f0.iterrows()
    ]
    total_failures = sum(failure_counts)
    rows: list[dict[str, float | int | str]] = []
    for (_, row), n_fail in zip(post_f0.iterrows(), failure_counts):
        share_pct = (n_fail / total_failures * 100.0) if total_failures else 0.0
        rows.append(
            {
                "Short_Name": row["Short_Name"],
                "N_Failures": n_fail,
                "Failure_Share_Pct": share_pct,
            }
        )
    return pd.DataFrame(rows)


def compute_cumulative_random_baseline(accuracy: pd.DataFrame) -> float:
    """Product of per-filter random retention rates (post Filter 0), as %."""
    cumulative = 1.0
    for pct in accuracy["Random_Hit_Rate_Pct"]:
        cumulative *= float(pct) / 100.0
    return cumulative * 100.0


def compute_residue_cumulative_random_baseline(residue_filters: pd.DataFrame) -> float:
    """Residue-specific cumulative random baseline from Filter_Residue_Type rows."""
    post_f0 = residue_filters[~residue_filters["_Filter_Name"].map(is_filter0)]
    cumulative = 1.0
    for _, row in post_f0.iterrows():
        if _has_if_match_n(row):
            total_before = _get_total_n(row, "Total_N_Before_If_Match")
            total_after = _get_total_n(row, "Total_N_After_If_Match")
        else:
            total_before = _get_total_n(row, "Total_N_Before")
            total_after = _get_total_n(row, "Total_N_After")
        if total_before > 0:
            cumulative *= 1.0 - total_after / total_before
    return cumulative * 100.0


def compute_residue_random_baselines(
    df: pd.DataFrame,
    filters: pd.DataFrame,
    residues: tuple[str, ...] = PANEL_D_RESIDUES,
) -> dict[str, float]:
    """Cumulative random baseline per residue from Filter_Residue_Type rows."""
    baselines: dict[str, float] = {}
    for residue in residues:
        residue_filters = get_residue_filter_rows(df, filters, residue)
        baselines[residue] = compute_residue_cumulative_random_baseline(residue_filters)
    return baselines


def get_residue_filter_rows(
    df: pd.DataFrame,
    filters: pd.DataFrame,
    residue: str,
) -> pd.DataFrame:
    """Ordered filter rows for one residue from Filter_Residue_Type summary rows."""
    subset = df[
        (df["Group"] == "Filter_Residue_Type") & (df["Category"] == residue)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No Filter_Residue_Type rows found for residue {residue}. "
            "Re-run baby_frank.py with --output-summary to regenerate the summary."
        )
    if "Filter" not in subset.columns or subset["Filter"].isna().all():
        raise ValueError(
            "Filter_Residue_Type rows require a Filter column. "
            "Re-run baby_frank.py with --output-summary to regenerate the summary."
        )

    filter_order = filters["_Filter_Name"].tolist()
    ordered_rows: list[pd.Series] = []
    for fname in filter_order:
        match = subset[subset["Filter"] == fname]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["_Filter_Name"] = fname
        row["Short_Name"] = short_filter_name(fname)
        ordered_rows.append(row)

    if not ordered_rows:
        raise ValueError(f"No filter rows matched overall filter order for residue {residue}.")

    return pd.DataFrame(ordered_rows)


def compute_panel_d_stats(residues: pd.DataFrame) -> dict[str, float | int]:
    panel_res = residues[residues["Category"].isin(PANEL_D_RESIDUES)].copy()

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
        "overall_six_miss_pct": 100.0 - overall_six_pct,
        "non_cys_hits": non_cys_hits,
        "non_cys_matchable": non_cys_matchable,
        "non_cys_pct": non_cys_pct,
    }


def build_sankey_pipeline(filters: pd.DataFrame) -> dict:
    """Build post–Filter 0 Sankey stage data from full Total_N_Before/After."""
    f0 = _filter0_row(filters)
    baseline = _get_full_total_n(f0, "Total_N_After")
    post = filters[~filters["_Filter_Name"].map(is_filter0)]

    steps: list[dict] = []
    for _, row in post.iterrows():
        before = _get_full_total_n(row, "Total_N_Before")
        after = _get_full_total_n(row, "Total_N_After")
        removed = before - after
        steps.append({
            "short_name": str(row["Short_Name"]).replace("\n", " "),
            "label_name": str(row["Short_Name"]),
            "total_before": before,
            "total_after": after,
            "removed": removed,
            "abs_pct": removed / baseline * 100.0 if baseline else 0.0,
            "rel_pct": removed / before * 100.0 if before else 0.0,
            "pass_frac": after / baseline if baseline else 0.0,
        })

    final_after = steps[-1]["total_after"] if steps else baseline
    return {
        "baseline_n": baseline,
        "steps": steps,
        "final_after": final_after,
        "final_pct": final_after / baseline * 100.0 if baseline else 0.0,
    }


def _fmt_n(n: float) -> str:
    return f"{n:,.0f}"


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _sankey_column_x(column_index: int, n_columns: int) -> float:
    """Spread filter columns to reduce horizontal label overlap."""
    if n_columns <= 1:
        return 1.0
    return 0.24 + column_index * (0.72 / (n_columns - 1))


def _sankey_node_y(column_index: int, n_columns: int, *, is_pass: bool) -> float:
    """Stagger pass/fail rows so labels on nearby columns do not collide."""
    if is_pass:
        return 0.10 + column_index * 0.07
    # Push later fail labels further down; last filter lowest.
    return 0.86 + column_index * 0.05


def _build_plotly_sankey_figure(
    pipeline: dict,
    f0_to_last_reduction: float,
    overall_reduction: float,
) -> tuple[go.Figure, list[dict[str, object]]]:
    """Plotly Sankey (no node labels — labels are overlaid in matplotlib on top)."""
    steps = pipeline["steps"]
    n = len(steps)
    if n == 0:
        raise ValueError("No post–Filter 0 rows for Sankey diagram.")

    baseline = pipeline["baseline_n"]
    label_specs: list[dict[str, object]] = [
        {
            "text": "Post-Nucleophilic\nPool\n(100%)",
            "x": -3,
            "y": 0.5,
            "kind": "pool",
        }
    ]
    node_colors = [SANKEY_PASS_COLOR]
    node_x = [0.0]
    node_y = [0.5]
    node_hover: list[str] = [f"N={_fmt_n(baseline)}"]

    for i, step in enumerate(steps):
        x = _sankey_column_x(i, n)
        name = step["label_name"]
        pool_pct = step["total_after"] / baseline * 100.0 if baseline else 0.0
        pass_y = _sankey_node_y(i, n, is_pass=True)
        fail_y = _sankey_node_y(i, n, is_pass=False)
        if i == n - 1:
            pass_label = f"Final\nCandidates\n({pipeline['final_pct']:.1f}%)"
            pass_kind = "final"
            pass_hover = (
                f"Final candidates<br>N={_fmt_n(pipeline['final_after'])}<br>"
                f"Post-F0 reduction: {f0_to_last_reduction:.1f}%"
            )
        else:
            pass_label = f"Pass\n{name}\n({pool_pct:.1f}%)"
            pass_kind = "pass"
            pass_hover = (
                f"Pass {step['short_name']}<br>"
                f"N={_fmt_n(step['total_after'])} ({pool_pct:.1f}% of pool)"
            )
        fail_label = f"Fail\n{name}\n(−{step['rel_pct']:.1f}%)"
        fail_hover = (
            f"Fail {step['short_name']}<br>"
            f"N {_fmt_n(step['total_before'])}→{_fmt_n(step['total_after'])}<br>"
            f"rel −{step['rel_pct']:.1f}%  abs −{step['abs_pct']:.1f}%"
        )
        label_specs.extend([
            {"text": pass_label, "x": x, "y": pass_y, "kind": pass_kind},
            {"text": fail_label, "x": x, "y": fail_y, "kind": "fail"},
        ])
        node_colors.extend([SANKEY_PASS_COLOR, SANKEY_FAIL_COLOR])
        node_x.extend([x, x])
        node_y.extend([pass_y, fail_y])
        node_hover.extend([pass_hover, fail_hover])

    pass_node_indices = [1 + 2 * i for i in range(n)]

    empty_labels = [""] * len(node_colors)

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    customdata: list[list[str]] = []

    pass_rgba = _hex_rgba(SANKEY_PASS_COLOR, 0.30)
    fail_rgba = _hex_rgba(SANKEY_FAIL_COLOR, 0.30)

    for i, step in enumerate(steps):
        src = 0 if i == 0 else pass_node_indices[i - 1]
        pass_idx = pass_node_indices[i]
        fail_idx = pass_idx + 1
        pass_n = step["total_after"]
        fail_n = step["removed"]

        sources.extend([src, src])
        targets.extend([pass_idx, fail_idx])
        values.extend([pass_n, fail_n])
        link_colors.extend([pass_rgba, fail_rgba])
        customdata.extend([
            [
                f"Pass: N {_fmt_n(pass_n)}  "
                f"rel −{step['rel_pct']:.1f}%  abs −{step['abs_pct']:.1f}%"
            ],
            [
                f"Fail: N {_fmt_n(fail_n)}  "
                f"rel −{step['rel_pct']:.1f}%  abs −{step['abs_pct']:.1f}%"
            ],
        ])

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                domain=dict(x=[0, 1], y=[0, 1]),
                node=dict(
                    pad=50,
                    thickness=30,
                    line=dict(color="white", width=1),
                    label=empty_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    customdata=node_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=customdata,
                    hovertemplate="%{customdata[0]}<extra></extra>",
                ),
            )
        ],
        layout=dict(
            font=dict(size=SANKEY_FONT_SIZE, color="#1a1a1a", family="Arial, Helvetica, sans-serif"),
            margin=dict(l=64, r=48, t=4, b=48),
            paper_bgcolor="white",
            plot_bgcolor="white",
        ),
    )
    return fig, label_specs


def _overlay_sankey_labels(
    ax: plt.Axes,
    label_specs: list[dict[str, object]],
    *,
    plot_left: float,
    plot_right: float,
    top_reserve: float,
) -> None:
    """Draw Sankey node labels above the rasterized diagram (no background box)."""
    plot_w = plot_right - plot_left
    plot_h = 1.0 - top_reserve
    plot_top = 1.0 - top_reserve

    for spec in label_specs:
        kind = str(spec["kind"])
        py = plot_top - float(spec["y"]) * plot_h

        if kind == "pool":
            x, y, ha, va = SANKEY_POOL_LABEL_X, py, "left", "center"
        else:
            px = plot_left + float(spec["x"]) * plot_w
            if kind in ("pass", "final"):
                x, y, ha, va = px + 0.016, py - 0.035 * plot_h, "left", "center"
            else:  # fail
                x, y, ha, va = px + 0.016, py + 0.035 * plot_h, "left", "center"

        ax.text(
            x,
            y,
            str(spec["text"]),
            transform=ax.transAxes,
            fontsize=SANKEY_LABEL_FONT_SIZE,
            color="#1a1a1a",
            ha=ha,
            va=va,
            zorder=10,
            clip_on=False,
            linespacing=1.12,
        )


def plot_search_space_sankey(
    ax: plt.Axes,
    filters: pd.DataFrame,
    f0_to_last_reduction: float,
    overall_reduction: float,
) -> None:
    """Embed a Plotly Sankey of post-nucleophilic pool → filters → final candidates."""
    pipeline = build_sankey_pipeline(filters)
    fig, label_specs = _build_plotly_sankey_figure(
        pipeline, f0_to_last_reduction, overall_reduction
    )
    img = imread(
        BytesIO(
            to_image(
                fig,
                format="png",
                scale=2,
                width=SANKEY_IMG_WIDTH,
                height=SANKEY_IMG_HEIGHT,
            )
        )
    )

    plot_left = SANKEY_LEFT_LABEL_RESERVE
    plot_right = 1.0 - SANKEY_RIGHT_INSET
    top_reserve = 0.14
    ax.imshow(
        img,
        aspect="auto",
        extent=(plot_left, plot_right, 0.0, 1 - top_reserve),
        origin="upper",
        zorder=1,
    )
    _overlay_sankey_labels(
        ax,
        label_specs,
        plot_left=plot_left,
        plot_right=plot_right,
        top_reserve=top_reserve,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.margins(0)
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "A  Search-space reduction (post-nucleophilic selection)",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
        va="top",
        ha="left",
        zorder=11,
    )
    ax.text(
        0.0,
        0.935,
        (
            f"Overall reduction incl. nucleophilic selection: {overall_reduction:.1f}%  |  "
            f"Post-F0 reduction: {f0_to_last_reduction:.1f}%"
        ),
        transform=ax.transAxes,
        fontsize=10,
        color="#444444",
        va="top",
        ha="left",
        zorder=11,
    )


def plot_filter_accuracy_lines(
    ax: plt.Axes,
    accuracy: pd.DataFrame,
    *,
    panel_label: str = "B",
    title: str = "Filter accuracy vs random baseline",
) -> None:
    """Filter retention vs random baseline across post-nucleophilic filters."""
    names = accuracy["Short_Name"].tolist()
    filter_vals = accuracy["Filter_Hit_Rate_Pct"].tolist()
    random_vals = accuracy["Random_Hit_Rate_Pct"].tolist()

    x = np.arange(len(names))

    ax.plot(
        x,
        filter_vals,
        "o-",
        linewidth=2,
        markersize=6,
        color=FILTER_HIT_COLOR,
        label="Filter retention",
    )
    ax.plot(
        x,
        random_vals,
        "s--",
        linewidth=1.5,
        markersize=5,
        color=RANDOM_BASELINE_COLOR,
        label="Random baseline",
    )

    ymax = max(max(filter_vals, default=0), max(random_vals, default=0))
    for xi, val in zip(x, filter_vals):
        ax.text(
            xi,
            val + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=FILTER_HIT_COLOR,
        )
    for xi, val in zip(x, random_vals):
        ax.text(
            xi,
            val - 2.5,
            f"{val:.1f}%",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#666666",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Target retention (%)")
    ax.set_ylim(0, min(ymax + 14, 115))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{panel_label}  {title}", loc="left", fontweight="bold", fontsize=11)
    ax.text(
        0.0,
        -0.28,
        (
            
        ),
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_failure_attribution_bars(
    ax: plt.Axes,
    attribution: pd.DataFrame,
    total_failures: int,
) -> None:
    """Bar chart of N_Failure share per filter (post–Filter 0)."""
    names = attribution["Short_Name"].tolist()
    shares = attribution["Failure_Share_Pct"].tolist()
    n_failures = attribution["N_Failures"].tolist()
    x = np.arange(len(names))
    width = 0.55

    bars = ax.bar(
        x,
        shares,
        width,
        color=FAILURE_ATTRIBUTION_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ymax = max(shares) if shares else 0.0
    for bar, share, n_fail in zip(bars, shares, n_failures):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{share:.1f}%\n(n={int(n_fail)})",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Share of failures (%)")
    ax.set_ylim(0, ymax + 14)
    ax.set_title(
        "C  Failure attribution by filter",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax.text(
        0.0,
        -0.22,
        f"N_Failures per filter / Σ N_Failures (n={total_failures}).",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_accuracy_bars(
    ax: plt.Axes,
    stats: dict[str, float | int],
    overall_reduction: float,
) -> None:
    per_res = stats["per_residue_pct"]

    categories: list[str] = [RESIDUE_LABELS[r] for r in PANEL_D_RESIDUES]
    values: list[float] = [per_res.get(r, 0.0) for r in PANEL_D_RESIDUES]

    categories.extend([
        "Overall\n(6 res.)",
        "Random\nbaseline",
        "Non-Cys\n(all res.)",
        "Cys-only\nscreeners*",
    ])
    values.extend([
        float(stats["overall_six_pct"]),
        float(stats["cumulative_random_pct"]),
        float(stats["non_cys_pct"]),
        CYS_ONLY_BENCHMARK_PCT,
    ])

    colors = ["#4C72B0"] * len(PANEL_D_RESIDUES)
    colors.extend(["#55A868", RANDOM_BASELINE_COLOR, "#C44E52", "#AAAAAA"])

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
    ax.set_ylim(0, max(values) + 9)
    ax.set_title("D  Accuracy by residue", loc="left", fontweight="bold", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _radar_grid_rings(r_max_pct: float, *, step: int = RADAR_GRID_STEP) -> list[float]:
    """Linear grid ring positions (unlabeled) up to r_max_pct."""
    top = min(100, int(np.ceil(r_max_pct / step) * step))
    return list(range(0, top + 1, step))


def _random_baseline_legend_label(random_rates: list[float]) -> str:
    rand_min = min(random_rates)
    rand_max = max(random_rates)
    if abs(rand_max - rand_min) < 0.05:
        return f"Random baseline* ({rand_min:.1f}%)"
    return f"Random baseline* ({rand_min:.1f}–{rand_max:.1f}%)"


def _radar_label_ha_va(theta: float) -> tuple[str, str]:
    """Text alignment so polar labels sit outside the chart along each spoke."""
    t = theta % (2 * np.pi)
    if t <= np.pi / 4 or t >= 7 * np.pi / 4:
        return "center", "bottom"
    if t < 3 * np.pi / 4:
        return "left", "center"
    if t < 5 * np.pi / 4:
        return "center", "top"
    return "right", "center"


def plot_accuracy_radar(
    ax: plt.Axes,
    stats: dict[str, float | int | dict[str, float]],
    overall_reduction: float,
) -> None:
    """Radar chart: hit rate vs cumulative random baseline per residue + overall."""
    per_res = stats["per_residue_pct"]
    residue_random = stats["residue_random_baselines"]

    labels = [RESIDUE_LABELS[r] for r in PANEL_D_RESIDUES] + ["Overall"]
    hit_rates = [float(per_res.get(r, 0.0)) for r in PANEL_D_RESIDUES]
    hit_rates.append(float(stats["overall_six_pct"]))

    random_rates = [float(residue_random[r]) for r in PANEL_D_RESIDUES]
    random_rates.append(float(stats["cumulative_random_pct"]))

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    hit_closed = hit_rates + [hit_rates[0]]
    random_closed = random_rates + [random_rates[0]]

    ax.plot(
        angles_closed,
        hit_closed,
        "o-",
        linewidth=2,
        color=HIT_RATE_COLOR,
        markersize=5,
        zorder=4,
    )
    ax.fill(angles_closed, hit_closed, alpha=0.20, color=HIT_RATE_COLOR, zorder=3)

    ax.plot(
        angles_closed,
        random_closed,
        "o-",
        linewidth=2,
        color="#999999",
        markersize=5,
        zorder=4,
    )
    ax.fill(
        angles_closed,
        random_closed,
        color=RANDOM_BASELINE_COLOR,
        alpha=0.40,
        zorder=2,
    )

    r_data_max = max(hit_rates + random_rates)
    chart_r = min(r_data_max * 1.12, 100.0)
    label_pad = max(5.0, chart_r * 0.06)

    ax.set_ylim(0, chart_r + label_pad * 1.6)
    ax.set_yticks(_radar_grid_rings(chart_r))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.5)

    for angle, hit, rand in zip(angles, hit_rates, random_rates):
        spoke_outer = max(hit, rand)
        label_r = min(spoke_outer + label_pad, chart_r + label_pad * 1.4)
        ha, va = _radar_label_ha_va(angle)
        ax.text(
            angle,
            label_r,
            f"{hit:.1f}%",
            ha=ha,
            va=va,
            fontsize=7,
            fontweight="bold",
            color=HIT_RATE_COLOR,
            zorder=5,
        )

    rand_note = _random_baseline_legend_label(random_rates)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", pad=14)
    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor=HIT_RATE_COLOR,
                edgecolor=HIT_RATE_COLOR,
                alpha=0.45,
                label="Hit rate",
            ),
            mpatches.Patch(
                facecolor=RANDOM_BASELINE_COLOR,
                edgecolor="#999999",
                alpha=0.45,
                label=rand_note,
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
        framealpha=0.9,
        ncol=2,
    )
    ax.set_title(
        "D  Hit rate vs random baseline",
        loc="left",
        fontweight="bold",
        fontsize=11,
        pad=28,
        y=1.08,
    )


def create_figure(
    csv_path: Path,
    output_path: Path | None = None,
    skip_filter0: bool = False,
    radar: bool = False,
) -> Path:
    df = load_summary(csv_path)
    filters = get_filter_rows(df)
    f0_to_last_reduction = compute_f0_to_last_reduction(filters)
    overall_reduction = compute_overall_reduction(filters)
    accuracy = compute_filter_accuracy_stats(filters)
    failure_attribution = compute_failure_attribution(filters)
    total_failures = int(failure_attribution["N_Failures"].sum())

    residues = get_residue_rows(df)
    stats = compute_panel_d_stats(residues)
    stats["cumulative_random_pct"] = compute_cumulative_random_baseline(accuracy)
    if radar:
        if df[df["Group"] == "Filter_Residue_Type"].empty:
            raise ValueError(
                "Summary CSV has no Filter_Residue_Type rows required for radar panel D. "
                "Re-run baby_frank.py with --output-summary."
            )
        stats["residue_random_baselines"] = compute_residue_random_baselines(df, filters)

    if radar:
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_RADAR))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1, 1],
            width_ratios=[1, 1],
            hspace=0.42,
            wspace=0.34,
        )
    else:
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
    if radar:
        ax_d = fig.add_subplot(gs[1, 1], projection="polar")
    else:
        ax_d = fig.add_subplot(gs[1, 1])

    plot_search_space_sankey(ax_a, filters, f0_to_last_reduction, overall_reduction)
    plot_filter_accuracy_lines(ax_b, accuracy)
    plot_failure_attribution_bars(ax_c, failure_attribution, total_failures)
    if radar:
        plot_accuracy_radar(ax_d, stats, overall_reduction)
    else:
        plot_accuracy_bars(ax_d, stats, overall_reduction)
    if radar:
        fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.07)
    else:
        fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.08)

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=300, facecolor="white")
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
        help="Deprecated: Panel A always excludes Filter 0 (post-nucleophilic steps only).",
    )
    parser.add_argument(
        "--radar",
        action="store_true",
        help=(
            "Render panel D as a radar chart of hit rate vs cumulative random baseline "
            "for each nucleophilic residue type plus overall (6-residue aggregate)."
        ),
    )
    args = parser.parse_args()

    if not args.summary_csv.exists():
        raise SystemExit(f"File not found: {args.summary_csv}")

    out = create_figure(
        args.summary_csv,
        args.output,
        skip_filter0=args.skip_filter0,
        radar=args.radar,
    )
    print(f"[INFO] Figure written to: {out}")


if __name__ == "__main__":
    main()
