#!/usr/bin/env python3
"""fpocket analysis figure: hit-rate tradeoff (A) + two effect-size forests (B, C).

Panel A — CovSite vs fpocket top-1 / top-3 / all: hit rate vs median search
space reduction (from test_fpocket_summary.csv).

With optional --covsite-top1-hit-rate / --covsite-top3-hit-rate, Panel A also
plots those CovSite points (SSR from summary / label-results). Omitted flags
are simply left out of the plot.

Panel B — All fpocket top-1 pockets (hit or miss) vs the actual target-containing
pocket whenever fpocket finds it (top-1 if that pocket has the site, else
fpocket_best_*).

Panel C — All fpocket top-1 pockets vs fpocket_best_* only for
top-1 miss ∩ CovSite hit ∩ overall hit (CovSite-rediscovered target pockets).

Both forests use Cohen's d (top-1 − comparator) as the primary signal; Mann–Whitney
p-values are BH-corrected within each panel across all tested features; asterisks
mark q < 0.05 on the curated geometry / chemistry subsets.

Also writes companion stats CSVs next to the figure:
  <stem>_panel_a_stats.csv, <stem>_panel_b_stats.csv, <stem>_panel_c_stats.csv

Usage (from scoring-analysis/ or helper/):
    python helper/figure_f_pocket.py
    python helper/figure_f_pocket.py \
        --summary fpocket_analysis/test_fpocket_summary.csv \
        --detail fpocket_analysis/test_fpocket_detail.csv \
        --output fpocket_analysis/fpocket_figure.png
    python helper/figure_f_pocket.py \
        --summary fpocket_analysis/test_fpocket_summary.csv \
        --detail fpocket_analysis/test_fpocket_detail.csv \
        --covsite-top1-hit-rate 0.85 \
        --covsite-top3-hit-rate 0.92 \
        --output fpocket_analysis/fpocket_figure.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = HERE.parent / "fpocket_analysis"

# Curated forest features.
GEOMETRY_FEATURES = [
    "score",
    "druggability_score",
    "n_alpha_spheres",
    "volume",
    "total_sasa",
]
CHEMISTRY_FEATURES = [
    "hydrophobicity_score",
    "charge_score",
    "flexibility",
    "prop_polar_atm",
]
CURATED_FEATURES = GEOMETRY_FEATURES + CHEMISTRY_FEATURES

# Full feature set for BH correction (matches check_f_pocket top-1 analysis).
BH_FEATURES = [
    "score",
    "druggability_score",
    "n_alpha_spheres",
    "total_sasa",
    "polar_sasa",
    "apolar_sasa",
    "volume",
    "mean_local_hydrophobic_density",
    "mean_alpha_sphere_radius",
    "mean_alpha_sphere_solvent_access",
    "apolar_alpha_sphere_proportion",
    "hydrophobicity_score",
    "volume_score",
    "polarity_score",
    "charge_score",
    "flexibility",
    "prop_polar_atm",
    "nucleophilic_count",
    "nucleophilic_cys",
    "nucleophilic_ser",
    "nucleophilic_his",
    "nucleophilic_lys",
    "nucleophilic_tyr",
    "nucleophilic_thr",
]

FEATURE_LABELS = {
    "score": "fpocket score",
    "druggability_score": "Druggability score",
    "n_alpha_spheres": "N alpha-spheres",
    "volume": "Volume",
    "total_sasa": "Total SASA",
    "hydrophobicity_score": "Hydrophobicity score",
    "charge_score": "Charge score",
    "flexibility": "Flexibility",
    "prop_polar_atm": "Prop. polar atoms",
}

COVSITE_COLOR = "#2a78d6"
FPOCKET_COLORS = {
    "fpocket top-1": "#e34948",
    "fpocket top-3": "#DD8452",
    "fpocket all": "#8172B3",
}
GEOMETRY_COLOR = "#2C7A4B"
CHEMISTRY_COLOR = "#73726c"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fpocket A/B/C analysis figure")
    p.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_ANALYSIS / "test_fpocket_summary.csv",
    )
    p.add_argument(
        "--detail",
        type=Path,
        default=DEFAULT_ANALYSIS / "test_fpocket_detail.csv",
        help="Per-label fpocket detail CSV (top-1 and best-pocket descriptors)",
    )
    p.add_argument(
        "--label-results",
        type=Path,
        default=None,
        help="Optional CovSite label-results CSV for Panel A SSR fallback",
    )
    p.add_argument(
        "--covsite-top1-hit-rate",
        type=float,
        default=None,
        help=(
            "Optional CovSite top-1%% hit rate for Panel A. "
            "Omit to exclude the CovSite top 1%% point."
        ),
    )
    p.add_argument(
        "--covsite-top3-hit-rate",
        type=float,
        default=None,
        help=(
            "Optional CovSite top-3%% hit rate for Panel A. "
            "Omit to exclude the CovSite top 3%% point."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANALYSIS / "fpocket_figure.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    for name, val in (
        ("--covsite-top1-hit-rate", args.covsite_top1_hit_rate),
        ("--covsite-top3-hit-rate", args.covsite_top3_hit_rate),
    ):
        if val is not None and not (0.0 <= val <= 1.0):
            p.error(f"{name} must be in [0, 1], got {val}")
    return args


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=12, pad=8)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return BH FDR q-values for a 1-d array of p-values (NaNs preserved)."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q_ranked
    q[valid] = out
    return q


def cohens_d(mean_a: float, std_a: float, n_a: float,
             mean_b: float, std_b: float, n_b: float) -> tuple[float, float]:
    """Cohen's d (A − B) and analytical SE for unequal-n pooled SD."""
    n_a, n_b = float(n_a), float(n_b)
    df = n_a + n_b - 2.0
    if df <= 0 or n_a < 2 or n_b < 2:
        return float("nan"), float("nan")
    pooled = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / df)
    if pooled == 0 or not np.isfinite(pooled):
        return float("nan"), float("nan")
    d = (mean_a - mean_b) / pooled
    se = np.sqrt((n_a + n_b) / (n_a * n_b) + d**2 / (2.0 * (n_a + n_b)))
    return float(d), float(se)


def _median_fraction_row(summary: pd.DataFrame, scope: str) -> pd.Series:
    rows = summary[
        (summary["section"] == "search_space_reduction")
        & (summary["metric"] == "median_fraction")
        & (summary["scope"] == scope)
    ]
    if rows.empty:
        raise KeyError(
            f"Missing search_space_reduction/median_fraction row with scope={scope!r}"
        )
    return rows.iloc[0]


def _fpocket_reduction_from_remaining(fraction: float) -> float:
    """fpocket summary median_fraction = nucleophilic kept / n_residues; plot 1 − f."""
    return 1.0 - float(fraction)


def _covsite_search_space_reduction(
    summary: pd.DataFrame,
    label_results: Path | None = None,
) -> float:
    """
    Median CovSite search-space reduction.

    Prefer summary scope=covsite (already a reduction). Else fall back to median
    search_space_reduction in the label-results CSV.
    """
    rows = summary[
        (summary["section"] == "search_space_reduction")
        & (summary["metric"] == "median_fraction")
        & (summary["scope"] == "covsite")
    ]
    if not rows.empty and pd.notna(rows.iloc[0]["top1"]):
        return float(rows.iloc[0]["top1"])

    candidates: list[Path] = []
    if label_results is not None:
        candidates.append(label_results)
    candidates.append(HERE.parent / "test_label_results.csv")

    for path in candidates:
        if not path.is_file():
            continue
        labels = pd.read_csv(path, low_memory=False)
        if "search_space_reduction" not in labels.columns:
            continue
        vals = pd.to_numeric(labels["search_space_reduction"], errors="coerce").dropna()
        if vals.empty:
            continue
        print(
            f"[INFO] CovSite SSR not in summary; using median from {path.name} "
            f"= {float(vals.median()):.6f}"
        )
        return float(vals.median())

    raise KeyError(
        "Missing CovSite search-space reduction: add scope=covsite to the summary "
        "or provide --label-results with a search_space_reduction column"
    )


def load_panel_a_points(
    summary: pd.DataFrame,
    label_results: Path | None = None,
    covsite_top1_hit_rate: float | None = None,
    covsite_top3_hit_rate: float | None = None,
) -> pd.DataFrame:
    """Hit rate vs median search-space reduction for CovSite and fpocket scopes.

    Each optional CovSite hit-rate flag adds its own point; omitted flags are
    excluded. Without either flag, Panel A is fpocket-only.
    """
    hit = summary[
        (summary["section"] == "hit_rate")
        & (summary["metric"] == "fpocket_target_hit_rate")
    ].iloc[0]
    ss = _median_fraction_row(summary, "when_hit")

    rows: list[dict] = []
    if covsite_top1_hit_rate is not None or covsite_top3_hit_rate is not None:
        cov_reduction = _covsite_search_space_reduction(summary, label_results)
        if covsite_top1_hit_rate is not None:
            rows.append(
                {
                    "method": "CovSite top 1%",
                    "hit_rate": float(covsite_top1_hit_rate),
                    "search_space_reduction": float(cov_reduction),
                    "color": COVSITE_COLOR,
                    "marker": "D",
                    "size": 90,
                }
            )
        if covsite_top3_hit_rate is not None:
            rows.append(
                {
                    "method": "CovSite top 3%",
                    "hit_rate": float(covsite_top3_hit_rate),
                    "search_space_reduction": float(cov_reduction),
                    "color": COVSITE_COLOR,
                    "marker": "P",
                    "size": 100,
                }
            )

    rows.extend(
        [
            {
                "method": "fpocket top-1",
                "hit_rate": float(hit["top1"]),
                "search_space_reduction": _fpocket_reduction_from_remaining(ss["top1"]),
                "color": FPOCKET_COLORS["fpocket top-1"],
                "marker": "o",
                "size": 90,
            },
            {
                "method": "fpocket top-3",
                "hit_rate": float(hit["top3"]),
                "search_space_reduction": _fpocket_reduction_from_remaining(ss["top3"]),
                "color": FPOCKET_COLORS["fpocket top-3"],
                "marker": "s",
                "size": 90,
            },
            {
                "method": "fpocket all",
                "hit_rate": float(hit["overall"]),
                "search_space_reduction": _fpocket_reduction_from_remaining(ss["overall"]),
                "color": FPOCKET_COLORS["fpocket all"],
                "marker": "^",
                "size": 100,
            },
        ]
    )
    return pd.DataFrame(rows)


def _series_top1(detail: pd.DataFrame, feature: str) -> pd.Series:
    col = f"fpocket_top1_{feature}"
    if col not in detail.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(detail[col], errors="coerce").dropna()


def _series_target_pocket(detail: pd.DataFrame, feature: str) -> pd.Series:
    """Target-containing pocket: top-1 if it hits, else best pocket when overall hit."""
    top1_col = f"fpocket_top1_{feature}"
    best_col = f"fpocket_best_{feature}"
    if top1_col not in detail.columns or best_col not in detail.columns:
        return pd.Series(dtype=float)

    top1_hit = detail["fpocket_top1_hit"].astype(int) == 1
    overall_hit = detail["fpocket_overall_hit"].astype(int) == 1
    vals = pd.Series(np.nan, index=detail.index, dtype=float)
    vals.loc[top1_hit] = pd.to_numeric(detail.loc[top1_hit, top1_col], errors="coerce")
    miss_but_found = (~top1_hit) & overall_hit
    vals.loc[miss_but_found] = pd.to_numeric(
        detail.loc[miss_but_found, best_col], errors="coerce"
    )
    return vals.dropna()


def _series_recovered_best(detail: pd.DataFrame, feature: str) -> pd.Series:
    """fpocket_best_* for top-1 miss ∩ CovSite hit ∩ overall hit."""
    best_col = f"fpocket_best_{feature}"
    if best_col not in detail.columns:
        return pd.Series(dtype=float)
    mask = (
        (detail["fpocket_top1_hit"].astype(int) == 0)
        & (detail["covsite_hit_at_top_pct"].astype(int) == 1)
        & (detail["fpocket_overall_hit"].astype(int) == 1)
    )
    return pd.to_numeric(detail.loc[mask, best_col], errors="coerce").dropna()


def _mannwhitney_p(a: pd.Series, b: pd.Series) -> float:
    if scipy_stats is None:
        raise RuntimeError("scipy is required for Mann-Whitney tests")
    if len(a) < 1 or len(b) < 1:
        return float("nan")
    return float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def build_effect_table(
    detail: pd.DataFrame,
    comparator: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Cohen's d + BH q-values for top-1 vs a comparator group.

    comparator:
      'target_pocket' — any fpocket-found target pocket (Panel B)
      'recovered_best' — CovSite-rediscovered misses only (Panel C)

    Returns (curated_for_plot, all_features_for_export, sample_sizes).
    """
    records_all: list[dict] = []
    n_top1 = n_comp = 0
    curated_set = set(CURATED_FEATURES)
    group_of = {
        **{f: "Geometry / druggability" for f in GEOMETRY_FEATURES},
        **{f: "Chemistry / composition" for f in CHEMISTRY_FEATURES},
    }

    for feat in BH_FEATURES:
        a = _series_top1(detail, feat)
        if comparator == "target_pocket":
            b = _series_target_pocket(detail, feat)
            comp_label = "target_pocket"
        elif comparator == "recovered_best":
            b = _series_recovered_best(detail, feat)
            comp_label = "covsite_rediscovered_target_pocket"
        else:
            raise ValueError(f"Unknown comparator: {comparator!r}")

        if feat == "score":
            n_top1, n_comp = len(a), len(b)

        if len(a) < 2 or len(b) < 2:
            continue

        mean_a, std_a = float(a.mean()), float(a.std(ddof=1))
        mean_b, std_b = float(b.mean()), float(b.std(ddof=1))
        d, se = cohens_d(mean_a, std_a, len(a), mean_b, std_b, len(b))
        records_all.append(
            {
                "panel_comparison": comparator,
                "group_a": "all_fpocket_top1",
                "group_b": comp_label,
                "feature": feat,
                "feature_label": FEATURE_LABELS.get(feat, feat),
                "feature_group": group_of.get(feat, ""),
                "shown_in_figure": feat in curated_set,
                "n_top1": len(a),
                "mean_top1": mean_a,
                "median_top1": float(a.median()),
                "std_top1": std_a,
                "n_comp": len(b),
                "mean_comp": mean_b,
                "median_comp": float(b.median()),
                "std_comp": std_b,
                "cohens_d": d,
                "cohens_d_se": se,
                "ci95_low": d - 1.96 * se,
                "ci95_high": d + 1.96 * se,
                "mannwhitney_pvalue": _mannwhitney_p(a, b),
            }
        )

    all_df = pd.DataFrame(records_all)
    ns = {"n_top1": n_top1, "n_comp": n_comp}
    if all_df.empty:
        return all_df, all_df, ns

    all_df["qvalue_bh"] = benjamini_hochberg(all_df["mannwhitney_pvalue"].to_numpy())
    all_df["significant_q_0.05"] = all_df["qvalue_bh"] < 0.05

    curated: list[dict] = []
    for group_name, features, color in (
        ("Geometry / druggability", GEOMETRY_FEATURES, GEOMETRY_COLOR),
        ("Chemistry / composition", CHEMISTRY_FEATURES, CHEMISTRY_COLOR),
    ):
        for feat in features:
            row = all_df.loc[all_df["feature"] == feat]
            if row.empty:
                raise KeyError(f"Feature missing from detail-derived effects: {feat}")
            r = row.iloc[0]
            curated.append(
                {
                    "feature": feat,
                    "label": FEATURE_LABELS.get(feat, feat),
                    "group": group_name,
                    "color": color,
                    "cohens_d": float(r["cohens_d"]),
                    "se": float(r["cohens_d_se"]),
                    "ci_low": float(r["ci95_low"]),
                    "ci_high": float(r["ci95_high"]),
                    "pvalue": float(r["mannwhitney_pvalue"]),
                    "qvalue": float(r["qvalue_bh"]),
                }
            )
    return pd.DataFrame(curated), all_df, ns


def export_figure_stats(
    out_dir: Path,
    stem: str,
    points: pd.DataFrame,
    stats_b: pd.DataFrame,
    stats_c: pd.DataFrame,
    ns_b: dict[str, int],
    ns_c: dict[str, int],
) -> list[Path]:
    """Write Panel A–C stats CSVs next to the figure for manuscript tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    panel_a = points[["method", "hit_rate", "search_space_reduction"]].copy()
    panel_a.insert(0, "panel", "A")
    cov_from_flags = points["method"].isin(["CovSite top 1%", "CovSite top 3%"]).any()
    panel_a["note"] = (
        (
            "CovSite hit_rate from --covsite-top1/top3-hit-rate flags; "
            if cov_from_flags
            else "hit_rate from summary; "
        )
        + "search_space_reduction is median "
        "(CovSite: reduction; fpocket: 1 - remaining nucleophilic fraction)"
    )
    path_a = out_dir / f"{stem}_panel_a_stats.csv"
    panel_a.to_csv(path_a, index=False)
    written.append(path_a)

    def _finalize_panel(stats: pd.DataFrame, panel: str, comparison_note: str) -> pd.DataFrame:
        out = stats.copy()
        out.insert(0, "panel", panel)
        out["comparison_note"] = comparison_note
        # Put curated figure features first, preserving BH_FEATURES order within.
        out["__order"] = out["feature"].map(
            {f: i for i, f in enumerate(BH_FEATURES)}
        )
        out = out.sort_values(
            ["shown_in_figure", "__order"],
            ascending=[False, True],
        ).drop(columns="__order")
        return out

    path_b = out_dir / f"{stem}_panel_b_stats.csv"
    _finalize_panel(
        stats_b,
        "B",
        (
            f"all fpocket top-1 (n={ns_b['n_top1']}) vs target-containing pocket "
            f"when fpocket finds it — top-1 if hit else fpocket_best (n={ns_b['n_comp']}); "
            "Cohen's d = top-1 − comparator; BH q over all listed features"
        ),
    ).to_csv(path_b, index=False)
    written.append(path_b)

    path_c = out_dir / f"{stem}_panel_c_stats.csv"
    _finalize_panel(
        stats_c,
        "C",
        (
            f"all fpocket top-1 (n={ns_c['n_top1']}) vs fpocket_best for "
            f"top-1 miss ∩ CovSite hit ∩ overall hit (n={ns_c['n_comp']}); "
            "Cohen's d = top-1 − comparator; BH q over all listed features"
        ),
    ).to_csv(path_c, index=False)
    written.append(path_c)

    return written


def plot_panel_a(ax: plt.Axes, points: pd.DataFrame) -> None:
    # CovSite label offsets: single point above; if both share an x, put the
    # higher-hit-rate label above and the lower below so offsets do not cross.
    cov_pct = points[points["method"].isin(["CovSite top 1%", "CovSite top 3%"])]
    cov_label_dy: dict[str, float] = {}
    if len(cov_pct) == 1:
        cov_label_dy[str(cov_pct.iloc[0]["method"])] = 3.0
    elif len(cov_pct) == 2:
        ordered = cov_pct.sort_values("hit_rate", ascending=False)
        cov_label_dy[str(ordered.iloc[0]["method"])] = 4.0
        cov_label_dy[str(ordered.iloc[1]["method"])] = -4.0

    for _, row in points.iterrows():
        x = 100.0 * float(row["search_space_reduction"])
        y = 100.0 * float(row["hit_rate"])
        ax.scatter(
            x,
            y,
            s=row["size"],
            c=row["color"],
            marker=row["marker"],
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        dx, dy = 3.5, 0.0
        ha = "left"
        method = str(row["method"])
        if method in cov_label_dy:
            dx, dy, ha = -3.0, cov_label_dy[method], "right"
        elif method == "fpocket all":
            dx, dy, ha = 3.0, -4.0, "left"
        elif method == "fpocket top-3":
            dx, dy, ha = -3.0, 3.5, "right"
        elif method == "fpocket top-1":
            dx, dy, ha = -3.0, -4.0, "right"
        ax.annotate(
            method,
            (x, y),
            xytext=(x + dx, y + dy),
            textcoords="data",
            fontsize=8.5,
            color=row["color"],
            ha=ha,
            va="center",
            fontweight="medium",
        )

    ax.set_xlabel("Median search space reduction (%)", fontsize=10)
    ax.set_ylabel("Hit rate (%)", fontsize=10)
    ax.set_xlim(-5, 108)
    ax.set_ylim(25, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    ax.axvline(0.0, color="#dddddd", lw=0.8, zorder=0)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _panel_letter(ax, "A")


def plot_forest(
    ax: plt.Axes,
    effects: pd.DataFrame,
    *,
    letter: str,
    xlabel: str,
    show_legend: bool,
) -> None:
    """Forest plot of Cohen's d with BH q < 0.05 asterisks.

    Feature order is fixed (GEOMETRY_FEATURES then CHEMISTRY_FEATURES) so
    panels B and C share the same y-axis labeling.
    """
    y_labels: list[str] = []
    y_pos: list[float] = []
    feature_rows: list[tuple[float, pd.Series]] = []
    y = 0.0
    for group_name, feat_order in (
        ("Geometry / druggability", GEOMETRY_FEATURES),
        ("Chemistry / composition", CHEMISTRY_FEATURES),
    ):
        if y_pos:
            y += 0.6
        y_labels.append(group_name)
        y_pos.append(y)
        y += 1.0
        block = effects[effects["group"] == group_name].set_index("feature")
        for feat in feat_order:
            if feat not in block.index:
                continue
            row = block.loc[feat]
            y_labels.append(str(row["label"]))
            y_pos.append(y)
            feature_rows.append((y, row))
            y += 1.0

    ax.axvline(0.0, color="#bbbbbb", lw=1.0, zorder=0)

    x_lo, x_hi = -0.4, 0.4
    for ypos, row in feature_rows:
        ax.errorbar(
            row["cohens_d"],
            ypos,
            xerr=1.96 * row["se"],
            fmt="none",
            ecolor=row["color"],
            elinewidth=1.4,
            capsize=3,
            capthick=1.2,
            zorder=2,
        )
        ax.scatter(
            row["cohens_d"],
            ypos,
            s=55,
            c=row["color"],
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
        if np.isfinite(row["qvalue"]) and row["qvalue"] < 0.05:
            x_star = row["ci_high"] + 0.06 if row["cohens_d"] >= 0 else row["ci_low"] - 0.06
            ha = "left" if row["cohens_d"] >= 0 else "right"
            ax.text(
                x_star,
                ypos,
                "*",
                ha=ha,
                va="center",
                fontsize=12,
                fontweight="bold",
                color=row["color"],
            )
        x_lo = min(x_lo, row["ci_low"] - 0.15)
        x_hi = max(x_hi, row["ci_high"] + 0.15)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    for tick, label in zip(ax.get_yticklabels(), y_labels):
        if label in ("Geometry / druggability", "Chemistry / composition"):
            tick.set_fontweight("bold")
            tick.set_fontsize(9)
            tick.set_color("#222222")
        else:
            tick.set_color("#333333")

    ax.set_xlabel(xlabel, fontsize=9.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.5, y - 0.4)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        geom_patch = mpatches.Patch(color=GEOMETRY_COLOR, label="Geometry / druggability")
        chem_patch = mpatches.Patch(color=CHEMISTRY_COLOR, label="Chemistry / composition")
        ax.legend(
            handles=[geom_patch, chem_patch],
            loc="lower right",
            fontsize=8,
            frameon=False,
            title="* BH q < 0.05",
            title_fontsize=8,
        )
    _panel_letter(ax, letter)


def _print_effects(tag: str, effects: pd.DataFrame, ns: dict[str, int]) -> None:
    print(f"\n{tag} (n_top1={ns['n_top1']}, n_comp={ns['n_comp']}):")
    show = effects[["group", "label", "cohens_d", "pvalue", "qvalue"]].copy()
    show["cohens_d"] = show["cohens_d"].map(lambda x: f"{x:.3f}")
    show["pvalue"] = show["pvalue"].map(lambda x: f"{x:.2e}")
    show["qvalue"] = show["qvalue"].map(lambda x: f"{x:.2e}")
    print(show.to_string(index=False))


def main() -> None:
    args = parse_args()
    if scipy_stats is None:
        raise SystemExit("[ERROR] scipy is required (Mann-Whitney + forest panels)")

    summary = pd.read_csv(args.summary)
    detail = pd.read_csv(args.detail, low_memory=False)

    points = load_panel_a_points(
        summary,
        args.label_results,
        covsite_top1_hit_rate=args.covsite_top1_hit_rate,
        covsite_top3_hit_rate=args.covsite_top3_hit_rate,
    )
    if (
        args.covsite_top1_hit_rate is not None
        or args.covsite_top3_hit_rate is not None
    ):
        parts = []
        if args.covsite_top1_hit_rate is not None:
            parts.append(f"top 1%={args.covsite_top1_hit_rate:.6f}")
        if args.covsite_top3_hit_rate is not None:
            parts.append(f"top 3%={args.covsite_top3_hit_rate:.6f}")
        print(f"[INFO] Panel A CovSite hit rates from flags: {', '.join(parts)}")
    effects_b, stats_b, ns_b = build_effect_table(detail, "target_pocket")
    effects_c, stats_c, ns_c = build_effect_table(detail, "recovered_best")

    print(f"BH correction within each panel over {len(BH_FEATURES)} Mann-Whitney features.")
    print("\nPanel A points:")
    print(points[["method", "hit_rate", "search_space_reduction"]].to_string(index=False))
    _print_effects("Panel B  all top-1 vs target pocket", effects_b, ns_b)
    _print_effects(
        "Panel C  all top-1 vs CovSite-rediscovered best pocket",
        effects_c,
        ns_c,
    )

    fig = plt.figure(figsize=(12.5, 9.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.0],
                          wspace=0.55, hspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis("off")

    plot_panel_a(ax_a, points)
    plot_forest(
        ax_b,
        effects_b,
        letter="B",
        xlabel="Cohen's d  (all top-1 − target pocket)",
        show_legend=True,
    )
    plot_forest(
        ax_c,
        effects_c,
        letter="C",
        xlabel="Cohen's d  (all top-1 − CovSite-rediscovered target)",
        show_legend=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nWrote {args.output}")

    for path in export_figure_stats(
        args.output.parent,
        args.output.stem,
        points,
        stats_b,
        stats_c,
        ns_b,
        ns_c,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
