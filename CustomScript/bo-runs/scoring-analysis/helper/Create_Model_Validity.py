#!/usr/bin/env python3
"""Create a four-panel model-validity figure (Panels A–D).

Panel A: Feature-ablation dumbbell chart — hit rate @ top 10% (test).
Panel B: Feature-ablation dumbbell chart — NDCG@K (test).
Panel C: SHAP beeswarm for physicochemical descriptors (residue one-hots omitted).
Panel D: CV fold distribution vs held-out test hit rate @ top 10%.

Usage:
    python helper/Create_Model_Validity.py \
  --full-summary normal_summary.json \
  --cv-folds normal_cv_fold_results.csv \
  --shap-candidates normal_test_shap_per_candidate.csv \
  --ablation "Without SASA=sasa_ablation_summary.json" \
  --ablation "Without deprotonation=deprot_ablation_summary.json" \
  --ablation "Without residue type=residue_type_ablation_summary.json" \
  --ablation "Without QM reactivity=reactivity_ablation_summary.json" \
  --ablation "Without N-terminus=N_terminus_ablation_summary.json" \
  --ablation "Without QM reactivity and deprotonation=reactivity_deprot_ablation_summary.json" \
  --output model_validity.png
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

# Physicochemical model inputs (residue one-hots intentionally excluded).
PHYSICOCHEMICAL_FEATURES: list[tuple[str, str]] = [
    ("Rel_Side_SASA", "Relative SASA"),
    ("deprotonation_prob", "Deprotonation probability"),
    ("Partial_Charge_Deprotonated", "Partial charge"),
    ("Fukui_Deprotonated", "Fukui index"),
    ("Nucleophile_HOMO_Deprotonated", "HOMO"),
    ("Electrophile_LUMO_Deprotonated", "LUMO"),
    ("HOMO_LUMO_Gap_Deprotonated", "HOMO–LUMO gap"),
    ("Nucleophilicity_Index_Deprotonated", "Nucleophilicity"),
    ("N_Terminal", "N-terminal"),
]

DEFAULT_ABLATION_STEMS: list[tuple[str, str]] = [
    ("Without SASA", "ablation_no_sasa_summary.json"),
    ("Without deprotonation", "ablation_no_deprotonation_summary.json"),
    ("Without residue type", "ablation_no_residue_type_summary.json"),
    ("Without QM reactivity", "ablation_no_qm_summary.json"),
]

FULL_MODEL_COLOR = "#2C3E50"
ABLATION_COLOR = "#C44E52"
ABLATION_LINE_COLOR = "#AAAAAA"
CV_BOX_COLOR = "#4C72B0"
TEST_MARKER_COLOR = "#D62728"
FIG_WIDTH = 14.85
FIG_HEIGHT = 11.0


def _panel_title(ax: plt.Axes, panel_label: str, title: str) -> None:
    ax.set_title(f"{panel_label}  {title}", loc="left", fontweight="bold", fontsize=11)


def load_test_overall(summary_path: Path) -> dict:
    with summary_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    test = data.get("test_overall", {})
    if not test:
        raise KeyError(f"{summary_path}: missing test_overall section")
    return test


def load_test_metric(summary_path: Path, key: str) -> float:
    test = load_test_overall(summary_path)
    if key not in test:
        raise KeyError(f"{summary_path}: missing test_overall.{key}")
    return float(test[key])


def load_hit_at_top_pct(summary_path: Path) -> float:
    return load_test_metric(summary_path, "hit_at_top_pct")


def load_ndcg(summary_path: Path) -> float:
    return load_test_metric(summary_path, "ndcg")


def parse_ablation_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Ablation spec must be LABEL=PATH, got {spec!r}"
        )
    label, path_str = spec.split("=", 1)
    label = label.strip()
    path = Path(path_str.strip())
    if not label:
        raise ValueError(f"Empty ablation label in {spec!r}")
    return label, path


def resolve_ablations(
    full_summary: Path,
    cli_ablations: list[str],
) -> list[tuple[str, Path]]:
    if cli_ablations:
        return [parse_ablation_arg(spec) for spec in cli_ablations]

    base_dir = full_summary.parent
    resolved: list[tuple[str, Path]] = []
    for label, stem in DEFAULT_ABLATION_STEMS:
        candidate = base_dir / stem
        if candidate.is_file():
            resolved.append((label, candidate))
    return resolved


def _format_ablation_labels(labels: list[str], *, width: int = 20) -> list[str]:
    """Wrap long ablation names so y-tick text does not intrude into the plot."""
    wrapped: list[str] = []
    for label in labels:
        if len(label) <= width:
            wrapped.append(label)
            continue
        if label.lower().startswith("without ") and " and " in label:
            wrapped.append(label.replace(" and ", "\nand ", 1))
            continue
        wrapped.append(textwrap.fill(label, width=width, break_long_words=False))
    return wrapped


def _order_ablations(
    full_value: float,
    ablations: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    drops = [full_value - value for _, value in ablations]
    order = np.argsort(drops)[::-1]
    return [ablations[i] for i in order]


def plot_ablation_dumbbell(
    ax: plt.Axes,
    full_value: float,
    ablations: list[tuple[str, float]],
    *,
    panel_label: str,
    title: str,
    xlabel: str,
    as_percent: bool = False,
    show_y_labels: bool = True,
) -> None:
    """Horizontal dumbbell: full model (left) → ablated (right) per feature group."""
    if not ablations:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No ablation summary JSONs found.\n"
            "Pass --ablation LABEL=PATH for each leave-one-out run.",
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
            transform=ax.transAxes,
        )
        _panel_title(ax, panel_label, title)
        return

    labels = [label for label, _ in ablations]
    ablated_values = [value for _, value in ablations]

    labels = _format_ablation_labels(labels)
    y_pos = np.arange(len(labels))
    scale = 100.0 if as_percent else 1.0
    full_scaled = full_value * scale
    ablated_scaled = [v * scale for v in ablated_values]

    for yi, abl in zip(y_pos, ablated_scaled):
        ax.plot(
            [full_scaled, abl],
            [yi, yi],
            color=ABLATION_LINE_COLOR,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            abl,
            yi,
            s=70,
            color=ABLATION_COLOR,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    ax.scatter(
        [full_scaled] * len(y_pos),
        y_pos,
        s=70,
        color=FULL_MODEL_COLOR,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="Full model",
    )

    xmin = min(ablated_scaled + [full_scaled]) - 0.03 * scale
    xmax = max(ablated_scaled + [full_scaled]) + 0.03 * scale
    if as_percent:
        ax.set_xlim(max(0.0, xmin), min(100.0, xmax))
    else:
        ax.set_xlim(max(0.0, xmin), xmax)
    ax.set_yticks(y_pos)
    if show_y_labels:
        ax.set_yticklabels(labels, fontsize=8, ha="right")
        ax.tick_params(axis="y", pad=6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.axvline(full_scaled, color=FULL_MODEL_COLOR, linestyle="--", linewidth=0.9, alpha=0.35)
    ax.legend(
        handles=[
            mpatches.Patch(color=FULL_MODEL_COLOR, label="Full model"),
            mpatches.Patch(color=ABLATION_COLOR, label="Feature removed"),
        ],
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )
    _panel_title(ax, panel_label, title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _subsample_shap_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _beeswarm_jitter(n: int, row_height: float = 0.35, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * row_height


def plot_shap_beeswarm_matplotlib(
    ax: plt.Axes,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
) -> None:
    """Matplotlib beeswarm fallback when shap is unavailable."""
    n_rows = len(feature_names)
    cmap = cm.coolwarm
    norm = mcolors.Normalize(
        vmin=np.nanpercentile(feature_values, 5),
        vmax=np.nanpercentile(feature_values, 95),
    )

    for row_idx, name in enumerate(feature_names):
        shap_row = shap_values[:, row_idx]
        feat_row = feature_values[:, row_idx]
        y = row_idx + _beeswarm_jitter(len(shap_row), seed=row_idx + 7)
        colors = cmap(norm(feat_row))
        ax.scatter(
            shap_row,
            y,
            c=colors,
            s=6,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#CCCCCC", linewidth=0.8, zorder=0)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Feature value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def plot_shap_beeswarm(
    ax: plt.Axes,
    shap_path: Path,
    *,
    max_rows: int,
    seed: int,
) -> None:
    usecols = ["Name"]
    for feat, _ in PHYSICOCHEMICAL_FEATURES:
        usecols.extend([feat, f"shap_{feat}"])
    df = pd.read_csv(shap_path, usecols=usecols)
    n_total = len(df)
    df = _subsample_shap_rows(df, max_rows=max_rows, seed=seed)
    subsampled = len(df) < n_total

    feat_cols = [feat for feat, _ in PHYSICOCHEMICAL_FEATURES]
    shap_cols = [f"shap_{feat}" for feat in feat_cols]
    display_names = [label for _, label in PHYSICOCHEMICAL_FEATURES]

    missing = [c for c in feat_cols + shap_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{shap_path}: missing SHAP columns {missing[:5]}")

    shap_values = df[shap_cols].to_numpy(dtype=np.float32)
    feature_values = df[feat_cols].to_numpy(dtype=np.float32)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    shap_values = shap_values[:, order]
    feature_values = feature_values[:, order]
    ordered_names = [display_names[i] for i in order]

    if shap is None:
        plot_shap_beeswarm_matplotlib(ax, shap_values, feature_values, ordered_names)
    else:
        explanation = shap.Explanation(
            values=shap_values,
            data=feature_values,
            feature_names=ordered_names,
        )
        plt.sca(ax)
        shap.plots.beeswarm(
            explanation,
            max_display=len(ordered_names),
            show=False,
            plot_size=None,
        )
    ax.set_xlabel("SHAP value (impact on model score)")
    ax.set_ylabel("")
    _panel_title(ax, "C", "SHAP summary")
    note = f"Physicochemical descriptors only; n={len(df):,} candidate residues"
    if subsampled:
        note += f" (subsampled from {n_total:,})"
    ax.text(
        0.0,
        -0.22,
        note,
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
    )


def plot_cv_vs_test(
    ax: plt.Axes,
    cv_path: Path,
    test_rate: float,
) -> None:
    cv_df = pd.read_csv(cv_path)
    fold_col = cv_df["fold"]
    is_fold = fold_col.apply(lambda v: str(v).strip().lower() not in {"mean", "avg", "average"})
    folds = cv_df.loc[is_fold, "hit_at_top_pct"].astype(float)
    if folds.empty:
        raise ValueError(f"{cv_path}: no numeric CV fold rows found")

    fold_pct = folds.to_numpy() * 100.0
    test_pct = test_rate * 100.0

    bp = ax.boxplot(
        [fold_pct],
        widths=0.45,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": CV_BOX_COLOR, "alpha": 0.55, "edgecolor": CV_BOX_COLOR},
        whiskerprops={"color": CV_BOX_COLOR},
        capprops={"color": CV_BOX_COLOR},
    )
    _ = bp  # silence linter

    ax.scatter(
        1.0,
        test_pct,
        marker="D",
        s=110,
        color=TEST_MARKER_COLOR,
        edgecolors="white",
        linewidths=1.0,
        zorder=5,
        label="Held-out test (20%)",
    )

    ymin = min(fold_pct.min(), test_pct) - 4.0
    ymax = max(fold_pct.max(), test_pct) + 4.0
    ax.set_ylim(max(0.0, ymin), min(100.0, ymax))
    ax.set_xticks([1])
    ax.set_xticklabels([f"CV folds (n={len(fold_pct)})"], fontsize=9)
    ax.set_ylabel("Hit rate @ top 10%")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _panel_title(ax, "D", "CV vs held-out test")
    ax.text(
        0.0,
        -0.22,
        f"Sequence-clustered folds; test marker = {test_pct:.1f}%",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figure(
    *,
    full_summary: Path,
    cv_folds: Path,
    shap_candidates: Path,
    ablation_specs: list[str],
    output_path: Path,
    shap_max_rows: int,
    seed: int,
) -> None:
    full_test = load_test_overall(full_summary)
    full_hit = float(full_test["hit_at_top_pct"])
    full_ndcg = float(full_test["ndcg"])
    ndcg_k = int(full_test.get("k", 15))

    ablation_paths = resolve_ablations(full_summary, ablation_specs)
    ablation_hits: list[tuple[str, float]] = []
    ablation_ndcgs: list[tuple[str, float]] = []
    for label, path in ablation_paths:
        if not path.is_file():
            print(f"[WARN] Ablation summary not found, skipping: {path}")
            continue
        hit = load_hit_at_top_pct(path)
        ndcg = load_ndcg(path)
        ablation_hits.append((label, hit))
        ablation_ndcgs.append((label, ndcg))
        print(
            f"  Ablation '{label}': hit@top10% = {hit * 100:.1f}%, "
            f"NDCG@{ndcg_k} = {ndcg:.3f}"
        )

    print(f"  Full model test hit@top10% = {full_hit * 100:.1f}%")
    print(f"  Full model test NDCG@{ndcg_k} = {full_ndcg:.3f}")

    ablation_hits = _order_ablations(full_hit, ablation_hits)
    label_order = [label for label, _ in ablation_hits]
    ndcg_by_label = dict(ablation_ndcgs)
    ablation_ndcgs = [(label, ndcg_by_label[label]) for label in label_order]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.08, 1],
        width_ratios=[1.12, 1],
        hspace=0.40,
        wspace=0.22,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_ablation_dumbbell(
        ax_a,
        full_hit,
        ablation_hits,
        panel_label="A",
        title="Feature ablation (hit rate)",
        xlabel="Hit rate @ top 10%",
        as_percent=True,
    )
    plot_ablation_dumbbell(
        ax_b,
        full_ndcg,
        ablation_ndcgs,
        panel_label="B",
        title="Feature ablation (NDCG)",
        xlabel=f"NDCG@{ndcg_k} (test)",
        as_percent=False,
        show_y_labels=False,
    )
    plot_shap_beeswarm(ax_c, shap_candidates, max_rows=shap_max_rows, seed=seed)
    plot_cv_vs_test(ax_d, cv_folds, full_hit)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved figure -> {output_path}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    analysis_dir = here.parent

    p = argparse.ArgumentParser(
        description="Create four-panel model validity figure (ablation, SHAP, CV vs test)."
    )
    p.add_argument(
        "--full-summary",
        type=Path,
        default=analysis_dir / "normal_summary.json",
        help="Full-model summary JSON (test_overall.hit_at_top_pct).",
    )
    p.add_argument(
        "--cv-folds",
        type=Path,
        default=analysis_dir / "normal_cv_fold_results.csv",
        help="CV fold results CSV (hit_at_top_pct per fold).",
    )
    p.add_argument(
        "--shap-candidates",
        type=Path,
        default=analysis_dir / "normal_test_shap_per_candidate.csv",
        help="Per-candidate SHAP CSV from Training_Cov_Screen.py.",
    )
    p.add_argument(
        "--ablation",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Leave-one-out ablation summary JSON. Repeat for each condition. "
             "If omitted, auto-discovers ablation_*_summary.json next to --full-summary.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=analysis_dir / "model_validity.png",
        help="Output figure path (default: scoring-analysis/model_validity.png).",
    )
    p.add_argument(
        "--shap-max-rows",
        type=int,
        default=4000,
        help="Max rows for SHAP beeswarm (random subsample; default 4000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SHAP subsampling.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.full_summary, args.cv_folds, args.shap_candidates):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    create_figure(
        full_summary=args.full_summary,
        cv_folds=args.cv_folds,
        shap_candidates=args.shap_candidates,
        ablation_specs=args.ablation,
        output_path=args.output,
        shap_max_rows=args.shap_max_rows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
