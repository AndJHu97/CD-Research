#!/usr/bin/env python3
"""Create a four-panel model-validity figure (Panels A–D).

Panel A: Feature-ablation dumbbell chart — hit rate @ top X% (test) with 95% Wilson CI.
Panel B: Feature-ablation dumbbell chart — NDCG (test) with 95% bootstrap CI
         (query-group rows when available, else CV fold SEM).
         Optional --ndcg-k recomputes NDCG@K from target_rank, or
         --ndcg-top-pct recomputes NDCG@top-X% with a per-query cutoff
         from n_residues (same window as hit_at_top_pct).
Panel C: SHAP beeswarm for physicochemical descriptors (residue one-hots omitted).
Panel D: CV fold distribution vs held-out test hit rate @ top X%
         (both use hit_at_top_pct; X from summary top_pct_threshold).

Usage:
    python helper/Create_Model_Validity.py \
  --full-summary normal_summary.json \
  --cv-folds normal_cv_fold_results.csv \
  --shap-candidates normal_test_shap_per_candidate.csv \
  --ablation "Without SASA=sasa_ablation_summary.json" \
  --ablation "Without deprotonation=deprot_ablation_summary.json" \
  --ablation "Without QM reactivity=reactivity_ablation_summary.json" \
  --ablation "Without non-covalent interactions=noncovalent_ablation_summary.json" \
  --ablation "Without QM reactivity and deprotonation=reactivity_deprot_ablation_summary.json" \
  --ablation "Warhead only=warhead_ablation_summary.json" \
  --ablation "Non-covalent only=noncovalent_only_ablation_summary.json" \
  --ndcg-top-pct 1 \
  --test-results test_results.csv \
  --output model_validity.png
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

# Physicochemical model inputs (residue one-hots and N_Terminal excluded).
PHYSICOCHEMICAL_FEATURES: list[tuple[str, str]] = [
    ("Rel_Side_SASA", "Relative SASA"),
    ("deprotonation_prob", "Deprotonation probability"),
    ("Partial_Charge_Deprotonated", "Partial charge"),
    ("Fukui_Deprotonated", "Fukui index"),
    ("Nucleophile_HOMO_Deprotonated", "HOMO"),
    ("Electrophile_LUMO_Deprotonated", "LUMO"),
    ("HOMO_LUMO_Gap_Deprotonated", "HOMO–LUMO gap"),
    ("Nucleophilicity_Index_Deprotonated", "Nucleophilicity"),
    # Non-covalent interaction / complementarity fits
    ("Geo_Fit", "Geometric Fit"),
    ("Hydrogen_DPAL_Fit", "Hydrogen DPAL Fit"),
    ("Hydrogen_APDL_Fit", "Hydrogen APDL Fit"),
    ("Electro_Fit", "Electro Fit"),
]

DEFAULT_ABLATION_STEMS: list[tuple[str, str]] = [
    ("Without SASA", "ablation_no_sasa_summary.json"),
    ("Without deprotonation", "ablation_no_deprotonation_summary.json"),
    ("Without QM reactivity", "ablation_no_qm_summary.json"),
    ("Without non-covalent interactions", "ablation_no_noncovalent_summary.json"),
    ("Warhead only", "ablation_warhead_only_summary.json"),
    ("Non-covalent only", "ablation_noncovalent_only_summary.json"),
]

FULL_MODEL_COLOR = "#2C3E50"
ABLATION_COLOR = "#C44E52"
ABLATION_LINE_COLOR = "#AAAAAA"
CV_BOX_COLOR = "#A8B58A"  # light army green (Panel D)
TEST_MARKER_COLOR = "#D62728"
FIG_WIDTH = 14.85
FIG_HEIGHT = 11.0
DEFAULT_CI_LEVEL = 0.95
_BOOTSTRAP_DRAWS = 2000
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def _panel_title(ax: plt.Axes, panel_label: str, title: str = "") -> None:
    _ = title
    ax.set_title(panel_label, loc="left", fontweight="bold", fontsize=11)


def _style_axes(ax: plt.Axes) -> None:
    """Match Panel D axis weight: hide top/right; thicken left/bottom."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.4)
    ax.tick_params(width=1.2)


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


def load_top_pct_threshold(summary_path: Path, default: float = 1.0) -> float:
    """Return training top-X% threshold from summary (root or test_overall)."""
    with summary_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for section in (data, data.get("test_overall", {})):
        if isinstance(section, dict) and "top_pct_threshold" in section:
            return float(section["top_pct_threshold"])
    return float(default)


def hit_rate_axis_label(top_pct: float) -> str:
    """Format e.g. top_pct=1 -> 'Hit rate @ top 1%'."""
    if float(top_pct).is_integer():
        return f"Hit rate @ top {int(top_pct)}%"
    return f"Hit rate @ top {top_pct:g}%"


def ndcg_axis_label(*, k: int | None = None, top_pct: float | None = None) -> str:
    """Panel B x-axis label for fixed-K or top-X% NDCG."""
    if top_pct is not None:
        if float(top_pct).is_integer():
            return f"NDCG@top {int(top_pct)}% (test)"
        return f"NDCG@top {top_pct:g}% (test)"
    if k is None:
        return "NDCG (test)"
    return f"NDCG@{int(k)} (test)"


def ndcg_metric_tag(*, k: int | None = None, top_pct: float | None = None) -> str:
    """Short tag for logs, e.g. NDCG@15 or NDCG@top 1%."""
    if top_pct is not None:
        if float(top_pct).is_integer():
            return f"NDCG@top {int(top_pct)}%"
        return f"NDCG@top {top_pct:g}%"
    if k is None:
        return "NDCG"
    return f"NDCG@{int(k)}"


def load_hit_at_top_pct(summary_path: Path) -> float:
    return load_test_metric(summary_path, "hit_at_top_pct")


def load_ndcg(summary_path: Path) -> float:
    return load_test_metric(summary_path, "ndcg")


def top_pct_cutoff_rank(n_residues: float, top_pct: float) -> float:
    """
    Max qualifying rank for Hit/NDCG@top-X%, matching Training_Cov_Screen
    hit_at_top_pct: rank_frac = (rank-1)/(n-1) <= top_pct/100.
    """
    n = float(n_residues)
    p_frac = float(top_pct) / 100.0
    if n <= 1:
        return 1.0
    if p_frac <= 0:
        return 0.0
    return float(np.floor(p_frac * (n - 1.0) + 1.0))


def ndcg_at_k_from_ranks(ranks: np.ndarray, k: int) -> np.ndarray:
    """
    Per-query-group NDCG@K for binary relevance with one true target.

    NDCG@K = 1/log2(rank+1) if rank <= K else 0 (IDCG@K = 1).
    """
    if k <= 0:
        raise ValueError(f"ndcg k must be positive, got {k}")
    ranks = np.asarray(ranks, dtype=float)
    out = np.zeros(ranks.shape, dtype=float)
    in_window = np.isfinite(ranks) & (ranks >= 1) & (ranks <= k)
    out[in_window] = 1.0 / np.log2(ranks[in_window] + 1.0)
    return out


def ndcg_at_top_pct_from_ranks(
    ranks: np.ndarray,
    n_residues: np.ndarray,
    top_pct: float,
) -> np.ndarray:
    """
    Per-query-group NDCG@top-X% (binary, one true target).

    Uses the same top-X% window as hit_at_top_pct: credit only when
    (rank-1)/(n-1) <= top_pct/100, with DCG = 1/log2(rank+1).
    """
    if top_pct <= 0:
        raise ValueError(f"ndcg top_pct must be positive, got {top_pct}")
    ranks = np.asarray(ranks, dtype=float)
    n_residues = np.asarray(n_residues, dtype=float)
    if ranks.shape != n_residues.shape:
        raise ValueError(
            f"ranks and n_residues length mismatch: {ranks.shape} vs {n_residues.shape}"
        )
    cutoffs = np.array(
        [top_pct_cutoff_rank(n, top_pct) for n in n_residues],
        dtype=float,
    )
    out = np.zeros(ranks.shape, dtype=float)
    in_window = np.isfinite(ranks) & (ranks >= 1) & (ranks <= cutoffs)
    out[in_window] = 1.0 / np.log2(ranks[in_window] + 1.0)
    return out


def load_rank_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "target_rank" not in df.columns:
        raise ValueError(
            f"{csv_path}: missing target_rank column "
            "(expected test_results / test_query_groups export)."
        )
    ranks = pd.to_numeric(df["target_rank"], errors="coerce")
    if ranks.size == 0 or not np.isfinite(ranks.to_numpy(dtype=float)).any():
        raise ValueError(f"{csv_path}: no valid target_rank values")
    out = df.copy()
    out["target_rank"] = ranks
    if "n_residues" in out.columns:
        out["n_residues"] = pd.to_numeric(out["n_residues"], errors="coerce")
    return out


def load_target_ranks(csv_path: Path) -> np.ndarray:
    return load_rank_table(csv_path)["target_rank"].to_numpy(dtype=float)


def resolve_test_query_groups(
    summary_path: Path,
    explicit: Path | None = None,
) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Rank CSV not found: {explicit}")
        return explicit
    stem = summary_path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    candidates = [
        summary_path.with_name(f"{stem}_test_query_groups.csv"),
        summary_path.with_name(f"{stem}_test_results.csv"),
        summary_path.with_name(f"{stem}_results.csv"),
        summary_path.parent / "test_query_groups.csv",
        summary_path.parent / "test_results.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_ablation_test_results_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Ablation test-results spec must be LABEL=PATH, got {spec!r}"
        )
    label, path_str = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty ablation label in {spec!r}")
    return label, Path(path_str.strip())


def load_ndcg_from_ranks(
    csv_path: Path,
    *,
    k: int | None = None,
    top_pct: float | None = None,
) -> float:
    if (k is None) == (top_pct is None):
        raise ValueError("Provide exactly one of k or top_pct for NDCG recompute.")
    if k is not None:
        return float(ndcg_at_k_from_ranks(load_target_ranks(csv_path), k).mean())
    df = load_rank_table(csv_path)
    if "n_residues" not in df.columns:
        raise ValueError(
            f"{csv_path}: missing n_residues column (required for NDCG@top-X%)."
        )
    values = ndcg_at_top_pct_from_ranks(
        df["target_rank"].to_numpy(dtype=float),
        df["n_residues"].to_numpy(dtype=float),
        float(top_pct),
    )
    return float(np.nanmean(values))


def load_ndcg_ci_from_ranks(
    csv_path: Path,
    *,
    k: int | None = None,
    top_pct: float | None = None,
    alpha: float = DEFAULT_CI_LEVEL,
    seed: int = 42,
) -> tuple[float, float, str]:
    if (k is None) == (top_pct is None):
        raise ValueError("Provide exactly one of k or top_pct for NDCG CI.")
    if k is not None:
        values = ndcg_at_k_from_ranks(load_target_ranks(csv_path), k)
        tag = f"bootstrap (NDCG@{k} from target_rank)"
    else:
        df = load_rank_table(csv_path)
        if "n_residues" not in df.columns:
            raise ValueError(
                f"{csv_path}: missing n_residues column (required for NDCG@top-X%)."
            )
        values = ndcg_at_top_pct_from_ranks(
            df["target_rank"].to_numpy(dtype=float),
            df["n_residues"].to_numpy(dtype=float),
            float(top_pct),
        )
        tag = (
            f"bootstrap ({ndcg_metric_tag(top_pct=float(top_pct))} from target_rank)"
        )
    lo, hi = _bootstrap_mean_ci(values, alpha=alpha, seed=seed)
    return (lo, hi, tag)


def _t_critical(df: int, alpha: float) -> float:
    if df <= 0:
        return 1.96
    if abs(alpha - 0.05) < 1e-9 and df in _T_CRIT_95:
        return _T_CRIT_95[df]
    return 1.96


def wilson_ci(successes: int, n: int, alpha: float = DEFAULT_CI_LEVEL) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else 1.96
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_hit_rate_ci(summary_path: Path, alpha: float = DEFAULT_CI_LEVEL) -> tuple[float, float]:
    test = load_test_overall(summary_path)
    n_labels = int(test.get("n_labels", 0))
    if n_labels <= 0:
        return (float(test["hit_at_top_pct"]), float(test["hit_at_top_pct"]))
    successes = int(round(float(test["hit_at_top_pct"]) * n_labels))
    successes = min(max(successes, 0), n_labels)
    return wilson_ci(successes, n_labels, alpha=alpha)


def _cv_fold_values(summary_path: Path, metric: str) -> np.ndarray:
    with summary_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    values: list[float] = []
    for row in data.get("cv_folds", []):
        fold = str(row.get("fold", "")).strip().lower()
        if fold in {"mean", "avg", "average"}:
            continue
        if metric not in row:
            continue
        values.append(float(row[metric]))
    return np.asarray(values, dtype=float)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    alpha: float = DEFAULT_CI_LEVEL,
    seed: int = 42,
    n_draws: int = _BOOTSTRAP_DRAWS,
) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        sample = rng.choice(values, size=values.size, replace=True)
        boots[i] = sample.mean()
    lo = float(np.percentile(boots, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def load_ndcg_ci(
    summary_path: Path,
    *,
    alpha: float = DEFAULT_CI_LEVEL,
    seed: int = 42,
) -> tuple[float, float, str]:
    """
    Return (lo, hi, method) for test NDCG.

    Prefer bootstrap over per-query-group test rows; fall back to CV fold SEM
    centered on the held-out test NDCG when query-group exports are unavailable
    or inconsistent with summary NDCG / n_query_groups.
    """
    test = load_test_overall(summary_path)
    test_ndcg = float(test["ndcg"])
    n_qg_expected = int(test.get("n_query_groups", 0))
    qg_path = resolve_test_query_groups(summary_path)
    if qg_path is not None:
        df = pd.read_csv(qg_path)
        values: np.ndarray | None = None
        if "ndcg_at_k" in df.columns:
            values = df["ndcg_at_k"].astype(float).to_numpy()
        elif "ndcg" in df.columns:
            values = df["ndcg"].astype(float).to_numpy()
        if values is not None and values.size > 0:
            mean_ndcg = float(np.nanmean(values))
            n_rows = int(values.size)
            matches_n = n_qg_expected <= 0 or n_rows == n_qg_expected
            matches_mean = abs(mean_ndcg - test_ndcg) <= 0.02
            if matches_n and matches_mean:
                lo, hi = _bootstrap_mean_ci(values, alpha=alpha, seed=seed)
                return (lo, hi, "bootstrap (query groups)")
            print(
                f"[WARN] Ignoring rank CSV for NDCG CI ({qg_path.name}): "
                f"n={n_rows} (expected {n_qg_expected or '?'}), "
                f"mean ndcg={mean_ndcg:.3f} (summary={test_ndcg:.3f}). "
                "Falling back to CV fold SEM."
            )

    fold_vals = _cv_fold_values(summary_path, "ndcg")
    if fold_vals.size >= 2:
        df = int(fold_vals.size - 1)
        hw = _t_critical(df, alpha) * float(fold_vals.std(ddof=1)) / np.sqrt(fold_vals.size)
        return (
            max(0.0, test_ndcg - hw),
            min(1.0, test_ndcg + hw),
            "CV fold SEM",
        )

    return (test_ndcg, test_ndcg, "point estimate")


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


def _test_set_shape(summary_path: Path) -> tuple[int, int]:
    test = load_test_overall(summary_path)
    return int(test.get("n_labels", 0)), int(test.get("n_query_groups", 0))


def _intervals_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return not (hi_a < lo_b or hi_b < lo_a)


def report_ablation_comparability(
    *,
    full_summary: Path,
    ablation_paths: list[tuple[str, Path]],
    full_hit_ci: tuple[float, float],
    full_ndcg_ci: tuple[float, float],
    ablation_hit_cis: dict[str, tuple[float, float]],
    ablation_ndcg_cis: dict[str, tuple[float, float]],
) -> None:
    """Warn when ablation summaries use a different test set than the full model."""
    full_labels, full_qg = _test_set_shape(full_summary)
    mismatches: list[str] = []
    for label, path in ablation_paths:
        if not path.is_file():
            continue
        n_labels, n_qg = _test_set_shape(path)
        if n_labels != full_labels or n_qg != full_qg:
            mismatches.append(
                f"    {label}: n_labels={n_labels}, n_query_groups={n_qg} "
                f"(full model: {full_labels}, {full_qg})"
            )

    if mismatches:
        print(
            "\n[WARN] Ablation summaries do not all use the same held-out test set "
            "as the full model. Point estimates and Wilson CIs are not strictly "
            "paired-comparable until all runs share identical cluster splits and "
            "label/query-group counts.\n"
            + "\n".join(mismatches)
        )
    else:
        print(
            f"\n  Test-set check: all ablations match full model "
            f"(n_labels={full_labels}, n_query_groups={full_qg})."
        )

    print("\n  95% CI overlap with full model (non-overlap ⇒ statistically separable "
          "under each CI's assumptions):")
    for label, path in ablation_paths:
        if label not in ablation_hit_cis:
            continue
        hit_lo, hit_hi = ablation_hit_cis[label]
        nd_lo, nd_hi = ablation_ndcg_cis[label]
        hit_ov = _intervals_overlap(*full_hit_ci, hit_lo, hit_hi)
        nd_ov = _intervals_overlap(*full_ndcg_ci, nd_lo, nd_hi)
        print(
            f"    {label}: hit@top% overlap={hit_ov}, NDCG overlap={nd_ov}"
        )


def _plot_xerr(
    ax: plt.Axes,
    center: float,
    y: float,
    lo: float,
    hi: float,
    *,
    color: str,
) -> None:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi <= lo:
        return
    ax.errorbar(
        center,
        y,
        xerr=[[center - lo], [hi - center]],
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=2.5,
        capthick=1.0,
        zorder=2,
    )


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
    full_ci: tuple[float, float] | None = None,
    ablation_cis: list[tuple[float, float] | None] | None = None,
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
    if ablation_cis is None:
        ablation_cis = [None] * len(ablated_scaled)
    full_lo = full_hi = None
    if full_ci is not None:
        full_lo, full_hi = (full_ci[0] * scale, full_ci[1] * scale)

    for yi, abl, ci in zip(y_pos, ablated_scaled, ablation_cis):
        ax.plot(
            [full_scaled, abl],
            [yi, yi],
            color=ABLATION_LINE_COLOR,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=1,
        )
        if ci is not None:
            _plot_xerr(ax, abl, yi, ci[0] * scale, ci[1] * scale, color=ABLATION_COLOR)
        ax.scatter(
            abl,
            yi,
            s=70,
            color=ABLATION_COLOR,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    for yi in y_pos:
        if full_lo is not None and full_hi is not None:
            _plot_xerr(ax, full_scaled, yi, full_lo, full_hi, color=FULL_MODEL_COLOR)
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

    ci_bounds = [full_scaled]
    if full_lo is not None and full_hi is not None:
        ci_bounds.extend([full_lo, full_hi])
    for abl, ci in zip(ablated_scaled, ablation_cis):
        ci_bounds.append(abl)
        if ci is not None:
            ci_bounds.extend([ci[0] * scale, ci[1] * scale])
    xmin = min(ci_bounds) - 0.03 * scale
    xmax = max(ci_bounds) + 0.03 * scale
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
    _style_axes(ax)


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
    *,
    alpha: float = 0.85,
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
            alpha=alpha,
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
    alpha: float = 0.85,
) -> None:
    header = pd.read_csv(shap_path, nrows=0).columns.tolist()
    available: list[tuple[str, str]] = []
    skipped: list[str] = []
    for feat, label in PHYSICOCHEMICAL_FEATURES:
        if feat in header and f"shap_{feat}" in header:
            available.append((feat, label))
        else:
            skipped.append(feat)
    if not available:
        raise ValueError(
            f"{shap_path}: none of the physicochemical SHAP features are present"
        )
    if skipped:
        print(
            f"[Create_Model_Validity] Skipping SHAP features missing from CSV: "
            f"{', '.join(skipped)}"
        )

    usecols = ["Name"] + [c for feat, _ in available for c in (feat, f"shap_{feat}")]
    df = pd.read_csv(shap_path, usecols=usecols)
    df = _subsample_shap_rows(df, max_rows=max_rows, seed=seed)

    feat_cols = [feat for feat, _ in available]
    shap_cols = [f"shap_{feat}" for feat in feat_cols]
    display_names = [label for _, label in available]

    shap_values = df[shap_cols].to_numpy(dtype=np.float32)
    feature_values = df[feat_cols].to_numpy(dtype=np.float32)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    shap_values = shap_values[:, order]
    feature_values = feature_values[:, order]
    ordered_names = [display_names[i] for i in order]

    if shap is None:
        plot_shap_beeswarm_matplotlib(
            ax, shap_values, feature_values, ordered_names, alpha=alpha,
        )
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
            alpha=alpha,
            ax=ax,
        )
    ax.set_xlabel("SHAP value (impact on model score)")
    ax.set_ylabel("")
    _panel_title(ax, "C")
    _style_axes(ax)


def plot_cv_vs_test(
    ax: plt.Axes,
    cv_path: Path,
    test_rate: float,
    *,
    top_pct: float = 1.0,
) -> None:
    """Panel D: CV-fold hit_at_top_pct boxplot vs held-out test hit_at_top_pct."""
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
        widths=0.36,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={
            "facecolor": CV_BOX_COLOR,
            "alpha": 0.85,
            "edgecolor": "#6F7D52",
            "linewidth": 1.2,
        },
        whiskerprops={"color": "#6F7D52", "linewidth": 1.2},
        capprops={"color": "#6F7D52", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": CV_BOX_COLOR,
            "markeredgecolor": "#6F7D52",
            "markersize": 4,
        },
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
    ax.set_ylabel(hit_rate_axis_label(top_pct))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _panel_title(ax, "D")
    _style_axes(ax)


def create_figure(
    *,
    full_summary: Path,
    cv_folds: Path,
    shap_candidates: Path,
    ablation_specs: list[str],
    output_path: Path,
    shap_max_rows: int,
    shap_alpha: float,
    seed: int,
    ci_level: float = DEFAULT_CI_LEVEL,
    ndcg_k_override: int | None = None,
    ndcg_top_pct_override: float | None = None,
    test_results: Path | None = None,
    ablation_test_results: list[str] | None = None,
) -> None:
    _configure_figure_fonts()
    full_test = load_test_overall(full_summary)
    full_hit = float(full_test["hit_at_top_pct"])
    summary_ndcg_k = int(full_test.get("k", 15))
    top_pct = load_top_pct_threshold(full_summary)
    hit_xlabel = hit_rate_axis_label(top_pct)

    if ndcg_k_override is not None and ndcg_top_pct_override is not None:
        raise ValueError("Pass only one of --ndcg-k or --ndcg-top-pct.")

    recompute_ndcg = (
        ndcg_k_override is not None or ndcg_top_pct_override is not None
    )
    ndcg_k: int | None = None
    ndcg_top_pct: float | None = None
    if ndcg_top_pct_override is not None:
        ndcg_top_pct = float(ndcg_top_pct_override)
    elif ndcg_k_override is not None:
        ndcg_k = int(ndcg_k_override)
    else:
        ndcg_k = summary_ndcg_k

    ndcg_xlabel = ndcg_axis_label(k=ndcg_k, top_pct=ndcg_top_pct)
    ndcg_tag = ndcg_metric_tag(k=ndcg_k, top_pct=ndcg_top_pct)

    ablation_rank_map = {
        label: path
        for label, path in (
            parse_ablation_test_results_arg(spec)
            for spec in (ablation_test_results or [])
        )
    }

    full_rank_csv: Path | None = None
    if recompute_ndcg:
        full_rank_csv = resolve_test_query_groups(full_summary, explicit=test_results)
        if full_rank_csv is None:
            raise FileNotFoundError(
                f"{ndcg_tag} recompute requires a rank CSV for the full model. "
                "Pass --test-results or place test_results.csv / "
                "test_query_groups.csv next to --full-summary."
            )
        full_ndcg = load_ndcg_from_ranks(
            full_rank_csv, k=ndcg_k, top_pct=ndcg_top_pct,
        )
        print(
            f"  Recomputing {ndcg_tag} from ranks "
            f"(summary had NDCG@{summary_ndcg_k}={float(full_test['ndcg']):.3f})"
        )
        print(f"  Full-model ranks: {full_rank_csv}")
    else:
        full_ndcg = float(full_test["ndcg"])
        if test_results is not None:
            print(
                "[WARN] --test-results is ignored unless --ndcg-k or "
                "--ndcg-top-pct is set; using summary NDCG."
            )

    ablation_paths = resolve_ablations(full_summary, ablation_specs)
    ablation_hits: list[tuple[str, float]] = []
    ablation_ndcgs: list[tuple[str, float]] = []
    ablation_hit_cis: dict[str, tuple[float, float]] = {}
    ablation_ndcg_cis: dict[str, tuple[float, float]] = {}
    ndcg_ci_method = ""
    for label, path in ablation_paths:
        if not path.is_file():
            print(f"[WARN] Ablation summary not found, skipping: {path}")
            continue
        hit = load_hit_at_top_pct(path)
        hit_ci = load_hit_rate_ci(path, alpha=ci_level)

        if recompute_ndcg:
            explicit_rank = ablation_rank_map.get(label)
            rank_csv = resolve_test_query_groups(path, explicit=explicit_rank)
            if rank_csv is None:
                print(
                    f"[WARN] Ablation '{label}': no test_results/query_groups CSV; "
                    f"skipping (needed for {ndcg_tag})."
                )
                continue
            ndcg = load_ndcg_from_ranks(
                rank_csv, k=ndcg_k, top_pct=ndcg_top_pct,
            )
            ndcg_lo, ndcg_hi, method = load_ndcg_ci_from_ranks(
                rank_csv,
                k=ndcg_k,
                top_pct=ndcg_top_pct,
                alpha=ci_level,
                seed=seed,
            )
        else:
            ndcg = load_ndcg(path)
            ndcg_lo, ndcg_hi, method = load_ndcg_ci(
                path, alpha=ci_level, seed=seed,
            )

        if not ndcg_ci_method:
            ndcg_ci_method = method
        ablation_hits.append((label, hit))
        ablation_ndcgs.append((label, ndcg))
        ablation_hit_cis[label] = hit_ci
        ablation_ndcg_cis[label] = (ndcg_lo, ndcg_hi)
        print(
            f"  Ablation '{label}': {hit_xlabel} = {hit * 100:.1f}% "
            f"[{hit_ci[0] * 100:.1f}, {hit_ci[1] * 100:.1f}], "
            f"{ndcg_tag} = {ndcg:.3f} [{ndcg_lo:.3f}, {ndcg_hi:.3f}]"
        )

    full_hit_ci = load_hit_rate_ci(full_summary, alpha=ci_level)
    if recompute_ndcg:
        assert full_rank_csv is not None
        full_ndcg_lo, full_ndcg_hi, full_ndcg_method = load_ndcg_ci_from_ranks(
            full_rank_csv,
            k=ndcg_k,
            top_pct=ndcg_top_pct,
            alpha=ci_level,
            seed=seed,
        )
    else:
        full_ndcg_lo, full_ndcg_hi, full_ndcg_method = load_ndcg_ci(
            full_summary, alpha=ci_level, seed=seed,
        )
    ndcg_ci_method = full_ndcg_method or ndcg_ci_method

    print(
        f"  Full model test {hit_xlabel} = {full_hit * 100:.1f}% "
        f"[{full_hit_ci[0] * 100:.1f}, {full_hit_ci[1] * 100:.1f}]"
    )
    print(
        f"  Full model test {ndcg_tag} = {full_ndcg:.3f} "
        f"[{full_ndcg_lo:.3f}, {full_ndcg_hi:.3f}] ({ndcg_ci_method})"
    )

    ablation_hits = _order_ablations(full_hit, ablation_hits)
    label_order = [label for label, _ in ablation_hits]
    ndcg_by_label = dict(ablation_ndcgs)
    ablation_ndcgs = [(label, ndcg_by_label[label]) for label in label_order]
    ordered_hit_cis = [ablation_hit_cis[label] for label in label_order]
    ordered_ndcg_cis = [ablation_ndcg_cis[label] for label in label_order]

    report_ablation_comparability(
        full_summary=full_summary,
        ablation_paths=ablation_paths,
        full_hit_ci=full_hit_ci,
        full_ndcg_ci=(full_ndcg_lo, full_ndcg_hi),
        ablation_hit_cis=ablation_hit_cis,
        ablation_ndcg_cis=ablation_ndcg_cis,
    )

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
        xlabel=hit_xlabel,
        as_percent=True,
        full_ci=full_hit_ci,
        ablation_cis=ordered_hit_cis,
    )
    plot_ablation_dumbbell(
        ax_b,
        full_ndcg,
        ablation_ndcgs,
        panel_label="B",
        title="Feature ablation (NDCG)",
        xlabel=ndcg_xlabel,
        as_percent=False,
        full_ci=(full_ndcg_lo, full_ndcg_hi),
        ablation_cis=ordered_ndcg_cis,
    )
    plot_shap_beeswarm(
        ax_c,
        shap_candidates,
        max_rows=shap_max_rows,
        seed=seed,
        alpha=shap_alpha,
    )
    plot_cv_vs_test(ax_d, cv_folds, full_hit, top_pct=top_pct)

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
        "--ndcg-k",
        type=int,
        default=None,
        help=(
            "Recompute Panel B NDCG@K from target_rank in test_results / "
            "test_query_groups CSVs (e.g. 15). Mutually exclusive with "
            "--ndcg-top-pct. Requires --test-results for the full model "
            "(or a sibling CSV next to --full-summary), and a rank CSV "
            "for each ablation (auto-discovered or via --ablation-test-results)."
        ),
    )
    p.add_argument(
        "--ndcg-top-pct",
        type=float,
        default=None,
        nargs="?",
        const=-1.0,
        metavar="X",
        help=(
            "Recompute Panel B as NDCG@top-X%% from target_rank and n_residues "
            "(same top-X%% window as hit_at_top_pct). Pass a value (e.g. 1) or "
            "omit the value to use top_pct_threshold from --full-summary. "
            "Mutually exclusive with --ndcg-k."
        ),
    )
    p.add_argument(
        "--test-results",
        type=Path,
        default=None,
        help=(
            "Full-model test_results.csv or test_query_groups.csv with target_rank. "
            "Used with --ndcg-k or --ndcg-top-pct to recompute NDCG."
        ),
    )
    p.add_argument(
        "--ablation-test-results",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Optional rank CSV for an ablation arm (LABEL must match --ablation). "
            "Repeat as needed. If omitted, looks next to each ablation summary."
        ),
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
        "--shap-alpha",
        type=float,
        default=0.85,
        help="Point opacity for Panel C SHAP beeswarm (0–1; higher keeps color when points overlap; default 0.85).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SHAP subsampling and bootstrap CIs.",
    )
    p.add_argument(
        "--ci-level",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help="Confidence level for Panels A/B error bars (default 0.95).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.full_summary, args.cv_folds, args.shap_candidates):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    if not (0.0 < args.ci_level < 1.0):
        raise ValueError("--ci-level must be between 0 and 1.")
    if not (0.0 < args.shap_alpha <= 1.0):
        raise ValueError("--shap-alpha must be in (0, 1].")
    if args.ndcg_k is not None and args.ndcg_k <= 0:
        raise ValueError("--ndcg-k must be a positive integer.")
    if args.ndcg_k is not None and args.ndcg_top_pct is not None:
        raise ValueError("Pass only one of --ndcg-k or --ndcg-top-pct.")
    ndcg_top_pct_override = args.ndcg_top_pct
    if ndcg_top_pct_override is not None:
        if abs(ndcg_top_pct_override + 1.0) < 1e-12:
            ndcg_top_pct_override = load_top_pct_threshold(args.full_summary)
        if ndcg_top_pct_override <= 0:
            raise ValueError("--ndcg-top-pct must be positive.")
    if args.test_results is not None and not args.test_results.is_file():
        raise FileNotFoundError(f"--test-results not found: {args.test_results}")

    create_figure(
        full_summary=args.full_summary,
        cv_folds=args.cv_folds,
        shap_candidates=args.shap_candidates,
        ablation_specs=args.ablation,
        output_path=args.output,
        shap_max_rows=args.shap_max_rows,
        shap_alpha=args.shap_alpha,
        seed=args.seed,
        ci_level=args.ci_level,
        ndcg_k_override=args.ndcg_k,
        ndcg_top_pct_override=ndcg_top_pct_override,
        test_results=args.test_results,
        ablation_test_results=args.ablation_test_results,
    )


if __name__ == "__main__":
    main()
