#!/usr/bin/env python3
"""Create the ablation main figure (A–F) plus an SI per-feature heatmap.

Panel A: Radar of four ablation models, each with that model's panel-B category column.
Panel B: Category-level mean |SHAP| heatmap (nucleophile / covalent / non-covalent).
Panel C: Pie of Yes / Limited / No strong noncovalent recognition.
Panel D: Radar recomputed on Limited + No PDBs only.
Panel E: Hit rate at top 1% vs noncovalent pre-organization (Full and Noncovalent).
Panel F: Paired rank wins (which model ranked the true site better).
SI: Per-feature mean |SHAP| heatmap.

Metrics
-------
* Hit rate @ top 1% and @ top 10% — from Training_Cov_Screen summary JSON
  (``test_overall.hit_at_top_pct``). Each model directory may contain two
  summaries, one trained/evaluated at each threshold. Missing thresholds are
  recomputed from ``target_rank`` / ``n_residues`` in test_label_results.
* Median search-space reduction — ``test_overall.median_search_space_reduction``
  (independent of the top-X% threshold).
* Adjusted logAUC — per label row, then averaged. Uses the Mysinger/Shoichet
  LogAUC_0.001 integrator from TOPSIS_VS.py on a single-hit residue ranking
  (true site at ``target_rank`` among ``n_residues``).
  adjusted_logAUC = logAUC - 0.14462.
* BEDROC (α=20) — per label row, then averaged. RDKit / Truchon & Bayly 2007.

Expected input (named subdirectory per model)::

    ablation_dir/
      Full Model/                 # or Full, normal, ...
        *summary*.json            # top 1% and/or top 10%
        test_label_results.csv
        feature_importance.csv    # optional; SHAP in the summary is preferred
      Noncovalent/
      Warhead and nucleophile/
      Covalent reactivity only/

Usage:
    python helper/figure/Create_Ablation_Figure.py \\
        --input-dir path/to/ablation_dir \\
        --output ablation_figure.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Figure style (matches Create_Model_Validity / Create_Residue_Specificity)
# ---------------------------------------------------------------------------

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


# Mysinger & Shoichet LogAUC (copied from VS_eval/TOPSIS_VS.py).
LOGAUC_LAMBDA = 0.001
RANDOM_LOGAUC = 0.14462
DEFAULT_BEDROC_ALPHA = 20.0

FIG_WIDTH = 19.5
FIG_HEIGHT = 32.0

UNUSED_FACE = "#E4E4E4"
UNUSED_TEXT = "#9A9A9A"
HEADER_FACE = "#F4F4F4"
GROUP_FACES = {
    "Nucleophile": "#EEF3F7",
    "Covalent reactivity": "#F8F1EA",
    "Non-covalent": "#EAF4EF",
    "Other": "#F5F5F5",
}


# Canonical ablation arms. Keys are stable IDs used throughout the script.
MODEL_ORDER: tuple[str, ...] = (
    "full",
    "noncovalent",
    "warhead",
    "reactivity",
)

MODEL_LABELS: dict[str, str] = {
    "full": "Full Model",
    "noncovalent": "Noncovalent Model",
    "warhead": "Reactive Chemistry Model",
    "reactivity": "Intrinsic Reactivity Model",
}

MODEL_HEADER_LABELS: dict[str, str] = {
    "full": "Full\nModel",
    "noncovalent": "Noncovalent\nModel",
    "warhead": "Reactive\nChemistry Model",
    "reactivity": "Intrinsic\nReactivity Model",
}

MODEL_COLORS: dict[str, str] = {
    "full": "#CC5800",
    "noncovalent": "#5A5A5A",
    "warhead": "#4C72B0",
    "reactivity": "#8172B3",
}

MODEL_LINESTYLES: dict[str, str] = {
    "full": "-",
    "noncovalent": "--",
    "warhead": "-.",
    "reactivity": ":",
}

MODEL_MARKERS: dict[str, str] = {
    "full": "o",
    "noncovalent": "s",
    "warhead": "D",
    "reactivity": "^",
}

# Directory / file-stem aliases (normalized: lowercased, punctuation stripped).
_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "full": (
        "full model",
        "full",
        "full_model",
        "normal",
        "all features",
        "all",
    ),
    "noncovalent": (
        "noncovalent",
        "non covalent",
        "non covalent interactions",
        "noncovalent interactions",
        "noncovalent only",
        "non covalent only",
        "nc only",
        "nc_only",
        "noncovalent_only",
    ),
    "warhead": (
        "warhead",
        "warhead and nucleophile",
        "warhead nucleophile",
        "warhead + nucleophile",
        "warhead only",
        "no noncovalent",
        "no non covalent",
        "without noncovalent",
        "without non covalent",
        "without non covalent interactions",
        "warhead and nucleophile no non covalent features",
    ),
    "reactivity": (
        "covalent reactivity only",
        "covalent reactivity",
        "reactivity only",
        "reactivity",
        "qm only",
        "qm reactivity",
        "covalent only",
    ),
}

# Longest-first stems for a *flat* Training_Cov_Screen output directory.
# Note: plain "reactivity_*" leave-one-out summaries are NOT mapped here —
# those drop QM features, whereas "Covalent reactivity only" keeps only QM.
_FLAT_STEMS: tuple[tuple[str, str], ...] = (
    ("noncovalent_only", "noncovalent"),
    ("nc_only", "noncovalent"),
    ("reactivity_only", "reactivity"),
    ("covalent_reactivity", "reactivity"),
    ("qm_only", "reactivity"),
    ("warhead", "warhead"),
    ("normal", "full"),
    ("full", "full"),
)

_FLAT_FILE_ALIASES: dict[str, tuple[str, ...]] = {
    "full": ("normal", "full"),
    "noncovalent": ("noncovalent_only", "nc_only", "nc"),
    "warhead": ("warhead",),
    "reactivity": ("reactivity_only", "covalent_reactivity", "qm_only"),
}

RADAR_METRICS: tuple[tuple[str, str], ...] = (
    ("hit_top_1", "Hit at 1%"),
    ("hit_top_10", "Hit at 10%"),
    ("median_ssr", "Med. SSR"),
    ("adjusted_logauc", "Adj. logAUC"),
    ("bedroc", "BEDROC"),
)

# Physicochemical features shown in Panel B (residue one-hots omitted).
# Names must match Training_Cov_Screen / test_shap_global feature strings.
FEATURE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Rel_Side_SASA", "Relative SASA", "Nucleophile"),
    ("deprotonation_prob", "Deprotonation probability", "Nucleophile"),
    ("N_Terminal", "N-terminal", "Nucleophile"),
    ("Partial_Charge_Deprotonated", "Partial charge", "Covalent reactivity"),
    ("Fukui_Deprotonated", "Fukui index", "Covalent reactivity"),
    ("Nucleophile_HOMO_Deprotonated", "HOMO", "Covalent reactivity"),
    ("Electrophile_LUMO_Deprotonated", "LUMO", "Covalent reactivity"),
    ("HOMO_LUMO_Gap_Deprotonated", "HOMO–LUMO gap", "Covalent reactivity"),
    ("Nucleophilicity_Index_Deprotonated", "Nucleophilicity", "Covalent reactivity"),
    ("Geo_Fit", "Geometric fit", "Non-covalent"),
    ("Hydrogen_DPAL_Fit", "Hydrogen DPAL fit", "Non-covalent"),
    ("Hydrogen_APDL_Fit", "Hydrogen APDL fit", "Non-covalent"),
    ("Electro_Fit", "Electrostatic fit", "Non-covalent"),
    ("Hydrophobic_Fit", "Hydrophobic fit", "Non-covalent"),
)

_RESIDUE_ONEHOT_RE = re.compile(r"^res_[A-Z]{3}$")

CATEGORY_ORDER: tuple[str, ...] = (
    "Nucleophile",
    "Covalent reactivity",
    "Non-covalent",
)
CATEGORY_ROW_LABELS: tuple[str, ...] = (
    "Nucleophilic\nfeatures",
    "Covalent\nreactivity",
    "Non-covalent\ninteractions",
)
NC_CLASS_ORDER: tuple[str, ...] = ("Yes", "Limited", "No")
NC_CLASS_COLORS: dict[str, str] = {
    "Yes": "#2E8B6E",
    "Limited": "#D4A017",
    "No": "#C44E52",
}
NC_CLASS_LABELS: dict[str, str] = {
    "Yes": "Strong",
    "Limited": "Limited",
    "No": "Absent",
}
# Panels E/F: strong vs not-strong (Limited+No), plus the purely covalent No subset.
COMPARE_CLASS_ORDER: tuple[str, ...] = ("Yes", "Limited+No", "No")
COMPARE_CLASS_MEMBERS: dict[str, tuple[str, ...]] = {
    "Yes": ("Yes",),
    "Limited+No": ("Limited", "No"),
    "No": ("No",),
}
COMPARE_CLASS_LABELS: dict[str, str] = {
    "Yes": "Strong",
    "Limited+No": "Limited/Absent",
    "No": "Absent",
}
ANNOTATION_STEMS: tuple[str, ...] = (
    "TestSet_Noncovalent_Annotation_v2",
    "TestSet_Noncovalent_Annotation",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _analysis_dir() -> Path:
    here = Path(__file__).resolve().parent
    if here.name == "figure":
        return here.parent.parent
    if here.name == "helper":
        return here.parent
    return here


def _norm_name(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ").replace("+", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_model_id(name: str) -> str | None:
    """Map a directory or file stem to a canonical model id."""
    key = _norm_name(name)
    if not key:
        return None
    # Prefer longer / more specific aliases first.
    ranked: list[tuple[int, str, str]] = []
    for model_id, aliases in _MODEL_ALIASES.items():
        for alias in aliases:
            alias_n = _norm_name(alias)
            if key == alias_n or key.startswith(alias_n + " ") or alias_n in key:
                ranked.append((len(alias_n), alias_n, model_id))
    if not ranked:
        return None
    ranked.sort(key=lambda row: row[0], reverse=True)
    # "noncovalent only" must beat plain "noncovalent"; already handled by length.
    # "covalent reactivity only" should beat "reactivity".
    return ranked[0][2]


def _panel_title(
    ax: plt.Axes,
    panel_label: str,
    *,
    pad: float = 10,
    y: float | None = None,
    fontsize: float = 15,
) -> None:
    kwargs: dict[str, object] = {
        "loc": "left",
        "fontweight": "bold",
        "fontsize": fontsize,
        "pad": pad,
    }
    if y is not None:
        kwargs["y"] = y
    ax.set_title(panel_label, **kwargs)


# ---------------------------------------------------------------------------
# LogAUC (TOPSIS_VS.py) and BEDROC (RDKit / Truchon & Bayly)
# ---------------------------------------------------------------------------

def logauc_from_fpr_tpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    lambda_min: float = LOGAUC_LAMBDA,
) -> float:
    """Mysinger/Shoichet LogAUC_λ; identical to TOPSIS_VS.logauc_from_fpr_tpr."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.size < 2 or tpr.size < 2:
        return float("nan")

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    last_idx = []
    for value in np.unique(fpr):
        matches = np.where(fpr == value)[0]
        last_idx.append(int(matches[-1]))
    fpr = fpr[last_idx]
    tpr = tpr[last_idx]

    if float(fpr.max()) < lambda_min:
        return float("nan")

    tpr_at_lambda = float(np.interp(lambda_min, fpr, tpr))
    keep = fpr > lambda_min
    fpr_seg = np.concatenate(([lambda_min], fpr[keep]))
    tpr_seg = np.concatenate(([tpr_at_lambda], tpr[keep]))

    if fpr_seg[-1] < 1.0 - 1e-15:
        fpr_seg = np.concatenate((fpr_seg, [1.0]))
        tpr_seg = np.concatenate((tpr_seg, [float(tpr_seg[-1])]))

    log_x = np.log10(fpr_seg)
    area = 0.0
    for i in range(len(log_x) - 1):
        area += (log_x[i + 1] - log_x[i]) * 0.5 * (tpr_seg[i + 1] + tpr_seg[i])
    denom = np.log10(1.0 / lambda_min)
    if denom <= 0:
        return float("nan")
    return float(area / denom)


def single_hit_step_roc(rank: float, n_residues: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Step ROC for one true site at 1-based ``rank`` among ``n_residues``.

    FPR jumps to (rank-1)/(N-1) with TPR still 0, then TPR jumps to 1.
    This is the ranking ROC; sklearn's default drop_intermediate=True would
    skip the horizontal segment and inflate logAUC on a log-FPR axis.
    """
    n = float(n_residues)
    r = float(rank)
    if not np.isfinite(n) or not np.isfinite(r) or n <= 1 or r < 1:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    r = min(max(r, 1.0), n)
    fpr_hit = (r - 1.0) / (n - 1.0)
    fpr = np.array([0.0, fpr_hit, fpr_hit, 1.0], dtype=float)
    tpr = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    return fpr, tpr


def adjusted_logauc_from_rank(rank: float, n_residues: float) -> float:
    """adjusted_logAUC = logAUC_0.001 - 0.14462 for a single-hit ranking."""
    fpr, tpr = single_hit_step_roc(rank, n_residues)
    logauc = logauc_from_fpr_tpr(fpr, tpr, LOGAUC_LAMBDA)
    if not np.isfinite(logauc):
        return float("nan")
    return float(logauc - RANDOM_LOGAUC)


def bedroc_from_rank(
    rank: float,
    n_residues: float,
    alpha: float = DEFAULT_BEDROC_ALPHA,
) -> float:
    """
    BEDROC for a single active at 1-based rank ``rank`` among N candidates.

    Matches RDKit ``CalcBEDROC`` (Truchon & Bayly 2007): 1-indexed ranks in
    the exponential, RIE min/max normalization.
    """
    n_total = float(n_residues)
    r = float(rank)
    n_actives = 1.0
    if (
        not np.isfinite(n_total)
        or not np.isfinite(r)
        or n_total <= 1
        or r < 1
        or r > n_total
        or alpha <= 0
    ):
        return float("nan")
    if n_actives >= n_total:
        return 1.0

    denom = (1.0 / n_total) * (
        (1.0 - math.exp(-alpha)) / (math.exp(alpha / n_total) - 1.0)
    )
    if denom == 0.0:
        return float("nan")
    rie = math.exp(-(alpha * r) / n_total) / denom
    ratio = n_actives / n_total
    rie_max = (1.0 - math.exp(-alpha * ratio)) / (ratio * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * ratio)) / (ratio * (1.0 - math.exp(alpha)))
    span = rie_max - rie_min
    if span == 0.0:
        return 1.0
    return float((rie - rie_min) / span)


def hit_at_top_pct_from_rank(rank: float, n_residues: float, top_pct: float) -> float:
    """Same window as Training_Cov_Screen: (rank-1)/(N-1) <= top_pct/100."""
    n = float(n_residues)
    r = float(rank)
    p_frac = float(top_pct) / 100.0
    if not np.isfinite(n) or not np.isfinite(r) or r < 1:
        return float("nan")
    if n <= 1:
        return 1.0
    if p_frac <= 0:
        return 0.0
    rank_frac = (r - 1.0) / (n - 1.0)
    return 1.0 if rank_frac <= p_frac else 0.0


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _summary_threshold(data: dict) -> float | None:
    for section in (data, data.get("test_overall", {})):
        if isinstance(section, dict) and "top_pct_threshold" in section:
            return float(section["top_pct_threshold"])
    return None


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _glob_one(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(directory.glob(pattern))
        files = [p for p in hits if p.is_file()]
        if files:
            return files[0]
    return None


def load_rank_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "target_rank" not in df.columns:
        raise ValueError(
            f"{csv_path}: missing target_rank "
            "(expected test_label_results / test_results export)."
        )
    out = df.copy()
    out["target_rank"] = pd.to_numeric(out["target_rank"], errors="coerce")
    if "n_residues" in out.columns:
        out["n_residues"] = pd.to_numeric(out["n_residues"], errors="coerce")
    if "search_space_reduction" in out.columns:
        out["search_space_reduction"] = pd.to_numeric(
            out["search_space_reduction"], errors="coerce"
        )
    ranks = out["target_rank"].to_numpy(dtype=float)
    if ranks.size == 0 or not np.isfinite(ranks).any():
        raise ValueError(f"{csv_path}: no valid target_rank values")
    return out


@dataclass
class ModelFiles:
    model_id: str
    directory: Path
    summaries: dict[float, tuple[Path, dict]] = field(default_factory=dict)
    rank_csv: Path | None = None
    importance_csv: Path | None = None
    shap_global_csv: Path | None = None


@dataclass
class ModelMetrics:
    model_id: str
    label: str
    hit_top_1: float
    hit_top_10: float
    median_ssr: float
    adjusted_logauc: float
    bedroc: float
    n_rows: int
    sources: dict[str, str]


def discover_model_files(model_dir: Path, model_id: str) -> ModelFiles:
    files = ModelFiles(model_id=model_id, directory=model_dir)
    json_paths = []
    seen: set[Path] = set()
    for pattern in ("*summary*.json", "*.json"):
        for path in sorted(model_dir.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            json_paths.append(path)

    for path in json_paths:
        try:
            data = _load_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"[WARN] Skipping JSON {path}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        if "test_overall" not in data and "top_pct_threshold" not in data:
            continue
        threshold = _summary_threshold(data)
        if threshold is None:
            continue
        files.summaries[float(threshold)] = (path, data)

    files.rank_csv = _glob_one(
        model_dir,
        [
            "test_label_results.csv",
            "*_test_label_results.csv",
            "*label_results*.csv",
            "test_query_groups.csv",
            "*_test_query_groups.csv",
            "test_results.csv",
            "*_test_results.csv",
        ],
    )
    files.importance_csv = _glob_one(
        model_dir,
        ["feature_importance.csv", "*feature_importance*.csv"],
    )
    files.shap_global_csv = _glob_one(
        model_dir,
        ["test_shap_global.csv", "*shap_global*.csv"],
    )
    return files


def _pick_summary(
    summaries: dict[float, tuple[Path, dict]],
    target: float,
    *,
    tol: float,
) -> tuple[Path, dict] | None:
    if not summaries:
        return None
    best_thr = min(summaries, key=lambda thr: abs(thr - target))
    if abs(best_thr - target) <= tol:
        return summaries[best_thr]
    return None


def mean_rank_metrics(
    rank_df: pd.DataFrame,
    *,
    bedroc_alpha: float,
) -> tuple[float, float, int]:
    """Mean adjusted logAUC and BEDROC over label/query rows."""
    if "n_residues" not in rank_df.columns:
        raise ValueError("Rank CSV is missing n_residues (required for logAUC/BEDROC).")
    logaucs: list[float] = []
    bedrocs: list[float] = []
    for rank, n_res in zip(
        rank_df["target_rank"].to_numpy(dtype=float),
        rank_df["n_residues"].to_numpy(dtype=float),
    ):
        la = adjusted_logauc_from_rank(rank, n_res)
        bd = bedroc_from_rank(rank, n_res, alpha=bedroc_alpha)
        if np.isfinite(la):
            logaucs.append(la)
        if np.isfinite(bd):
            bedrocs.append(bd)
    n = int(rank_df["target_rank"].notna().sum())
    mean_la = float(np.mean(logaucs)) if logaucs else float("nan")
    mean_bd = float(np.mean(bedrocs)) if bedrocs else float("nan")
    return mean_la, mean_bd, n


def hit_rate_from_ranks(rank_df: pd.DataFrame, top_pct: float) -> float:
    if "n_residues" not in rank_df.columns:
        raise ValueError("Rank CSV is missing n_residues (required to recompute hit rate).")
    values = [
        hit_at_top_pct_from_rank(rank, n_res, top_pct)
        for rank, n_res in zip(
            rank_df["target_rank"].to_numpy(dtype=float),
            rank_df["n_residues"].to_numpy(dtype=float),
        )
    ]
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def median_ssr_from_summary_or_ranks(
    summary: dict | None,
    rank_df: pd.DataFrame | None,
) -> tuple[float, str]:
    if summary is not None:
        test = summary.get("test_overall", {})
        if "median_search_space_reduction" in test:
            return float(test["median_search_space_reduction"]), "summary test_overall"
    if rank_df is not None and "search_space_reduction" in rank_df.columns:
        vals = pd.to_numeric(rank_df["search_space_reduction"], errors="coerce")
        vals = vals[np.isfinite(vals.to_numpy(dtype=float))]
        if not vals.empty:
            return float(vals.median()), "median of rank CSV"
    return float("nan"), "missing"


def compute_model_metrics(
    files: ModelFiles,
    *,
    bedroc_alpha: float,
) -> ModelMetrics:
    rank_df = load_rank_table(files.rank_csv) if files.rank_csv is not None else None

    sources: dict[str, str] = {}
    hit1 = hit10 = float("nan")

    sum1 = _pick_summary(files.summaries, 1.0, tol=0.51)
    sum10 = _pick_summary(files.summaries, 10.0, tol=1.0)
    any_summary = next(iter(files.summaries.values()))[1] if files.summaries else None

    if sum1 is not None:
        hit1 = float(sum1[1]["test_overall"]["hit_at_top_pct"])
        sources["hit_top_1"] = f"{sum1[0].name} (top_pct={_summary_threshold(sum1[1])})"
    elif rank_df is not None:
        hit1 = hit_rate_from_ranks(rank_df, 1.0)
        sources["hit_top_1"] = f"recomputed from {files.rank_csv.name}"
    else:
        sources["hit_top_1"] = "missing"

    if sum10 is not None:
        hit10 = float(sum10[1]["test_overall"]["hit_at_top_pct"])
        sources["hit_top_10"] = (
            f"{sum10[0].name} (top_pct={_summary_threshold(sum10[1])})"
        )
    elif rank_df is not None:
        hit10 = hit_rate_from_ranks(rank_df, 10.0)
        sources["hit_top_10"] = f"recomputed from {files.rank_csv.name}"
    else:
        sources["hit_top_10"] = "missing"

    ssr_summary = (sum1 or sum10 or (None, any_summary))[1]
    median_ssr, ssr_src = median_ssr_from_summary_or_ranks(ssr_summary, rank_df)
    sources["median_ssr"] = ssr_src

    if rank_df is None:
        raise FileNotFoundError(
            f"{files.directory}: need test_label_results.csv (or test_results.csv) "
            "to compute adjusted logAUC and BEDROC from target_rank."
        )
    adj_logauc, bedroc, n_rows = mean_rank_metrics(rank_df, bedroc_alpha=bedroc_alpha)
    sources["adjusted_logauc"] = f"mean of {n_rows} rows in {files.rank_csv.name}"
    sources["bedroc"] = sources["adjusted_logauc"]

    return ModelMetrics(
        model_id=files.model_id,
        label=MODEL_LABELS[files.model_id],
        hit_top_1=hit1,
        hit_top_10=hit10,
        median_ssr=median_ssr,
        adjusted_logauc=adj_logauc,
        bedroc=bedroc,
        n_rows=n_rows,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _importance_from_records(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in records:
        feat = str(row.get("feature", "")).strip()
        if not feat:
            continue
        if "mean_abs_shap" in row and row["mean_abs_shap"] is not None:
            val = float(row["mean_abs_shap"])
        elif "importance" in row and row["importance"] is not None:
            val = float(row["importance"])
        elif "lgbm_gain_importance" in row and row["lgbm_gain_importance"] is not None:
            val = float(row["lgbm_gain_importance"])
        else:
            continue
        if np.isfinite(val):
            out[feat] = val
    return out


def load_feature_importance(files: ModelFiles) -> dict[str, float]:
    """
    Prefer test_shap_global mean |SHAP| (summary JSON or CSV).
    Fall back to feature_importance.csv (LGBM gain).
    """
    # Newest / most complete summary first (doesn't matter which threshold).
    for _thr, (_path, data) in sorted(files.summaries.items()):
        records = data.get("test_shap_global")
        if isinstance(records, list) and records:
            parsed = _importance_from_records(records)
            if parsed:
                return parsed

    if files.shap_global_csv is not None:
        df = pd.read_csv(files.shap_global_csv)
        if "feature" in df.columns:
            recs = df.to_dict(orient="records")
            parsed = _importance_from_records(recs)
            if parsed:
                return parsed

    if files.importance_csv is not None:
        df = pd.read_csv(files.importance_csv)
        if "feature" in df.columns:
            recs = df.to_dict(orient="records")
            parsed = _importance_from_records(recs)
            if parsed:
                return parsed
    return {}


def collect_feature_table(
    models: dict[str, ModelFiles],
) -> tuple[list[tuple[str, str, str]], dict[str, dict[str, float]]]:
    """
    Return (rows, values) where rows are (feature, display, group) and
    values[model][feat] is the importance when that feature was used.

    Include every catalog feature used by *any* ablation arm (so e.g.
    N_Terminal from warhead + nucleophile still appears even if the full
    model drops it). Residue one-hots are omitted.
    """
    importance: dict[str, dict[str, float]] = {}
    for model_id, files in models.items():
        importance[model_id] = load_feature_importance(files)

    catalog = {feat: (label, group) for feat, label, group in FEATURE_ROWS}
    used_anywhere = {
        feat
        for mapping in importance.values()
        for feat in mapping
        if not _RESIDUE_ONEHOT_RE.match(feat)
    }

    ordered: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for feat, label, group in FEATURE_ROWS:
        if feat in seen or feat not in used_anywhere:
            continue
        ordered.append((feat, label, group))
        seen.add(feat)

    leftover = [feat for feat in used_anywhere if feat not in seen]
    # Prefer full-model order for uncatalogued extras, then the rest.
    if "full" in importance:
        leftover = [
            feat for feat in importance["full"] if feat in leftover
        ] + [feat for feat in leftover if feat not in importance.get("full", {})]
    leftover = list(dict.fromkeys(leftover))
    for feat in leftover:
        label, group = catalog.get(feat, (feat.replace("_", " "), "Other"))
        ordered.append((feat, label, group))
        seen.add(feat)

    return ordered, importance


def aggregate_category_importance(
    feature_rows: list[tuple[str, str, str]],
    importance: dict[str, dict[str, float]],
    present_models: list[str],
) -> np.ndarray:
    """Sum mean |SHAP| of features in each of the three physicochemical groups."""
    matrix = np.full((len(CATEGORY_ORDER), len(present_models)), np.nan, dtype=float)
    for j, model_id in enumerate(present_models):
        mapping = importance.get(model_id, {})
        for i, category in enumerate(CATEGORY_ORDER):
            vals = [
                float(mapping[feat])
                for feat, _label, group in feature_rows
                if group == category
                and feat in mapping
                and mapping[feat] is not None
                and np.isfinite(mapping[feat])
            ]
            if vals:
                matrix[i, j] = float(np.sum(vals))
    return matrix


# ---------------------------------------------------------------------------
# Noncovalent-recognition annotation (Yes / Limited / No)
# ---------------------------------------------------------------------------

def pdb_from_name(name: object) -> str:
    text = str(name).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.split("-")[0].split("_")[0].upper()


def _normalize_nc_class(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("limit"):
        return "Limited"
    if text.startswith("no"):
        return "No"
    return None


def find_annotation_path(input_dir: Path) -> Path | None:
    for stem in ANNOTATION_STEMS:
        for ext in (".csv", ".xlsx", ".xls"):
            path = input_dir / f"{stem}{ext}"
            if path.is_file():
                return path
    matches = sorted(input_dir.glob("TestSet_Noncovalent_Annotation*"))
    for path in matches:
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            return path
    return None


def load_nc_annotation(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    pdb_col = next(
        (c for c in df.columns if str(c).strip().lower() == "pdb"),
        None,
    )
    nc_col = next(
        (
            c
            for c in df.columns
            if "strong" in str(c).lower() and "noncovalent" in str(c).lower()
        ),
        None,
    )
    if pdb_col is None or nc_col is None:
        raise ValueError(
            f"{path}: need a PDB column and a 'Strong noncovalent?' column. "
            f"Found {list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "pdb": df[pdb_col].map(lambda v: str(v).strip().upper()),
            "nc_class": df[nc_col].map(_normalize_nc_class),
        }
    )
    out = out[out["pdb"].ne("") & out["nc_class"].notna()]
    out = out.drop_duplicates(subset=["pdb"], keep="first")
    return out


def annotate_rank_df(rank_df: pd.DataFrame, pdb_to_class: dict[str, str]) -> pd.DataFrame:
    out = rank_df.copy()
    if "Name" in out.columns:
        out["pdb"] = out["Name"].map(pdb_from_name)
    elif "PDB" in out.columns:
        out["pdb"] = out["PDB"].map(lambda v: str(v).strip().upper())
    else:
        out["pdb"] = ""
    out["nc_class"] = out["pdb"].map(pdb_to_class)
    return out


def compute_metrics_from_rank_df(
    rank_df: pd.DataFrame,
    *,
    model_id: str,
    bedroc_alpha: float,
    source: str,
) -> ModelMetrics:
    hit1 = hit_rate_from_ranks(rank_df, 1.0)
    hit10 = hit_rate_from_ranks(rank_df, 10.0)
    median_ssr, ssr_src = median_ssr_from_summary_or_ranks(None, rank_df)
    if not np.isfinite(median_ssr) and "n_residues" in rank_df.columns:
        ssr = 1.0 - rank_df["target_rank"] / rank_df["n_residues"]
        median_ssr = float(np.nanmedian(ssr.to_numpy(dtype=float)))
        ssr_src = "1 - rank/n_residues"
    adj_logauc, bedroc, n_rows = mean_rank_metrics(rank_df, bedroc_alpha=bedroc_alpha)
    return ModelMetrics(
        model_id=model_id,
        label=MODEL_LABELS[model_id],
        hit_top_1=hit1,
        hit_top_10=hit10,
        median_ssr=median_ssr,
        adjusted_logauc=adj_logauc,
        bedroc=bedroc,
        n_rows=n_rows,
        sources={
            "hit_top_1": source,
            "hit_top_10": source,
            "median_ssr": f"{ssr_src}; {source}",
            "adjusted_logauc": source,
            "bedroc": source,
        },
    )


def _pair_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    keys = [
        col
        for col in ("Name", "Residue", "ResNum", "Chain")
        if col in left.columns and col in right.columns
    ]
    return keys or ["Name"]


def pair_full_vs_nc(full_df: pd.DataFrame, nc_df: pd.DataFrame) -> pd.DataFrame:
    keys = _pair_keys(full_df, nc_df)
    keep_full = keys + [
        c for c in ("target_rank", "n_residues", "nc_class", "pdb") if c in full_df.columns
    ]
    keep_nc = keys + [c for c in ("target_rank", "n_residues") if c in nc_df.columns]
    left = full_df[keep_full].copy()
    right = nc_df[keep_nc].copy()
    for col in keys:
        left[col] = left[col].astype(str).str.strip()
        right[col] = right[col].astype(str).str.strip()
    left = left.drop_duplicates(subset=keys, keep="first")
    right = right.drop_duplicates(subset=keys, keep="first")
    merged = left.merge(right, on=keys, suffixes=("_full", "_nc"), how="inner")
    merged["full_better"] = merged["target_rank_full"] < merged["target_rank_nc"]
    merged["nc_better"] = merged["target_rank_nc"] < merged["target_rank_full"]
    merged["tie"] = merged["target_rank_full"] == merged["target_rank_nc"]
    return merged


def _wilcoxon_p(full_ranks: np.ndarray, nc_ranks: np.ndarray) -> float | None:
    if full_ranks.size < 8:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    try:
        result = wilcoxon(full_ranks, nc_ranks, alternative="two-sided", zero_method="wilcox")
        return float(result.pvalue)
    except ValueError:
        return None


def summarize_nc_comparison(
    full_df: pd.DataFrame,
    nc_df: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    groups: dict[str, tuple[str, ...]] | None = None,
    group_order: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Hit@1% and paired rank wins by recognition class (or collapsed groups)."""
    if groups is None:
        groups = {cls: (cls,) for cls in NC_CLASS_ORDER}
    if group_order is None:
        group_order = tuple(groups)
    out: dict[str, dict[str, float | int | None]] = {}
    for cls in group_order:
        members = groups[cls]
        full_sub = full_df[full_df["nc_class"].isin(members)]
        nc_sub = nc_df[nc_df["nc_class"].isin(members)]
        pair_sub = (
            paired[paired["nc_class"].isin(members)]
            if "nc_class" in paired.columns
            else paired.iloc[0:0]
        )
        full_ranks = pair_sub["target_rank_full"].to_numpy(dtype=float) if not pair_sub.empty else np.array([])
        nc_ranks = pair_sub["target_rank_nc"].to_numpy(dtype=float) if not pair_sub.empty else np.array([])
        out[cls] = {
            "n_full_rows": int(len(full_sub)),
            "n_nc_rows": int(len(nc_sub)),
            "n_unique_pdb": int(full_sub["pdb"].nunique()) if "pdb" in full_sub.columns else 0,
            "n_paired": int(len(pair_sub)),
            "hit_top_1_full": hit_rate_from_ranks(full_sub, 1.0) if not full_sub.empty else float("nan"),
            "hit_top_1_nc": hit_rate_from_ranks(nc_sub, 1.0) if not nc_sub.empty else float("nan"),
            "mean_rank_full": float(np.nanmean(full_ranks)) if full_ranks.size else float("nan"),
            "mean_rank_nc": float(np.nanmean(nc_ranks)) if nc_ranks.size else float("nan"),
            "median_rank_full": float(np.nanmedian(full_ranks)) if full_ranks.size else float("nan"),
            "median_rank_nc": float(np.nanmedian(nc_ranks)) if nc_ranks.size else float("nan"),
            "wins_full": int(pair_sub["full_better"].sum()) if not pair_sub.empty else 0,
            "wins_nc": int(pair_sub["nc_better"].sum()) if not pair_sub.empty else 0,
            "ties": int(pair_sub["tie"].sum()) if not pair_sub.empty else 0,
            "wilcoxon_p": _wilcoxon_p(full_ranks, nc_ranks) if full_ranks.size else None,
        }
    return out


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def discover_models(input_dir: Path) -> dict[str, ModelFiles]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    found: dict[str, ModelFiles] = {}
    subdirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    for sub in subdirs:
        model_id = match_model_id(sub.name)
        if model_id is None:
            continue
        files = discover_model_files(sub, model_id)
        if not files.summaries and files.rank_csv is None:
            print(f"[WARN] Skipping {sub}: no summary JSON or rank CSV.")
            continue
        if model_id in found:
            print(
                f"[WARN] Duplicate model '{MODEL_LABELS[model_id]}': "
                f"keeping {found[model_id].directory.name}, ignoring {sub.name}"
            )
            continue
        found[model_id] = files

    if found:
        return found

    # Flat directory of Training_Cov_Screen outputs (prefix_summary.json).
    print(
        f"[INFO] No named model subdirectories in {input_dir}; "
        "trying filename-prefix discovery."
    )
    jsons = sorted(input_dir.glob("*summary*.json"))
    claimed: set[Path] = set()
    for stem, model_id in _FLAT_STEMS:
        if model_id in found:
            continue
        matches = [
            p for p in jsons
            if p.name.lower().startswith(stem) and p.resolve() not in claimed
        ]
        if not matches:
            continue
        # Build a synthetic "directory" using the parent plus this prefix.
        files = ModelFiles(model_id=model_id, directory=input_dir)
        for path in matches:
            try:
                data = _load_json(path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            thr = _summary_threshold(data)
            if thr is None:
                continue
            files.summaries[float(thr)] = (path, data)
            claimed.add(path.resolve())
        rank_candidates: list[Path] = []
        fi_candidates: list[Path] = []
        shap_candidates: list[Path] = []
        for alias in _FLAT_FILE_ALIASES.get(model_id, (stem,)):
            rank_candidates.extend(
                [
                    input_dir / f"{alias}_test_label_results.csv",
                    input_dir / f"{alias}_label_results.csv",
                    input_dir / f"{alias}_test_query_groups.csv",
                    input_dir / f"{alias}_test_results.csv",
                ]
            )
            fi_candidates.append(input_dir / f"{alias}_feature_importance.csv")
            shap_candidates.extend(
                [
                    input_dir / f"{alias}_test_shap_global.csv",
                    input_dir / f"{alias}_shap_global.csv",
                ]
            )
        files.rank_csv = _first_existing(rank_candidates)
        files.importance_csv = _first_existing(fi_candidates)
        files.shap_global_csv = _first_existing(shap_candidates)
        if files.summaries or files.rank_csv is not None:
            found[model_id] = files

    return found


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _radar_label_ha_va(theta: float) -> tuple[str, str]:
    t = theta % (2 * np.pi)
    if t <= np.pi / 5 or t >= 9 * np.pi / 5:
        return "center", "bottom"
    if t < 2 * np.pi / 5:
        return "left", "bottom"
    if t < 3 * np.pi / 5:
        return "left", "center"
    if t < 4 * np.pi / 5:
        return "left", "top"
    if t < 6 * np.pi / 5:
        return "center", "top"
    if t < 7 * np.pi / 5:
        return "right", "top"
    if t < 8 * np.pi / 5:
        return "right", "center"
    return "right", "bottom"


def plot_radar_panel(
    ax: plt.Axes,
    model_id: str,
    row: ModelMetrics,
    *,
    show_metric_labels: bool,
    show_title: bool = True,
) -> None:
    """Single-model polar radar. Vertex numbers replace the radial 0–100 scale."""
    del show_metric_labels  # kept so existing callers stay valid
    keys = [key for key, _ in RADAR_METRICS]
    labels = [label for _, label in RADAR_METRICS]
    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([])
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    grid_color = "#7A7A7A"
    for radius in (20, 40, 60, 80, 100):
        ax.plot(
            angles_closed,
            np.full(n + 1, radius),
            color=grid_color,
            linestyle=(0, (2.4, 2.1)),
            linewidth=1.15,
            zorder=0,
        )
    for angle in angles:
        ax.plot(
            [angle, angle],
            [0.0, 100.0],
            color=grid_color,
            linestyle=(0, (2.4, 2.1)),
            linewidth=1.15,
            zorder=0,
        )

    for angle, label in zip(angles, labels):
        ha, va = _radar_label_ha_va(float(angle))
        ax.text(
            angle,
            124.0,
            label,
            ha=ha,
            va=va,
            fontsize=10.5,
            color="#222222",
            clip_on=False,
            zorder=5,
        )

    raw = [float(getattr(row, key)) for key in keys]
    values = [v * 100.0 if np.isfinite(v) else 0.0 for v in raw]
    closed = values + [values[0]]
    color = MODEL_COLORS[model_id]
    ax.fill(angles_closed, closed, color=color, alpha=0.35, zorder=2)
    ax.plot(
        angles_closed,
        closed,
        color=color,
        linestyle="-",
        linewidth=1.5,
        zorder=4,
    )
    for angle, radius, raw_v in zip(angles, values, raw):
        if not np.isfinite(raw_v):
            text = "—"
        else:
            text = f"{raw_v:.3f}"
        ha, va = _radar_label_ha_va(float(angle))
        # Sit just outside the vertex, and stay inside the spoke name.
        r_text = min(max(radius, 0.0) + 8.0, 110.0)
        ax.text(
            angle,
            r_text,
            text,
            ha=ha,
            va=va,
            fontsize=9.5,
            color="#333333",
            clip_on=False,
            zorder=5,
        )
    if show_title:
        ax.set_title(
            MODEL_LABELS[model_id],
            fontsize=12.5,
            fontweight="bold",
            color=color,
            pad=34,
        )


def _ylorrd_cmap():
    """YlOrRd pulled toward a warm cream so the cells read pastel."""
    try:
        base = plt.colormaps["YlOrRd"]
    except (AttributeError, KeyError):  # pragma: no cover
        base = plt.cm.get_cmap("YlOrRd")
    samples = base(np.linspace(0.0, 1.0, 256))
    warm = np.array([1.0, 0.91, 0.78])
    mix = np.linspace(0.22, 0.36, 256)[:, None]
    samples[:, :3] = samples[:, :3] * (1.0 - mix) + warm * mix
    cmap = mcolors.LinearSegmentedColormap.from_list("ylorrd_pastel", samples)
    cmap.set_bad(UNUSED_FACE)
    return cmap


def plot_category_row(
    ax: plt.Axes,
    values: np.ndarray,
    vmax: float,
    *,
    show_xlabels: bool = True,
    show_ylabel: bool = True,
    header: str | None = None,
) -> None:
    """Category |SHAP| under a radar. A header row carries the model name."""
    row = np.asarray(values, dtype=float).reshape(1, -1)
    n_cols = row.shape[1]
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    cmap = _ylorrd_cmap()
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    if header:
        data_y = 1.0
        ax.imshow(
            np.ma.masked_invalid(row),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
            extent=(-0.5, n_cols - 0.5, 0.5, 1.5),
        )
        ax.add_patch(
            plt.Rectangle(
                (-0.5, -0.5),
                n_cols,
                1.0,
                facecolor="#D5DEE8",
                edgecolor="none",
                zorder=3,
                clip_on=False,
            )
        )
        ax.plot(
            [-0.5, n_cols - 0.5],
            [0.5, 0.5],
            color="#222222",
            linewidth=0.9,
            solid_capstyle="butt",
            zorder=4,
            clip_on=False,
        )
        ax.text(
            (n_cols - 1) / 2.0,
            0.0,
            header,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#222222",
            zorder=5,
            clip_on=False,
        )
        ax.set_ylim(1.5, -0.5)
    else:
        data_y = 0.0
        ax.imshow(
            np.ma.masked_invalid(row),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_ylim(0.5, -0.5)
    for j, val in enumerate(row[0]):
        if not np.isfinite(val):
            text, color = "—", UNUSED_TEXT
        else:
            text, color = _format_importance(val), _heatmap_text_color(val, vmax)
        ax.text(
            j, data_y, text,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=color,
            zorder=4,
        )
    if show_xlabels:
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(
            list(CATEGORY_ROW_LABELS[:n_cols]),
            fontsize=9.0,
            fontweight="bold",
        )
        ax.tick_params(axis="x", length=0, pad=2)
    else:
        ax.set_xticks([])
    if show_ylabel and header is None:
        ax.set_yticks([data_y])
        ax.set_yticklabels(["mean |SHAP|"], fontsize=9.5, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=2)
    else:
        ax.set_yticks([])
    ax.set_xlim(-0.5, n_cols - 0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#222222")


def plot_radar_row(
    fig: plt.Figure,
    outer_spec,
    metrics: dict[str, ModelMetrics],
    *,
    panel_label: str = "A",
    row_note: str | None = None,
    category_matrix: np.ndarray | None = None,
    category_models: list[str] | None = None,
) -> None:
    """One radar per ablation arm. Panel A adds a category |SHAP| row under each radar."""
    present = [m for m in MODEL_ORDER if m in metrics]
    n = max(len(present), 1)
    show_columns = category_matrix is not None and category_models is not None
    column_vmax = 1.0
    if show_columns:
        finite = category_matrix[np.isfinite(category_matrix)]
        if finite.size:
            column_vmax = float(np.nanmax(finite))
        if column_vmax <= 0:
            column_vmax = 1.0
    model_index = {model_id: j for j, model_id in enumerate(category_models or [])}

    def _add_shared_colorbar(slot_spec) -> plt.Axes:
        """Color bar on the left, spanning the Full Model radar and its heatmap row."""
        slot = fig.add_subplot(slot_spec)
        slot.set_axis_off()
        cax = slot.inset_axes([0.22, 0.02, 0.22, 0.84])
        norm = mcolors.Normalize(vmin=0.0, vmax=column_vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=_ylorrd_cmap())
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label(
            "mean |SHAP|",
            fontsize=10.5,
            fontweight="bold",
        )
        cbar.ax.tick_params(labelsize=9, length=2)
        return slot

    def _draw_radar(
        cell_spec,
        model_id: str,
        *,
        show_ticks: bool,
        show_xlabels: bool = True,
        show_ylabel: bool = True,
    ) -> plt.Axes:
        if show_columns and model_id in model_index:
            pair = cell_spec.subgridspec(
                2, 1,
                height_ratios=[0.892, 0.108],
                hspace=0.08,
            )
            ax_radar = fig.add_subplot(pair[0, 0], polar=True)
            ax_heat = fig.add_subplot(pair[1, 0])
            plot_category_row(
                ax_heat,
                category_matrix[:, model_index[model_id]],
                column_vmax,
                show_xlabels=show_xlabels,
                show_ylabel=show_ylabel,
                header=MODEL_LABELS[model_id],
            )
            plot_radar_panel(
                ax_radar,
                model_id,
                metrics[model_id],
                show_metric_labels=show_ticks,
                show_title=False,
            )
            return ax_radar
        ax_radar = fig.add_subplot(cell_spec, polar=True)
        plot_radar_panel(
            ax_radar,
            model_id,
            metrics[model_id],
            show_metric_labels=show_ticks,
        )
        return ax_radar

    if show_columns:
        # All radar + heatmap pairs on one row. The color bar sits left of
        # Full Model and spans that radar plus the heatmap row.
        shell = outer_spec.subgridspec(
            1, 2, width_ratios=[0.07, 0.93], wspace=0.14,
        )
        cbar_ax = _add_shared_colorbar(shell[0, 0])
        inner = shell[0, 1].subgridspec(1, n, wspace=0.651)
        for idx, model_id in enumerate(present):
            if model_id not in model_index:
                continue
            ax = _draw_radar(
                inner[0, idx],
                model_id,
                show_ticks=(idx == 0),
                show_xlabels=True,
                show_ylabel=(idx == 0),
            )
            if idx == 0:
                cbar_ax.text(
                    -0.05,
                    1.02,
                    panel_label,
                    transform=cbar_ax.transAxes,
                    fontsize=15,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    clip_on=False,
                )
                if row_note:
                    ax.text(
                        0.08,
                        1.22,
                        row_note,
                        transform=ax.transAxes,
                        fontsize=11.5,
                        fontweight="bold",
                        color="#333333",
                        ha="left",
                        va="bottom",
                        clip_on=False,
                    )
    else:
        inner = outer_spec.subgridspec(1, n, wspace=0.55)
        for idx, model_id in enumerate(present):
            ax = _draw_radar(inner[0, idx], model_id, show_ticks=(idx == 0))
            if idx == 0:
                ax.text(
                    -0.22,
                    1.22,
                    panel_label,
                    transform=ax.transAxes,
                    fontsize=15,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    clip_on=False,
                )
                if row_note:
                    ax.text(
                        0.08,
                        1.22,
                        row_note,
                        transform=ax.transAxes,
                        fontsize=11.5,
                        fontweight="bold",
                        color="#333333",
                        ha="left",
                        va="bottom",
                        clip_on=False,
                    )


def _format_importance(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    abs_v = abs(value)
    if abs_v >= 100:
        return f"{value:.0f}"
    if abs_v >= 10:
        return f"{value:.1f}"
    if abs_v >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _heatmap_text_color(value: float, vmax: float) -> str:
    """Dark text on light cells; white on saturated cells."""
    if not np.isfinite(value) or vmax <= 0:
        return "#111111"
    return "#FFFFFF" if (value / vmax) >= 0.72 else "#111111"


def plot_importance_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    feature_rows: list[tuple[str, str, str]],
    importance: dict[str, dict[str, float]],
    present_models: list[str],
    *,
    panel_label: str = "SI",
) -> None:
    """Per-feature mean |SHAP| heatmap; unused features are grey."""
    _panel_title(ax, panel_label, pad=6)

    n_rows = len(feature_rows)
    n_cols = len(present_models)
    if n_rows == 0 or n_cols == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No feature importance available.", ha="center", va="center")
        return

    matrix = np.full((n_rows, n_cols), np.nan, dtype=float)
    groups: list[str] = []
    feat_labels: list[str] = []
    for i, (feat, label, group) in enumerate(feature_rows):
        groups.append(group)
        feat_labels.append(label)
        for j, model_id in enumerate(present_models):
            val = importance.get(model_id, {}).get(feat)
            if val is not None and np.isfinite(val):
                matrix[i, j] = float(val)

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    cmap = _ylorrd_cmap()
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color=UNUSED_TEXT,
                )
            else:
                ax.text(
                    j,
                    i,
                    _format_importance(val),
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color=_heatmap_text_color(val, vmax),
                )

    # Group separators.
    for i in range(1, n_rows):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="#222222", linewidth=1.3, zorder=5)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [MODEL_HEADER_LABELS[m] for m in present_models],
        fontsize=11.5,
        fontweight="bold",
    )
    for tick, model_id in zip(ax.get_xticklabels(), present_models):
        tick.set_color(MODEL_COLORS[model_id])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=4)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(feat_labels, fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Group labels along the left margin.
    i = 0
    while i < n_rows:
        j = i + 1
        while j < n_rows and groups[j] == groups[i]:
            j += 1
        mid = 0.5 * (i + j - 1)
        ax.text(
            -0.85,
            mid,
            groups[i],
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#333333",
            rotation=90,
            clip_on=False,
            transform=ax.transData,
        )
        i = j

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#222222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean |SHAP|", fontsize=11.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)


def plot_category_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: np.ndarray,
    present_models: list[str],
) -> None:
    """Panel B: summed mean |SHAP| for the three feature categories."""
    _panel_title(ax, "B", pad=6)
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No feature importance available.", ha="center", va="center")
        return

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    cmap = _ylorrd_cmap()
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(
                    j, i, "—",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=UNUSED_TEXT,
                )
            else:
                ax.text(
                    j, i, _format_importance(val),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=_heatmap_text_color(val, vmax),
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [MODEL_HEADER_LABELS[m] for m in present_models],
        fontsize=11.5,
        fontweight="bold",
    )
    for tick, model_id in zip(ax.get_xticklabels(), present_models):
        tick.set_color(MODEL_COLORS[model_id])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(CATEGORY_ORDER), fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Sum of per-feature mean |SHAP|", fontsize=11.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)


def plot_nc_pie(ax: plt.Axes, counts: dict[str, int]) -> None:
    _panel_title(ax, "C", pad=6, fontsize=19)
    sizes = [int(counts.get(cls, 0)) for cls in NC_CLASS_ORDER]
    total = sum(sizes)
    if total == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No annotation PDBs found.", ha="center", va="center")
        return
    colors = [NC_CLASS_COLORS[cls] for cls in NC_CLASS_ORDER]
    wedges, _texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 1.1, "edgecolor": "white"},
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 3 else "",
        pctdistance=0.62,
    )
    for txt in autotexts:
        txt.set_fontsize(16)
        txt.set_fontweight("bold")
        txt.set_color("white")
    ax.legend(
        wedges,
        [
            f"{NC_CLASS_LABELS[cls]}  (n={counts.get(cls, 0)})"
            for cls in NC_CLASS_ORDER
        ],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        frameon=False,
        fontsize=17,
        prop={"weight": "bold"},
    )
    ax.set_aspect("equal")
    ax.text(
        0.5,
        -0.12,
        "Noncovalent Pre-organization in the Test Set",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15.5,
        fontweight="bold",
    )


def plot_hitrate_xy(
    ax: plt.Axes,
    comparison: dict[str, dict[str, float | int | None]],
) -> None:
    """Hit rate at top 1% across ordered pre-organization categories."""
    _panel_title(ax, "E", pad=6, fontsize=19)
    classes = [cls for cls in COMPARE_CLASS_ORDER if cls in comparison]
    x = np.arange(len(classes), dtype=float)
    series = (
        (
            "full",
            [
                100.0 * float(comparison[cls]["hit_top_1_full"])
                if np.isfinite(comparison[cls]["hit_top_1_full"])
                else 0.0
                for cls in classes
            ],
        ),
        (
            "noncovalent",
            [
                100.0 * float(comparison[cls]["hit_top_1_nc"])
                if np.isfinite(comparison[cls]["hit_top_1_nc"])
                else 0.0
                for cls in classes
            ],
        ),
    )
    for model_id, vals in series:
        ax.plot(
            x,
            vals,
            color=MODEL_COLORS[model_id],
            linestyle=MODEL_LINESTYLES[model_id],
            linewidth=1.6,
            marker=MODEL_MARKERS[model_id],
            markersize=7.5,
            markerfacecolor=MODEL_COLORS[model_id],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=MODEL_LABELS[model_id],
            zorder=3,
        )
    # Full sits just left of each point, noncovalent just right, so close rates stay readable.
    label_offsets = {"full": (-8, 6), "noncovalent": (8, 6)}
    label_ha = {"full": "right", "noncovalent": "left"}
    for model_id, vals in series:
        for xi, val in zip(x, vals):
            ax.annotate(
                f"{val:.1f}",
                (xi, val),
                textcoords="offset points",
                xytext=label_offsets[model_id],
                ha=label_ha[model_id],
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=MODEL_COLORS[model_id],
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [COMPARE_CLASS_LABELS.get(cls, cls) for cls in classes],
        fontsize=17,
        fontweight="bold",
    )
    ax.set_xlim(-0.45, max(len(classes) - 0.55, 0.45))
    ax.tick_params(axis="x", pad=8)
    ax.set_xlabel(
        "Noncovalent Pre-organization",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel("Hit rate at top 1% (%)", fontsize=19, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(
        frameon=False,
        fontsize=17,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.14),
        borderaxespad=0.0,
    )
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(17)


def plot_rank_wins(
    ax: plt.Axes,
    comparison: dict[str, dict[str, float | int | None]],
) -> None:
    """Grouped bars: share of non-tie pairs where each model ranked the site better."""
    _panel_title(ax, "F", pad=6, fontsize=19)
    classes = [cls for cls in COMPARE_CLASS_ORDER if cls in comparison]
    y = np.arange(len(classes), dtype=float)
    height = 0.34
    full_share: list[float] = []
    nc_share: list[float] = []
    for cls in classes:
        fw = int(comparison[cls]["wins_full"])
        nw = int(comparison[cls]["wins_nc"])
        decisive = fw + nw
        full_share.append(100.0 * fw / decisive if decisive else 0.0)
        nc_share.append(100.0 * nw / decisive if decisive else 0.0)

    ax.barh(
        y + height / 2,
        full_share,
        height,
        color=MODEL_COLORS["full"],
        edgecolor="white",
        linewidth=0.6,
        label="Full ranks site better",
        zorder=3,
    )
    ax.barh(
        y - height / 2,
        nc_share,
        height,
        color=MODEL_COLORS["noncovalent"],
        edgecolor="white",
        linewidth=0.6,
        label="Non-covalent ranks site better",
        zorder=3,
    )
    def _win_label(ax, share, y_pos, color):
        if share >= 86:
            ax.text(
                share - 1.6, y_pos, f"{share:.0f}%",
                ha="right", va="center", fontsize=17, fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                share + 1.2, y_pos, f"{share:.0f}%",
                ha="left", va="center", fontsize=17, fontweight="bold",
                color=color,
            )

    for yi, fs, ns in zip(y, full_share, nc_share):
        _win_label(ax, fs, yi + height / 2, MODEL_COLORS["full"])
        _win_label(ax, ns, yi - height / 2, MODEL_COLORS["noncovalent"])
    ax.set_yticks(y)
    ax.set_yticklabels(
        [COMPARE_CLASS_LABELS.get(cls, cls) for cls in classes],
        fontsize=17,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Noncovalent Pre-organization",
        fontsize=20,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "% of pairs with a better target rank",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xlim(0, 108)
    ax.legend(frameon=False, fontsize=17, loc="upper right")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(17)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Write a 300 dpi PNG and a matching SVG, then close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.22,
    }
    png_path = path.with_suffix(".png")
    svg_path = path.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, **save_kwargs)
    fig.savefig(svg_path, format="svg", **save_kwargs)
    plt.close(fig)
    print(f"Saved figure -> {png_path}")
    print(f"Saved figure -> {svg_path}")


def _save_panel_figures(
    output_path: Path,
    *,
    metrics: dict[str, ModelMetrics],
    subset_metrics: dict[str, ModelMetrics],
    category_matrix: np.ndarray,
    present_models: list[str],
    pie_counts: dict[str, int],
    comparison: dict[str, dict[str, float | int | None]],
    feature_rows: list[tuple[str, str, str]],
    importance: dict[str, dict[str, float]],
) -> None:
    """One high-resolution PNG and SVG for each panel, beside the combined figure."""
    panel_dir = output_path.parent / f"{output_path.stem}_panels"

    fig = plt.figure(figsize=(FIG_WIDTH, 7.0))
    plot_radar_row(
        fig, fig.add_gridspec(1, 1)[0, 0], metrics,
        panel_label="A",
        category_matrix=category_matrix,
        category_models=present_models,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08)
    _save_figure(fig, panel_dir / "A")

    fig = plt.figure(figsize=(12.4, 4.8))
    ax = fig.add_subplot(111)
    plot_category_heatmap(fig, ax, category_matrix, present_models)
    fig.subplots_adjust(left=0.16, right=0.90, top=0.88, bottom=0.16)
    _save_figure(fig, panel_dir / "B")

    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)
    plot_nc_pie(ax, pie_counts)
    fig.subplots_adjust(left=0.04, right=0.72, top=0.92, bottom=0.12)
    _save_figure(fig, panel_dir / "C")

    fig = plt.figure(figsize=(FIG_WIDTH, 6.0))
    plot_radar_row(
        fig,
        fig.add_gridspec(1, 1)[0, 0],
        subset_metrics if subset_metrics else metrics,
        panel_label="D",
        row_note="Limited / No strong noncovalent recognition",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08)
    _save_figure(fig, panel_dir / "D")

    fig = plt.figure(figsize=(9.6, 5.2))
    ax = fig.add_subplot(111)
    if comparison:
        plot_hitrate_xy(ax, comparison)
    else:
        ax.axis("off")
        _panel_title(ax, "E", pad=6, fontsize=19)
        ax.text(0.5, 0.5, "Comparison data unavailable.", ha="center", va="center")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.16)
    _save_figure(fig, panel_dir / "E")

    fig = plt.figure(figsize=(FIG_WIDTH, 4.6))
    ax = fig.add_subplot(111)
    if comparison:
        plot_rank_wins(ax, comparison)
    else:
        ax.axis("off")
        _panel_title(ax, "F", pad=6, fontsize=19)
        ax.text(0.5, 0.5, "Comparison data unavailable.", ha="center", va="center")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.18)
    _save_figure(fig, panel_dir / "F")

    fig = plt.figure(figsize=(FIG_WIDTH, 8.4))
    ax = fig.add_subplot(111)
    plot_importance_heatmap(
        fig, ax, feature_rows, importance, present_models, panel_label="SI",
    )
    fig.subplots_adjust(left=0.18, right=0.92, top=0.94, bottom=0.06)
    _save_figure(fig, panel_dir / "SI")


def create_figure(
    *,
    input_dir: Path,
    output_path: Path,
    bedroc_alpha: float = DEFAULT_BEDROC_ALPHA,
    metrics_json: Path | None = None,
) -> None:
    _configure_figure_fonts()
    models = discover_models(input_dir)
    if not models:
        raise FileNotFoundError(
            f"No ablation model directories or summary files found in {input_dir}. "
            "Expected subdirectories named like 'Full Model', 'Noncovalent', "
            "'Warhead and nucleophile', 'Covalent reactivity only'."
        )

    missing = [MODEL_LABELS[m] for m in MODEL_ORDER if m not in models]
    if missing:
        print("[WARN] Missing ablation arm(s): " + ", ".join(missing))

    metrics: dict[str, ModelMetrics] = {}
    for model_id in MODEL_ORDER:
        if model_id not in models:
            continue
        files = models[model_id]
        print(f"\n[{MODEL_LABELS[model_id]}]  {files.directory}")
        if files.summaries:
            for thr, (path, _) in sorted(files.summaries.items()):
                print(f"  summary @ top {thr:g}%: {path.name}")
        else:
            print("  summary: (none)")
        print(f"  ranks:   {files.rank_csv.name if files.rank_csv else '(none)'}")
        row = compute_model_metrics(files, bedroc_alpha=bedroc_alpha)
        metrics[model_id] = row
        print(
            f"  Hit@1%={row.hit_top_1 * 100:.1f}%  "
            f"Hit@10%={row.hit_top_10 * 100:.1f}%  "
            f"median SSR={row.median_ssr * 100:.1f}%  "
            f"adj. logAUC={row.adjusted_logauc:.3f}  "
            f"BEDROC={row.bedroc:.3f}  (n={row.n_rows})"
        )
        for key, src in row.sources.items():
            print(f"    {key}: {src}")

    feature_rows, importance = collect_feature_table(models)
    present_models = [m for m in MODEL_ORDER if m in models]
    category_matrix = aggregate_category_importance(
        feature_rows, importance, present_models,
    )

    rank_tables: dict[str, pd.DataFrame] = {}
    for model_id, files in models.items():
        if files.rank_csv is None:
            continue
        rank_tables[model_id] = load_rank_table(files.rank_csv)

    annotation_path = find_annotation_path(input_dir)
    pdb_to_class: dict[str, str] = {}
    pie_counts: dict[str, int] = {cls: 0 for cls in NC_CLASS_ORDER}
    subset_metrics: dict[str, ModelMetrics] = {}
    comparison: dict[str, dict[str, float | int | None]] = {}
    comparison_detail: dict[str, dict[str, float | int | None]] = {}
    annotated_pdbs: set[str] = set()
    subset_pdbs: set[str] = set()

    if annotation_path is None:
        print(
            "[WARN] No TestSet_Noncovalent_Annotation_v2 (.csv/.xlsx) in "
            f"{input_dir}; skipping panels C–F."
        )
    else:
        print(f"\nAnnotation: {annotation_path}")
        annot = load_nc_annotation(annotation_path)
        pdb_to_class = dict(zip(annot["pdb"], annot["nc_class"]))
        for model_id in list(rank_tables):
            rank_tables[model_id] = annotate_rank_df(rank_tables[model_id], pdb_to_class)

        test_pdbs: set[str] = set()
        if "full" in rank_tables:
            test_pdbs = set(rank_tables["full"]["pdb"].dropna().astype(str))
        elif rank_tables:
            first = next(iter(rank_tables.values()))
            test_pdbs = set(first["pdb"].dropna().astype(str))
        annotated_pdbs = set(pdb_to_class) & test_pdbs if test_pdbs else set(pdb_to_class)
        unmatched_test = sorted(test_pdbs - set(pdb_to_class))
        unused_annot = sorted(set(pdb_to_class) - test_pdbs)
        print(
            f"  annotated PDBs: {len(pdb_to_class)}  "
            f"matched in test labels: {len(annotated_pdbs)}"
        )
        if unmatched_test:
            print(f"  [WARN] {len(unmatched_test)} test PDBs have no annotation")
        if unused_annot:
            print(f"  [WARN] {len(unused_annot)} annotation PDBs not in test labels")

        for cls in NC_CLASS_ORDER:
            pie_counts[cls] = sum(
                1 for pdb in annotated_pdbs if pdb_to_class.get(pdb) == cls
            )
        print(
            "  unique PDBs in pie: "
            + ", ".join(f"{cls}={pie_counts[cls]}" for cls in NC_CLASS_ORDER)
        )

        subset_pdbs = {
            pdb for pdb in annotated_pdbs if pdb_to_class.get(pdb) in {"Limited", "No"}
        }
        print(f"  Limited+No PDBs for panel D: {len(subset_pdbs)}")
        source_note = (
            f"recomputed from test_label_results on {len(subset_pdbs)} "
            "Limited/No PDBs"
        )
        for model_id in MODEL_ORDER:
            if model_id not in rank_tables:
                continue
            sub = rank_tables[model_id][
                rank_tables[model_id]["pdb"].isin(subset_pdbs)
            ]
            if sub.empty:
                print(f"  [WARN] {MODEL_LABELS[model_id]} has no Limited/No rows")
                continue
            row = compute_metrics_from_rank_df(
                sub,
                model_id=model_id,
                bedroc_alpha=bedroc_alpha,
                source=source_note,
            )
            subset_metrics[model_id] = row
            print(
                f"  D [{MODEL_LABELS[model_id]}]  "
                f"Hit@1%={row.hit_top_1 * 100:.1f}%  "
                f"Hit@10%={row.hit_top_10 * 100:.1f}%  "
                f"median SSR={row.median_ssr * 100:.1f}%  "
                f"adj. logAUC={row.adjusted_logauc:.3f}  "
                f"BEDROC={row.bedroc:.3f}  (n={row.n_rows})"
            )

        if "full" in rank_tables and "noncovalent" in rank_tables:
            paired = pair_full_vs_nc(rank_tables["full"], rank_tables["noncovalent"])
            comparison_detail = summarize_nc_comparison(
                rank_tables["full"], rank_tables["noncovalent"], paired,
            )
            comparison = summarize_nc_comparison(
                rank_tables["full"],
                rank_tables["noncovalent"],
                paired,
                groups=COMPARE_CLASS_MEMBERS,
                group_order=COMPARE_CLASS_ORDER,
            )
            print("\nFull vs non-covalent by strong noncovalent class:")
            for cls in NC_CLASS_ORDER:
                stats = comparison_detail[cls]
                pval = stats["wilcoxon_p"]
                p_txt = f"{pval:.4g}" if isinstance(pval, float) else "n/a"
                print(
                    f"  {cls}: n_rows={stats['n_full_rows']}  "
                    f"n_pdb={stats['n_unique_pdb']}  n_paired={stats['n_paired']}\n"
                    f"    hit@1% full={100 * float(stats['hit_top_1_full']):.1f}%  "
                    f"nc={100 * float(stats['hit_top_1_nc']):.1f}%\n"
                    f"    mean rank full={stats['mean_rank_full']:.2f}  "
                    f"nc={stats['mean_rank_nc']:.2f}\n"
                    f"    wins full={stats['wins_full']}  nc={stats['wins_nc']}  "
                    f"ties={stats['ties']}  Wilcoxon p={p_txt}"
                )
            print("\nFull vs non-covalent for panels E–F (Yes vs Limited+No):")
            for cls in COMPARE_CLASS_ORDER:
                stats = comparison[cls]
                pval = stats["wilcoxon_p"]
                p_txt = f"{pval:.4g}" if isinstance(pval, float) else "n/a"
                print(
                    f"  {cls}: n_rows={stats['n_full_rows']}  "
                    f"n_pdb={stats['n_unique_pdb']}  n_paired={stats['n_paired']}\n"
                    f"    hit@1% full={100 * float(stats['hit_top_1_full']):.1f}%  "
                    f"nc={100 * float(stats['hit_top_1_nc']):.1f}%\n"
                    f"    mean rank full={stats['mean_rank_full']:.2f}  "
                    f"nc={stats['mean_rank_nc']:.2f}\n"
                    f"    wins full={stats['wins_full']}  nc={stats['wins_nc']}  "
                    f"ties={stats['ties']}  Wilcoxon p={p_txt}"
                )
        else:
            print("[WARN] Need Full and Noncovalent rank tables for panels E–F.")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        6,
        1,
        height_ratios=[1.45, 0.82, 1.12, 0.78, 0.72, 1.18],
        hspace=0.62,
    )
    plot_radar_row(
        fig, gs[0, 0], metrics,
        panel_label="A",
        category_matrix=category_matrix,
        category_models=present_models,
    )

    row_bc = gs[1, 0].subgridspec(1, 2, width_ratios=[1.20, 0.80], wspace=0.42)
    ax_b = fig.add_subplot(row_bc[0, 0])
    ax_c = fig.add_subplot(row_bc[0, 1])
    plot_category_heatmap(fig, ax_b, category_matrix, present_models)
    plot_nc_pie(ax_c, pie_counts)

    plot_radar_row(
        fig,
        gs[2, 0],
        subset_metrics if subset_metrics else metrics,
        panel_label="D",
        row_note="Limited / No strong noncovalent recognition",
    )

    # Three categories do not need the full figure width. The open right side
    # is left for a later layout pass.
    row_e = gs[3, 0].subgridspec(1, 2, width_ratios=[0.46, 0.54], wspace=0.08)
    ax_e = fig.add_subplot(row_e[0, 0])
    ax_f = fig.add_subplot(gs[4, 0])
    if comparison:
        plot_hitrate_xy(ax_e, comparison)
        plot_rank_wins(ax_f, comparison)
    else:
        ax_e.axis("off")
        ax_f.axis("off")
        _panel_title(ax_e, "E", pad=6, fontsize=19)
        _panel_title(ax_f, "F", pad=6, fontsize=19)
        ax_e.text(0.5, 0.5, "Comparison data unavailable.", ha="center", va="center")
        ax_f.text(0.5, 0.5, "Comparison data unavailable.", ha="center", va="center")

    ax_si = fig.add_subplot(gs[5, 0])
    plot_importance_heatmap(
        fig, ax_si, feature_rows, importance, present_models, panel_label="SI",
    )

    fig.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.04)
    _save_figure(fig, output_path)

    _save_panel_figures(
        output_path,
        metrics=metrics,
        subset_metrics=subset_metrics,
        category_matrix=category_matrix,
        present_models=present_models,
        pie_counts=pie_counts,
        comparison=comparison,
        feature_rows=feature_rows,
        importance=importance,
    )

    def _metrics_payload(rows: dict[str, ModelMetrics]) -> dict:
        return {
            model_id: {
                "label": row.label,
                "hit_rate_top_1pct": row.hit_top_1,
                "hit_rate_top_10pct": row.hit_top_10,
                "median_search_space_reduction": row.median_ssr,
                "adjusted_logAUC": row.adjusted_logauc,
                "BEDROC": row.bedroc,
                "n_rank_rows": row.n_rows,
                "sources": row.sources,
            }
            for model_id, row in rows.items()
        }

    payload = {
        "bedroc_alpha": bedroc_alpha,
        "logauc_lambda": LOGAUC_LAMBDA,
        "random_logauc": RANDOM_LOGAUC,
        "models": _metrics_payload(metrics),
        "subset_limited_or_no": {
            "n_pdbs": len(subset_pdbs),
            "models": _metrics_payload(subset_metrics),
        },
        "annotation": {
            "path": str(annotation_path) if annotation_path is not None else None,
            "unique_pdb_counts": pie_counts,
        },
        "full_vs_noncovalent_by_class": comparison,
        "full_vs_noncovalent_by_class_detail": comparison_detail,
        "category_importance": {
            MODEL_LABELS[model_id]: {
                cat: (
                    None
                    if not np.isfinite(category_matrix[i, j])
                    else float(category_matrix[i, j])
                )
                for i, cat in enumerate(CATEGORY_ORDER)
            }
            for j, model_id in enumerate(present_models)
        },
        "feature_importance": {
            MODEL_LABELS[model_id]: values
            for model_id, values in importance.items()
        },
    }
    out_json = metrics_json
    if out_json is None:
        out_json = output_path.with_suffix(".metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    def _jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"Saved metrics -> {out_json}")


def parse_args() -> argparse.Namespace:
    analysis_dir = _analysis_dir()
    p = argparse.ArgumentParser(
        description=(
            "Create ablation figure: full radars, category SHAP heatmap, "
            "noncovalent-recognition pie, Limited/No radars, hit@1% and "
            "rank-win bars, plus SI per-feature heatmap."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Input directory should contain one named folder per ablation arm, e.g.
              Full Model/, Noncovalent/, Warhead and nucleophile/,
              Covalent reactivity only/
            Each folder needs Training_Cov_Screen outputs: summary JSON(s) for
            top 1% and top 10% hit rates, and test_label_results.csv for ranks.
            """
        ),
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=analysis_dir / "Ablation_Inputs",
        help="Directory of named ablation-model folders "
             "(default: scoring-analysis/Ablation_Inputs).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=analysis_dir / "ablation_figure.png",
        help="Output figure path (default: scoring-analysis/ablation_figure.png).",
    )
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional metrics JSON path (default: alongside --output).",
    )
    p.add_argument(
        "--bedroc-alpha",
        type=float,
        default=DEFAULT_BEDROC_ALPHA,
        help="BEDROC α (default 20, Truchon & Bayly / RDKit).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.bedroc_alpha <= 0:
        raise ValueError("--bedroc-alpha must be positive.")
    create_figure(
        input_dir=args.input_dir,
        output_path=args.output,
        bedroc_alpha=args.bedroc_alpha,
        metrics_json=args.metrics_json,
    )


if __name__ == "__main__":
    main()
