"""
LGBM Classifier for Nucleophilic Residue Hit Probability
========================================================
Pointwise binary classifier (relevance 1/0) with cluster-grouped CV.
Sibling to Training_Cov_Screen.py (ranker): reuses load/cluster/split/features.

Primary metrics: pooled + group-mean AUC/PR-AUC, Brier, ECE (+ reliability curve).
Secondary (sanity only): within-protein Hit@K / Hit@top-% using pred_prob.

Usage:
    python Training_Cov_Classifier.py \\
        --training training.csv --labels labels.csv \\
        --pdb_folder ./pdbs --deep-cluster \\
        --output_dir lgbm_classifier_output

    # Reuse ranker splits / clusters for paired comparison:
    python Training_Cov_Classifier.py ... --deep-cluster \\
        --clusters-tsv mmseqs_output/clusters.tsv \\
        --split-csv query_group_splits.csv

    # Optional XGBoost side-by-side on the same folds:
    python Training_Cov_Classifier.py ... --compare-xgb

Calibration (isotonic on a held-out train-cluster slice):
    default: apply if OOF ECE > --calibrate-ece-threshold (0.05)
    --no-calibrate / --force-calibrate override
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from Training_Cov_Screen import (
    DEFAULT_RANK_BONUS_EPSILON,
    REWARD_MODES,
    aggregate_nucleophile_totals,
    analyze_residue_composition,
    apply_large_pool_prefilter,
    assign_splits,
    build_cluster_map,
    build_features,
    evaluate_predictions,
    export_shap_csvs,
    load_and_merge,
    parse_simple_filters,
    print_residue_composition,
    print_results,
    summarize_eval_metrics,
)

warnings.filterwarnings("ignore")

DEPTH_GRID_DEFAULT = (3, 4, 5, 6)
DEFAULT_ECE_THRESHOLD = 0.05
DEFAULT_CALIB_FRACTION = 0.15


# ─────────────────────────────────────────────
# MODEL PARAMS
# ─────────────────────────────────────────────

def get_lgbm_classifier_params(max_depth: int = 6) -> dict:
    return {
        "objective": "binary",
        "metric": ["binary_logloss"],
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": max_depth,
        "num_leaves": min(31, 2 ** max_depth - 1),
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_classifier_params(
    scale_pos_weight: float,
    max_depth: int = 6,
) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": max_depth,
        "max_leaves": min(31, 2 ** max_depth - 1),
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }


def scale_pos_weight_from_y(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos <= 0:
        return 1.0
    return float(n_neg) / float(n_pos)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probs))


def _safe_pr_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, probs))


def _safe_logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-7, 1.0 - 1e-7)
    try:
        return float(log_loss(y_true, probs, labels=[0, 1]))
    except ValueError:
        return float("nan")


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, dict]:
    """
    ECE via sklearn calibration_curve (quantile bins when possible).
    Returns (ece, reliability_payload).
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    payload = {
        "n_bins": n_bins,
        "strategy": "quantile",
        "prob_true": [],
        "prob_pred": [],
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
    }
    if len(y_true) < n_bins or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        payload["strategy"] = "skipped"
        return float("nan"), payload

    try:
        prob_true, prob_pred = calibration_curve(
            y_true, probs, n_bins=n_bins, strategy="quantile",
        )
    except ValueError:
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, probs, n_bins=n_bins, strategy="uniform",
            )
            payload["strategy"] = "uniform"
        except ValueError:
            payload["strategy"] = "failed"
            return float("nan"), payload

    payload["prob_true"] = [float(x) for x in prob_true]
    payload["prob_pred"] = [float(x) for x in prob_pred]
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return ece, payload


def classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    ece, reliability = expected_calibration_error(y_true, probs, n_bins=n_bins)
    return {
        "auc": _safe_auc(y_true, probs),
        "pr_auc": _safe_pr_auc(y_true, probs),
        "brier": float(brier_score_loss(y_true, probs)),
        "logloss": _safe_logloss(y_true, probs),
        "ece": ece,
        "reliability": reliability,
        "n": int(len(y_true)),
        "n_pos": int(np.asarray(y_true).astype(int).sum()),
        "n_neg": int((np.asarray(y_true).astype(int) == 0).sum()),
    }


def pool_size_distribution(groups: np.ndarray | pd.Series) -> dict:
    """Summarize query-group (pocket) sizes for reviewer skew checks."""
    sizes = pd.Series(groups).astype(str).value_counts()
    if sizes.empty:
        return {
            "n_groups": 0,
            "min": 0, "median": 0.0, "mean": 0.0, "max": 0,
            "p90": 0.0, "p95": 0.0, "sum": 0,
        }
    vals = sizes.to_numpy(dtype=float)
    return {
        "n_groups": int(len(sizes)),
        "min": int(vals.min()),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "max": int(vals.max()),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "sum": int(vals.sum()),
    }


def group_averaged_clf_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray | pd.Series,
) -> dict:
    """
    Mean AUC / PR-AUC across query groups (equal weight per pocket).

    Groups with <2 classes are skipped (AUC undefined). This counters
    instance-weighting where large pockets dominate pooled metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    groups = np.asarray(groups)
    aucs: list[float] = []
    prs: list[float] = []
    n_skipped = 0

    for g in pd.unique(groups):
        mask = groups == g
        yg = y_true[mask]
        pg = probs[mask]
        if len(np.unique(yg)) < 2:
            n_skipped += 1
            continue
        aucs.append(_safe_auc(yg, pg))
        prs.append(_safe_pr_auc(yg, pg))

    return {
        "auc_group_mean": float(np.nanmean(aucs)) if aucs else float("nan"),
        "pr_auc_group_mean": float(np.nanmean(prs)) if prs else float("nan"),
        "n_groups": int(len(pd.unique(groups))),
        "n_groups_scored": int(len(aucs)),
        "n_groups_skipped_single_class": int(n_skipped),
    }


def classification_metrics_with_groups(
    y_true: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray | pd.Series | None = None,
    n_bins: int = 10,
) -> dict:
    """Pooled metrics + optional group-averaged AUC/PR-AUC and pool-size stats."""
    out = classification_metrics(y_true, probs, n_bins=n_bins)
    if groups is None:
        return out
    out.update(group_averaged_clf_metrics(y_true, probs, groups))
    out["pool_size"] = pool_size_distribution(groups)
    return out


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    # fallback
    return np.asarray(model.predict(X), dtype=float)


# ─────────────────────────────────────────────
# FIT HELPERS
# ─────────────────────────────────────────────

def fit_lgbm_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_depth: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        **get_lgbm_classifier_params(max_depth=max_depth)
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=["binary_logloss"],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


def fit_xgb_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    max_depth: int,
):
    try:
        import xgboost as xgb
    except ImportError:
        sys.exit(
            "[ERROR] --compare-xgb requires xgboost. "
            "Install with: pip install xgboost"
        )

    model = xgb.XGBClassifier(
        **get_xgb_classifier_params(scale_pos_weight, max_depth=max_depth)
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def unwrap_tree_estimator(model):
    """
    Return the underlying tree model for feature_importances_ / TreeSHAP.

    CalibratedClassifierCV stores the prefit base on each inner calibrated
    classifier as `.estimator` (sklearn >=1.2) or `.base_estimator` (older).
    Prefer an explicit hasattr chain over try/except so we never silently
    fall back to the calibration wrapper.
    """
    if not hasattr(model, "calibrated_classifiers_"):
        return model

    inners = getattr(model, "calibrated_classifiers_", None) or []
    if not inners:
        print("[WARN] Calibrated model has empty calibrated_classifiers_; "
              "using wrapper for importance/SHAP.")
        return model

    inner = inners[0]
    for attr in ("estimator", "base_estimator"):
        if hasattr(inner, attr):
            est = getattr(inner, attr)
            if est is not None:
                return est

    # Prefit API also keeps the original estimator on the outer object
    for attr in ("estimator", "base_estimator"):
        if hasattr(model, attr):
            est = getattr(model, attr)
            if est is not None and est is not model:
                return est

    print("[WARN] Could not unwrap calibrated tree estimator; "
          "importance/SHAP may target the calibration wrapper.")
    return model


def _mean_abs_shap(model, X: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    try:
        import shap
    except ImportError:
        return np.full(len(feature_cols), np.nan)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # binary: use positive class
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.asarray(shap_values, dtype=np.float64)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return np.abs(shap_values).mean(axis=0)


# ─────────────────────────────────────────────
# DEPTH GRID + CV (single pass — no retrain of best depth)
# ─────────────────────────────────────────────

def _cv_one_depth(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    max_depth: int,
    backend: str = "lgbm",
    collect_shap: bool = True,
) -> tuple[dict, pd.DataFrame, np.ndarray, list, pd.DataFrame]:
    """
    One GroupKFold CV at a fixed depth.

    Returns cv_summary, fold_df, oof_probs, fold_models, shap_stability_df.
    """
    X = train_df[feature_cols].values
    y = train_df["relevance"].values.astype(int)
    groups_cluster = train_df["cluster_id"].values
    qg = train_df["query_group"].values
    oof_probs = np.full(len(train_df), np.nan, dtype=float)

    gkf = GroupKFold(n_splits=n_folds)
    fold_rows: list[dict] = []
    fold_models: list = []
    shap_by_fold: dict[int, np.ndarray] = {}

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups_cluster)):
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        if backend == "lgbm":
            model = fit_lgbm_classifier(
                X[tr_idx], y_tr, X[val_idx], y_val, max_depth,
            )
            spw = float("nan")
        else:
            spw = scale_pos_weight_from_y(y_tr)
            model = fit_xgb_classifier(
                X[tr_idx], y_tr, X[val_idx], y_val, spw, max_depth,
            )

        probs = predict_proba_positive(model, X[val_idx])
        oof_probs[val_idx] = probs
        m = classification_metrics_with_groups(y_val, probs, groups=qg[val_idx])
        fold_models.append(model)

        if collect_shap:
            shap_by_fold[fold + 1] = _mean_abs_shap(model, X[val_idx], feature_cols)

        fold_rows.append({
            "fold": fold + 1,
            "n_val": m["n"],
            "n_pos": m["n_pos"],
            "scale_pos_weight": spw,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "auc": m["auc"],
            "pr_auc": m["pr_auc"],
            "auc_group_mean": m.get("auc_group_mean", float("nan")),
            "pr_auc_group_mean": m.get("pr_auc_group_mean", float("nan")),
            "n_groups_scored": m.get("n_groups_scored", 0),
            "brier": m["brier"],
            "logloss": m["logloss"],
            "ece": m["ece"],
        })

    fold_df = pd.DataFrame(fold_rows)
    oof_mask = ~np.isnan(oof_probs)
    oof_metrics = classification_metrics_with_groups(
        y[oof_mask], oof_probs[oof_mask], groups=qg[oof_mask],
    )

    cv_summary = {
        "backend": backend,
        "max_depth": max_depth,
        "auc": float(fold_df["auc"].mean()),
        "pr_auc": float(fold_df["pr_auc"].mean()),
        "auc_group_mean": float(fold_df["auc_group_mean"].mean()),
        "pr_auc_group_mean": float(fold_df["pr_auc_group_mean"].mean()),
        "brier": float(fold_df["brier"].mean()),
        "logloss": float(fold_df["logloss"].mean()),
        "ece": float(fold_df["ece"].mean()),
        "oof_auc": oof_metrics["auc"],
        "oof_pr_auc": oof_metrics["pr_auc"],
        "oof_auc_group_mean": oof_metrics.get("auc_group_mean", float("nan")),
        "oof_pr_auc_group_mean": oof_metrics.get("pr_auc_group_mean", float("nan")),
        "oof_n_groups_scored": oof_metrics.get("n_groups_scored", 0),
        "oof_n_groups_skipped_single_class": oof_metrics.get(
            "n_groups_skipped_single_class", 0
        ),
        "oof_brier": oof_metrics["brier"],
        "oof_logloss": oof_metrics["logloss"],
        "oof_ece": oof_metrics["ece"],
        "oof_reliability": oof_metrics["reliability"],
        "oof_pool_size": oof_metrics.get("pool_size", {}),
        "n_folds": n_folds,
    }

    stability = pd.DataFrame()
    if collect_shap and shap_by_fold:
        stability = pd.DataFrame(
            {f"fold_{k}": v for k, v in shap_by_fold.items()},
            index=feature_cols,
        )
        fold_cols = [c for c in stability.columns if c.startswith("fold_")]
        stability["mean_abs_shap"] = stability[fold_cols].mean(axis=1)
        stability["std_abs_shap"] = stability[fold_cols].std(axis=1)
        stability["cv"] = stability["std_abs_shap"] / (
            stability["mean_abs_shap"] + 1e-12
        )
        stability = stability.sort_values("mean_abs_shap", ascending=False)

    return cv_summary, fold_df, oof_probs, fold_models, stability


def _shap_stability_from_fold_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    fold_models: list,
) -> pd.DataFrame:
    """Reconstruct GroupKFold val sets and compute mean |SHAP| per fold model."""
    if not fold_models:
        return pd.DataFrame()

    X = train_df[feature_cols].values
    y = train_df["relevance"].values.astype(int)
    groups = train_df["cluster_id"].values
    gkf = GroupKFold(n_splits=n_folds)
    shap_by_fold: dict[int, np.ndarray] = {}

    for fold, ((_tr_idx, val_idx), model) in enumerate(
        zip(gkf.split(X, y, groups=groups), fold_models)
    ):
        shap_by_fold[fold + 1] = _mean_abs_shap(
            unwrap_tree_estimator(model), X[val_idx], feature_cols,
        )

    stability = pd.DataFrame(
        {f"fold_{k}": v for k, v in shap_by_fold.items()},
        index=feature_cols,
    )
    fold_cols = [c for c in stability.columns if c.startswith("fold_")]
    if fold_cols:
        stability["mean_abs_shap"] = stability[fold_cols].mean(axis=1)
        stability["std_abs_shap"] = stability[fold_cols].std(axis=1)
        stability["cv"] = stability["std_abs_shap"] / (
            stability["mean_abs_shap"] + 1e-12
        )
        stability = stability.sort_values("mean_abs_shap", ascending=False)
    return stability


def select_max_depth_and_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    depth_grid: list[int] | None,
    fixed_depth: int | None = None,
    backend: str = "lgbm",
) -> tuple[int, pd.DataFrame, dict, pd.DataFrame, np.ndarray, list, pd.DataFrame]:
    """
    Depth search + full CV in one pass.

    Runs CV once per candidate depth; keeps fold models / OOF / SHAP only for
    the selected depth (no second CV at the winner).
    Selection criterion: mean fold pooled PR-AUC (group-mean also reported).
    """
    if fixed_depth is not None:
        depths = [int(fixed_depth)]
        print(f"\n[5a/8] Using fixed --max-depth={fixed_depth} (grid skipped)")
    else:
        depths = list(depth_grid or DEPTH_GRID_DEFAULT)
        print(
            f"\n[5a/8] Depth selection over {depths} "
            f"(criterion: pooled PR-AUC; also tracking group-mean)"
        )

    depth_rows: list[dict] = []
    best_depth = depths[0]
    best_pr = float("-inf")
    best_pack: tuple | None = None

    for depth in depths:
        label = "LightGBM" if backend == "lgbm" else "XGBoost"
        print(f"\n  [{label}] CV depth={depth}...")
        # Skip SHAP during the grid; compute once on the winner below.
        cv_summary, fold_df, oof_probs, fold_models, _stability = _cv_one_depth(
            train_df, feature_cols, n_folds, depth, backend=backend,
            collect_shap=False,
        )
        for _, row in fold_df.iterrows():
            depth_rows.append({
                "max_depth": depth,
                **row.to_dict(),
            })
        depth_rows.append({
            "max_depth": depth,
            "fold": "mean",
            "auc": cv_summary["auc"],
            "pr_auc": cv_summary["pr_auc"],
            "auc_group_mean": cv_summary["auc_group_mean"],
            "pr_auc_group_mean": cv_summary["pr_auc_group_mean"],
            "brier": cv_summary["brier"],
            "logloss": cv_summary["logloss"],
            "ece": cv_summary["ece"],
            "n_val": float("nan"),
            "n_pos": float("nan"),
            "scale_pos_weight": float("nan"),
            "n_groups_scored": float("nan"),
        })
        print(
            f"  depth={depth}: pooled PR-AUC={cv_summary['pr_auc']:.4f}  "
            f"group-mean PR-AUC={cv_summary['pr_auc_group_mean']:.4f}  "
            f"AUC={cv_summary['auc']:.4f}  "
            f"group-mean AUC={cv_summary['auc_group_mean']:.4f}"
        )

        score = cv_summary["pr_auc"]
        if pd.isna(score):
            score = float("-inf")
        if score > best_pr or best_pack is None:
            best_pr = score
            best_depth = depth
            best_pack = (cv_summary, fold_df, oof_probs, fold_models)

    assert best_pack is not None
    cv_summary, fold_df, oof_probs, fold_models = best_pack
    cv_summary = dict(cv_summary)
    cv_summary["max_depth"] = best_depth

    # SHAP stability only for the selected depth (reuse fold models; same GroupKFold)
    print(f"\n  Computing SHAP fold stability for max_depth={best_depth}...")
    stability = _shap_stability_from_fold_models(
        train_df, feature_cols, n_folds, fold_models,
    )

    print(
        f"\n  Selected max_depth={best_depth} "
        f"(best mean pooled PR-AUC={best_pr:.4f})"
    )
    print(
        f"  CV Mean → pooled PR-AUC={cv_summary['pr_auc']:.4f}  "
        f"group-mean PR-AUC={cv_summary['pr_auc_group_mean']:.4f}  "
        f"Brier={cv_summary['brier']:.4f}  ECE={cv_summary['ece']:.4f}"
    )
    print(
        f"  OOF     → pooled PR-AUC={cv_summary['oof_pr_auc']:.4f}  "
        f"group-mean PR-AUC={cv_summary['oof_pr_auc_group_mean']:.4f}  "
        f"ECE={cv_summary['oof_ece']:.4f}"
    )
    ps = cv_summary.get("oof_pool_size") or {}
    if ps:
        print(
            f"  OOF pool sizes → n_groups={ps.get('n_groups')}  "
            f"median={ps.get('median')}  mean={ps.get('mean'):.1f}  "
            f"max={ps.get('max')}  p95={ps.get('p95')}"
        )

    depth_df = pd.DataFrame(depth_rows)
    return (
        best_depth, depth_df, cv_summary, fold_df,
        oof_probs, fold_models, stability,
    )


def train_classifier_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    max_depth: int,
    backend: str = "lgbm",
) -> tuple[None, dict, pd.DataFrame, np.ndarray, list, pd.DataFrame]:
    """CV at a fixed depth (used for --compare-xgb side-by-side)."""
    label = "LightGBM" if backend == "lgbm" else "XGBoost"
    print(f"\n[5b/8] {label} cross-validation ({n_folds} folds, depth={max_depth})...")
    cv_summary, fold_df, oof_probs, fold_models, stability = _cv_one_depth(
        train_df, feature_cols, n_folds, max_depth, backend=backend,
        collect_shap=(backend == "lgbm"),
    )
    print(
        f"\n  CV Mean → pooled PR-AUC={cv_summary['pr_auc']:.4f}  "
        f"group-mean PR-AUC={cv_summary['pr_auc_group_mean']:.4f}  "
        f"ECE={cv_summary['ece']:.4f}"
    )
    return None, cv_summary, fold_df, oof_probs, fold_models, stability


def decide_calibration(
    oof_ece: float,
    ece_threshold: float,
    no_calibrate: bool,
    force_calibrate: bool,
) -> tuple[bool, str]:
    if no_calibrate and force_calibrate:
        sys.exit("[ERROR] Use either --no-calibrate or --force-calibrate, not both.")
    if no_calibrate:
        return False, "forced_off"
    if force_calibrate:
        return True, "forced_on"
    if pd.isna(oof_ece):
        return False, "skipped_nan_ece"
    if oof_ece > ece_threshold:
        return True, f"auto_ece_gt_{ece_threshold:g}"
    return False, f"auto_ece_le_{ece_threshold:g}"


def fit_final_classifier(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    max_depth: int,
    apply_calibration: bool,
    calib_fraction: float,
    random_state: int,
) -> tuple[object, dict]:
    """
    Final refit on train. If calibrating: hold out calib_fraction of clusters,
    train base on remainder, isotonic CalibratedClassifierCV(cv='prefit').
    """
    print("\n[5c/8] Final model refit on training set...")
    X_all = train_df[feature_cols].values
    y_all = train_df["relevance"].values.astype(int)
    clusters = train_df["cluster_id"].astype(str)
    unique_clusters = clusters.drop_duplicates().tolist()

    calib_meta: dict = {
        "calibration_applied": False,
        "method": None,
        "reason": None,
        "ece_before": float("nan"),
        "ece_after": float("nan"),
        "reliability_before": None,
        "reliability_after": None,
        "n_calib_clusters": 0,
        "n_fit_clusters": len(unique_clusters),
        "is_unbalance": True,
        "scale_pos_weight": None,
        "max_depth": max_depth,
    }

    if not apply_calibration:
        rng = np.random.RandomState(random_state)
        n_es = max(1, int(round(0.1 * len(unique_clusters))))
        es_clusters = set(
            rng.choice(
                unique_clusters,
                size=min(n_es, len(unique_clusters)),
                replace=False,
            )
        )
        es_mask = clusters.isin(es_clusters).values
        fit_mask = ~es_mask
        if fit_mask.sum() == 0 or es_mask.sum() == 0:
            fit_mask = np.ones(len(train_df), dtype=bool)
            es_mask = fit_mask

        es_model = fit_lgbm_classifier(
            X_all[fit_mask], y_all[fit_mask],
            X_all[es_mask], y_all[es_mask],
            max_depth,
        )
        best_iter = int(getattr(es_model, "best_iteration_", 0) or 0)
        if best_iter <= 0:
            best_iter = int(get_lgbm_classifier_params(max_depth)["n_estimators"])

        # Refit on full train with early-stopped tree count.
        # Class imbalance: is_unbalance=True (not scale_pos_weight / rarity boost).
        params = get_lgbm_classifier_params(max_depth)
        params["n_estimators"] = best_iter
        model = lgb.LGBMClassifier(**params)
        model.fit(X_all, y_all)
        calib_meta["reason"] = "no_calibration"
        calib_meta["best_iteration"] = best_iter
        print(
            f"  Final LGBMClassifier (uncalibrated), "
            f"is_unbalance=True, n_estimators={best_iter}"
        )
        return model, calib_meta

    # Calibration path: carve cluster holdout
    rng = np.random.RandomState(random_state)
    n_calib = max(1, int(round(calib_fraction * len(unique_clusters))))
    n_calib = min(n_calib, max(1, len(unique_clusters) - 1))
    calib_clusters = set(
        rng.choice(unique_clusters, size=n_calib, replace=False)
    )
    calib_mask = clusters.isin(calib_clusters).values
    fit_mask = ~calib_mask

    X_fit, y_fit = X_all[fit_mask], y_all[fit_mask]
    X_cal, y_cal = X_all[calib_mask], y_all[calib_mask]

    # Early-stopping slice inside fit set
    fit_clusters = clusters[fit_mask].drop_duplicates().tolist()
    n_es = max(1, int(round(0.1 * len(fit_clusters))))
    es_clusters = set(
        rng.choice(fit_clusters, size=min(n_es, len(fit_clusters)), replace=False)
    )
    # indices relative to full train
    es_mask_full = clusters.isin(es_clusters).values & fit_mask
    train_only_mask = fit_mask & ~clusters.isin(es_clusters).values
    if train_only_mask.sum() == 0:
        train_only_mask = fit_mask
        es_mask_full = fit_mask

    base = fit_lgbm_classifier(
        X_all[train_only_mask], y_all[train_only_mask],
        X_all[es_mask_full], y_all[es_mask_full],
        max_depth,
    )
    probs_before = predict_proba_positive(base, X_cal)
    ece_before, rel_before = expected_calibration_error(y_cal, probs_before)
    calib_meta["ece_before"] = ece_before
    calib_meta["reliability_before"] = rel_before
    calib_meta["n_calib_clusters"] = len(calib_clusters)
    calib_meta["n_fit_clusters"] = int(fit_mask.sum() and len(fit_clusters))

    calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    probs_after = predict_proba_positive(calibrated, X_cal)
    ece_after, rel_after = expected_calibration_error(y_cal, probs_after)

    calib_meta.update({
        "calibration_applied": True,
        "method": "isotonic",
        "reason": "applied",
        "ece_after": ece_after,
        "reliability_after": rel_after,
    })
    print(
        f"  Calibrated (isotonic) on {n_calib} clusters: "
        f"ECE {ece_before:.4f} → {ece_after:.4f}"
    )
    return calibrated, calib_meta


# ─────────────────────────────────────────────
# SECONDARY RANK DIAGNOSTICS + PER-RESIDUE
# ─────────────────────────────────────────────

def secondary_rank_diagnostics(
    df: pd.DataFrame,
    score_col: str,
    k: int,
    top_pct: float,
    reward_mode: str,
    epsilon: float,
) -> tuple[pd.DataFrame, dict, dict]:
    """Within-protein ranking sanity check (not primary objective)."""
    eval_df = evaluate_predictions(
        df, score_col, k=k, top_pct_threshold=top_pct,
        reward_mode=reward_mode, epsilon=epsilon,
    )
    summary = summarize_eval_metrics(
        eval_df, k, top_pct, reward_mode, epsilon,
    )
    composition = analyze_residue_composition(
        df, score_col, k=k, top_pct_threshold=top_pct,
    )
    return eval_df, summary, composition


def per_residue_type_clf_metrics(
    df: pd.DataFrame,
    score_col: str = "pred_prob",
) -> pd.DataFrame:
    """Stratified AUC/PR-AUC/Brier by target residue type (label site type)."""
    rows = []
    if "target_residue_type" not in df.columns:
        # fall back to Residue on positive rows only is wrong for AUC;
        # use target_residue_type from merge when available
        return pd.DataFrame()

    for res, grp in df.groupby("target_residue_type", sort=True):
        y = grp["relevance"].values.astype(int)
        p = grp[score_col].values.astype(float)
        m = classification_metrics(y, p)
        rows.append({
            "target_residue_type": res,
            "n": m["n"],
            "n_pos": m["n_pos"],
            "auc": m["auc"],
            "pr_auc": m["pr_auc"],
            "brier": m["brier"],
            "ece": m["ece"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_classifier_outputs(
    model,
    feature_cols: list[str],
    output_dir: str,
    *,
    max_depth: int,
    calib_meta: dict,
    cv_summary: dict,
    cv_folds_df: pd.DataFrame,
    depth_grid_df: pd.DataFrame,
    shap_stability_df: pd.DataFrame,
    test_metrics: dict,
    test_reliability: dict,
    test_df: pd.DataFrame,
    test_rank_eval: pd.DataFrame,
    test_rank_summary: dict,
    composition: dict | None,
    per_res_clf: pd.DataFrame,
    xgb_comparison: dict | None,
    include_residue_type: bool,
    large_pool_prefilter: dict | None,
    export_shap: bool,
    shap_max_rows: int | None,
    k: int,
    top_pct: float,
    reward_mode: str,
    epsilon: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "lgbm_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_cols,
            "model_type": "classifier",
            "include_residue_type": include_residue_type,
            "max_depth": max_depth,
            "is_unbalance": True,
            "scale_pos_weight": None,
            "calibration": {
                "applied": bool(calib_meta.get("calibration_applied")),
                "method": calib_meta.get("method"),
                "reason": calib_meta.get("reason"),
                "ece_before": calib_meta.get("ece_before"),
                "ece_after": calib_meta.get("ece_after"),
            },
            "k": k,
            "reward_mode": reward_mode,
            "top_pct_threshold": top_pct,
            "rank_bonus_epsilon": epsilon,
            "large_pool_prefilter": large_pool_prefilter,
        }, f)
    print(f"  Model saved:   {model_path}")

    # Feature importance (base estimator if calibrated)
    base_model = unwrap_tree_estimator(model)
    if hasattr(base_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": base_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_path = os.path.join(output_dir, "feature_importance.csv")
        fi.to_csv(fi_path, index=False)
        print(f"  Feature importance: {fi_path}")
    else:
        print("[WARN] Unwrapped estimator has no feature_importances_; "
              "skipping importance CSV.")

    if not depth_grid_df.empty:
        p = os.path.join(output_dir, "depth_grid_results.csv")
        depth_grid_df.to_csv(p, index=False)
        print(f"  Depth grid: {p}")

    if not cv_folds_df.empty:
        p = os.path.join(output_dir, "cv_fold_results.csv")
        mean_row = {"fold": "mean"}
        for col in cv_folds_df.columns:
            if col == "fold":
                continue
            if pd.api.types.is_numeric_dtype(cv_folds_df[col]):
                mean_row[col] = float(cv_folds_df[col].mean())
        out = pd.concat([cv_folds_df, pd.DataFrame([mean_row])], ignore_index=True)
        out.to_csv(p, index=False)
        print(f"  CV fold results: {p}")

    if shap_stability_df is not None and not shap_stability_df.empty:
        p = os.path.join(output_dir, "shap_fold_stability.csv")
        shap_stability_df.to_csv(p)
        print(f"  SHAP fold stability: {p}")

    # Reliability curves
    rel_rows = []
    for split_name, rel in [
        ("cv_oof", cv_summary.get("oof_reliability")),
        ("test", test_reliability),
        ("calib_before", calib_meta.get("reliability_before")),
        ("calib_after", calib_meta.get("reliability_after")),
    ]:
        if not rel or not rel.get("prob_true"):
            continue
        for pt, pp in zip(rel["prob_true"], rel["prob_pred"]):
            rel_rows.append({
                "split": split_name,
                "strategy": rel.get("strategy"),
                "prob_true": pt,
                "prob_pred": pp,
            })
    if rel_rows:
        p = os.path.join(output_dir, "reliability_curves.csv")
        pd.DataFrame(rel_rows).to_csv(p, index=False)
        print(f"  Reliability curves: {p}")

    # Test row scores
    score_path = os.path.join(output_dir, "test_scores.csv")
    cols = [c for c in [
        "Name", "Residue", "ResNum", "Chain", "Warhead", "query_group",
        "relevance", "target_residue_type", "pred_prob",
    ] if c in test_df.columns]
    test_df[cols].to_csv(score_path, index=False)
    print(f"  Test scores: {score_path}")

    # Pool-size distribution (test + OOF summary already in JSON)
    if "query_group" in test_df.columns:
        pool_sizes = (
            test_df.groupby("query_group", sort=False)
            .size()
            .rename("n_candidates")
            .reset_index()
            .sort_values("n_candidates", ascending=False)
        )
        p = os.path.join(output_dir, "test_pool_sizes.csv")
        pool_sizes.to_csv(p, index=False)
        print(f"  Test pool sizes: {p}")

    if not test_rank_eval.empty:
        p = os.path.join(output_dir, "test_rank_sanity.csv")
        test_rank_eval.to_csv(p, index=False)
        print(f"  Rank sanity (secondary): {p}")

    if not per_res_clf.empty:
        p = os.path.join(output_dir, "test_per_residue_clf_metrics.csv")
        per_res_clf.to_csv(p, index=False)
        print(f"  Per-residue clf metrics: {p}")

    if export_shap:
        scored = test_df.copy()
        if "pred_score" not in scored.columns and "pred_prob" in scored.columns:
            scored["pred_score"] = scored["pred_prob"]
        shap_model = unwrap_tree_estimator(model)
        export_shap_csvs(
            model=shap_model,
            df=scored,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix="test",
            max_rows=shap_max_rows,
        )

    summary = {
        "model_type": "classifier",
        "model": model_path,
        "max_depth": max_depth,
        "is_unbalance": True,
        "calibration_applied": bool(calib_meta.get("calibration_applied")),
        "calibration": calib_meta,
        "cv": {
            k: v for k, v in cv_summary.items()
            if k not in ("oof_reliability",)
        },
        "test": {
            "auc_pooled": test_metrics.get("auc"),
            "pr_auc_pooled": test_metrics.get("pr_auc"),
            "auc_group_mean": test_metrics.get("auc_group_mean"),
            "pr_auc_group_mean": test_metrics.get("pr_auc_group_mean"),
            "n_groups_scored": test_metrics.get("n_groups_scored"),
            "n_groups_skipped_single_class": test_metrics.get(
                "n_groups_skipped_single_class"
            ),
            "brier": test_metrics.get("brier"),
            "logloss": test_metrics.get("logloss"),
            "ece": test_metrics.get("ece"),
            "n": test_metrics.get("n"),
            "n_pos": test_metrics.get("n_pos"),
            "pool_size": test_metrics.get("pool_size"),
        },
        "test_rank_sanity": test_rank_summary,
        "note": (
            "Primary metrics: pooled and group-mean AUC/PR-AUC, Brier, ECE. "
            "Pooled metrics weight large pockets more; group-mean gives equal "
            "weight per query_group. test_rank_sanity Hit@K-style numbers are "
            "secondary diagnostics only."
        ),
    }
    if composition and composition.get("n_query_groups"):
        summary["residue_composition"] = {
            k: v for k, v in composition.items() if k != "detail"
        }
    if xgb_comparison is not None:
        summary["xgb_comparison"] = xgb_comparison
    if not per_res_clf.empty:
        summary["per_residue_clf"] = per_res_clf.to_dict(orient="records")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  Summary JSON: {summary_path}")


# ─────────────────────────────────────────────
# CLI + MAIN
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LGBM binary classifier for nucleophilic residue hits "
                    "(sibling to Training_Cov_Screen ranker)"
    )
    p.add_argument("--training", required=True, help="Path to training CSV")
    p.add_argument("--labels", required=True, help="Path to labels CSV")
    p.add_argument("--features", nargs="+", default=[
        "Abs_Side_SASA", "Rel_Side_SASA",
        "deprotonation_prob", "pKa_shift",
        "Accessibility_Score",
        "Reactivity_Score",
        "Reactivity_Score_Protonated",
        "Reactivity_Score_Deprotonated",
    ])
    p.add_argument("--residue_specific_features", nargs="+", default=[
        "Reactivity_Score_Protonated",
        "Reactivity_Score_Deprotonated",
        "Fukui_Protonated", "Fukui_Deprotonated",
        "Nucleophilicity_Index_Protonated",
        "Nucleophilicity_Index_Deprotonated",
    ])
    p.add_argument("--pdb_folder", default=None)
    p.add_argument("--mmseqs_dir", default="mmseqs_output")
    p.add_argument("--seq_identity", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--deep-cluster", action="store_true")
    p.add_argument("--deep-cluster-min-chain-length", type=int, default=30)
    p.add_argument("--clusters-tsv", default=None, metavar="PATH")
    p.add_argument("--split-csv", default=None, metavar="PATH")
    p.add_argument("--export-split", default=None, metavar="PATH")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--normalize_within_protein", action="store_true")
    p.add_argument("--res-num", type=int, nargs="+", default=None, metavar="N")
    p.add_argument("--no-residue-type", action="store_true")
    p.add_argument("--no-residue-specific-features", action="store_true")
    p.add_argument("--large-pool-prefilter", action="store_true")
    p.add_argument("--large-pool-min-size", type=int, default=500)
    p.add_argument("--large-pool-filter", nargs="+", default=None, metavar="SPEC")
    p.add_argument("--output_dir", default="lgbm_classifier_output")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--shap-max-rows", type=int, default=None)
    p.add_argument("--random_state", type=int, default=42)

    # Classifier-specific
    p.add_argument(
        "--depth-grid", type=int, nargs="+", default=list(DEPTH_GRID_DEFAULT),
        help="Candidate max_depth values (default: 3 4 5 6); select by PR-AUC",
    )
    p.add_argument(
        "--max-depth", type=int, default=None,
        help="Skip depth grid and use this max_depth",
    )
    p.add_argument(
        "--calibrate-ece-threshold", type=float, default=DEFAULT_ECE_THRESHOLD,
        help=f"Auto-calibrate if OOF ECE exceeds this (default {DEFAULT_ECE_THRESHOLD})",
    )
    p.add_argument(
        "--no-calibrate", action="store_true",
        help="Never apply isotonic calibration",
    )
    p.add_argument(
        "--force-calibrate", action="store_true",
        help="Always apply isotonic calibration",
    )
    p.add_argument(
        "--calib-fraction", type=float, default=DEFAULT_CALIB_FRACTION,
        help="Fraction of train clusters held out for calibration (default 0.15)",
    )
    p.add_argument(
        "--compare-xgb", action="store_true",
        help="Also run XGBoost classifier CV for side-by-side metrics",
    )

    # Secondary rank diagnostics
    p.add_argument("--topk", type=int, default=10,
                   help="K for secondary within-protein Hit@K sanity check")
    p.add_argument("--reward-mode", choices=REWARD_MODES, default="hit_at_k")
    p.add_argument("--top-pct", type=float, default=10.0)
    p.add_argument("--rank-bonus-epsilon", type=float,
                   default=DEFAULT_RANK_BONUS_EPSILON)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_pct <= 0 or args.top_pct > 100:
        sys.exit("[ERROR] --top-pct must be in (0, 100].")
    if args.topk < 1:
        sys.exit("[ERROR] --topk must be >= 1.")
    if args.split_csv and args.export_split:
        sys.exit("[ERROR] Use either --split-csv or --export-split, not both.")
    if args.no_calibrate and args.force_calibrate:
        sys.exit("[ERROR] Use either --no-calibrate or --force-calibrate, not both.")
    if args.calib_fraction <= 0 or args.calib_fraction >= 1:
        sys.exit("[ERROR] --calib-fraction must be in (0, 1).")

    large_pool_prefilter_meta = None
    pool_prefilter_filters = None
    if args.large_pool_prefilter:
        if not args.large_pool_filter:
            sys.exit(
                "[ERROR] --large-pool-filter is required when "
                "--large-pool-prefilter is set."
            )
        pool_prefilter_filters = parse_simple_filters(args.large_pool_filter)

    print("=" * 60)
    print("  Training_Cov_Classifier — LGBM binary hit classifier")
    print("=" * 60)

    # 1. Load & merge
    merged = load_and_merge(args.training, args.labels)

    # 2. Cluster
    cluster_map = build_cluster_map(
        merged, args.pdb_folder, args.mmseqs_dir,
        args.seq_identity, args.coverage,
        deep_cluster=args.deep_cluster,
        min_chain_length=args.deep_cluster_min_chain_length,
        clusters_tsv=args.clusters_tsv,
    )

    # 3. Split
    merged = assign_splits(
        merged, cluster_map,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_state=args.random_state,
        split_csv=args.split_csv,
        export_split=args.export_split,
    )

    # 4. Features
    residue_specific_cols = (
        [] if args.no_residue_specific_features else args.residue_specific_features
    )
    all_req_features = list(dict.fromkeys(args.features + residue_specific_cols))
    merged, feature_cols = build_features(
        merged,
        feature_cols=all_req_features,
        residue_specific_cols=residue_specific_cols,
        normalize_within_protein=args.normalize_within_protein,
        resnum_equals=args.res_num,
        include_residue_type=not args.no_residue_type,
    )

    train_df = merged[merged["split"] == "train"].copy()
    test_df = merged[merged["split"] == "test"].copy()
    train_df = train_df.sort_values("query_group").reset_index(drop=True)
    test_df = test_df.sort_values("query_group").reset_index(drop=True)

    train_pool_totals_before_filter = aggregate_nucleophile_totals(train_df)
    _ = train_pool_totals_before_filter  # retained for parity / future summary

    if args.large_pool_prefilter:
        train_df, large_pool_prefilter_meta = apply_large_pool_prefilter(
            train_df,
            pool_prefilter_filters,
            args.large_pool_min_size,
        )
        train_df = train_df.sort_values("query_group").reset_index(drop=True)

    # 5. Depth selection + CV (single pass; no retrain of best depth)
    (
        best_depth, depth_grid_df, cv_summary, cv_folds_df,
        oof_probs, fold_models, shap_stability,
    ) = select_max_depth_and_cv(
        train_df,
        feature_cols,
        args.n_folds,
        depth_grid=None if args.max_depth is not None else list(args.depth_grid),
        fixed_depth=args.max_depth,
        backend="lgbm",
    )
    _ = fold_models  # retained for potential future export

    xgb_comparison = None
    if args.compare_xgb:
        _, xgb_cv, xgb_folds, _, _, _ = train_classifier_cv(
            train_df, feature_cols, args.n_folds, best_depth, backend="xgb",
        )
        xgb_comparison = {
            "max_depth": best_depth,
            "lgbm": {
                "pr_auc_pooled": cv_summary["pr_auc"],
                "pr_auc_group_mean": cv_summary["pr_auc_group_mean"],
                "auc_pooled": cv_summary["auc"],
                "auc_group_mean": cv_summary["auc_group_mean"],
                "brier": cv_summary["brier"],
                "ece": cv_summary["ece"],
                "oof_pr_auc_pooled": cv_summary["oof_pr_auc"],
                "oof_pr_auc_group_mean": cv_summary["oof_pr_auc_group_mean"],
                "oof_ece": cv_summary["oof_ece"],
            },
            "xgb": {
                "pr_auc_pooled": xgb_cv["pr_auc"],
                "pr_auc_group_mean": xgb_cv["pr_auc_group_mean"],
                "auc_pooled": xgb_cv["auc"],
                "auc_group_mean": xgb_cv["auc_group_mean"],
                "brier": xgb_cv["brier"],
                "ece": xgb_cv["ece"],
                "oof_pr_auc_pooled": xgb_cv["oof_pr_auc"],
                "oof_pr_auc_group_mean": xgb_cv["oof_pr_auc_group_mean"],
                "oof_ece": xgb_cv["oof_ece"],
            },
            "xgb_fold_results": xgb_folds.to_dict(orient="records"),
        }
        print("\n  XGB comparison (CV mean):")
        print(
            f"    LGBM  pooled PR-AUC={cv_summary['pr_auc']:.4f}  "
            f"group-mean={cv_summary['pr_auc_group_mean']:.4f}  "
            f"ECE={cv_summary['ece']:.4f}"
        )
        print(
            f"    XGB   pooled PR-AUC={xgb_cv['pr_auc']:.4f}  "
            f"group-mean={xgb_cv['pr_auc_group_mean']:.4f}  "
            f"ECE={xgb_cv['ece']:.4f}"
        )

    # Calibration decision from OOF ECE
    apply_cal, cal_reason = decide_calibration(
        cv_summary.get("oof_ece", float("nan")),
        args.calibrate_ece_threshold,
        args.no_calibrate,
        args.force_calibrate,
    )
    print(
        f"\n[5c/8] Calibration decision: apply={apply_cal} "
        f"(reason={cal_reason}, OOF ECE={cv_summary.get('oof_ece')})"
    )

    final_model, calib_meta = fit_final_classifier(
        train_df, feature_cols, best_depth,
        apply_calibration=apply_cal,
        calib_fraction=args.calib_fraction,
        random_state=args.random_state,
    )
    calib_meta["decision_reason"] = cal_reason
    if calib_meta.get("reason") == "no_calibration":
        calib_meta["reason"] = cal_reason

    # 6. Test evaluation
    print("\n[6/8] Evaluating on held-out test set...")
    X_test = test_df[feature_cols].values
    y_test = test_df["relevance"].values.astype(int)
    test_probs = predict_proba_positive(final_model, X_test)
    test_df = test_df.copy()
    test_df["pred_prob"] = test_probs
    test_df["pred_score"] = test_probs  # alias for shared rank helpers

    test_metrics = classification_metrics_with_groups(
        y_test, test_probs, groups=test_df["query_group"].values,
    )
    print(
        f"  Test pooled   PR-AUC={test_metrics['pr_auc']:.4f}  "
        f"AUC={test_metrics['auc']:.4f}"
    )
    print(
        f"  Test group-mean PR-AUC={test_metrics.get('pr_auc_group_mean', float('nan')):.4f}  "
        f"AUC={test_metrics.get('auc_group_mean', float('nan')):.4f}  "
        f"(scored {test_metrics.get('n_groups_scored', 0)}/"
        f"{test_metrics.get('n_groups', 0)} groups; "
        f"skipped single-class={test_metrics.get('n_groups_skipped_single_class', 0)})"
    )
    print(
        f"  Test Brier={test_metrics['brier']:.4f}  "
        f"ECE={test_metrics['ece']:.4f}"
    )
    ps = test_metrics.get("pool_size") or {}
    if ps:
        print(
            f"  Test pool sizes → n_groups={ps.get('n_groups')}  "
            f"median={ps.get('median')}  mean={ps.get('mean'):.1f}  "
            f"max={ps.get('max')}  p95={ps.get('p95')}"
        )

    per_res_clf = per_residue_type_clf_metrics(test_df, "pred_prob")
    if not per_res_clf.empty:
        print("\n  Per target-residue-type (primary clf metrics):")
        print(per_res_clf.to_string(index=False, float_format="%.4f"))

    print(
        "\n[6b/8] Secondary within-protein rank sanity "
        "(NOT primary objective)..."
    )
    test_rank_eval, test_rank_summary, composition = secondary_rank_diagnostics(
        test_df, "pred_prob",
        k=args.topk, top_pct=args.top_pct,
        reward_mode=args.reward_mode, epsilon=args.rank_bonus_epsilon,
    )
    print_results(
        test_rank_eval,
        k=args.topk,
        top_pct_threshold=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
    )
    print_residue_composition(
        composition, k=args.topk, top_pct_threshold=args.top_pct,
    )

    # 7. Save
    print(f"\n[7/8] Saving outputs to '{args.output_dir}'...")
    save_classifier_outputs(
        final_model, feature_cols, args.output_dir,
        max_depth=best_depth,
        calib_meta=calib_meta,
        cv_summary=cv_summary,
        cv_folds_df=cv_folds_df,
        depth_grid_df=depth_grid_df,
        shap_stability_df=shap_stability,
        test_metrics=test_metrics,
        test_reliability=test_metrics.get("reliability", {}),
        test_df=test_df,
        test_rank_eval=test_rank_eval,
        test_rank_summary=test_rank_summary,
        composition=composition,
        per_res_clf=per_res_clf,
        xgb_comparison=xgb_comparison,
        include_residue_type=not args.no_residue_type,
        large_pool_prefilter=large_pool_prefilter_meta,
        export_shap=not args.no_shap,
        shap_max_rows=args.shap_max_rows,
        k=args.topk,
        top_pct=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
    )

    # Persist OOF probs for audit
    oof_path = os.path.join(args.output_dir, "cv_oof_predictions.csv")
    oof_df = train_df[[
        c for c in [
            "Name", "Residue", "ResNum", "Chain", "Warhead",
            "query_group", "cluster_id", "relevance", "target_residue_type",
        ] if c in train_df.columns
    ]].copy()
    oof_df["oof_pred_prob"] = oof_probs
    oof_df.to_csv(oof_path, index=False)
    print(f"  OOF predictions: {oof_path}")

    print("\n[8/8] Done.")


if __name__ == "__main__":
    main()
