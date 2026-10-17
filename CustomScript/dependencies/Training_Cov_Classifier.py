"""
LGBM Classifier for Nucleophilic Residue Hit Probability
========================================================
Pointwise binary classifier (relevance 1/0) with cluster-grouped CV.
Sibling to Training_Cov_Screen.py (ranker): reuses load/cluster/split/features.

Primary metrics: pooled + group-mean AUC/PR-AUC, Brier, ECE (+ reliability curve).
Ranking metrics in ranking_metrics.csv: EF@1/5/10%, LogAUC, adjusted
LogAUC, BEDROC, median search-space reduction, and Hit@K / Hit@top-%.

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

CovSite ligand-screening mode:
    python Training_Cov_Classifier.py \
        --vs-candidates VS/covsite_candidates_combined.csv \
        --vs-hits VS/all_hits.csv \
        --features Geo_Fit Hydrophobic_Fit Hydrogen_DPAL_Fit Electro_Fit \
        --pdb_folder Existing_Structures --deep-cluster \
        --seq_identity 0.5 --keep-warhead "Acrylamide warhead" \
        --collapse-active-protomers \
        --output_dir VS/covsite_vs_classifier

This alternate mode classifies known active ligands (1) versus decoys (0) at
fixed CovSite sites. Held-out probabilities are ranked per site and pooled to
report EF@1/5/10%, LogAUC, adjusted LogAUC, BEDROC, median search-space
reduction, Hit@K/Hit@top-%, AUROC, AUPRC, ACC, F1, and MCC. The classifier
cutoff is selected only from training OOF predictions by maximum F1 and is
locked before test evaluation; 0.5 is reported as a baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
    VALID_RESIDUES,
)

warnings.filterwarnings("ignore")

DEPTH_GRID_DEFAULT = (3, 4, 5, 6)
DEFAULT_ECE_THRESHOLD = 0.05
DEFAULT_CALIB_FRACTION = 0.15
HERE = Path(__file__).resolve().parent
CUSTOM_SCRIPT_ROOT = HERE.parent.parent


def _locate_helper(relative_paths: tuple[str, ...]) -> Path:
    """First existing helper path, so this file runs from any directory.

    It may live in osc_runs/scoring-analysis or in dependencies, which sit at
    different depths. Returns the first relative path for the error message
    when nothing matches; --topsis-vs can still override it.
    """
    roots = (HERE, HERE.parent, HERE.parent.parent)
    for root in roots:
        for relative in relative_paths:
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return HERE / relative_paths[0]


DEFAULT_COVSITE_VS = _locate_helper((
    "Training_CovSite_VS.py",
    "osc_runs/scoring-analysis/Training_CovSite_VS.py",
))
DEFAULT_TOPSIS_VS = _locate_helper((
    "VS_eval/Topsis_VS.py",
    "osc_runs/scoring-analysis/VS_eval/Topsis_VS.py",
    "Topsis_VS.py",
))
# Same Mysinger/Shoichet definitions as VS_eval/Topsis_VS.py.
EF_FRACTIONS = ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct"))
LOGAUC_LAMBDA = 0.001
RANDOM_LOGAUC = 0.14462


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
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    predicted = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()
    ece, reliability = expected_calibration_error(y_true, probs, n_bins=n_bins)
    return {
        "auc": _safe_auc(y_true, probs),
        "pr_auc": _safe_pr_auc(y_true, probs),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted)
        ),
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_threshold": float(threshold),
        "brier": float(brier_score_loss(y_true, probs)),
        "logloss": _safe_logloss(y_true, probs),
        "ece": ece,
        "reliability": reliability,
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
    }


def select_f1_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    """Select a cutoff on training OOF scores by maximum F1.

    Ties are resolved by higher recall, then higher precision, then the
    larger threshold. The held-out test labels are never consulted.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if thresholds.size == 0:
        return {
            "threshold": 0.5,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    candidates = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }).sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    best = candidates.iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
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
    threshold: float = 0.5,
) -> dict:
    """Pooled metrics + optional group-averaged AUC/PR-AUC and pool-size stats."""
    out = classification_metrics(
        y_true, probs, n_bins=n_bins, threshold=threshold,
    )
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
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "mcc": m["mcc"],
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
        "accuracy": float(fold_df["accuracy"].mean()),
        "f1": float(fold_df["f1"].mean()),
        "mcc": float(fold_df["mcc"].mean()),
        "auc_group_mean": float(fold_df["auc_group_mean"].mean()),
        "pr_auc_group_mean": float(fold_df["pr_auc_group_mean"].mean()),
        "brier": float(fold_df["brier"].mean()),
        "logloss": float(fold_df["logloss"].mean()),
        "ece": float(fold_df["ece"].mean()),
        "oof_auc": oof_metrics["auc"],
        "oof_pr_auc": oof_metrics["pr_auc"],
        "oof_accuracy": oof_metrics["accuracy"],
        "oof_f1": oof_metrics["f1"],
        "oof_mcc": oof_metrics["mcc"],
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
            "accuracy": cv_summary["accuracy"],
            "f1": cv_summary["f1"],
            "mcc": cv_summary["mcc"],
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
# COVSITE LIGAND-SCREENING MODE
# ─────────────────────────────────────────────

def load_python_module(name: str, path: Path) -> ModuleType:
    """Load a project helper without depending on the launch directory."""
    path = path.resolve()
    if not path.is_file():
        sys.exit(f"[ERROR] Python helper not found: {path}")
    root = str(CUSTOM_SCRIPT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        sys.exit(f"[ERROR] Could not load Python helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_covsite_ligand_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], list[dict[str, str]], ModuleType, ModuleType]:
    """Label CovSite VS candidates as active ligands (1) or decoys (0)."""
    vs = load_python_module("training_covsite_vs_helpers", DEFAULT_COVSITE_VS)
    topsis_path = (
        Path(args.topsis_vs).resolve()
        if args.topsis_vs
        else DEFAULT_TOPSIS_VS
    )
    tvs = vs.load_topsis_vs(topsis_path)

    candidates_path = Path(args.vs_candidates).resolve()
    hits_path = Path(args.vs_hits).resolve()
    if not candidates_path.is_file():
        sys.exit(f"[ERROR] --vs-candidates not found: {candidates_path}")
    if not hits_path.is_file():
        sys.exit(f"[ERROR] --vs-hits not found: {hits_path}")

    # Training_CovSite_VS accepts a candidate table as its own library table:
    # PDB_ID/electrophile_smiles/Residue/ResNum/Chain are all present.
    helper_args = SimpleNamespace(
        pdb_column=args.pdb_column,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        residue_column=args.residue_column,
        resnum_column=args.resnum_column,
        chain_column=args.chain_column,
        hit_residue_column=args.hit_residue_column,
        hit_resnum_column=args.hit_resnum_column,
        hit_chain_column=args.hit_chain_column,
        pdb_hits=args.pdb_hits,
        smiles_hit_column=args.smiles_hit_column,
        name_hit_column=args.name_hit_column,
        name_miss_column=args.name_miss_column,
        smiles_miss_column=args.smiles_miss_column,
        warhead_type_column=args.warhead_type_column,
    )
    library = vs.load_library(candidates_path, helper_args)
    hits_by_key, hit_key_level = vs.load_hits(hits_path, tvs, helper_args)
    candidates = pd.read_csv(candidates_path, low_memory=False)
    if candidates.empty:
        sys.exit("[ERROR] --vs-candidates has no rows")
    if args.keep_warhead:
        candidates = vs.filter_candidates_to_warheads(
            candidates, tvs, args.keep_warhead
        )

    features = list(dict.fromkeys(args.features))
    missing = [feature for feature in features if feature not in candidates.columns]
    if missing:
        sys.exit(
            "[ERROR] Features missing from --vs-candidates:\n  "
            + ", ".join(missing)
        )

    selected, unmatched = vs.label_candidates(
        candidates,
        library,
        hits_by_key,
        tvs,
        hit_key_level=hit_key_level,
    )
    if args.collapse_active_protomers:
        selected["_parent_key"] = selected["electrophile_smiles"].map(
            lambda smiles: vs.parent_compound_key(smiles, tvs)
        )

    data = tvs.build_site_candidates(selected, features)
    if "_parent_key" in selected.columns:
        data = vs.attach_parent_keys(data, selected)
    data["PDB_ID"] = data["Site"].astype(str).str.split("|").str[0]
    data["pdb_id"] = data["PDB_ID"].astype(str).str.upper()
    data["query_group"] = data["Site"]
    data["relevance"] = data["is_hit"].astype(int)
    data["target_residue_type"] = data["Residue"].astype(str).str.upper()

    valid_sites = vs.valid_sites(data)
    binary_sites: list[str] = []
    for site, group in data.loc[data["Site"].isin(valid_sites)].groupby(
        "Site", sort=True
    ):
        n_hits = int(group["is_hit"].sum())
        if n_hits >= len(group):
            print(
                f"[WARN] Skipping site {site}: no decoys "
                "(binary classification and EF are undefined)"
            )
            continue
        binary_sites.append(str(site))
    data = data.loc[data["Site"].isin(binary_sites)].copy()
    if data.empty:
        sys.exit("[ERROR] No valid VS sites remain after labeling")

    # Unlike LambdaRank, a pooled classifier may use features that are
    # constant within a site but differ between proteins/sites.
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    if not args.no_residue_type:
        residues = ("CYS", "SER", "THR", "TYR", "LYS", "HIS")
        for residue in residues:
            column = f"res_{residue}"
            data[column] = (data["Residue"] == residue).astype(int)
            if column not in features:
                features.append(column)
    if args.res_num:
        normalized = data["ResNum"].astype(str).str.strip()
        for number in args.res_num:
            column = f"resnum_eq_{number}"
            data[column] = (normalized == str(number)).astype(int)
            if column not in features:
                features.append(column)
    if args.normalize_within_protein:
        base_features = list(features)
        for feature in base_features:
            if feature.startswith("res_") or feature.startswith("resnum_eq_"):
                continue
            column = f"{feature}_norm"
            data[column] = data.groupby("Site")[feature].transform(
                lambda values: (
                    values - values.min()
                ) / (values.max() - values.min() + 1e-8)
            )
            if column not in features:
                features.append(column)

    print(
        f"[INFO] CovSite ligand data: {len(data):,} rows, "
        f"{data['Site'].nunique():,} sites, "
        f"{int(data['relevance'].sum()):,} hit rows"
    )
    return data.reset_index(drop=True), features, unmatched, vs, tvs


def covsite_ligand_rankings(
    scored: pd.DataFrame,
    args: argparse.Namespace,
    vs: ModuleType,
    tvs: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank held-out ligand probabilities and calculate EF/LogAUC."""
    per_site_parts: list[pd.DataFrame] = []
    enrichment_parts: list[pd.DataFrame] = []
    for site, group in scored.groupby("Site", sort=True):
        ranked = vs.rank_frame(group, "pred_prob", "site_rank")
        if args.best_warhead:
            ranked = tvs.keep_best_warhead_per_electrophile(
                ranked,
                score_column="pred_prob",
                rank_column="site_rank",
                higher_is_better=True,
            )
        if args.collapse_active_protomers:
            ranked = vs.collapse_active_protomers(
                ranked,
                tvs,
                score_column="pred_prob",
                rank_column="site_rank",
                higher_is_better=True,
            )
        per_site_parts.append(ranked)
        enrichment_parts.append(
            tvs.enrichment_any_candidate_per_ligand(
                ranked,
                "classifier_test_site",
                rank_column="site_rank",
                metadata={
                    "analysis_scope": "per_site",
                    "Site": site,
                    "PDB_ID": group["PDB_ID"].iloc[0],
                    "Residue": group["Residue"].iloc[0],
                    "ResNum": group["ResNum"].iloc[0],
                    "Chain": group["Chain"].iloc[0],
                },
                multiple_hits=args.multiple_hits,
            )
        )
    per_site = pd.concat(per_site_parts, ignore_index=True)

    pooled = vs.rank_frame(scored, "pred_prob", "overall_rank")
    if args.best_warhead:
        pooled = tvs.keep_best_warhead_per_electrophile(
            pooled,
            score_column="pred_prob",
            rank_column="overall_rank",
            higher_is_better=True,
        )
    if args.collapse_active_protomers:
        pooled = vs.collapse_active_protomers(
            pooled,
            tvs,
            score_column="pred_prob",
            rank_column="overall_rank",
            higher_is_better=True,
        )
    pooled_enrichment = tvs.enrichment_any_candidate_per_ligand(
        pooled,
        "classifier_test_pooled",
        rank_column="overall_rank",
        metadata={
            "analysis_scope": "pooled",
            "Site": "ALL",
            "PDB_ID": "ALL",
            "Residue": "ALL",
            "ResNum": "ALL",
            "Chain": "ALL",
        },
        multiple_hits=args.multiple_hits,
    )
    enrichment = pd.concat(
        [*enrichment_parts, pooled_enrichment], ignore_index=True
    )
    return per_site, pooled, enrichment


def covsite_classifier_metric_rows(
    scored: pd.DataFrame,
    threshold: float,
    *,
    data_split: str,
    threshold_source: str,
) -> pd.DataFrame:
    """Traditional classifier metrics pooled and separately by site."""
    rows: list[dict[str, object]] = []
    groups = [("pooled", "ALL", scored)]
    groups.extend(
        ("per_site", str(site), group)
        for site, group in scored.groupby("Site", sort=True)
    )
    for scope, site, group in groups:
        metrics = classification_metrics(
            group["relevance"].to_numpy(),
            group["pred_prob"].to_numpy(),
            threshold=threshold,
        )
        rows.append({
            "data_split": data_split,
            "threshold_source": threshold_source,
            "analysis_scope": scope,
            "Site": site,
            "n": metrics["n"],
            "n_pos": metrics["n_pos"],
            "n_neg": metrics["n_neg"],
            "AUROC": metrics["auc"],
            "AUPRC": metrics["pr_auc"],
            "ACC": metrics["accuracy"],
            "balanced_ACC": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "F1": metrics["f1"],
            "MCC": metrics["mcc"],
            "TN": metrics["tn"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "TP": metrics["tp"],
            "classification_threshold": threshold,
        })
    return pd.DataFrame(rows)


def logauc_from_fpr_tpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    lambda_min: float = LOGAUC_LAMBDA,
) -> float:
    """Mysinger/Shoichet LogAUC; identical to VS_eval/Topsis_VS.py."""
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


def filter_to_residues(
    df: pd.DataFrame,
    residues: list[str],
) -> pd.DataFrame:
    """Keep one residue class and the ligands annotated at that class.

    Other nucleophiles are removed from every candidate list. A ligand whose
    annotated site is a different residue is dropped, so its remaining rows
    are not trained as an all-negative group.
    """
    wanted: list[str] = []
    for residue in residues:
        token = str(residue).strip().upper()
        if token not in VALID_RESIDUES:
            allowed = ", ".join(sorted(VALID_RESIDUES))
            sys.exit(
                f"[ERROR] --residue {residue!r} is not one of {allowed}."
            )
        if token not in wanted:
            wanted.append(token)

    residue_names = df["Residue"].astype(str).str.strip().str.upper()
    before_rows = len(df)
    before_pos = int(df["relevance"].astype(int).sum())
    before_groups = int(df["query_group"].nunique())
    kept = df.loc[residue_names.isin(wanted)].copy()
    positive_groups = set(
        kept.loc[kept["relevance"].astype(int) == 1, "query_group"].astype(str)
    )
    groups_without_positive = int(
        kept.loc[
            ~kept["query_group"].astype(str).isin(positive_groups),
            "query_group",
        ].nunique()
    )
    kept = kept.loc[
        kept["query_group"].astype(str).isin(positive_groups)
    ].copy()
    if kept.empty or int(kept["relevance"].astype(int).sum()) == 0:
        sys.exit(
            "[ERROR] --residue left no labeled target rows. "
            "Check that the labels include that residue."
        )
    print(
        f"\n[INFO] --residue {' '.join(wanted)}: "
        f"kept {len(kept):,} / {before_rows:,} rows "
        f"({int(kept['relevance'].astype(int).sum()):,} / {before_pos:,} "
        f"positives), "
        f"{kept['query_group'].nunique():,} / {before_groups:,} query groups. "
        f"Dropped {groups_without_positive:,} query group(s) whose annotated "
        f"site is not in this residue set."
    )
    return kept


def residue_screen_metrics(
    labels_in_rank_order: np.ndarray,
) -> dict[str, float]:
    """EF@1/5/10% plus LogAUC for one best-to-worst residue ranking.

    Each row is one candidate. EF_x = (hits in top x / top-x size) /
    (all hits / all candidates). adjusted_logAUC = logAUC_0.001 - 0.14462.

    The ROC keeps intermediate points. Dropping them skips the horizontal
    step before a single hit and inflates LogAUC on a log-FPR axis.
    """
    labels = np.asarray(labels_in_rank_order).astype(int)
    metrics = {
        "EF_1pct": float("nan"),
        "EF_5pct": float("nan"),
        "EF_10pct": float("nan"),
        "logAUC": float("nan"),
        "adjusted_logAUC": float("nan"),
    }
    n_total = int(len(labels))
    n_hits = int(labels.sum())
    if n_total == 0 or n_hits == 0:
        return metrics

    prevalence = n_hits / n_total
    for fraction, label in EF_FRACTIONS:
        top_n = max(1, int(np.ceil(fraction * n_total)))
        hits_top = int(labels[:top_n].sum())
        metrics[f"EF_{label}"] = float((hits_top / top_n) / prevalence)

    if n_hits == n_total:
        return metrics
    ranks = np.arange(1, n_total + 1, dtype=float)
    fpr, tpr, _ = roc_curve(
        labels, -ranks, drop_intermediate=False
    )
    logauc = logauc_from_fpr_tpr(fpr, tpr, LOGAUC_LAMBDA)
    if np.isfinite(logauc):
        metrics["logAUC"] = float(logauc)
        metrics["adjusted_logAUC"] = float(logauc - RANDOM_LOGAUC)
    return metrics


def safe_bedroc(
    labels_in_rank_order: np.ndarray,
    alpha: float,
) -> float:
    """BEDROC for an already best-to-worst ordered binary ranking."""
    labels = np.asarray(labels_in_rank_order).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    try:
        from rdkit.ML.Scoring import Scoring

        records = [[int(label)] for label in labels]
        return float(Scoring.CalcBEDROC(records, 0, alpha))
    except (ImportError, ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


def covsite_ranking_metric_rows(
    per_site: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    data_split: str,
    topk: int,
    top_pct: float,
    bedroc_alpha: float,
) -> pd.DataFrame:
    """BEDROC, search-space reduction, and early-hit metrics."""
    rows: list[dict[str, object]] = []
    ranking_groups: list[tuple[str, str, pd.DataFrame, str]] = [
        ("pooled", "ALL", pooled, "overall_rank"),
    ]
    ranking_groups.extend(
        ("per_site", str(site), group, "site_rank")
        for site, group in per_site.groupby("Site", sort=True)
    )
    for scope, site, group, rank_col in ranking_groups:
        ranked = group.sort_values(rank_col, kind="mergesort").reset_index(
            drop=True
        )
        n_candidates = len(ranked)
        hit_ranks = (
            ranked.index[ranked["is_hit"].astype(int) == 1].to_numpy() + 1
        )
        n_hits = len(hit_ranks)
        topk_n = min(topk, n_candidates)
        top_pct_n = max(
            1, min(n_candidates, int(np.ceil(top_pct * n_candidates / 100.0)))
        )
        if n_hits:
            search_space_reduction = 1.0 - (
                hit_ranks.astype(float) / float(n_candidates)
            )
            median_ssr = float(np.median(search_space_reduction))
            median_hit_rank = float(np.median(hit_ranks))
            best_hit_rank = int(hit_ranks.min())
            hits_topk = int((hit_ranks <= topk_n).sum())
            hits_top_pct = int((hit_ranks <= top_pct_n).sum())
        else:
            median_ssr = float("nan")
            median_hit_rank = float("nan")
            best_hit_rank = 0
            hits_topk = 0
            hits_top_pct = 0
        rows.append({
            "data_split": data_split,
            "analysis_scope": scope,
            "Site": site,
            "n_candidates": n_candidates,
            "n_hits": n_hits,
            "BEDROC": safe_bedroc(
                ranked["is_hit"].to_numpy(), bedroc_alpha
            ),
            "bedroc_alpha": bedroc_alpha,
            "median_hit_rank": median_hit_rank,
            "best_hit_rank": best_hit_rank,
            "median_search_space_reduction": median_ssr,
            "topk": topk,
            "topk_n": topk_n,
            "hit_at_topk": int(hits_topk > 0),
            "hits_in_topk": hits_topk,
            "fraction_hits_recovered_topk": (
                hits_topk / n_hits if n_hits else float("nan")
            ),
            "top_pct": top_pct,
            "top_pct_n": top_pct_n,
            "hit_at_top_pct": int(hits_top_pct > 0),
            "hits_in_top_pct": hits_top_pct,
            "fraction_hits_recovered_top_pct": (
                hits_top_pct / n_hits if n_hits else float("nan")
            ),
        })

    result = pd.DataFrame(rows)
    site_rows = result.loc[result["analysis_scope"] == "per_site"]
    if not site_rows.empty:
        summary = {
            "data_split": data_split,
            "analysis_scope": "site_summary",
            "Site": "MEAN_OR_MEDIAN",
            "n_candidates": int(site_rows["n_candidates"].sum()),
            "n_hits": int(site_rows["n_hits"].sum()),
            "BEDROC": float(site_rows["BEDROC"].mean()),
            "bedroc_alpha": bedroc_alpha,
            "median_hit_rank": float(site_rows["median_hit_rank"].median()),
            "best_hit_rank": float(site_rows["best_hit_rank"].median()),
            "median_search_space_reduction": float(
                site_rows["median_search_space_reduction"].median()
            ),
            "topk": topk,
            "topk_n": float("nan"),
            "hit_at_topk": float(site_rows["hit_at_topk"].mean()),
            "hits_in_topk": int(site_rows["hits_in_topk"].sum()),
            "fraction_hits_recovered_topk": float(
                site_rows["fraction_hits_recovered_topk"].mean()
            ),
            "top_pct": top_pct,
            "top_pct_n": float("nan"),
            "hit_at_top_pct": float(site_rows["hit_at_top_pct"].mean()),
            "hits_in_top_pct": int(site_rows["hits_in_top_pct"].sum()),
            "fraction_hits_recovered_top_pct": float(
                site_rows["fraction_hits_recovered_top_pct"].mean()
            ),
        }
        result = pd.concat(
            [result, pd.DataFrame([summary])], ignore_index=True
        )
    return result


def residue_classifier_metric_rows(
    scored: pd.DataFrame,
    threshold: float,
    *,
    data_split: str,
    threshold_source: str,
) -> pd.DataFrame:
    """Classifier metrics pooled and within each ligand×warhead query."""
    rows: list[dict[str, object]] = []
    groups = [("pooled", "ALL", scored)]
    groups.extend(
        ("per_query_group", str(query_group), group)
        for query_group, group in scored.groupby("query_group", sort=True)
    )
    for scope, query_group, group in groups:
        metrics = classification_metrics(
            group["relevance"].to_numpy(),
            group["pred_prob"].to_numpy(),
            threshold=threshold,
        )
        rows.append({
            "data_split": data_split,
            "threshold_source": threshold_source,
            "analysis_scope": scope,
            "query_group": query_group,
            "n": metrics["n"],
            "n_pos": metrics["n_pos"],
            "n_neg": metrics["n_neg"],
            "AUROC": metrics["auc"],
            "AUPRC": metrics["pr_auc"],
            "ACC": metrics["accuracy"],
            "balanced_ACC": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "F1": metrics["f1"],
            "MCC": metrics["mcc"],
            "TN": metrics["tn"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "TP": metrics["tp"],
            "classification_threshold": threshold,
        })
    return pd.DataFrame(rows)


def residue_ranking_metric_rows(
    scored: pd.DataFrame,
    *,
    data_split: str,
    topk: int,
    top_pct: float,
    bedroc_alpha: float,
) -> pd.DataFrame:
    """Early-recognition metrics for target-residue rankings.

    Includes EF@1/5/10%, LogAUC, adjusted LogAUC, BEDROC, median
    search-space reduction, and Hit@K / Hit@top-%.
    """
    rows: list[dict[str, object]] = []
    for query_group, group in scored.groupby("query_group", sort=True):
        ranked = group.sort_values(
            "pred_prob", ascending=False, kind="mergesort"
        ).reset_index(drop=True)
        n_candidates = len(ranked)
        hit_ranks = (
            ranked.index[ranked["relevance"].astype(int) == 1].to_numpy() + 1
        )
        n_hits = len(hit_ranks)
        topk_n = min(topk, n_candidates)
        top_pct_n = max(
            1, min(n_candidates, int(np.ceil(top_pct * n_candidates / 100.0)))
        )
        if n_hits:
            median_rank = float(np.median(hit_ranks))
            best_rank = int(hit_ranks.min())
            median_ssr = float(
                np.median(
                    1.0 - hit_ranks.astype(float) / float(n_candidates)
                )
            )
            hits_topk = int((hit_ranks <= topk_n).sum())
            hits_top_pct = int((hit_ranks <= top_pct_n).sum())
        else:
            median_rank = float("nan")
            best_rank = 0
            median_ssr = float("nan")
            hits_topk = 0
            hits_top_pct = 0
        screen = residue_screen_metrics(ranked["relevance"].to_numpy())
        rows.append({
            "data_split": data_split,
            "analysis_scope": "per_query_group",
            "query_group": str(query_group),
            "n_candidates": n_candidates,
            "n_hits": n_hits,
            **screen,
            "BEDROC": safe_bedroc(
                ranked["relevance"].to_numpy(), bedroc_alpha
            ),
            "bedroc_alpha": bedroc_alpha,
            "median_hit_rank": median_rank,
            "best_hit_rank": best_rank,
            "median_search_space_reduction": median_ssr,
            "topk": topk,
            "topk_n": topk_n,
            "hit_at_topk": int(hits_topk > 0),
            "hits_in_topk": hits_topk,
            "top_pct": top_pct,
            "top_pct_n": top_pct_n,
            "hit_at_top_pct": int(hits_top_pct > 0),
            "hits_in_top_pct": hits_top_pct,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        summary = {
            "data_split": data_split,
            "analysis_scope": "query_group_summary",
            "query_group": "MEAN_OR_MEDIAN",
            "n_candidates": int(result["n_candidates"].sum()),
            "n_hits": int(result["n_hits"].sum()),
            "EF_1pct": float(result["EF_1pct"].mean()),
            "EF_5pct": float(result["EF_5pct"].mean()),
            "EF_10pct": float(result["EF_10pct"].mean()),
            "logAUC": float(result["logAUC"].mean()),
            "adjusted_logAUC": float(result["adjusted_logAUC"].mean()),
            "BEDROC": float(result["BEDROC"].mean()),
            "bedroc_alpha": bedroc_alpha,
            "median_hit_rank": float(result["median_hit_rank"].median()),
            "best_hit_rank": float(result["best_hit_rank"].median()),
            "median_search_space_reduction": float(
                result["median_search_space_reduction"].median()
            ),
            "topk": topk,
            "topk_n": float("nan"),
            "hit_at_topk": float(result["hit_at_topk"].mean()),
            "hits_in_topk": int(result["hits_in_topk"].sum()),
            "top_pct": top_pct,
            "top_pct_n": float("nan"),
            "hit_at_top_pct": float(result["hit_at_top_pct"].mean()),
            "hits_in_top_pct": int(result["hits_in_top_pct"].sum()),
        }
        result = pd.concat(
            [result, pd.DataFrame([summary])], ignore_index=True
        )
    return result


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
    task_mode: str = "residue_site_classifier",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "lgbm_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_cols,
            "model_type": "classifier",
            "task_mode": task_mode,
            "include_residue_type": include_residue_type,
            "max_depth": max_depth,
            "is_unbalance": True,
            "scale_pos_weight": None,
            "classification_threshold": test_metrics.get(
                "classification_threshold", 0.5
            ),
            "classification_threshold_source": test_metrics.get(
                "classification_threshold_source", "fixed"
            ),
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
        "Name", "PDB_ID", "pdb_id", "Residue", "ResNum", "Chain",
        "Warhead", "electrophile_smiles", "Site", "query_group",
        "cluster_id", "relevance", "is_hit", "target_residue_type",
        "pred_prob", "pred_class",
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
        "task_mode": task_mode,
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
            "accuracy": test_metrics.get("accuracy"),
            "f1": test_metrics.get("f1"),
            "mcc": test_metrics.get("mcc"),
            "classification_threshold": test_metrics.get(
                "classification_threshold"
            ),
            "classification_threshold_source": test_metrics.get(
                "classification_threshold_source"
            ),
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
        description=(
            "LGBM binary classifier for either nucleophilic residue hits or "
            "active ligands in CovSite virtual-screening candidates"
        )
    )
    p.add_argument("--training", default=None, help="Residue-mode training CSV")
    p.add_argument("--labels", default=None, help="Residue-mode labels CSV")
    p.add_argument(
        "--vs-candidates",
        default=None,
        metavar="CSV",
        help=(
            "CovSite candidates CSV for ligand-classifier mode. Use with "
            "--vs-hits; the candidate table also supplies the library rows."
        ),
    )
    p.add_argument(
        "--vs-hits",
        default=None,
        metavar="CSV",
        help=(
            "all_hits-style CSV for ligand-classifier mode. Positives are "
            "matched by canonical SMILES and site."
        ),
    )
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
    p.add_argument(
        "--residue",
        nargs="+",
        default=None,
        metavar="RES",
        help=(
            "Train and evaluate only these residue types "
            "(e.g. --residue CYS). Other nucleophiles are removed, and "
            "ligands annotated at a different residue are dropped."
        ),
    )
    p.add_argument("--no-residue-type", action="store_true")
    p.add_argument("--no-residue-specific-features", action="store_true")
    p.add_argument("--large-pool-prefilter", action="store_true")
    p.add_argument("--large-pool-min-size", type=int, default=500)
    p.add_argument("--large-pool-filter", nargs="+", default=None, metavar="SPEC")
    p.add_argument("--output_dir", default="lgbm_classifier_output")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--shap-max-rows", type=int, default=None)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for ACC, F1, and MCC (default: 0.5)",
    )

    # CovSite ligand-classifier matching/evaluation
    p.add_argument("--topsis-vs", default=None)
    p.add_argument("--pdb-column", default=None)
    p.add_argument("--smiles-column", default=None)
    p.add_argument("--name-column", default=None)
    p.add_argument("--residue-column", default=None)
    p.add_argument("--resnum-column", default=None)
    p.add_argument("--chain-column", default=None)
    p.add_argument("--hit-residue-column", default=None)
    p.add_argument("--hit-resnum-column", default=None)
    p.add_argument("--hit-chain-column", default=None)
    p.add_argument("--pdb-hits", action="store_true")
    p.add_argument("--smiles-hit-column", default="electrophile_hit")
    p.add_argument("--name-hit-column", default="Name_Hit")
    p.add_argument("--name-miss-column", default="Name_Miss")
    p.add_argument("--smiles-miss-column", default="electrophile_miss")
    p.add_argument("--warhead-type-column", default="warhead_type")
    p.add_argument("--keep-warhead", nargs="+", default=None, metavar="WARHEAD")
    p.add_argument("--best-warhead", action="store_true")
    p.add_argument("--collapse-active-protomers", action="store_true")
    p.add_argument("--multiple-hits", action="store_true")

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
    p.add_argument(
        "--bedroc-alpha",
        type=float,
        default=20.0,
        help="BEDROC early-recognition alpha for ligand mode (default: 20)",
    )
    p.add_argument("--rank-bonus-epsilon", type=float,
                   default=DEFAULT_RANK_BONUS_EPSILON)
    return p.parse_args()


def main_covsite_ligand(args: argparse.Namespace) -> None:
    """Train/test a pointwise active-ligand classifier on CovSite features."""
    if args.large_pool_prefilter:
        sys.exit(
            "[ERROR] --large-pool-prefilter is not supported with "
            "--vs-candidates; filter the candidate CSV before training."
        )
    if args.compare_xgb:
        sys.exit(
            "[ERROR] --compare-xgb is not yet supported with --vs-candidates."
        )
    if args.residue_specific_features and not args.no_residue_specific_features:
        print(
            "[INFO] CovSite ligand mode uses --features only; "
            "--residue_specific_features is ignored."
        )

    print("=" * 68)
    print("  Training_Cov_Classifier — CovSite active-ligand classifier")
    print("=" * 68)

    data, feature_cols, unmatched, vs, tvs = load_covsite_ligand_data(args)

    cluster_map = build_cluster_map(
        data,
        args.pdb_folder,
        args.mmseqs_dir,
        args.seq_identity,
        args.coverage,
        deep_cluster=args.deep_cluster,
        min_chain_length=args.deep_cluster_min_chain_length,
        clusters_tsv=args.clusters_tsv,
    )
    data = assign_splits(
        data,
        cluster_map,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_state=args.random_state,
        split_csv=args.split_csv,
        export_split=args.export_split,
    )
    train_df = (
        data.loc[data["split"] == "train"]
        .sort_values("query_group")
        .reset_index(drop=True)
    )
    test_df = (
        data.loc[data["split"] == "test"]
        .sort_values("query_group")
        .reset_index(drop=True)
    )
    if train_df.empty or test_df.empty:
        sys.exit(
            "[ERROR] Cluster split produced an empty train or test set. "
            "Adjust --test_size or provide --split-csv."
        )
    for split_name, frame in (("train", train_df), ("test", test_df)):
        if frame["relevance"].nunique() < 2:
            sys.exit(
                f"[ERROR] The {split_name} split has only one class. "
                "Use a different cluster split."
            )
    n_train_clusters = int(train_df["cluster_id"].nunique())
    if args.n_folds > n_train_clusters:
        sys.exit(
            f"[ERROR] --n_folds={args.n_folds} exceeds the "
            f"{n_train_clusters} training cluster(s)."
        )

    (
        best_depth,
        depth_grid_df,
        cv_summary,
        cv_folds_df,
        oof_probs,
        _fold_models,
        shap_stability,
    ) = select_max_depth_and_cv(
        train_df,
        feature_cols,
        args.n_folds,
        depth_grid=None if args.max_depth is not None else list(args.depth_grid),
        fixed_depth=args.max_depth,
        backend="lgbm",
    )

    train_oof_df = train_df.copy()
    train_oof_df["pred_prob"] = oof_probs
    train_oof_df["pred_score"] = oof_probs
    threshold_selection = select_f1_threshold(
        train_oof_df["relevance"].to_numpy(), oof_probs
    )
    selected_threshold = threshold_selection["threshold"]
    print(
        "\nOOF-selected classification cutoff (test labels not used):\n"
        f"  threshold={selected_threshold:.6g}  "
        f"precision={threshold_selection['precision']:.4f}  "
        f"recall={threshold_selection['recall']:.4f}  "
        f"F1={threshold_selection['f1']:.4f}"
    )

    # The optimized cutoff is learned on raw fold-model OOF probabilities.
    # Keep the final model on that same score scale; isotonic calibration
    # would remap probabilities and invalidate the transferred numeric cutoff.
    apply_cal = False
    cal_reason = "disabled_for_oof_threshold_transfer"
    print(
        "[INFO] Isotonic calibration disabled in ligand mode so the OOF "
        "cutoff and final-model probabilities use the same score scale."
    )
    final_model, calib_meta = fit_final_classifier(
        train_df,
        feature_cols,
        best_depth,
        apply_calibration=apply_cal,
        calib_fraction=args.calib_fraction,
        random_state=args.random_state,
    )
    calib_meta["decision_reason"] = cal_reason
    if calib_meta.get("reason") == "no_calibration":
        calib_meta["reason"] = cal_reason

    X_test = test_df[feature_cols].to_numpy()
    test_probs = predict_proba_positive(final_model, X_test)
    test_df = test_df.copy()
    test_df["pred_prob"] = test_probs
    test_df["pred_score"] = test_probs
    test_df["pred_class"] = (
        test_probs >= selected_threshold
    ).astype(int)
    test_metrics = classification_metrics_with_groups(
        test_df["relevance"].to_numpy(),
        test_probs,
        groups=test_df["Site"].to_numpy(),
        threshold=selected_threshold,
    )
    test_metrics["classification_threshold_source"] = "training_oof_max_f1"
    print(
        "\nHeld-out ligand classification:\n"
        f"  AUROC={test_metrics['auc']:.4f}  "
        f"AUPRC={test_metrics['pr_auc']:.4f}  "
        f"ACC={test_metrics['accuracy']:.4f}  "
        f"F1={test_metrics['f1']:.4f}  "
        f"MCC={test_metrics['mcc']:.4f}"
    )

    test_per_site, test_pooled, test_enrichment = covsite_ligand_rankings(
        test_df, args, vs, tvs
    )
    train_per_site, train_pooled, train_enrichment = covsite_ligand_rankings(
        train_oof_df, args, vs, tvs
    )
    classifier_rows = pd.concat(
        [
            covsite_classifier_metric_rows(
                train_oof_df,
                selected_threshold,
                data_split="training_oof",
                threshold_source="training_oof_max_f1",
            ),
            covsite_classifier_metric_rows(
                train_oof_df,
                0.5,
                data_split="training_oof",
                threshold_source="fixed_0.5",
            ),
            covsite_classifier_metric_rows(
                test_df,
                selected_threshold,
                data_split="held_out_test",
                threshold_source="training_oof_max_f1",
            ),
            covsite_classifier_metric_rows(
                test_df,
                0.5,
                data_split="held_out_test",
                threshold_source="fixed_0.5",
            ),
        ],
        ignore_index=True,
    )
    ranking_rows = pd.concat(
        [
            covsite_ranking_metric_rows(
                train_per_site,
                train_pooled,
                data_split="training_oof",
                topk=args.topk,
                top_pct=args.top_pct,
                bedroc_alpha=args.bedroc_alpha,
            ),
            covsite_ranking_metric_rows(
                test_per_site,
                test_pooled,
                data_split="held_out_test",
                topk=args.topk,
                top_pct=args.top_pct,
                bedroc_alpha=args.bedroc_alpha,
            ),
        ],
        ignore_index=True,
    )
    per_res_clf = per_residue_type_clf_metrics(test_df, "pred_prob")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_classifier_outputs(
        final_model,
        feature_cols,
        str(output_dir),
        max_depth=best_depth,
        calib_meta=calib_meta,
        cv_summary=cv_summary,
        cv_folds_df=cv_folds_df,
        depth_grid_df=depth_grid_df,
        shap_stability_df=shap_stability,
        test_metrics=test_metrics,
        test_reliability=test_metrics.get("reliability", {}),
        test_df=test_df,
        test_rank_eval=pd.DataFrame(),
        test_rank_summary={},
        composition=None,
        per_res_clf=per_res_clf,
        xgb_comparison=None,
        include_residue_type=not args.no_residue_type,
        large_pool_prefilter=None,
        export_shap=not args.no_shap,
        shap_max_rows=args.shap_max_rows,
        k=args.topk,
        top_pct=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
        task_mode="covsite_ligand_classifier",
    )

    test_per_site.to_csv(
        output_dir / "test_per_site_ranking.csv", index=False
    )
    test_pooled.to_csv(
        output_dir / "test_pooled_pair_ranking.csv", index=False
    )
    test_pooled_ligands = vs.collapse_ligand_ranking(
        tvs, test_pooled, args.multiple_hits
    )
    test_pooled_ligands.to_csv(
        output_dir / "test_pooled_ligand_ranking.csv", index=False
    )
    test_per_site.loc[test_per_site["is_hit"] == 1].to_csv(
        output_dir / "test_hit_per_site_ranks.csv", index=False
    )
    test_enrichment.to_csv(
        output_dir / "test_enrichment_factors.csv", index=False
    )
    train_per_site.to_csv(
        output_dir / "train_oof_per_site_ranking.csv", index=False
    )
    train_pooled.to_csv(
        output_dir / "train_oof_pooled_pair_ranking.csv", index=False
    )
    train_enrichment.to_csv(
        output_dir / "train_oof_enrichment_factors.csv", index=False
    )
    classifier_rows.to_csv(
        output_dir / "test_classifier_metrics.csv", index=False
    )
    classifier_rows.to_csv(
        output_dir / "classifier_metrics.csv", index=False
    )
    ranking_rows.to_csv(
        output_dir / "ranking_metrics.csv", index=False
    )
    pd.DataFrame(
        unmatched, columns=["Hit", "Reason", "Detail"]
    ).to_csv(output_dir / "unmatched_hit_names.csv", index=False)

    oof_df = train_df[[
        column for column in [
            "Name", "PDB_ID", "Residue", "ResNum", "Chain", "Warhead",
            "electrophile_smiles", "Site", "query_group", "cluster_id",
            "relevance", "is_hit",
        ] if column in train_df.columns
    ]].copy()
    oof_df["oof_pred_prob"] = oof_probs
    oof_df["oof_pred_class_selected"] = (
        oof_probs >= selected_threshold
    ).astype(int)
    oof_df["oof_pred_class_0.5"] = (oof_probs >= 0.5).astype(int)
    oof_df.to_csv(output_dir / "cv_oof_predictions.csv", index=False)

    summary_path = output_dir / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    pooled_test_enrichment = test_enrichment.loc[
        test_enrichment["analysis_scope"] == "pooled"
    ]
    pooled_train_enrichment = train_enrichment.loc[
        train_enrichment["analysis_scope"] == "pooled"
    ]
    if not pooled_test_enrichment.empty:
        summary["test_enrichment"] = pooled_test_enrichment.to_dict(
            orient="records"
        )
    if not pooled_train_enrichment.empty:
        summary["train_oof_enrichment"] = pooled_train_enrichment.to_dict(
            orient="records"
        )
    summary["threshold_selection"] = {
        "source": "training_oof",
        "criterion": "maximum_f1",
        **threshold_selection,
        "baseline_threshold": 0.5,
        "test_labels_used_for_selection": False,
    }
    summary["classifier_metrics"] = classifier_rows.to_dict(orient="records")
    summary["ranking_metrics"] = ranking_rows.to_dict(orient="records")
    summary["unmatched_hits"] = len(unmatched)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("\nCovSite ligand classifier complete")
    print(f"  Model:          {output_dir / 'lgbm_classifier.pkl'}")
    print(f"  Class metrics:  {output_dir / 'classifier_metrics.csv'}")
    print(f"  Rank metrics:   {output_dir / 'ranking_metrics.csv'}")
    print(f"  Enrichment:     {output_dir / 'test_enrichment_factors.csv'}")
    print(f"  Site rankings:  {output_dir / 'test_per_site_ranking.csv'}")
    print(f"  Pooled ranking: {output_dir / 'test_pooled_pair_ranking.csv'}")
    print(f"  Ligand ranking: {output_dir / 'test_pooled_ligand_ranking.csv'}")


def main() -> None:
    args = parse_args()
    ligand_mode = bool(args.vs_candidates or args.vs_hits)
    if ligand_mode:
        if not (args.vs_candidates and args.vs_hits):
            sys.exit(
                "[ERROR] Use --vs-candidates and --vs-hits together."
            )
        if args.training or args.labels:
            sys.exit(
                "[ERROR] Do not combine --training/--labels with "
                "--vs-candidates/--vs-hits."
            )
    elif not (args.training and args.labels):
        sys.exit(
            "[ERROR] Residue mode requires --training and --labels; "
            "ligand mode requires --vs-candidates and --vs-hits."
        )
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
    if not 0 <= args.classification_threshold <= 1:
        sys.exit("[ERROR] --classification-threshold must be in [0, 1].")
    if args.bedroc_alpha <= 0:
        sys.exit("[ERROR] --bedroc-alpha must be > 0.")
    if args.force_calibrate:
        sys.exit(
            "[ERROR] --force-calibrate is incompatible with OOF cutoff "
            "selection because calibration remaps the numeric cutoff."
        )
    if ligand_mode:
        if args.residue:
            sys.exit(
                "[ERROR] --residue filters the residue-site classifier. "
                "Do not combine it with --vs-candidates."
            )
        main_covsite_ligand(args)
        return

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
    if args.residue:
        merged = filter_to_residues(merged, args.residue)

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
    train_oof_df = train_df.copy()
    train_oof_df["pred_prob"] = oof_probs
    train_oof_df["pred_score"] = oof_probs
    threshold_selection = select_f1_threshold(
        train_oof_df["relevance"].to_numpy(), oof_probs
    )
    selected_threshold = threshold_selection["threshold"]
    print(
        "\nOOF-selected classification cutoff (test labels not used):\n"
        f"  threshold={selected_threshold:.6g}  "
        f"precision={threshold_selection['precision']:.4f}  "
        f"recall={threshold_selection['recall']:.4f}  "
        f"F1={threshold_selection['f1']:.4f}"
    )

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

    # Keep final-model probabilities on the same numeric scale as the raw OOF
    # probabilities used to select the operational cutoff.
    apply_cal = False
    cal_reason = "disabled_for_oof_threshold_transfer"
    print(
        "\n[5c/8] Isotonic calibration disabled so the OOF-selected cutoff "
        "and final-model probabilities use the same score scale."
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
    test_df["pred_class"] = (
        test_probs >= selected_threshold
    ).astype(int)

    test_metrics = classification_metrics_with_groups(
        y_test, test_probs, groups=test_df["query_group"].values,
        threshold=selected_threshold,
    )
    test_metrics["classification_threshold_source"] = "training_oof_max_f1"
    print(
        f"  Test pooled   PR-AUC={test_metrics['pr_auc']:.4f}  "
        f"AUC={test_metrics['auc']:.4f}  "
        f"ACC={test_metrics['accuracy']:.4f}  "
        f"F1={test_metrics['f1']:.4f}  MCC={test_metrics['mcc']:.4f}"
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

    classifier_metric_rows = pd.concat(
        [
            residue_classifier_metric_rows(
                train_oof_df,
                selected_threshold,
                data_split="training_oof",
                threshold_source="training_oof_max_f1",
            ),
            residue_classifier_metric_rows(
                train_oof_df,
                0.5,
                data_split="training_oof",
                threshold_source="fixed_0.5",
            ),
            residue_classifier_metric_rows(
                test_df,
                selected_threshold,
                data_split="held_out_test",
                threshold_source="training_oof_max_f1",
            ),
            residue_classifier_metric_rows(
                test_df,
                0.5,
                data_split="held_out_test",
                threshold_source="fixed_0.5",
            ),
        ],
        ignore_index=True,
    )
    ranking_metric_rows = pd.concat(
        [
            residue_ranking_metric_rows(
                train_oof_df,
                data_split="training_oof",
                topk=args.topk,
                top_pct=args.top_pct,
                bedroc_alpha=args.bedroc_alpha,
            ),
            residue_ranking_metric_rows(
                test_df,
                data_split="held_out_test",
                topk=args.topk,
                top_pct=args.top_pct,
                bedroc_alpha=args.bedroc_alpha,
            ),
        ],
        ignore_index=True,
    )
    ranking_summary = ranking_metric_rows.loc[
        ranking_metric_rows["analysis_scope"].eq("query_group_summary"),
        [
            "data_split", "EF_1pct", "EF_5pct", "EF_10pct",
            "logAUC", "adjusted_logAUC", "BEDROC",
            "hit_at_topk", "hit_at_top_pct",
            "median_search_space_reduction",
        ],
    ]
    if not ranking_summary.empty:
        print("\n  Query-group ranking summary (mean across ligands):")
        print(ranking_summary.to_string(index=False, float_format="%.4f"))

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
    oof_df["oof_pred_class_selected"] = (
        oof_probs >= selected_threshold
    ).astype(int)
    oof_df["oof_pred_class_0.5"] = (oof_probs >= 0.5).astype(int)
    oof_df.to_csv(oof_path, index=False)
    print(f"  OOF predictions: {oof_path}")

    metric_path = os.path.join(args.output_dir, "classifier_metrics.csv")
    classifier_metric_rows.to_csv(metric_path, index=False)
    ranking_path = os.path.join(args.output_dir, "ranking_metrics.csv")
    ranking_metric_rows.to_csv(ranking_path, index=False)
    print(f"  Classifier metrics: {metric_path}")
    print(f"  Ranking metrics:    {ranking_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["threshold_selection"] = {
        "source": "training_oof",
        "criterion": "maximum_f1",
        **threshold_selection,
        "baseline_threshold": 0.5,
        "test_labels_used_for_selection": False,
    }
    summary["classifier_metrics"] = classifier_metric_rows.to_dict(
        orient="records"
    )
    summary["ranking_metrics"] = ranking_metric_rows.to_dict(orient="records")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("\n[8/8] Done.")


if __name__ == "__main__":
    main()
