"""
covalent_rank_optimizer.py

Combines four per-residue scores into a single ranking score S, then
optimises either the weights OR the per-score thresholds using Bayesian
Optimisation (BO) with k-fold cross-validation to maximise a composite
ranking objective.

SCORES (all higher = better, each scaled to [0, 1]):
    A  = Rel_Side_SASA                   (accessibility)    training CSV
    O  = Binary_Orbital_Deprotonated     (orbital, 0/1)     training CSV
    D  = deprotonation_prob              (deprotonation)    training CSV
    R  = R                               (reactivity)       reactivity CSV

SCALING:
    value >= threshold  →  1.0
    value <  threshold  →  value / threshold  (linear, clamped [0,1])
    Binary O            →  passed through as-is (or thresholded if --train-thresholds)

COMBINED SCORE:
    S = w_A*A + w_O*O + w_D*D + w_R*R

MODES:
    Default (no weight flags):
        Weights are optimised via BO; thresholds are fixed CLI values.

    Manual weights (--w-sasa, --w-orbital, --w-deprot, --w-react):
        Weights are fixed; BO is skipped entirely.

    Train thresholds (--train-thresholds, requires all --w-* flags):
        Weights are fixed; BO optimises the four per-score thresholds
        (thr_sasa, thr_orbital, thr_deprot, thr_react) to maximise the
        same CV objective.

WARHEAD MATCHING:
    Training column : "Warhead"               (Frankenstein terminology)
    Labels column   : "Frankenstein_Warhead"  (comma-separated, same terminology)
    Match = training Warhead in labels Frankenstein_Warhead set

CROSS-VALIDATION (--n-folds, default 10):
    Proteins are grouped by their base PDB name (stripping -2, -3 suffixes).
    Groups are split into k folds — the same base PDB never spans train/test.
    BO objective = mean held-out ranking metric across all k folds.

OBJECTIVE:
    obj = 0.6 * Hit@1 + 0.3 * Hit@K + 0.1 * MRR

Usage:
    # Optimise weights (default):
    python covalent_rank_optimizer.py \\
        --training   training.csv \\
        --labels     labels.csv \\
        --reactivity reactivity.csv \\
        --threshold-sasa  0.15 \\
        --threshold-deprot 0.5 \\
        --threshold-react  0.5 \\
        [--top-k 5] [--n-folds 10] [--n-calls 60] [--match-warheads] ...

    # Fix weights, optimise thresholds:
    python covalent_rank_optimizer.py \\
        --training   training.csv \\
        --labels     labels.csv \\
        --reactivity reactivity.csv \\
        --threshold-sasa  0.15 \\
        --threshold-deprot 0.5 \\
        --threshold-react  0.5 \\
        --w-sasa 1.2 --w-orbital 0.8 --w-deprot 1.5 --w-react 2.0 \\
        --train-thresholds \\
        [--top-k 5] [--n-folds 10] [--n-calls 60] ...
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

COL_SASA    = "Rel_Side_SASA"
COL_ORBITAL = "Binary_Orbital_Deprotonated"
COL_DEPROT  = "deprotonation_prob"
COL_R       = "R"
# Training has "Warhead" (Frankenstein terms)
# Labels   has "Frankenstein_Warhead" (comma-separated Frankenstein terms)
COL_TRAIN_WARHEAD = "Warhead"
COL_LABEL_WARHEAD = "Frankenstein_Warhead"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Cannot find {label} file: {path}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read {label} file: {e}")
    df.columns = df.columns.str.strip()
    return df


def norm_name(s) -> str:
    return str(s).strip().upper()


def make_key(name, residue, chain, resnum) -> tuple:
    return (norm_name(name), norm_name(str(residue)), str(chain).strip(), int(resnum))


def base_pdb(name: str) -> str:
    """
    Strip trailing -2, -3, ... suffixes so that '4G5J', '4G5J-2', '4G5J-3'
    all map to the same base PDB group '4G5J'.
    """
    return re.sub(r'-\d+$', '', norm_name(name))


def parse_frankenstein_warheads(raw: str) -> set[str]:
    """Split comma-separated Frankenstein_Warhead into a normalised set."""
    return {w.strip().lower() for w in str(raw).split(",") if w.strip()}


def scale_score(values: pd.Series, threshold: float) -> pd.Series:
    scaled = (values / threshold).clip(0.0, 1.0)
    scaled = scaled.where(values < threshold, 1.0)
    return scaled


# ---------------------------------------------------------------------------
# Data loading + joining
# ---------------------------------------------------------------------------

def load_and_join(
    training_path: str,
    labels_path: str,
    reactivity_path: str,
    thr_sasa: float,
    thr_deprot: float,
    thr_react: float,
    match_warheads: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        merged  : one row per training residue, scaled scores, _key, _labeled,
                  _base_pdb columns.
                  Raw score columns (COL_SASA etc.) are always retained so that
                  --train-thresholds can re-scale on the fly inside BO.
        targets : one row per labeled target site entry (replicates kept)
    """
    train  = load_csv(training_path,    "training")
    labels = load_csv(labels_path,      "labels")
    react  = load_csv(reactivity_path,  "reactivity")

    # ---- validate --------------------------------------------------------
    req_train = {"Name", "Residue", "Chain", "ResNum",
                 COL_SASA, COL_ORBITAL, COL_DEPROT}
    if match_warheads:
        req_train.add(COL_TRAIN_WARHEAD)
    missing = req_train - set(train.columns)
    if missing:
        sys.exit(f"[ERROR] Training CSV missing columns: {missing}")

    req_labels = {"Name", "Residue", "Chain", "ResNum"}
    if match_warheads:
        req_labels.add(COL_LABEL_WARHEAD)
    missing = req_labels - set(labels.columns)
    if missing:
        sys.exit(f"[ERROR] Labels CSV missing columns: {missing}")

    req_react = {"Name", "Residue", "Chain", "ResNum", COL_R}
    missing = req_react - set(react.columns)
    if missing:
        sys.exit(f"[ERROR] Reactivity CSV missing columns: {missing}")

    # ---- coerce types ----------------------------------------------------
    for df, cols in [
        (train,  ["ResNum", COL_SASA, COL_ORBITAL, COL_DEPROT]),
        (labels, ["ResNum"]),
        (react,  ["ResNum", COL_R]),
    ]:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for df in (train, labels, react):
        df["ResNum"] = df["ResNum"].astype("Int64")

    train.dropna(subset=["ResNum", COL_SASA, COL_ORBITAL, COL_DEPROT], inplace=True)
    react.dropna(subset=["ResNum", COL_R], inplace=True)

    # ---- join reactivity → training --------------------------------------
    react_key = ["Name", "Residue", "Chain", "ResNum", "Warhead"]
    react_dedup = react.drop_duplicates(subset=react_key, keep="first")
    merged = train.merge(react_dedup[react_key + [COL_R]], on=react_key, how="left")

    print("\n[INFO] INITIAL COUNTS")
    print(f"  Training rows : {len(train):,}")
    print(f"  Label rows    : {len(labels):,}")
    print(f"  React rows    : {len(react):,}")

    n_miss = merged[COL_R].isna().sum()
    if n_miss:
        print(f"[WARN] {n_miss:,} training rows had no reactivity score — filling with 0")
    merged[COL_R] = merged[COL_R].fillna(0.0)

    print(f"  After fill (0s): {(merged[COL_R] == 0).sum():,}")

    # ---- default-threshold scaled columns (used when weights are trained) -
    merged["A_scaled"] = scale_score(merged[COL_SASA].astype(float),   thr_sasa)
    merged["O_scaled"] = merged[COL_ORBITAL].astype(float).clip(0.0, 1.0)
    merged["D_scaled"] = scale_score(merged[COL_DEPROT].astype(float), thr_deprot)
    merged["R_scaled"] = scale_score(merged[COL_R].astype(float),      thr_react)

    # ---- also retain raw float columns for threshold training ------------
    merged["_raw_A"] = merged[COL_SASA].astype(float)
    merged["_raw_O"] = merged[COL_ORBITAL].astype(float)
    merged["_raw_D"] = merged[COL_DEPROT].astype(float)
    merged["_raw_R"] = merged[COL_R].astype(float)

    # ---- build keys ------------------------------------------------------
    merged["_key"] = list(zip(
        merged["Name"].str.strip().str.upper(),
        merged["Residue"].str.strip().str.upper(),
        merged["Chain"].str.strip(),
        merged["ResNum"].astype(int),
    ))
    merged["_name_upper"] = merged["Name"].str.strip().str.upper()
    merged["_base_pdb"]   = merged["_name_upper"].map(base_pdb)

    # ---- build target entries from labels --------------------------------
    target_rows = []
    for _, row in labels.iterrows():
        key = make_key(row["Name"], row["Residue"], row["Chain"], row["ResNum"])
        allowed_warheads: set[str] = set()
        if match_warheads:
            allowed_warheads = parse_frankenstein_warheads(
                row.get(COL_LABEL_WARHEAD, "")
            )
        target_rows.append({
            "_key":          key,
            "_base_pdb":     base_pdb(row["Name"]),
            "_label_name":   norm_name(row["Name"]),
            "_warhead_set":  allowed_warheads,
        })
    targets = pd.DataFrame(target_rows)

    # ---- mark labeled training rows --------------------------------------
    label_key_set = set(targets["_key"])
    merged["_labeled"] = merged["_key"].isin(label_key_set)

    if match_warheads:
        # Build key → union of allowed warheads across all label rows
        key_to_warheads: dict[tuple, set[str]] = {}
        for _, tr in targets.iterrows():
            k = tr["_key"]
            key_to_warheads[k] = key_to_warheads.get(k, set()) | tr["_warhead_set"]

        def _warhead_ok(row) -> bool:
            if not row["_labeled"]:
                return False
            allowed = key_to_warheads.get(row["_key"], set())
            if not allowed:
                return True   # no warhead info → don't penalise
            train_wh = str(row.get(COL_TRAIN_WARHEAD, "")).strip().lower()
            return train_wh in allowed

        base_n = merged["_labeled"].sum()
        merged["_labeled"] = merged.apply(_warhead_ok, axis=1)
        filtered = int(base_n - merged["_labeled"].sum())
        if filtered:
            print(f"[INFO] {filtered} label-matched rows excluded by warhead mismatch")

    n_matched   = merged["_labeled"].sum()
    matched_keys = set(merged.loc[merged["_labeled"], "_key"])
    n_unmatched = len(label_key_set - matched_keys)
    print(f"[INFO] Label target entries          : {len(targets):,}")
    print(f"[INFO] Matched in training           : {n_matched:,}")
    if n_unmatched:
        print(f"[WARN] {n_unmatched} label keys had no training match")

    print("\n[SUMMARY]")
    print(f"  Training rows (final) : {len(merged):,}")
    print(f"  Labeled rows (final)  : {merged['_labeled'].sum():,}")
    print(f"  React missing ratio   : {merged[COL_R].isna().mean():.4f}")

    return merged, targets


# ---------------------------------------------------------------------------
# Cross-validation fold builder
# ---------------------------------------------------------------------------

def make_cv_folds(
    merged: pd.DataFrame,
    targets: pd.DataFrame,
    n_folds: int,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split proteins into n_folds by BASE PDB name so that '4G5J', '4G5J-2',
    '4G5J-3' always land in the same fold.

    Returns list of (train_df, val_df, val_targets) tuples where each is a
    subset of merged.  Targets are filtered to match whichever names are in
    val_df.
    """
    rng = np.random.default_rng(seed)

    base_pdbs = sorted(merged["_base_pdb"].unique())
    rng.shuffle(base_pdbs)

    fold_assignments: dict[str, int] = {
        bp: i % n_folds for i, bp in enumerate(base_pdbs)
    }
    merged = merged.copy()
    merged["_fold"] = merged["_base_pdb"].map(fold_assignments)

    target_base_pdbs = targets["_base_pdb"].values

    folds = []
    for f in range(n_folds):
        val_mask   = merged["_fold"] == f
        train_mask = ~val_mask

        val_df   = merged[val_mask].copy()
        train_df = merged[train_mask].copy()

        val_base = set(val_df["_base_pdb"].unique())
        val_targets = targets[
            pd.Series(target_base_pdbs).isin(val_base).values
        ].copy()

        folds.append((train_df, val_df, val_targets))

    return folds


# ---------------------------------------------------------------------------
# Scoring + ranking
# ---------------------------------------------------------------------------

def apply_thresholds(df: pd.DataFrame,
                     thr_sasa: float, thr_orbital: float,
                     thr_deprot: float, thr_react: float) -> pd.DataFrame:
    """
    Re-scale raw score columns using the supplied thresholds.
    Returns a copy of df with updated *_scaled columns.
    Used inside the threshold-training BO loop.
    """
    df = df.copy()
    df["A_scaled"] = scale_score(df["_raw_A"], thr_sasa)
    df["O_scaled"] = scale_score(df["_raw_O"], thr_orbital)
    df["D_scaled"] = scale_score(df["_raw_D"], thr_deprot)
    df["R_scaled"] = scale_score(df["_raw_R"], thr_react)
    return df


def compute_combined_score(
    df: pd.DataFrame,
    w_A: float, w_O: float, w_D: float, w_R: float,
) -> pd.Series:
    return (
        w_A * df["A_scaled"] +
        w_O * df["O_scaled"] +
        w_D * df["D_scaled"] +
        w_R * df["R_scaled"]
    )


def rank_within_protein(df: pd.DataFrame, score_col: str = "_score") -> pd.Series:
    """Rank residues within each protein name descending. Rank 1 = best."""
    return df.groupby("_name_upper")[score_col].rank(method="min", ascending=False)


def evaluate_objective(
    df: pd.DataFrame,
    targets: pd.DataFrame,
    w_A: float, w_O: float, w_D: float, w_R: float,
    top_k: int,
) -> dict:
    """
    Score and rank df, then compute Hit@5, Hit@K, and clipped-MRR over targets.

    OBJECTIVE:
        obj = 0.75 * Hit@K + 0.15 * Hit@5 + 0.10 * clipped_MRR

    Clipped reciprocal rank (clipped_RR):
        rank <= top_k  →  1 / rank          (normal credit, gradient within K)
        rank >  top_k  →  1 / (top_k + 1)   (floored — all misses are equal)

    Rationale: Hit@K is the primary signal (find it somewhere in the shortlist).
    Hit@5 gives a mild bonus for finding it near the top without over-rewarding
    rank-1. Clipped MRR provides a smooth gradient inside the top-K window to
    help BO differentiate partial progress, but misses beyond K incur no extra
    penalty regardless of how far out they are.

    Each target row is an independent entry (multiple sites per protein each
    contribute their own score).
    """
    HIT5_K = 5  # fixed inner window for the mild top-5 bonus

    df = df.copy()
    df["_score"] = compute_combined_score(df, w_A, w_O, w_D, w_R)
    df["_rank"]  = rank_within_protein(df)

    key_to_rank: dict[tuple, float] = dict(zip(df["_key"], df["_rank"]))

    rr_floor = 1.0 / (top_k + 1)

    hitk_vals, hit5_vals, crr_vals = [], [], []
    for _, trow in targets.iterrows():
        rank = key_to_rank.get(trow["_key"], np.inf)
        hitk_vals.append(1.0 if rank <= top_k  else 0.0)
        hit5_vals.append(1.0 if rank <= HIT5_K else 0.0)
        if np.isfinite(rank):
            crr_vals.append(max(1.0 / rank, rr_floor))
        else:
            crr_vals.append(rr_floor)

    hitk = float(np.mean(hitk_vals)) if hitk_vals else 0.0
    hit5 = float(np.mean(hit5_vals)) if hit5_vals else 0.0
    crr  = float(np.mean(crr_vals))  if crr_vals  else 0.0
    obj  = 0.75 * hitk + 0.15 * hit5 + 0.10 * crr

    return {
        "hitk": hitk, "hit5": hit5, "clipped_mrr": crr,
        "objective": obj, "n_targets": len(hitk_vals),
    }


def cv_objective(
    folds: list,
    w_A: float, w_O: float, w_D: float, w_R: float,
    top_k: int,
) -> dict:
    """
    Evaluate weights on each validation fold and return mean metrics.
    Folds with no target entries are skipped.
    """
    fold_metrics = []
    for _train_df, val_df, val_targets in folds:
        if len(val_targets) == 0:
            continue
        m = evaluate_objective(val_df, val_targets, w_A, w_O, w_D, w_R, top_k)
        fold_metrics.append(m)

    if not fold_metrics:
        return {"hitk": 0.0, "hit5": 0.0, "clipped_mrr": 0.0,
                "objective": 0.0, "n_targets": 0}

    return {
        "hitk":        float(np.mean([m["hitk"]        for m in fold_metrics])),
        "hit5":        float(np.mean([m["hit5"]        for m in fold_metrics])),
        "clipped_mrr": float(np.mean([m["clipped_mrr"] for m in fold_metrics])),
        "objective":   float(np.mean([m["objective"]   for m in fold_metrics])),
        "n_targets":   sum(m["n_targets"] for m in fold_metrics),
    }


def saturation_penalty(df: pd.DataFrame, w_A: float, w_O: float,
                        w_D: float, w_R: float) -> float:
    """
    Returns a penalty in [0, 1] that is HIGH when individual score components
    are saturated at their ceiling (1.0), indicating thresholds are too small
    and the component contributes no discriminative information.

    Checks each non-zero-weight component independently: if >= 95% of residues
    have that scaled component == 1.0, that component is considered saturated.
    Returns the mean saturation fraction across all active components.

    Checking per-component rather than the combined score prevents solutions
    from gaming the penalty by suppressing one component (keeping the combined
    score below its theoretical ceiling) while saturating all others.
    """
    components = []
    if w_A > 0:
        components.append(df["A_scaled"])
    if w_O > 0:
        components.append(df["O_scaled"])
    if w_D > 0:
        components.append(df["D_scaled"])
    if w_R > 0:
        components.append(df["R_scaled"])
    if not components:
        return 0.0
    # Fraction of residues at ceiling per component, averaged across components
    sat_fracs = [(c >= 0.999).mean() for c in components]
    return float(np.mean(sat_fracs))


def cv_objective_thresholds(
    folds: list,
    w_A: float, w_O: float, w_D: float, w_R: float,
    thr_sasa: float, thr_orbital: float, thr_deprot: float, thr_react: float,
    top_k: int,
    sat_lambda: float = 0.5,
) -> dict:
    """
    Like cv_objective but re-scales each fold's val_df with the supplied
    thresholds before scoring.  Used when training thresholds with fixed weights.

    A saturation penalty (weighted by sat_lambda) is subtracted from the
    objective to prevent degenerate solutions where all thresholds collapse
    near zero, saturating every residue to the score ceiling and making
    rankings near-random.

    penalised_obj = raw_obj * (1 - sat_lambda * sat_fraction)

    sat_lambda=0.5 means a fully saturated solution (sat_frac=1.0) loses 50%
    of its apparent objective score.  Lower to 0.3 if the penalty seems too
    aggressive; raise toward 0.8 if degeneracy persists.
    """
    fold_metrics  = []
    fold_sat_frac = []

    for _train_df, val_df, val_targets in folds:
        if len(val_targets) == 0:
            continue
        val_df_rescaled = apply_thresholds(
            val_df, thr_sasa, thr_orbital, thr_deprot, thr_react
        )
        m = evaluate_objective(
            val_df_rescaled, val_targets, w_A, w_O, w_D, w_R, top_k
        )
        sat = saturation_penalty(val_df_rescaled, w_A, w_O, w_D, w_R)
        fold_metrics.append(m)
        fold_sat_frac.append(sat)

    if not fold_metrics:
        return {"hitk": 0.0, "hit5": 0.0, "clipped_mrr": 0.0,
                "objective": 0.0, "penalised_obj": 0.0,
                "sat_frac": 0.0, "n_targets": 0}

    raw_obj  = float(np.mean([m["objective"] for m in fold_metrics]))
    sat_frac = float(np.mean(fold_sat_frac))
    pen_obj  = raw_obj * (1.0 - sat_lambda * sat_frac)

    return {
        "hitk":          float(np.mean([m["hitk"]        for m in fold_metrics])),
        "hit5":          float(np.mean([m["hit5"]        for m in fold_metrics])),
        "clipped_mrr":   float(np.mean([m["clipped_mrr"] for m in fold_metrics])),
        "objective":     raw_obj,
        "penalised_obj": pen_obj,
        "sat_frac":      sat_frac,
        "n_targets":     sum(m["n_targets"] for m in fold_metrics),
    }


# ---------------------------------------------------------------------------
# Threshold-based TP/TN/FP/FN stats (full dataset)
# ---------------------------------------------------------------------------

def threshold_stats(df: pd.DataFrame, threshold: float) -> None:
    pred_pos = df["_score"] >= threshold
    labeled  = df["_labeled"]

    TP = int(( labeled &  pred_pos).sum())
    FN = int(( labeled & ~pred_pos).sum())
    FP = int((~labeled &  pred_pos).sum())
    TN = int((~labeled & ~pred_pos).sum())
    total = TP + TN + FP + FN

    recall      = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    precision   = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else float("nan"))
    accuracy    = (TP + TN) / total if total > 0 else float("nan")

    print(f"  Threshold (S ≥):              {threshold:.6f}")
    print(f"  Total residues evaluated      : {total:,}")
    print("-" * 62)
    print(f"  True  Positives (TP):  {TP:>9,}")
    print(f"  True  Negatives (TN):  {TN:>9,}")
    print(f"  False Positives (FP):  {FP:>9,}  ← not labeled, but S ≥ threshold")
    print(f"  False Negatives (FN):  {FN:>9,}  ← labeled, but S < threshold")
    print("-" * 62)
    print(f"  Sensitivity / Recall:  {recall:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    print(f"  Precision (PPV):       {precision:.4f}")
    print(f"  F1 Score:              {f1:.4f}")
    print(f"  Accuracy:              {accuracy:.4f}")


def propose_100pct_recall_threshold(df: pd.DataFrame) -> float:
    labeled_scores = df.loc[df["_labeled"], "_score"]
    return float(labeled_scores.min()) if not labeled_scores.empty else 0.0


# ---------------------------------------------------------------------------
# Bayesian Optimisation — weights
# ---------------------------------------------------------------------------

def run_bo(
    merged: pd.DataFrame,
    targets: pd.DataFrame,
    top_k: int,
    n_calls: int,
    n_folds: int,
) -> tuple[list[float], dict]:
    from skopt import gp_minimize
    from skopt.space import Real

    print(f"\n[BO] Building {n_folds}-fold CV splits (grouped by base PDB) ...")
    folds = make_cv_folds(merged, targets, n_folds)
    fold_sizes = [len(vt) for _, _, vt in folds]
    print(f"[BO] Fold target counts: {fold_sizes}")

    space = [
        Real(0.01, 5.0, name="w_A",  prior="log-uniform"),
        Real(0.01, 5.0, name="w_O",  prior="log-uniform"),
        Real(0.01, 5.0, name="w_D",  prior="log-uniform"),
        Real(0.01, 5.0, name="w_R",  prior="log-uniform"),
    ]

    # Chemistry-informed seeds
    x0 = [
        [1.0, 1.0, 1.0, 1.0],   # equal
        [2.0, 1.0, 1.5, 1.5],   # accessibility + deprot
        [1.0, 2.0, 1.0, 2.0],   # orbital + reactivity
        [1.5, 1.0, 2.0, 1.0],   # SASA + deprot
        [0.5, 3.0, 1.0, 2.5],   # orbital + reactivity heavy
    ]

    call_counter = [0]

    def objective(params):
        w_A, w_O, w_D, w_R = params
        metrics = cv_objective(folds, w_A, w_O, w_D, w_R, top_k)
        call_counter[0] += 1
        if call_counter[0] % 10 == 0 or call_counter[0] <= len(x0):
            print(f"  BO call {call_counter[0]:>3}/{n_calls}  "
                  f"cv_obj={metrics['objective']:.4f}  "
                  f"Hit@K={metrics['hitk']:.3f}  "
                  f"Hit@5={metrics['hit5']:.3f}  "
                  f"cMRR={metrics['clipped_mrr']:.4f}  "
                  f"w=[{w_A:.2f},{w_O:.2f},{w_D:.2f},{w_R:.2f}]")
        return -metrics["objective"]

    print(f"[BO] Running {n_calls} calls ({len(x0)} chemistry-informed seeds) ...")
    result = gp_minimize(
        objective, space,
        x0=x0,
        n_calls=n_calls,
        n_initial_points=len(x0),
        acq_func="EI",
        random_state=42,
        verbose=False,
    )

    best_params = result.x
    best_metrics = evaluate_objective(
        merged, targets,
        best_params[0], best_params[1], best_params[2], best_params[3],
        top_k,
    )
    cv_metrics = cv_objective(
        folds,
        best_params[0], best_params[1], best_params[2], best_params[3],
        top_k,
    )
    best_metrics["cv_objective"]   = cv_metrics["objective"]
    best_metrics["cv_hitk"]        = cv_metrics["hitk"]
    best_metrics["cv_hit5"]        = cv_metrics["hit5"]
    best_metrics["cv_clipped_mrr"] = cv_metrics["clipped_mrr"]

    return best_params, best_metrics


# ---------------------------------------------------------------------------
# Bayesian Optimisation — thresholds (fixed weights)
# ---------------------------------------------------------------------------

def run_bo_thresholds(
    merged: pd.DataFrame,
    targets: pd.DataFrame,
    w_A: float, w_O: float, w_D: float, w_R: float,
    thr_sasa_init: float,
    thr_deprot_init: float,
    thr_react_init: float,
    top_k: int,
    n_calls: int,
    n_folds: int,
) -> tuple[list[float], dict]:
    """
    Fix weights and use BO to find the best per-score thresholds.

    The orbital score is treated as a continuous value here (using
    scale_score like the other three) rather than a hard binary pass-through,
    which lets BO find the optimal cutoff.  If your orbital column is truly
    binary (0/1) the found threshold will naturally converge near 0.5.

    Search space for each threshold: [1e-4, 10.0] log-uniform.

    Returns:
        best_thresholds : [thr_sasa, thr_orbital, thr_deprot, thr_react]
        best_metrics    : full + CV metrics dict
    """
    from skopt import gp_minimize
    from skopt.space import Real

    print(f"\n[BO-THR] Building {n_folds}-fold CV splits (grouped by base PDB) ...")
    folds = make_cv_folds(merged, targets, n_folds)
    fold_sizes = [len(vt) for _, _, vt in folds]
    print(f"[BO-THR] Fold target counts: {fold_sizes}")
    print(f"[BO-THR] Fixed weights: "
          f"w_A={w_A:.4f}  w_O={w_O:.4f}  w_D={w_D:.4f}  w_R={w_R:.4f}")

    # Derive upper bounds from actual data so large-valued scores (e.g. SASA
    # in percent units) don't push seeds or the search space out of range.
    LO = 1e-4
    ub_sasa    = max(float(merged["_raw_A"].quantile(0.99)) * 2, thr_sasa_init  * 4, 1.0)
    ub_orbital = max(float(merged["_raw_O"].quantile(0.99)) * 2, 2.0)
    ub_deprot  = max(float(merged["_raw_D"].quantile(0.99)) * 2, thr_deprot_init * 4, 1.0)
    ub_react   = max(float(merged["_raw_R"].quantile(0.99)) * 2, thr_react_init  * 4, 1.0)

    print(f"[BO-THR] Search bounds (lo={LO}):")
    print(f"         thr_sasa    ∈ [{LO}, {ub_sasa:.4f}]")
    print(f"         thr_orbital ∈ [{LO}, {ub_orbital:.4f}]")
    print(f"         thr_deprot  ∈ [{LO}, {ub_deprot:.4f}]")
    print(f"         thr_react   ∈ [{LO}, {ub_react:.4f}]")

    space = [
        Real(LO, ub_sasa,    name="thr_sasa",    prior="log-uniform"),
        Real(LO, ub_orbital, name="thr_orbital", prior="log-uniform"),
        Real(LO, ub_deprot,  name="thr_deprot",  prior="log-uniform"),
        Real(LO, ub_react,   name="thr_react",   prior="log-uniform"),
    ]

    def clip_seed(s):
        """Clip a single seed point to lie strictly within the search bounds."""
        return [
            float(np.clip(s[0], LO, ub_sasa)),
            float(np.clip(s[1], LO, ub_orbital)),
            float(np.clip(s[2], LO, ub_deprot)),
            float(np.clip(s[3], LO, ub_react)),
        ]

    # Seeds: start from CLI-supplied values plus a handful of variants
    raw_seeds = [
        [thr_sasa_init,      0.5,             thr_deprot_init,     thr_react_init],
        [thr_sasa_init * 2,  0.5,             thr_deprot_init * 2, thr_react_init * 2],
        [thr_sasa_init / 2,  0.5,             thr_deprot_init / 2, thr_react_init / 2],
        [0.1,                0.5,             0.3,                 0.3],
        [0.3,                0.5,             0.7,                 0.7],
    ]
    x0 = [clip_seed(s) for s in raw_seeds]

    # Remove exact duplicates that arise when init values are the same
    seen = []
    x0_dedup = []
    for s in x0:
        key = tuple(round(v, 6) for v in s)
        if key not in seen:
            seen.append(key)
            x0_dedup.append(s)
    x0 = x0_dedup

    call_counter = [0]

    def objective(params):
        thr_s, thr_o, thr_d, thr_r = params
        metrics = cv_objective_thresholds(
            folds, w_A, w_O, w_D, w_R,
            thr_s, thr_o, thr_d, thr_r,
            top_k,
        )
        call_counter[0] += 1
        if call_counter[0] % 10 == 0 or call_counter[0] <= len(x0):
            print(f"  BO-THR call {call_counter[0]:>3}/{n_calls}  "
                  f"pen_obj={metrics['penalised_obj']:.4f}  "
                  f"raw_obj={metrics['objective']:.4f}  "
                  f"sat={metrics['sat_frac']:.3f}  "
                  f"Hit@K={metrics['hitk']:.3f}  "
                  f"Hit@5={metrics['hit5']:.3f}  "
                  f"cMRR={metrics['clipped_mrr']:.4f}  "
                  f"thr=[{thr_s:.4f},{thr_o:.4f},{thr_d:.4f},{thr_r:.4f}]")
        return -metrics["penalised_obj"]

    print(f"[BO-THR] Running {n_calls} calls ({len(x0)} seeds) ...")
    result = gp_minimize(
        objective, space,
        x0=x0,
        n_calls=n_calls,
        n_initial_points=len(x0),
        acq_func="EI",
        random_state=42,
        verbose=False,
    )

    best_thr = result.x  # [thr_sasa, thr_orbital, thr_deprot, thr_react]

    # Apply best thresholds to full dataset then evaluate
    merged_rescaled = apply_thresholds(merged, *best_thr)
    best_metrics = evaluate_objective(
        merged_rescaled, targets, w_A, w_O, w_D, w_R, top_k
    )

    # CV performance at best thresholds
    cv_metrics = cv_objective_thresholds(
        folds, w_A, w_O, w_D, w_R, *best_thr, top_k
    )
    best_metrics["cv_objective"]     = cv_metrics["objective"]
    best_metrics["cv_penalised_obj"] = cv_metrics["penalised_obj"]
    best_metrics["cv_sat_frac"]      = cv_metrics["sat_frac"]
    best_metrics["cv_hitk"]          = cv_metrics["hitk"]
    best_metrics["cv_hit5"]          = cv_metrics["hit5"]
    best_metrics["cv_clipped_mrr"]   = cv_metrics["clipped_mrr"]

    return best_thr, best_metrics, merged_rescaled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    training_path: str,
    labels_path: str,
    reactivity_path: str,
    thr_sasa: float,
    thr_deprot: float,
    thr_react: float,
    top_k: int,
    n_calls: int,
    n_folds: int,
    threshold_score: Optional[float],
    match_warheads: bool,
    verbose: bool,
    export_scores: Optional[str],
    w_A: Optional[float],
    w_O: Optional[float],
    w_D: Optional[float],
    w_R: Optional[float],
    train_thresholds: bool,
) -> None:

    print("=" * 62)
    print("  COVALENT SITE RANK OPTIMIZER")
    print("=" * 62)
    print(f"  Thresholds  SASA/Deprot/React : {thr_sasa} / {thr_deprot} / {thr_react}")
    print(f"  Top-K                          : {top_k}")
    print(f"  CV folds                       : {n_folds}")
    print(f"  Warhead matching               : {'enabled' if match_warheads else 'disabled'}")

    if train_thresholds:
        if not all(v is not None for v in [w_A, w_O, w_D, w_R]):
            sys.exit(
                "[ERROR] --train-thresholds requires all four weight flags to be set: "
                "--w-sasa, --w-orbital, --w-deprot, --w-react"
            )
        print(f"  Mode                           : THRESHOLD TRAINING (weights fixed)")
    elif all(v is not None for v in [w_A, w_O, w_D, w_R]):
        print(f"  Mode                           : MANUAL WEIGHTS (BO skipped)")
    else:
        print(f"  Mode                           : WEIGHT OPTIMISATION (BO)")

    merged, targets = load_and_join(
        training_path, labels_path, reactivity_path,
        thr_sasa, thr_deprot, thr_react,
        match_warheads,
    )

    n_proteins   = merged["_name_upper"].nunique()
    n_base_pdbs  = merged["_base_pdb"].nunique()
    n_target_ent = len(targets)
    print(f"[INFO] Unique Names (with suffixes)  : {n_proteins:,}")
    print(f"[INFO] Unique base PDBs              : {n_base_pdbs:,}")
    print(f"[INFO] Target site entries (for MRR) : {n_target_ent:,}")

    # -----------------------------------------------------------------------
    # Branch on mode
    # -----------------------------------------------------------------------

    if train_thresholds:
        # --- Threshold-training mode ---
        best_thr, best_metrics, merged = run_bo_thresholds(
            merged, targets,
            w_A, w_O, w_D, w_R,
            thr_sasa_init=thr_sasa,
            thr_deprot_init=thr_deprot,
            thr_react_init=thr_react,
            top_k=top_k,
            n_calls=n_calls,
            n_folds=n_folds,
        )
        thr_sasa_out, thr_orbital_out, thr_deprot_out, thr_react_out = best_thr
        w_A_b, w_O_b, w_D_b, w_R_b = w_A, w_O, w_D, w_R

        merged["_score"] = compute_combined_score(merged, w_A_b, w_O_b, w_D_b, w_R_b)
        merged["_rank"]  = rank_within_protein(merged)

        recall_threshold = propose_100pct_recall_threshold(merged)

        print("\n" + "=" * 62)
        print("  OPTIMISED THRESHOLDS  (weights fixed)")
        print("=" * 62)
        print(f"  w_A (Accessibility / SASA)    : {w_A_b:.4f}  [fixed]")
        print(f"  w_O (Orbital / Binary)        : {w_O_b:.4f}  [fixed]")
        print(f"  w_D (Deprotonation prob)      : {w_D_b:.4f}  [fixed]")
        print(f"  w_R (Reactivity score)        : {w_R_b:.4f}  [fixed]")
        print()
        print(f"  thr_sasa                      : {thr_sasa_out:.6f}")
        print(f"  thr_orbital                   : {thr_orbital_out:.6f}")
        print(f"  thr_deprot                    : {thr_deprot_out:.6f}")
        print(f"  thr_react                     : {thr_react_out:.6f}")
        print()
        print(f"  Saturation fraction (CV)      : {best_metrics.get('cv_sat_frac', float('nan')):.4f}  "
              f"(fraction of residues at score ceiling; lower = better differentiation)")
        print(f"  CV penalised objective        : {best_metrics.get('cv_penalised_obj', float('nan')):.4f}")

    elif all(v is not None for v in [w_A, w_O, w_D, w_R]):
        # --- Manual weights mode ---
        best_metrics = evaluate_objective(
            merged, targets, w_A, w_O, w_D, w_R, top_k
        )
        folds = make_cv_folds(merged, targets, n_folds)
        cv_m  = cv_objective(folds, w_A, w_O, w_D, w_R, top_k)
        best_metrics.update({
            "cv_objective":   cv_m["objective"],
            "cv_hitk":        cv_m["hitk"],
            "cv_hit5":        cv_m["hit5"],
            "cv_clipped_mrr": cv_m["clipped_mrr"],
        })
        w_A_b, w_O_b, w_D_b, w_R_b = w_A, w_O, w_D, w_R

        merged["_score"] = compute_combined_score(merged, w_A_b, w_O_b, w_D_b, w_R_b)
        merged["_rank"]  = rank_within_protein(merged)
        recall_threshold = propose_100pct_recall_threshold(merged)

        print("\n" + "=" * 62)
        print("  MANUAL WEIGHTS")
        print("=" * 62)
        print(f"  w_A (Accessibility / SASA)    : {w_A_b:.4f}")
        print(f"  w_O (Orbital / Binary)        : {w_O_b:.4f}")
        print(f"  w_D (Deprotonation prob)      : {w_D_b:.4f}")
        print(f"  w_R (Reactivity score)        : {w_R_b:.4f}")

    else:
        # --- Weight optimisation mode (BO) ---
        best_params, best_metrics = run_bo(merged, targets, top_k, n_calls, n_folds)
        w_A_b, w_O_b, w_D_b, w_R_b = best_params

        merged["_score"] = compute_combined_score(merged, w_A_b, w_O_b, w_D_b, w_R_b)
        merged["_rank"]  = rank_within_protein(merged)
        recall_threshold = propose_100pct_recall_threshold(merged)

        print("\n" + "=" * 62)
        print("  OPTIMISED WEIGHTS")
        print("=" * 62)
        print(f"  w_A (Accessibility / SASA)    : {w_A_b:.4f}")
        print(f"  w_O (Orbital / Binary)        : {w_O_b:.4f}")
        print(f"  w_D (Deprotonation prob)      : {w_D_b:.4f}")
        print(f"  w_R (Reactivity score)        : {w_R_b:.4f}")

    # ---- shared performance summary (all modes) --------------------------
    print()
    print(f"  --- Full dataset performance ---")
    print(f"  Hit@{top_k:<2}                        : {best_metrics['hitk']:.4f}")
    print(f"  Hit@5                         : {best_metrics['hit5']:.4f}")
    print(f"  Clipped MRR                   : {best_metrics['clipped_mrr']:.4f}")
    print(f"  Objective                     : {best_metrics['objective']:.4f}")
    print()
    print(f"  --- {n_folds}-fold CV held-out performance ---")
    print(f"  CV Hit@{top_k:<2}                      : {best_metrics['cv_hitk']:.4f}")
    print(f"  CV Hit@5                      : {best_metrics['cv_hit5']:.4f}")
    print(f"  CV Clipped MRR                : {best_metrics['cv_clipped_mrr']:.4f}")
    print(f"  CV Objective                  : {best_metrics['cv_objective']:.4f}")
    print(f"  N target entries              : {best_metrics['n_targets']}")
    print()
    print(f"  Proposed 100% recall cutoff   : {recall_threshold:.6f}")
    print("=" * 62)

    eval_threshold = threshold_score if threshold_score is not None else recall_threshold
    print(f"\n{'FIXED THRESHOLD STATS' if threshold_score else 'STATS AT 100% RECALL THRESHOLD'}")
    print("=" * 62)
    threshold_stats(merged, eval_threshold)
    print("=" * 62)

    if verbose:
        print(f"\n[VERBOSE] Target site ranks (best weights/thresholds, full dataset):")
        key_to_rank  = dict(zip(merged["_key"], merged["_rank"]))
        key_to_score = dict(zip(merged["_key"], merged["_score"]))
        rows = []
        for _, trow in targets.iterrows():
            key  = trow["_key"]
            rank = key_to_rank.get(key, np.inf)
            rr_floor = 1.0 / (top_k + 1)
            clipped_rr = max(1.0 / rank, rr_floor) if np.isfinite(rank) else rr_floor
            rows.append({
                "Name":        key[0],
                "Residue":     key[1],
                "Chain":       key[2],
                "ResNum":      key[3],
                "Score":       round(key_to_score.get(key, np.nan), 4),
                "Rank":        rank,
                "Hit@5":       rank <= 5,
                f"Hit@{top_k}": rank <= top_k,
                "Clipped_RR":  round(clipped_rr, 4),
            })
        vdf = pd.DataFrame(rows).sort_values("Rank")
        print(vdf.to_string(index=False))

    if export_scores:
        export_cols = [c for c in [
            "Name", "Residue", "Chain", "ResNum",
            "_score", "_rank", "_labeled", "_base_pdb",
            "A_scaled", "O_scaled", "D_scaled", "R_scaled",
            COL_SASA, COL_ORBITAL, COL_DEPROT, COL_R,
        ] if c in merged.columns]
        merged[export_cols].rename(columns={
            "_score": "Score", "_rank": "Rank",
            "_labeled": "Labeled", "_base_pdb": "Base_PDB",
        }).to_csv(export_scores, index=False)
        print(f"\n[INFO] All residue scores exported to: {export_scores}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian-optimised covalent site ranker with k-fold CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--training",          required=True)
    parser.add_argument("--labels",            required=True)
    parser.add_argument("--reactivity",        required=True)
    parser.add_argument("--threshold-sasa",    type=float, required=True)
    parser.add_argument("--threshold-deprot",  type=float, required=True)
    parser.add_argument("--threshold-react",   type=float, required=True)
    parser.add_argument("--top-k",             type=int,   default=5)
    parser.add_argument("--n-calls",           type=int,   default=60,
                        help="Total BO evaluations (default: 60).")
    parser.add_argument("--n-folds",           type=int,   default=10,
                        help="Number of CV folds (default: 10). Proteins sharing "
                             "the same base PDB (e.g. 4G5J, 4G5J-2) are always "
                             "kept in the same fold.")
    parser.add_argument("--threshold-score",   type=float, default=None,
                        help="Evaluate TP/FP/etc at this fixed score threshold "
                             "instead of the 100%% recall cutoff.")
    parser.add_argument("--match-warheads",    action="store_true",
                        help="Match training 'Warhead' against labels "
                             "'Frankenstein_Warhead' (comma-separated OR logic).")
    parser.add_argument("--verbose",           action="store_true")
    parser.add_argument("--export-scores",     default=None, metavar="PATH")

    # Manual weight overrides — all four must be supplied to skip BO
    parser.add_argument("--w-sasa",    type=float, default=None)
    parser.add_argument("--w-orbital", type=float, default=None)
    parser.add_argument("--w-deprot",  type=float, default=None)
    parser.add_argument("--w-react",   type=float, default=None)

    # Threshold training mode
    parser.add_argument(
        "--train-thresholds",
        action="store_true",
        help=(
            "Instead of optimising weights, fix weights (requires all --w-* "
            "flags) and use BO to find the best per-score scaling thresholds "
            "(thr_sasa, thr_orbital, thr_deprot, thr_react). The --threshold-* "
            "CLI values are used only as starting-point seeds for the search."
        ),
    )

    args = parser.parse_args()

    run(
        training_path    = args.training,
        labels_path      = args.labels,
        reactivity_path  = args.reactivity,
        thr_sasa         = args.threshold_sasa,
        thr_deprot       = args.threshold_deprot,
        thr_react        = args.threshold_react,
        top_k            = args.top_k,
        n_calls          = args.n_calls,
        n_folds          = args.n_folds,
        threshold_score  = args.threshold_score,
        match_warheads   = args.match_warheads,
        verbose          = args.verbose,
        export_scores    = args.export_scores,
        w_A              = args.w_sasa,
        w_O              = args.w_orbital,
        w_D              = args.w_deprot,
        w_R              = args.w_react,
        train_thresholds = args.train_thresholds,
    )


if __name__ == "__main__":
    main()