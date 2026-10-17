"""
reactivity_lgbm_ranker.py

Learns a LightGBM Learning-to-Rank model that ranks nucleophilic amino-acid
residue types within each protein × warhead query group, pushing the labeled
(covalent-hit) residue to rank 1 and ensuring it appears within the top-K.

Replaces the Bayesian-Optimisation weight search in reactivity_rank_optimizer.py
with a non-linear LightGBM lambdarank model while preserving all surrounding
infrastructure: prefilters, grouped CV splits, warhead matching, three-variant
Hit@K metrics, breakdown analysis, and score/breakdown export.

RELEVANCE LABEL SCHEME
───────────────────────
  2  →  labeled covalent-hit residue (push to rank 1)
  0  →  all other candidates

LightGBM lambdarank with label_gain=[0,0,1] trains NDCG so that only the
label-2 item matters; this is equivalent to asking for the labeled residue
to be ranked as high as possible.

QUERY GROUPS
────────────
Each (Name, Warhead) pair is one query.  Within a query the model sees up
to 6 rows (one per candidate residue type that passed prefilters).  The
model is trained across all queries; CV is protein-grouped so no base PDB
leaks between train and validation folds.

FEATURE ENGINEERING
────────────────────
Raw feature values are used directly (no within-group min-max scaling
needed — LightGBM is tree-based and invariant to monotone transforms).
Optional: within-group z-score normalisation is applied when
--normalize-features is passed, which can help when features have very
different scales but is generally not required for GBDT.

OBJECTIVE (reported, not optimised by CV directly — LightGBM does that)
────────────────────────────────────────────────────────────────────────
    Hit@K (all)       – targets whose labeled residue was prefiltered out
                        are counted as misses.
    Hit@K (rankable)  – computed only over targets whose labeled residue
                        survived the prefilter.
    % targets filtered – breakdown same as original script.

Usage
─────
    python reactivity_lgbm_ranker.py \\
        --training   training.csv \\
        --labels     labels.csv \\
        --features   Fukui_Deprotonated Nucleophilicity_Index_Deprotonated \\
                     HOMO_LUMO_Gap_Deprotonated \\
        --feature-direction higher higher lower \\
        --prefilter  Relative_SASA gte 0.2  Deprotonated_Fraction gt 0.5 \\
        --top-k 3 \\
        [--n-folds 5] \\
        [--n-estimators 300] [--num-leaves 31] [--learning-rate 0.05] \\
        [--normalize-features] \\
        [--match-warheads] [--verbose] \\
        [--export-scores PATH] [--export-breakdown PATH] \\
        [--export-model PATH]


        python reactivity_lgbm_analysis.py     --training  training.csv     --labels    batch_pdbs_bo_fixed.csv --features  Nucleophilicity_Index_Deprotonated HOMO_LUMO_Gap_Deprotonated Fukui_Deprotonated Nucleophilicity_Index_Deprotonated Partial_Charge_Deprotonated      --feature-direction higher lower higher higher higher  --top-k 3     --n-folds 5     --match-warheads   --prefilter Rel_Side_SASA gte 12 deprotonation_prob gte 0.14 --export-breakdown lgbm_reactivity_breakdown.csv --export-model lgbm_model.pkl
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── try importing lightgbm early so we fail fast ──────────────────────────────
try:
    import lightgbm as lgb
except ImportError:
    sys.exit(
        "[ERROR] lightgbm is not installed.\n"
        "        Install with:  pip install lightgbm"
    )

DEFAULT_RESIDUE_TYPES = {"CYS", "SER", "THR", "TYR", "LYS", "HIS"}

FILTER_OPS = {
    "gt":  lambda a, b: a >  b,
    "lt":  lambda a, b: a <  b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "eq":  lambda a, b: a == b,
}

# Relevance label for the covalent-hit residue.
# 2 = hit, 0 = non-hit.  label_gain will be set to [0, 0, 1] so only
# label-2 items contribute to NDCG.
RELEVANCE_HIT  = 2
RELEVANCE_MISS = 0

EPSILON = 0.05   # rank-bonus coefficient (same as original)


# ─────────────────────────────────────────────────────────────
# Helpers  (identical to original where possible)
# ─────────────────────────────────────────────────────────────

def load_csv(path: str, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Cannot find {label} file: {path}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read {label} file: {e}")
    df.columns = df.columns.str.strip()
    return df


def base_pdb(name: str) -> str:
    return re.sub(r"-\d+$", "", str(name).strip().upper())


def parse_frankenstein_warheads(raw: str) -> set[str]:
    return {w.strip().lower() for w in str(raw).split(",") if w.strip()}


def parse_prefilters(tokens: list[str]) -> list[tuple[str, str, float]]:
    if len(tokens) % 3 != 0:
        sys.exit(
            f"[ERROR] --prefilter requires groups of 3 tokens: COL OP VALUE. "
            f"Got {len(tokens)} token(s): {tokens}"
        )
    filters = []
    for i in range(0, len(tokens), 3):
        col, op, val_str = tokens[i], tokens[i + 1], tokens[i + 2]
        if op not in FILTER_OPS:
            sys.exit(f"[ERROR] Unknown operator '{op}'. Valid: {', '.join(FILTER_OPS)}")
        try:
            val = float(val_str)
        except ValueError:
            sys.exit(f"[ERROR] Prefilter value must be numeric, got '{val_str}'")
        filters.append((col, op, val))
    return filters


def apply_prefilters(df: pd.DataFrame,
                     filters: list[tuple[str, str, float]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, op, val in filters:
        if col not in df.columns:
            sys.exit(f"[ERROR] Prefilter column '{col}' not found in training CSV.")
        numeric = pd.to_numeric(df[col], errors="coerce")
        mask &= FILTER_OPS[op](numeric, val)
    return df[mask].copy()


def random_hitk(n_candidates: int, k: int) -> float:
    if n_candidates <= 0:
        return 0.0
    return min(k, n_candidates) / n_candidates


# ─────────────────────────────────────────────────────────────
# Data loading  (identical logic to original)
# ─────────────────────────────────────────────────────────────

def load_and_prepare(
    training_path: str,
    labels_path: str,
    feature_cols: list[str],
    feature_directions: list[bool],
    residue_types: set[str],
    match_warheads: bool,
    prefilters: list[tuple[str, str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        records : one row per (Name, Residue, Warhead) that passed all
                  prefilters, with _labeled, _base_pdb, _n_candidates,
                  and _relevance columns.
        targets : one row per label entry with _prefiltered_out flag.
    """
    train  = load_csv(training_path, "training")
    labels = load_csv(labels_path,   "labels")

    req_train = {"Name", "Residue", "Warhead"} | set(feature_cols)
    missing = req_train - set(train.columns)
    if missing:
        sys.exit(f"[ERROR] Training CSV missing columns: {missing}")

    req_labels = {"Name", "Residue"}
    if match_warheads:
        req_labels.add("Frankenstein_Warhead")
    missing = req_labels - set(labels.columns)
    if missing:
        sys.exit(f"[ERROR] Labels CSV missing columns: {missing}")

    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    train.dropna(subset=feature_cols, inplace=True)

    train["_res_upper"] = train["Residue"].str.strip().str.upper()
    before = len(train)
    train = train[train["_res_upper"].isin(residue_types)].copy()
    print(f"[INFO] Residue filter ({', '.join(sorted(residue_types))}): "
          f"{before:,} → {len(train):,} rows")

    key_cols = ["Name", "_res_upper", "Warhead"]
    train_dedup = train.drop_duplicates(subset=key_cols, keep="first").copy()
    print(f"[INFO] After dedup (Name × Residue × Warhead): {len(train_dedup):,} rows")

    train_dedup["_name_upper"]    = train_dedup["Name"].str.strip().str.upper()
    train_dedup["_warhead_lower"] = train_dedup["Warhead"].str.strip().str.lower()
    train_dedup["_base_pdb"]      = train_dedup["_name_upper"].map(base_pdb)

    # Track which keys existed before filtering (for prefiltered_out detection)
    pre_filter_keys: set[tuple] = set(
        zip(train_dedup["_name_upper"],
            train_dedup["_res_upper"],
            train_dedup["_warhead_lower"])
    )

    if prefilters:
        filter_desc = "  +  ".join(f"{c} {o} {v}" for c, o, v in prefilters)
        print(f"[INFO] Prefilters: {filter_desc}")
        before_pf = len(train_dedup)
        train_dedup = apply_prefilters(train_dedup, prefilters)
        print(f"[INFO] After prefilter: {before_pf:,} → {len(train_dedup):,} rows")

    # Candidate counts per group after filtering
    group_counts = (
        train_dedup.groupby(["_name_upper", "_warhead_lower"])
        .size()
        .reset_index(name="_n_candidates")
    )
    train_dedup = train_dedup.merge(
        group_counts, on=["_name_upper", "_warhead_lower"], how="left"
    )
    train_dedup["_n_candidates"] = train_dedup["_n_candidates"].fillna(0).astype(int)

    post_filter_keys: set[tuple] = set(
        zip(train_dedup["_name_upper"],
            train_dedup["_res_upper"],
            train_dedup["_warhead_lower"])
    )

    # ── build label target rows ───────────────────────────────────────────
    target_rows = []
    labels["_name_upper"] = labels["Name"].str.strip().str.upper()
    labels["_res_upper"]  = labels["Residue"].str.strip().str.upper()

    for _, row in labels.iterrows():
        name = row["_name_upper"]
        res  = row["_res_upper"]
        if res not in residue_types:
            continue
        wh_set: set[str] = set()
        if match_warheads and "Frankenstein_Warhead" in labels.columns:
            wh_set = parse_frankenstein_warheads(row.get("Frankenstein_Warhead", ""))
        elif "Warhead" in labels.columns:
            wh_set = {str(row["Warhead"]).strip().lower()}

        was_present = any(
            (name, res, wh) in pre_filter_keys
            for wh in (wh_set if wh_set else [""])
        ) if prefilters else True

        survived = any(
            (name, res, wh) in post_filter_keys
            for wh in (wh_set if wh_set else [""])
        )

        prefiltered_out = was_present and not survived

        target_rows.append({
            "_name":             name,
            "_residue":          res,
            "_warhead_set":      wh_set,
            "_base_pdb":         base_pdb(name),
            "_prefiltered_out":  prefiltered_out,
        })

    targets = pd.DataFrame(target_rows)
    if targets.empty:
        sys.exit("[ERROR] No label entries remained after residue-type filter.")

    # ── mark labeled rows & assign relevance scores ───────────────────────
    label_name_res: dict[tuple, list[set]] = {}
    for _, t in targets.iterrows():
        k = (t["_name"], t["_residue"])
        label_name_res.setdefault(k, []).append(t["_warhead_set"])

    def _is_labeled(row) -> bool:
        k = (row["_name_upper"], row["_res_upper"])
        if k not in label_name_res:
            return False
        if not match_warheads:
            return True
        train_wh = row["_warhead_lower"]
        for wh_set in label_name_res[k]:
            if not wh_set or train_wh in wh_set:
                return True
        return False

    train_dedup["_labeled"]    = train_dedup.apply(_is_labeled, axis=1)
    train_dedup["_relevance"]  = train_dedup["_labeled"].map(
        {True: RELEVANCE_HIT, False: RELEVANCE_MISS}
    )

    # ── Drop groups whose labeled residue was prefiltered out ─────────────
    # These groups have no positive label — keeping them adds label-0 noise
    # from unresolvable groups that the model cannot learn from.
    if prefilters:
        pf_out_targets = targets[targets["_prefiltered_out"]]
        excluded_groups: set[tuple[str, str]] = set()
        for _, t in pf_out_targets.iterrows():
            name = t["_name"]
            wh_set = t["_warhead_set"]
            if wh_set:
                for wh in wh_set:
                    excluded_groups.add((name, wh))
            else:
                whs = train_dedup.loc[
                    train_dedup["_name_upper"] == name, "_warhead_lower"
                ].unique()
                for wh in whs:
                    excluded_groups.add((name, wh))

        if excluded_groups:
            before_excl = len(train_dedup)
            keep_mask = ~train_dedup.apply(
                lambda r: (r["_name_upper"], r["_warhead_lower"]) in excluded_groups,
                axis=1,
            )
            train_dedup = train_dedup[keep_mask].copy()
            # Recompute candidate counts now that groups have been removed
            group_counts = (
                train_dedup.groupby(["_name_upper", "_warhead_lower"])
                .size()
                .reset_index(name="_n_candidates")
            )
            train_dedup = train_dedup.drop(columns=["_n_candidates"]).merge(
                group_counts, on=["_name_upper", "_warhead_lower"], how="left"
            )
            train_dedup["_n_candidates"] = train_dedup["_n_candidates"].fillna(0).astype(int)
            print(
                f"[INFO] Excluded {len(excluded_groups)} group(s) whose labeled residue "
                f"was prefiltered out: {before_excl:,} → {len(train_dedup):,} rows"
            )


    n_labeled   = train_dedup["_labeled"].sum()
    n_pf_out    = targets["_prefiltered_out"].sum()
    n_tgt       = len(targets)
    pf_pct      = 100.0 * n_pf_out / n_tgt if n_tgt > 0 else 0.0

    print(f"[INFO] Label target entries              : {n_tgt:,}")
    print(f"[INFO] Labeled training rows (matched)   : {n_labeled:,}")
    if prefilters:
        print(f"[INFO] Targets prefiltered out           : "
              f"{n_pf_out:,} / {n_tgt:,}  ({pf_pct:.1f}%)")
        if n_pf_out > 0 and "_warhead_set" in targets.columns:
            wh_rows = []
            for _, t in targets.iterrows():
                wh_label = ",".join(sorted(t["_warhead_set"])) if t["_warhead_set"] else "unknown"
                wh_rows.append({"warhead": wh_label, "filtered_out": t["_prefiltered_out"]})
            wh_df = pd.DataFrame(wh_rows)
            wh_summary = (
                wh_df.groupby("warhead")["filtered_out"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "n_filtered", "count": "n_total"})
            )
            wh_summary["pct"] = (
                100.0 * wh_summary["n_filtered"] / wh_summary["n_total"]
            ).round(1)
            wh_summary = wh_summary[wh_summary["n_filtered"] > 0].sort_values(
                "pct", ascending=False
            )
            if not wh_summary.empty:
                print(f"[INFO] Filter-out rate by warhead:")
                for wh, row2 in wh_summary.iterrows():
                    print(f"       {wh:<40} "
                          f"{int(row2['n_filtered']):>4}/{int(row2['n_total']):<4} "
                          f"({row2['pct']:.1f}%)")
    else:
        unmatched = n_tgt - n_labeled
        if unmatched > 0:
            print(f"[WARN] Up to {unmatched} label entries had no training match "
                  f"(warhead mismatch or residue absent in protein)")

    return train_dedup, targets


# ─────────────────────────────────────────────────────────────
# Feature matrix helpers
# ─────────────────────────────────────────────────────────────

def _flip_lower_better(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
) -> pd.DataFrame:
    """
    For 'lower is better' features, negate so LightGBM always sees
    higher = better.  Trees are invariant to this but it makes feature
    importance signs consistent.
    """
    df = df.copy()
    for col, hib in zip(feature_cols, feature_directions):
        if not hib:
            df[col] = -df[col]
    return df


def _group_znorm(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Optional within-query z-score normalisation."""
    df = df.copy()
    for col in feature_cols:
        gstd = df.groupby(["_name_upper", "_warhead_lower"])[col].transform("std")
        gmean = df.groupby(["_name_upper", "_warhead_lower"])[col].transform("mean")
        gstd = gstd.replace(0, 1)          # avoid division by zero
        df[col] = (df[col] - gmean) / gstd
    return df


def build_X_y_groups(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X, y (relevance), query_group_sizes, row_index_in_df.
    Rows are sorted by (Name, Warhead) so query groups are contiguous,
    as required by LightGBM Dataset.
    """
    df_s = df.sort_values(["_name_upper", "_warhead_lower"]).copy()
    df_s = _flip_lower_better(df_s, feature_cols, feature_directions)
    if normalize:
        df_s = _group_znorm(df_s, feature_cols)

    X   = df_s[feature_cols].values.astype(np.float32)
    y   = df_s["_relevance"].values.astype(np.int32)
    idx = df_s.index.values

    # LightGBM needs group sizes as an array of counts per query
    groups = (
        df_s.groupby(["_name_upper", "_warhead_lower"], sort=False)
        .size()
        .values
        .astype(np.int32)
    )
    return X, y, groups, idx


# ─────────────────────────────────────────────────────────────
# CV fold builder  (identical to original)
# ─────────────────────────────────────────────────────────────

def make_cv_folds(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    n_folds: int,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Returns list of (train_df, val_df, train_targets, val_targets).
    Base PDBs are grouped — the same protein never spans train/val.
    """
    rng = np.random.default_rng(seed)
    base_pdbs = sorted(records["_base_pdb"].unique())
    rng.shuffle(base_pdbs)
    fold_of = {bp: i % n_folds for i, bp in enumerate(base_pdbs)}

    records = records.copy()
    records["_fold"] = records["_base_pdb"].map(fold_of)
    tgt_bases = targets["_base_pdb"].values

    folds = []
    for f in range(n_folds):
        val_mask  = records["_fold"] == f
        val_df    = records[val_mask].copy()
        train_df  = records[~val_mask].copy()
        val_base  = set(val_df["_base_pdb"].unique())
        val_tgt   = targets[pd.Series(tgt_bases).isin(val_base).values].copy()
        train_tgt = targets[~pd.Series(tgt_bases).isin(val_base).values].copy()
        folds.append((train_df, val_df, train_tgt, val_tgt))
    return folds


# ─────────────────────────────────────────────────────────────
# Ranking and evaluation
# ─────────────────────────────────────────────────────────────

def rank_by_score(records: pd.DataFrame,
                  score_col: str = "_score") -> pd.Series:
    """Rank within (Name, Warhead) — higher score → rank 1."""
    return records.groupby(["_name_upper", "_warhead_lower"])[score_col]\
                  .rank(method="min", ascending=False)


def evaluate(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    top_k: int,
    match_warheads: bool,
) -> dict:
    """
    Identical metric logic to original evaluate().
    Expects records to have '_score' and '_rank' columns already set.
    """
    hitk_all_vals  = []
    hitk_rank_vals = []
    bonus_vals     = []
    rand_all_vals  = []
    rand_rank_vals = []
    n_filtered_out = 0

    for _, trow in targets.iterrows():
        name   = trow["_name"]
        res    = trow["_residue"]
        wh_set = trow["_warhead_set"]
        pf_out = trow.get("_prefiltered_out", False)

        mask  = records["_name_upper"] == name
        mask &= records["_res_upper"]  == res
        if match_warheads and wh_set:
            mask &= records["_warhead_lower"].isin(wh_set)

        matched = records[mask]

        if pf_out:
            n_filtered_out += 1
            hitk_all_vals.append(0.0)
            rand_all_vals.append(0.0)
            continue

        if matched.empty:
            hitk_all_vals.append(0.0)
            rand_all_vals.append(0.0)
            #hitk_rank_vals.append(0.0)
            #bonus_vals.append(0.0)
            #rand_rank_vals.append(0.0)
            continue

        rank    = float(matched["_rank"].min())
        n_cands = int(matched["_n_candidates"].iloc[0])
        rand_p  = random_hitk(n_cands, top_k)
        is_hit  = rank <= top_k

        hitk_all_vals.append(1.0 if is_hit else 0.0)
        rand_all_vals.append(rand_p)
        hitk_rank_vals.append(1.0 if is_hit else 0.0)
        bonus_vals.append(1.0 / rank if is_hit else 0.0)
        rand_rank_vals.append(rand_p)

    hitk_all  = float(np.mean(hitk_all_vals))  if hitk_all_vals  else 0.0
    hitk_rank = float(np.mean(hitk_rank_vals)) if hitk_rank_vals else 0.0
    bonus     = float(np.mean(bonus_vals))     if bonus_vals     else 0.0
    rand_all  = float(np.mean(rand_all_vals))  if rand_all_vals  else 0.0
    rand_rank = float(np.mean(rand_rank_vals)) if rand_rank_vals else 0.0
    n_total   = len(hitk_all_vals) + n_filtered_out
    pct_filt  = 100.0 * n_filtered_out / n_total if n_total > 0 else 0.0

    return {
        "hitk_all":                   hitk_all,
        "hitk_rankable":              hitk_rank,
        "rank_bonus":                 bonus,
        "objective":                  hitk_rank + EPSILON * bonus,
        "n_targets":                  n_total,
        "n_rankable":                 len(hitk_rank_vals),
        "n_filtered_out":             n_filtered_out,
        "pct_filtered":               pct_filt,
        "mean_random_hitk_all":       rand_all,
        "mean_random_hitk_rankable":  rand_rank,
    }


# ─────────────────────────────────────────────────────────────
# LightGBM training
# ─────────────────────────────────────────────────────────────

def _lgbm_params(n_estimators: int, num_leaves: int,
                 learning_rate: float, min_child_samples: int) -> dict:
    return {
        "objective":         "lambdarank",
        "metric":            "ndcg",
        "ndcg_eval_at":      [1, 3, 5],
        # Only label 2 contributes to NDCG gain — forces rank-1 for the hit
        "label_gain":        [0, 0, 1],
        "n_estimators":      n_estimators,
        "num_leaves":        num_leaves,
        "learning_rate":     learning_rate,
        "min_child_samples": min_child_samples,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "n_jobs":            -1,
        "random_state":      42,
        "verbosity":         -1,
    }


def train_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    normalize: bool,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    min_child_samples: int,
    val_df: Optional[pd.DataFrame] = None,
    early_stopping_rounds: Optional[int] = None,
) -> lgb.LGBMRanker:
    """Train a single LGBMRanker on train_df, optionally with early stopping."""
    X_tr, y_tr, g_tr, _ = build_X_y_groups(
        train_df, feature_cols, feature_directions, normalize
    )

    params = _lgbm_params(n_estimators, num_leaves, learning_rate, min_child_samples)
    model  = lgb.LGBMRanker(**params)

    fit_kwargs: dict = {
        "group": g_tr,
        "feature_name": feature_cols,
    }

    if val_df is not None and early_stopping_rounds is not None:
        X_va, y_va, g_va, _ = build_X_y_groups(
            val_df, feature_cols, feature_directions, normalize
        )
        fit_kwargs["eval_set"]               = [(X_va, y_va)]
        fit_kwargs["eval_group"]             = [g_va]
        fit_kwargs["eval_at"]                = [1, 3, 5]
        fit_kwargs["callbacks"]              = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

    model.fit(X_tr, y_tr, **fit_kwargs)
    return model


def predict_scores(
    model: lgb.LGBMRanker,
    records: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    normalize: bool,
) -> pd.Series:
    """
    Returns a Series of LightGBM scores aligned to records.index,
    with direction-flipping and normalisation matching training.
    """
    df_p = _flip_lower_better(records, feature_cols, feature_directions)
    if normalize:
        df_p = _group_znorm(df_p, feature_cols)
    scores = model.predict(df_p[feature_cols].values.astype(np.float32))
    return pd.Series(scores, index=records.index, name="_score")


# ─────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────

def run_cv(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    normalize: bool,
    top_k: int,
    match_warheads: bool,
    n_folds: int,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    min_child_samples: int,
    early_stopping_rounds: int,
) -> tuple[list[dict], list[lgb.LGBMRanker], pd.DataFrame]:
    """
    Train one model per fold (held-out = val), collect Hit@K metrics.
    Returns (fold_metrics_list, fold_models_list, oof_records).

    oof_records is the full records DataFrame with _score and _rank filled
    from each row's held-out fold model only — no row is ever scored by a
    model that trained on it.  Use this for breakdown_analysis and headline
    Hit@K numbers so you always see true held-out performance.
    """
    folds = make_cv_folds(records, targets, n_folds)
    fold_sizes = [len(vt) for _, _, _, vt in folds]
    print(f"[CV] Fold target counts : {fold_sizes}")

    fold_metrics = []
    fold_models  = []

    # Pre-allocate OOF columns; filled in fold by fold
    oof_records = records.copy()
    oof_records["_score"] = np.nan
    oof_records["_rank"]  = np.nan

    for f_idx, (tr_df, va_df, tr_tgt, va_tgt) in enumerate(folds):
        if va_tgt.empty:
            print(f"[CV] Fold {f_idx+1}/{n_folds}  — skipped (empty val targets)")
            continue

        model = train_model(
            tr_df, feature_cols, feature_directions, normalize,
            n_estimators, num_leaves, learning_rate, min_child_samples,
            val_df=va_df, early_stopping_rounds=early_stopping_rounds,
        )
        fold_models.append(model)

        # Score + rank the validation fold
        va_df = va_df.copy()
        va_df["_score"] = predict_scores(
            model, va_df, feature_cols, feature_directions, normalize
        )
        va_df["_rank"] = rank_by_score(va_df)

        m = evaluate(va_df, va_tgt, top_k, match_warheads)
        n_trees = model.best_iteration_ if model.best_iteration_ else n_estimators
        print(f"[CV] Fold {f_idx+1}/{n_folds}  "
              f"Hit@{top_k}(all)={m['hitk_all']:.3f}  "
              f"Hit@{top_k}(rank)={m['hitk_rankable']:.3f}  "
              f"filt={m['pct_filtered']:.1f}%  "
              f"trees={n_trees}")
        fold_metrics.append(m)

        # Write OOF predictions back by original index (no data leakage)
        oof_records.loc[va_df.index, "_score"] = va_df["_score"].values
        oof_records.loc[va_df.index, "_rank"]  = va_df["_rank"].values

    n_missing = int(oof_records["_score"].isna().sum())
    if n_missing > 0:
        print(f"[WARN] {n_missing} rows have no OOF score (fell in empty folds).")

    return fold_metrics, fold_models, oof_records


def aggregate_cv_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {k: 0.0 for k in [
            "hitk_all", "hitk_rankable", "rank_bonus", "objective",
            "n_targets", "n_rankable", "n_filtered_out", "pct_filtered",
            "mean_random_hitk_all", "mean_random_hitk_rankable",
        ]}
    def _mean(key): return float(np.mean([m[key] for m in fold_metrics]))
    def _sum(key):  return sum(m[key] for m in fold_metrics)
    return {
        "hitk_all":                  _mean("hitk_all"),
        "hitk_rankable":             _mean("hitk_rankable"),
        "rank_bonus":                _mean("rank_bonus"),
        "objective":                 _mean("objective"),
        "n_targets":                 _sum("n_targets"),
        "n_rankable":                _sum("n_rankable"),
        "n_filtered_out":            _sum("n_filtered_out"),
        "pct_filtered":              _mean("pct_filtered"),
        "mean_random_hitk_all":      _mean("mean_random_hitk_all"),
        "mean_random_hitk_rankable": _mean("mean_random_hitk_rankable"),
    }


# ─────────────────────────────────────────────────────────────
# Breakdown analysis  (identical to original)
# ─────────────────────────────────────────────────────────────

def breakdown_analysis(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    top_k: int,
    match_warheads: bool,
    has_pf: bool,
    export_path: Optional[str] = None,
) -> None:
    rows = []
    for _, trow in targets.iterrows():
        name     = trow["_name"]
        res      = trow["_residue"]
        wh_set   = trow["_warhead_set"]
        pf_out   = trow.get("_prefiltered_out", False)
        wh_label = ",".join(sorted(wh_set)) if wh_set else "unknown"

        if pf_out:
            rows.append({"residue": res, "warhead": wh_label,
                         "hit": False, "pf_out": True, "no_match": False, "rand_p": 0.0})
            continue

        mask  = records["_name_upper"] == name
        mask &= records["_res_upper"]  == res
        if match_warheads and wh_set:
            mask &= records["_warhead_lower"].isin(wh_set)
        matched = records[mask]

        if matched.empty:
            rows.append({"residue": res, "warhead": wh_label,
                         "hit": False, "pf_out": False, "no_match": True, "rand_p": 0.0})
            continue

        rank    = float(matched["_rank"].min())
        n_cands = int(matched["_n_candidates"].iloc[0])
        rows.append({"residue": res, "warhead": wh_label,
                     "hit": rank <= top_k, "pf_out": False, "no_match": False,
                     "rand_p": random_hitk(n_cands, top_k)})

    df = pd.DataFrame(rows)

    def _aggregate(df: pd.DataFrame, group_col: str,
                   group_label: str) -> pd.DataFrame:
        out_rows = []
        for grp, gdf in sorted(df.groupby(group_col), key=lambda x: x[0]):
            n_total   = len(gdf)
            n_filt    = int(gdf["pf_out"].sum())
            n_nomatch = int(gdf["no_match"].sum())
            n_rank    = int((~gdf["pf_out"] & ~gdf["no_match"]).sum())
            rankable  = gdf[~gdf["pf_out"] & ~gdf["no_match"]]

            def _pct(s): return round(s * 100, 1) if not np.isnan(s) else None

            row = {
                "Group":              group_label,
                "Category":           str(grp),
                "N_Targets":          n_total,
                "N_Rankable":         n_rank,
                "N_No_Warhead_Match": n_nomatch,
            }
            if has_pf:
                row["N_Prefiltered_Out"] = n_filt
                row["Pct_Prefiltered"]   = round(100.0 * n_filt / n_total, 1) if n_total else None
            row[f"Hit@{top_k}_All_Pct"]        = _pct(gdf["hit"].mean()        if n_total else float("nan"))
            row[f"Random_Hit@{top_k}_All_Pct"] = _pct(gdf["rand_p"].mean()     if n_total else float("nan"))
            if has_pf:
                row[f"Hit@{top_k}_Rankable_Pct"]        = _pct(rankable["hit"].mean()    if n_rank else float("nan"))
                row[f"Random_Hit@{top_k}_Rankable_Pct"] = _pct(rankable["rand_p"].mean() if n_rank else float("nan"))
            out_rows.append(row)
        return pd.DataFrame(out_rows)

    res_df  = _aggregate(df, "residue", "Residue_Type")
    wh_df   = _aggregate(df, "warhead",  "Warhead")
    summary = pd.concat([res_df, wh_df], ignore_index=True)

    if export_path:
        summary.to_csv(export_path, index=False)
        print(f"\n[INFO] Hit@{top_k} breakdown exported to: {export_path}")
    else:
        def _print_table(tdf: pd.DataFrame, label: str) -> None:
            print(f"\n  {label}")
            print("  " + "-" * 80)
            if has_pf:
                print(f"  {'Category':<42} {'N':>5}  {'Filt':>5}  {'NoWH':>5}  "
                      f"{'Hit@K(all)':>10}  {'rand':>6}  "
                      f"{'Hit@K(rnk)':>10}  {'rand':>6}")
            else:
                print(f"  {'Category':<42} {'N':>5}  {'NoWH':>5}  "
                      f"{'Hit@K':>10}  {'rand':>6}")
            print("  " + "-" * 80)

            def _pct(v):
                return f"{v:5.1f}%" if v is not None else "   n/a"

            for _, r in tdf.iterrows():
                cat = str(r["Category"])[:42]
                if has_pf:
                    print(f"  {cat:<42} {r['N_Targets']:>5}  "
                          f"{r.get('N_Prefiltered_Out', 0):>5}  "
                          f"{r.get('N_No_Warhead_Match', 0):>5}  "
                          f"{_pct(r[f'Hit@{top_k}_All_Pct']):>10}  "
                          f"{_pct(r[f'Random_Hit@{top_k}_All_Pct']):>6}  "
                          f"{_pct(r.get(f'Hit@{top_k}_Rankable_Pct')):>10}  "
                          f"{_pct(r.get(f'Random_Hit@{top_k}_Rankable_Pct')):>6}")
                else:
                    print(f"  {cat:<42} {r['N_Targets']:>5}  "
                          f"{r.get('N_No_Warhead_Match', 0):>5}  "
                          f"{_pct(r[f'Hit@{top_k}_All_Pct']):>10}  "
                          f"{_pct(r[f'Random_Hit@{top_k}_All_Pct']):>6}")
            print("  " + "-" * 80)

        print("\n" + "=" * 70)
        print(f"  HIT@{top_k} BREAKDOWN (full dataset)")
        print("=" * 70)
        _print_table(res_df, "By Amino Acid Residue Type")
        _print_table(wh_df,  "By Warhead Type")
        print("=" * 70)

# ─────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────

def print_feature_importance(model: lgb.LGBMRanker,
                              feature_cols: list[str],
                              feature_directions: list[bool]) -> None:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    print("\n  Feature Importances (gain, final model):")
    print("  " + "-" * 52)
    max_col = max(len(c) for c in feature_cols)
    for i in order:
        col = feature_cols[i]
        dir_tag = "↑" if feature_directions[i] else "↓"
        print(f"  {dir_tag} {col:<{max_col}}  {imp[i]:>8.1f}")
    print("  " + "-" * 52)


# ─────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────

def run(
    training_path: str,
    labels_path: str,
    feature_cols: list[str],
    feature_directions: list[bool],
    top_k: int,
    n_folds: int,
    match_warheads: bool,
    residue_types: set[str],
    verbose: bool,
    export_scores: Optional[str],
    prefilters: list[tuple[str, str, float]],
    export_breakdown: Optional[str],
    export_model: Optional[str],
    # LightGBM hyperparameters
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    min_child_samples: int,
    early_stopping_rounds: int,
    normalize_features: bool,
) -> None:

    n_feat = len(feature_cols)
    has_pf = bool(prefilters)

    print("=" * 70)
    print("  REACTIVITY LGBM RANKER")
    print("=" * 70)
    print(f"  Features ({n_feat})       : {', '.join(feature_cols)}")
    dir_str = ', '.join("↑" if d else "↓" for d in feature_directions)
    print(f"  Directions            : {dir_str}  (↑ higher better, ↓ lower better)")
    print(f"  Model                 : LightGBM lambdarank  "
          f"(label_gain=[0,0,1]  →  label-2 = hit residue)")
    print(f"  n_estimators          : {n_estimators}  "
          f"(early stopping at {early_stopping_rounds} rounds)")
    print(f"  num_leaves            : {num_leaves}")
    print(f"  learning_rate         : {learning_rate}")
    print(f"  min_child_samples     : {min_child_samples}")
    print(f"  Normalize features    : {'yes (within-group z-score)' if normalize_features else 'no'}")
    if has_pf:
        pf_str = "  +  ".join(f"{c} {o} {v}" for c, o, v in prefilters)
        print(f"  Prefilters            : {pf_str}")
    else:
        print(f"  Prefilters            : none")
    print(f"  Residue types         : {', '.join(sorted(residue_types))}")
    print(f"  Top-K                 : {top_k}")
    print(f"  CV folds              : {n_folds}")
    print(f"  Warhead matching      : {'enabled' if match_warheads else 'disabled'}")

    # ── Load data ─────────────────────────────────────────────────────────
    records, targets = load_and_prepare(
        training_path, labels_path,
        feature_cols, feature_directions,
        residue_types, match_warheads,
        prefilters,
    )

    print(f"\n[INFO] Unique proteins (Name)            : "
          f"{records['_name_upper'].nunique():,}")
    print(f"[INFO] Unique base PDBs                  : "
          f"{records['_base_pdb'].nunique():,}")
    print(f"[INFO] Candidate rows (Name×Res×Warhead) : {len(records):,}")
    print(f"[INFO] Target entries (labels)           : {len(targets):,}")

    # ── Sanity check: warn if training data looks like hits-only ──────────
    n_groups        = records.groupby(["_name_upper", "_warhead_lower"]).ngroups
    n_labeled_rows  = int(records["_labeled"].sum())
    n_total_rows    = len(records)
    solo_groups     = (
        records.groupby(["_name_upper", "_warhead_lower"])["_labeled"]
        .sum()
        .eq(records.groupby(["_name_upper", "_warhead_lower"]).size())
    )
    n_solo = int(solo_groups.sum())
    pct_solo = 100.0 * n_solo / n_groups if n_groups > 0 else 0.0

    if n_labeled_rows == n_total_rows:
        print()
        print("!" * 70)
        print("  [CRITICAL] Every row in the training CSV is a labeled hit residue.")
        print("  The model has NO non-hit competitors to rank against — ranking")
        print("  is trivially 1-out-of-1 and Hit@K will be artifically ~100%.")
        print()
        print("  Your training CSV must contain ALL candidate residue types per")
        print("  (Name, Warhead) group, not only the covalent-hit residue.")
        print()
        print("  Expected training CSV format:")
        print("    Name     Residue  Warhead      Feature1  Feature2 ...")
        print("    7AHA     CYS      MichaelAccep  0.92      0.14  ...")
        print("    7AHA     SER      MichaelAccep  0.31      0.88  ...")
        print("    7AHA     THR      MichaelAccep  0.44      0.51  ...")
        print("    7AHA     LYS      MichaelAccep  0.12      0.29  ...")
        print("    ...      (all residue types computed for this protein)")
        print()
        print("  Your labels CSV then says which Residue is the covalent hit.")
        print("  Without competitor rows, LightGBM cannot learn to discriminate.")
        print("!" * 70)
        print()
        sys.exit("[ERROR] Training data contains only labeled (hit) residues. "
                 "Re-run with a training CSV that includes all candidate residue "
                 "types per protein × warhead group.")
    elif pct_solo > 20.0:
        print()
        print(f"[WARN] {n_solo}/{n_groups} groups ({pct_solo:.1f}%) have ONLY the labeled")
        print(f"       residue in the training CSV — no competitors. Those groups")
        print(f"       trivially rank the hit at position 1 and inflate Hit@K.")
        print(f"       Ensure your training CSV has all candidate residue types.")
        print()

    # Candidate count distribution
    cand_dist = records["_n_candidates"].value_counts().sort_index()
    print(f"[INFO] Candidates per group distribution : "
          + "  ".join(f"{k}→{v}" for k, v in cand_dist.items()))

    # ── Cross-validation ──────────────────────────────────────────────────
    print(f"\n[CV] Running {n_folds}-fold CV ...")
    fold_metrics, _, oof_records = run_cv(
        records, targets,
        feature_cols, feature_directions, normalize_features,
        top_k, match_warheads, n_folds,
        n_estimators, num_leaves, learning_rate, min_child_samples,
        early_stopping_rounds,
    )
    cv_m = aggregate_cv_metrics(fold_metrics)

    # OOF records have _score/_rank from held-out predictions only — use
    # these for all reported metrics and breakdown (true generalisation).
    full_m = evaluate(oof_records, targets, top_k, match_warheads)

    # ── Final model on full data ──────────────────────────────────────────
    # Trained for export/deployment only; NOT used for any reported metric.
    print(f"\n[INFO] Training final model on full dataset (for export/deployment) ...")
    final_model = train_model(
        records, feature_cols, feature_directions, normalize_features,
        n_estimators, num_leaves, learning_rate, min_child_samples,
    )
    n_trees_final = final_model.best_iteration_ or n_estimators
    print(f"[INFO] Final model: {n_trees_final} trees")

    # ── Report ────────────────────────────────────────────────────────────
    def _fmt(key, d=None):
        v = d[key] if d else full_m[key]
        return f"{v:.4f}"

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  --- OOF held-out predictions (all {len(targets)} targets, no data leakage) ---")
    print(f"  Hit@{top_k} (all targets)          : {_fmt('hitk_all')}"
          f"  [random: {_fmt('mean_random_hitk_all')}]")
    if has_pf:
        pf    = full_m["pct_filtered"]
        nfilt = full_m["n_filtered_out"]
        nrank = full_m["n_rankable"]
        ntot  = full_m["n_targets"]
        print(f"  Hit@{top_k} (rankable only)        : {_fmt('hitk_rankable')}"
              f"  [random: {_fmt('mean_random_hitk_rankable')}]")
        print(f"  Targets prefiltered out         : "
              f"{nfilt}/{ntot}  ({pf:.1f}%)  — {nrank} rankable")
    print(f"  Rank bonus                      : {_fmt('rank_bonus')}  (ε={EPSILON})")
    print(f"  Objective                       : {_fmt('objective')}")

    print(f"\n  --- {n_folds}-fold CV held-out (mean over folds) ---")
    print(f"  CV Hit@{top_k} (all targets)       : {_fmt('hitk_all', cv_m)}"
          f"  [random: {_fmt('mean_random_hitk_all', cv_m)}]")
    if has_pf:
        cv_pf    = cv_m["pct_filtered"]
        cv_nfilt = cv_m["n_filtered_out"]
        cv_nrank = cv_m["n_rankable"]
        cv_ntot  = cv_m["n_targets"]
        print(f"  CV Hit@{top_k} (rankable only)     : {_fmt('hitk_rankable', cv_m)}"
              f"  [random: {_fmt('mean_random_hitk_rankable', cv_m)}]")
        print(f"  CV Targets prefiltered out      : "
              f"{cv_nfilt}/{cv_ntot}  ({cv_pf:.1f}%)  — {cv_nrank} rankable")
    print(f"  CV Rank bonus                   : {_fmt('rank_bonus', cv_m)}")
    print(f"  CV Objective                    : {_fmt('objective', cv_m)}")
    print(f"  N target entries (summed)       : {cv_m['n_targets']}")
    print("=" * 70)

    # ── Feature importance ────────────────────────────────────────────────
    print_feature_importance(final_model, feature_cols, feature_directions)

    # -- Breakdown by residue / warhead (OOF predictions) ------------------
    breakdown_analysis(
        oof_records, targets, top_k, match_warheads, has_pf, export_breakdown
    )

    # ── Verbose per-target ranks ──────────────────────────────────────────
    if verbose:
        print(f"\n[VERBOSE] Per-target ranks (OOF held-out predictions):")
        rows = []
        for _, trow in targets.iterrows():
            name   = trow["_name"]
            res    = trow["_residue"]
            wh_set = trow["_warhead_set"]
            pf_out = trow.get("_prefiltered_out", False)

            if pf_out:
                rows.append({
                    "Name": name, "Residue": res,
                    "Warheads": ",".join(sorted(wh_set)) if wh_set else "any",
                    "Score": np.nan, "Rank": "filtered",
                    f"Hit@{top_k}": False, "N_Candidates": 0,
                    "Random_Hit%": 0.0, "Prefiltered": True,
                })
                continue

            mask  = oof_records["_name_upper"] == name
            mask &= oof_records["_res_upper"]  == res
            if match_warheads and wh_set:
                mask &= oof_records["_warhead_lower"].isin(wh_set)
            matched = oof_records[mask]

            if matched.empty:
                rank, score, n_cands = np.inf, np.nan, 0
            else:
                rank    = float(matched["_rank"].min())
                score   = float(matched.loc[matched["_rank"].idxmin(), "_score"])
                n_cands = int(matched["_n_candidates"].iloc[0])

            rows.append({
                "Name":         name,
                "Residue":      res,
                "Warheads":     ",".join(sorted(wh_set)) if wh_set else "any",
                "Score":        round(score, 5) if not np.isnan(score) else np.nan,
                "Rank":         rank,
                f"Hit@{top_k}": rank <= top_k,
                "N_Candidates": n_cands,
                "Random_Hit%":  round(random_hitk(n_cands, top_k) * 100, 1),
                "Prefiltered":  False,
            })

        vdf = pd.DataFrame(rows).sort_values(
            "Rank",
            key=lambda s: pd.to_numeric(s, errors="coerce").fillna(9999)
        )
        print(vdf.to_string(index=False))

    # ── Export scores ─────────────────────────────────────────────────────
    if export_scores:
        export_cols = ["Name", "_res_upper", "Warhead", "_score", "_rank",
                       "_labeled", "_n_candidates"] + feature_cols
        out = oof_records[[c for c in export_cols if c in oof_records.columns]].copy()
        out["_random_hitk_pct"] = out["_n_candidates"].apply(
            lambda n: round(random_hitk(n, top_k) * 100, 1)
        )
        out = out.rename(columns={
            "_res_upper":       "Residue",
            "_score":           "Score_LGBM",
            "_rank":            "Rank",
            "_labeled":         "Labeled",
            "_n_candidates":    "N_Candidates_After_Filter",
            "_random_hitk_pct": f"Random_Hit@{top_k}_Pct",
        })
        out.to_csv(export_scores, index=False)
        print(f"\n[INFO] Scores exported to: {export_scores}")

    # ── Export model ──────────────────────────────────────────────────────
    if export_model:
        with open(export_model, "wb") as f:
            pickle.dump({"model": final_model,
                         "feature_cols": feature_cols,
                         "feature_directions": feature_directions,
                         "normalize": normalize_features,
                         "residue_types": residue_types,
                         "top_k": top_k}, f)
        print(f"[INFO] Model saved to: {export_model}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LightGBM lambdarank for nucleophilic residue type ranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── data args (identical to original) ────────────────────────────────
    parser.add_argument("--training",  required=True)
    parser.add_argument("--labels",    required=True)
    parser.add_argument("--features",  nargs="+", required=True)
    parser.add_argument(
        "--feature-direction", nargs="+", default=None,
        choices=["higher", "lower"], metavar="higher|lower",
        dest="feature_direction",
    )
    parser.add_argument(
        "--prefilter", nargs="+", default=None, metavar="TOKEN",
        help="Pre-scoring filter(s). Groups of 3: COL  gt|lt|gte|lte|eq  VALUE.",
    )
    parser.add_argument("--top-k",          type=int,   default=3)
    parser.add_argument("--n-folds",        type=int,   default=5)
    parser.add_argument("--match-warheads", action="store_true")
    parser.add_argument(
        "--residue-types", nargs="+",
        default=list(DEFAULT_RESIDUE_TYPES), metavar="RES",
    )
    parser.add_argument("--verbose",           action="store_true")
    parser.add_argument("--export-scores",     default=None, metavar="PATH")
    parser.add_argument("--export-breakdown",  default=None, metavar="PATH")
    parser.add_argument("--export-model",      default=None, metavar="PATH",
                        help="Pickle the final model + metadata to this path.")

    # ── LightGBM hyperparameters ──────────────────────────────────────────
    parser.add_argument("--n-estimators",        type=int,   default=500,
                        help="Max trees (default 500; early stopping applies).")
    parser.add_argument("--num-leaves",          type=int,   default=31,
                        help="Max leaves per tree (default 31).")
    parser.add_argument("--learning-rate",       type=float, default=0.05,
                        help="Learning rate (default 0.05).")
    parser.add_argument("--min-child-samples",   type=int,   default=5,
                        help="Min samples per leaf (default 5; reduce for small datasets).")
    parser.add_argument("--early-stopping-rounds", type=int, default=50,
                        help="Early stopping patience in rounds (default 50).")
    parser.add_argument("--normalize-features",  action="store_true",
                        help="Apply within-group z-score normalisation before ranking "
                             "(generally not needed for GBDT but available).")

    args = parser.parse_args()

    n_feat = len(args.features)
    if args.feature_direction is None:
        directions = [True] * n_feat
    else:
        if len(args.feature_direction) != n_feat:
            sys.exit(
                f"[ERROR] --feature-direction has {len(args.feature_direction)} "
                f"values but --features has {n_feat}."
            )
        directions = [d == "higher" for d in args.feature_direction]

    prefilters = parse_prefilters(args.prefilter) if args.prefilter else []

    run(
        training_path        = args.training,
        labels_path          = args.labels,
        feature_cols         = args.features,
        feature_directions   = directions,
        top_k                = args.top_k,
        n_folds              = args.n_folds,
        match_warheads       = args.match_warheads,
        residue_types        = {r.strip().upper() for r in args.residue_types},
        verbose              = args.verbose,
        export_scores        = args.export_scores,
        prefilters           = prefilters,
        export_breakdown     = args.export_breakdown,
        export_model         = args.export_model,
        n_estimators         = args.n_estimators,
        num_leaves           = args.num_leaves,
        learning_rate        = args.learning_rate,
        min_child_samples    = args.min_child_samples,
        early_stopping_rounds = args.early_stopping_rounds,
        normalize_features   = args.normalize_features,
    )


if __name__ == "__main__":
    main()