"""
reactivity_rank_optimizer.py

Learns optimal feature weights for a composite Reactivity Score (R) that
ranks the correct nucleophilic amino-acid type within the top-K candidates
for each protein × warhead combination, using Bayesian Optimisation (BO)
with k-fold cross-validation.

PROBLEM SETUP
─────────────
Each protein × warhead pair has up to 6 candidate amino-acid types
(CYS, SER, THR, TYR, LYS, HIS).  For each candidate the training CSV
provides one row of quantum-chemical feature scores.  The label CSV says
which amino-acid type is the covalent hit for that protein × warhead.

A "hit" = the labeled residue type ranks in top-K (user-specified K) by
composite score R among the candidate types present for that protein × warhead.

PRE-FILTERING
─────────────
Before ranking, rows can be filtered by one or more columns in the training
CSV using --prefilter.  Only rows that pass ALL filters are ranked; the rest
are excluded from scoring entirely.

Syntax: --prefilter  COL  gt|lt|gte|lte|eq  VALUE  [COL op VALUE ...]

Because filtering can remove the labeled residue from the candidate pool,
the script reports THREE Hit@K metrics:

  Hit@K (all)       : standard Hit@K over all target entries.
                      Targets whose labeled residue was prefiltered out
                      are counted as misses (rank = inf).

  Hit@K (rankable)  : Hit@K computed only over targets where the labeled
                      residue SURVIVED the prefilter and was rankable.
                      This isolates the model's discrimination ability
                      independent of accessibility/protonation gating.

  % targets filtered: percentage of target entries where the labeled
                      residue was removed by the prefilter, along with
                      per-warhead breakdown so you can see which warhead
                      types drive the filtering.

Both metrics and the filter-out percentage are reported for full dataset
and CV held-out performance.

COMPOSITE SCORE
───────────────
    R = alpha_g * (sum_i(w_i * scaled_i) / sum(w_i)) + bias_g

Each feature is scaled to [0,1] within each (Name, Warhead) group using
min-max normalisation.  For lower-is-better features the scaled value is
inverted so 1.0 always means "best".  Zero-range groups get 0.5 (neutral).

g ∈ {S, O, N} residue group:
    S = CYS  (soft sulfur)
    O = SER, THR, TYR  (oxygen nucleophiles)
    N = LYS, HIS  (nitrogen nucleophiles)

OBJECTIVE (maximised by BO)
───────────────────────────
    obj = Hit@K_macro  +  EPSILON * mean(1/rank for hits in rankable set)

    Hit@K_macro = mean over groups {S, O, N} of Hit@K(rankable) for that group.

    Each nucleophile group (CYS / SER+THR+TYR / LYS+HIS) contributes equally
    to the objective regardless of how many targets it has (macro-average).
    This prevents the most-abundant group from dominating weight learning.

CROSS-VALIDATION
────────────────
Proteins are grouped by base PDB name (stripping -2, -3, ... suffixes).
Groups are split into n-folds; the same base PDB never spans train/test.

WARHEAD MATCHING
────────────────
Labels carry a Frankenstein_Warhead column (comma-separated).

Usage
─────
    python reactivity_rank_optimizer.py \\
        --training   training.csv \\
        --labels     labels.csv \\
        --features   Fukui_Deprotonated Nucleophilicity_Index_Deprotonated \\
                     HOMO_LUMO_Gap_Deprotonated \\
        --feature-direction higher higher lower \\
        --prefilter  Relative_SASA gte 0.2  Deprotonated_Fraction gt 0.5 \\
        --top-k 3 \\
        [--n-folds 5] [--n-calls 60] [--match-warheads] [--verbose]
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

DEFAULT_RESIDUE_TYPES = {"CYS", "SER", "THR", "TYR", "LYS", "HIS"}

FILTER_OPS = {
    "gt":  lambda a, b: a >  b,
    "lt":  lambda a, b: a <  b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "eq":  lambda a, b: a == b,
}

EPSILON = 0.1

# Nucleophile groups used throughout
NUCL_GROUPS = ["S", "O", "N"]


# ─────────────────────────────────────────────────────────────
# Helpers
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
# Residue group mapping
# ─────────────────────────────────────────────────────────────

def residue_to_group(res: str) -> str:
    """Map a residue name to its HSAB nucleophile group.

    S = soft sulfur  (CYS)
    O = oxygen       (SER, THR, TYR)
    N = nitrogen     (LYS, HIS)
    """
    if res == "CYS":
        return "S"
    elif res in {"SER", "THR", "TYR"}:
        return "O"
    elif res in {"LYS", "HIS"}:
        return "N"
    else:
        return "UNK"


# ─────────────────────────────────────────────────────────────
# Data loading
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
                  and _group (HSAB nucleophile group: S / O / N / UNK).
        targets : one row per label entry, with _prefiltered_out flag
                  indicating whether the labeled residue was removed by
                  the prefilter for that protein × warhead group.
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
    train_dedup["_group"]         = train_dedup["_res_upper"].apply(residue_to_group)

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

    targets["_group"] = targets["_residue"].apply(residue_to_group)

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

    train_dedup["_labeled"] = train_dedup.apply(_is_labeled, axis=1)

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
                wh_rows.append({
                    "warhead": wh_label,
                    "filtered_out": t["_prefiltered_out"],
                })
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
# CV fold builder
# ─────────────────────────────────────────────────────────────

def make_cv_folds(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    n_folds: int,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    base_pdbs = sorted(records["_base_pdb"].unique())
    rng.shuffle(base_pdbs)
    fold_of = {bp: i % n_folds for i, bp in enumerate(base_pdbs)}

    records = records.copy()
    records["_fold"] = records["_base_pdb"].map(fold_of)
    tgt_bases = targets["_base_pdb"].values

    folds = []
    for f in range(n_folds):
        val_mask = records["_fold"] == f
        val_df   = records[val_mask].copy()
        val_base = set(val_df["_base_pdb"].unique())
        val_tgt  = targets[pd.Series(tgt_bases).isin(val_base).values].copy()
        folds.append((val_df, val_tgt))
    return folds


# ─────────────────────────────────────────────────────────────
# Scoring and ranking
# ─────────────────────────────────────────────────────────────

def _minmax_scale_group(
    group: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
) -> pd.DataFrame:
    group = group.copy()
    for col, hib in zip(feature_cols, feature_directions):
        lo  = group[col].min()
        hi  = group[col].max()
        rng = hi - lo
        if rng == 0:
            group[f"_scaled_{col}"] = 0.5
        else:
            scaled = (group[col] - lo) / rng
            group[f"_scaled_{col}"] = scaled if hib else 1.0 - scaled
    return group


def compute_R(
    records: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    weights: list[float],
    group_alpha: dict,
    group_bias: dict,
) -> pd.Series:
    """
    R = alpha_g * base_score + bias_g

    where base_score = weighted average of min-max scaled features,
    and g is the HSAB nucleophile group (S / O / N) of each residue.
    """
    scaled_records = records.groupby(
        ["_name_upper", "_warhead_lower"], group_keys=False
    ).apply(
        lambda g: _minmax_scale_group(g, feature_cols, feature_directions),
        include_groups=False,
    )

    scaled_records["_group"] = records["_group"]

    w_total = sum(weights)
    if w_total == 0:
        return pd.Series(0.0, index=scaled_records.index)

    base_score = sum(
        w * scaled_records[f"_scaled_{col}"]
        for w, col in zip(weights, feature_cols)
    ) / w_total

    group = scaled_records["_group"]
    alpha = group.map(group_alpha).fillna(1.0)
    bias  = group.map(group_bias).fillna(0.0)

    return alpha * base_score + bias


def rank_within_group(records: pd.DataFrame,
                      score_col: str = "_R") -> pd.Series:
    return records.groupby(["_name_upper", "_warhead_lower"])[score_col]\
                  .rank(method="min", ascending=False)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    weights: list[float],
    top_k: int,
    match_warheads: bool,
    group_alpha: dict = None,
    group_bias: dict = None,
) -> dict:
    """
    Returns metrics dict.

    Hit@K objective uses MACRO-AVERAGE over nucleophile groups (S/O/N):
        hitk_macro = mean( Hit@K_rankable_S, Hit@K_rankable_O, Hit@K_rankable_N )

    Groups with no rankable targets in this split are skipped from the macro
    average (rather than treated as 0), so CV folds with sparse minority
    groups don't artificially depress the signal.

    Objective:
        obj = hitk_macro  +  EPSILON * rank_bonus

    Additional keys (micro metrics, per-group metrics) are reported for
    transparency but are NOT used in the BO objective.
    """
    rec = records.copy()
    rec["_R"]    = compute_R(
        rec, feature_cols, feature_directions, weights,
        group_alpha or {}, group_bias or {},
    )
    rec["_rank"] = rank_within_group(rec)

    # Collect per-target results
    hitk_all_vals      = []
    bonus_vals         = []
    rand_all_vals      = []
    n_filtered_out     = 0

    # Per-group accumulators for rankable targets
    group_hits:   dict[str, list[float]] = {g: [] for g in NUCL_GROUPS}
    group_rand:   dict[str, list[float]] = {g: [] for g in NUCL_GROUPS}
    group_bonus:  dict[str, list[float]] = {g: [] for g in NUCL_GROUPS}

    for _, trow in targets.iterrows():
        name   = trow["_name"]
        res    = trow["_residue"]
        wh_set = trow["_warhead_set"]
        pf_out = trow.get("_prefiltered_out", False)
        grp    = trow.get("_group", residue_to_group(res))

        mask = rec["_name_upper"] == name
        mask &= rec["_res_upper"]  == res
        if match_warheads and wh_set:
            mask &= rec["_warhead_lower"].isin(wh_set)

        matched = rec[mask]

        if pf_out:
            n_filtered_out += 1
            hitk_all_vals.append(0.0)
            rand_all_vals.append(0.0)
            continue
        
        #This is when there is no match in warhead. Previously was counting this as a miss...
        if matched.empty:
            hitk_all_vals.append(0.0)
            rand_all_vals.append(0.0)
            """
            if grp in group_hits:
                group_hits[grp].append(0.0)
                group_rand[grp].append(0.0)
                group_bonus[grp].append(0.0)
            """
            continue

        rank    = float(matched["_rank"].min())
        n_cands = int(matched["_n_candidates"].iloc[0])
        rand_p  = random_hitk(n_cands, top_k)
        is_hit  = rank <= top_k

        hitk_all_vals.append(1.0 if is_hit else 0.0)
        rand_all_vals.append(rand_p)
        bonus_vals.append(1.0 / rank if is_hit else 0.0)

        if grp in group_hits:
            group_hits[grp].append(1.0 if is_hit else 0.0)
            group_rand[grp].append(rand_p)
            group_bonus[grp].append(1.0 / rank if is_hit else 0.0)

    # ── per-group Hit@K (rankable targets only) ───────────────────────────
    per_group_hitk: dict[str, float] = {}
    per_group_rand: dict[str, float] = {}
    per_group_n:    dict[str, int]   = {}
    macro_parts: list[float] = []

    for g in NUCL_GROUPS:
        vals = group_hits[g]
        per_group_n[g]    = len(vals)
        per_group_hitk[g] = float(np.mean(vals)) if vals else float("nan")
        per_group_rand[g] = float(np.mean(group_rand[g])) if group_rand[g] else float("nan")
        if vals:                          # only include groups that have data
            macro_parts.append(per_group_hitk[g])

    hitk_macro = float(np.mean(macro_parts)) if macro_parts else 0.0

    # ── micro (flat) aggregates kept for reporting ────────────────────────
    n_rankable      = len(hitk_all_vals) - sum(1 for v in hitk_all_vals if v == 0.0
                                               and True)  # recompute properly
    # recount properly:
    n_rankable_real = sum(len(v) for v in group_hits.values())
    hitk_all  = float(np.mean(hitk_all_vals))  if hitk_all_vals  else 0.0
    hitk_rank = float(np.mean(
        [v for g in NUCL_GROUPS for v in group_hits[g]]
    )) if n_rankable_real > 0 else 0.0
    bonus     = float(np.mean(bonus_vals))     if bonus_vals     else 0.0
    rand_all  = float(np.mean(rand_all_vals))  if rand_all_vals  else 0.0
    rand_rank = float(np.mean(
        [v for g in NUCL_GROUPS for v in group_rand[g]]
    )) if n_rankable_real > 0 else 0.0
    n_total   = len(hitk_all_vals) + n_filtered_out
    pct_filt  = 100.0 * n_filtered_out / n_total if n_total > 0 else 0.0

    return {
        # ── macro objective (used by BO) ──────────────────────────────────
        "hitk_macro":                 hitk_macro,
        "objective":                  hitk_macro + EPSILON * bonus,

        # ── per-group metrics ─────────────────────────────────────────────
        "hitk_S":                     per_group_hitk.get("S", float("nan")),
        "hitk_O":                     per_group_hitk.get("O", float("nan")),
        "hitk_N":                     per_group_hitk.get("N", float("nan")),
        "rand_S":                     per_group_rand.get("S", float("nan")),
        "rand_O":                     per_group_rand.get("O", float("nan")),
        "rand_N":                     per_group_rand.get("N", float("nan")),
        "n_S":                        per_group_n.get("S", 0),
        "n_O":                        per_group_n.get("O", 0),
        "n_N":                        per_group_n.get("N", 0),

        # ── micro / legacy metrics (reported, not optimised) ──────────────
        "hitk_all":                   hitk_all,
        "hitk_rankable":              hitk_rank,
        "rank_bonus":                 bonus,
        "n_targets":                  n_total,
        "n_rankable":                 n_rankable_real,
        "n_filtered_out":             n_filtered_out,
        "pct_filtered":               pct_filt,
        "mean_random_hitk_all":       rand_all,
        "mean_random_hitk_rankable":  rand_rank,
    }


def cv_evaluate(
    folds: list,
    targets: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    weights: list[float],
    top_k: int,
    match_warheads: bool,
    group_alpha: dict = None,
    group_bias: dict = None,
) -> dict:
    fold_metrics = []
    for val_df, val_tgt in folds:
        if val_tgt.empty:
            continue
        m = evaluate(
            val_df, val_tgt, feature_cols, feature_directions,
            weights, top_k, match_warheads,
            group_alpha, group_bias,
        )
        fold_metrics.append(m)
    if not fold_metrics:
        return {
            "hitk_macro": 0.0, "hitk_all": 0.0, "hitk_rankable": 0.0,
            "rank_bonus": 0.0, "objective": 0.0,
            "hitk_S": float("nan"), "hitk_O": float("nan"), "hitk_N": float("nan"),
            "rand_S": float("nan"), "rand_O": float("nan"), "rand_N": float("nan"),
            "n_S": 0, "n_O": 0, "n_N": 0,
            "n_targets": 0, "n_rankable": 0,
            "n_filtered_out": 0, "pct_filtered": 0.0,
            "mean_random_hitk_all": 0.0, "mean_random_hitk_rankable": 0.0,
        }

    def _mean(key):
        vals = [m[key] for m in fold_metrics if not np.isnan(m.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    def _sum(key):
        return sum(m[key] for m in fold_metrics)

    return {
        "hitk_macro":                _mean("hitk_macro"),
        "objective":                 _mean("objective"),
        "hitk_S":                    _mean("hitk_S"),
        "hitk_O":                    _mean("hitk_O"),
        "hitk_N":                    _mean("hitk_N"),
        "rand_S":                    _mean("rand_S"),
        "rand_O":                    _mean("rand_O"),
        "rand_N":                    _mean("rand_N"),
        "n_S":                       _sum("n_S"),
        "n_O":                       _sum("n_O"),
        "n_N":                       _sum("n_N"),
        "hitk_all":                  _mean("hitk_all"),
        "hitk_rankable":             _mean("hitk_rankable"),
        "rank_bonus":                _mean("rank_bonus"),
        "n_targets":                 _sum("n_targets"),
        "n_rankable":                _sum("n_rankable"),
        "n_filtered_out":            _sum("n_filtered_out"),
        "pct_filtered":              _mean("pct_filtered"),
        "mean_random_hitk_all":      _mean("mean_random_hitk_all"),
        "mean_random_hitk_rankable": _mean("mean_random_hitk_rankable"),
    }


# ─────────────────────────────────────────────────────────────
# Bayesian Optimisation
# ─────────────────────────────────────────────────────────────

def run_bo(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    feature_cols: list[str],
    feature_directions: list[bool],
    top_k: int,
    n_calls: int,
    n_folds: int,
    match_warheads: bool,
) -> tuple[list[float], dict, dict, dict]:
    from skopt import gp_minimize
    from skopt.space import Real

    n_feat = len(feature_cols)
    groups = ["S", "O", "N"]

    print(f"\n[BO] Building {n_folds}-fold CV splits ...")
    folds = make_cv_folds(records, targets, n_folds)
    fold_sizes = [len(vt) for _, vt in folds]
    print(f"[BO] Fold target counts : {fold_sizes}")
    print(f"[BO] Optimising {n_feat} feature weights + "
          f"{len(groups) * 2} group params over {n_calls} calls ...")
    print(f"[BO] Objective          : macro-Hit@K (equal weight per group S/O/N) "
          f"+ {EPSILON} * rank_bonus")

    space = []
    for col in feature_cols:
        space.append(Real(0.01, 5.0, name=f"w_{col}", prior="log-uniform"))
    for g in groups:
        space.append(Real(0.5, 2.0,  name=f"alpha_{g}"))
        space.append(Real(-1.0, 1.0, name=f"bias_{g}"))

    x0 = [[1.0] * n_feat + [1.0, 0.0] * len(groups)]
    rng = np.random.default_rng(0)
    for _ in range(min(4, n_calls - 1)):
        feat_init  = list(np.round(rng.uniform(0.1, 3.0, n_feat), 3))
        group_init = []
        for _ in groups:
            group_init.append(round(float(rng.uniform(0.5, 2.0)), 3))
            group_init.append(round(float(rng.uniform(-0.5, 0.5)), 3))
        x0.append(feat_init + group_init)

    call_counter = [0]

    def _decode(x):
        weights = list(x[:n_feat])
        group_alpha = {"S": x[n_feat + 0], "O": x[n_feat + 2], "N": x[n_feat + 4]}
        group_bias  = {"S": x[n_feat + 1], "O": x[n_feat + 3], "N": x[n_feat + 5]}
        return weights, group_alpha, group_bias

    def objective(params):
        weights, group_alpha, group_bias = _decode(params)
        metrics = cv_evaluate(
            folds, targets, feature_cols, feature_directions,
            weights, top_k, match_warheads,
            group_alpha, group_bias,
        )
        call_counter[0] += 1
        n = call_counter[0]
        if n % 10 == 0 or n <= len(x0):
            w_str = "[" + ",".join(f"{v:.2f}" for v in weights) + "]"
            a_str = "S={:.2f},O={:.2f},N={:.2f}".format(
                group_alpha["S"], group_alpha["O"], group_alpha["N"])
            b_str = "S={:.2f},O={:.2f},N={:.2f}".format(
                group_bias["S"], group_bias["O"], group_bias["N"])
            s_str = (f"S={metrics['hitk_S']:.3f}"
                     if not np.isnan(metrics['hitk_S']) else "S=n/a")
            o_str = (f"O={metrics['hitk_O']:.3f}"
                     if not np.isnan(metrics['hitk_O']) else "O=n/a")
            nn_str = (f"N={metrics['hitk_N']:.3f}"
                     if not np.isnan(metrics['hitk_N']) else "N=n/a")
            print(f"  BO call {n:>3}/{n_calls}  "
                  f"obj={metrics['objective']:.4f}  "
                  f"macro={metrics['hitk_macro']:.3f}  "
                  f"[{s_str} {o_str} {nn_str}]  "
                  f"filt={metrics['pct_filtered']:.1f}%  "
                  f"w={w_str}  α=[{a_str}]  b=[{b_str}]")
        return -metrics["objective"]

    result = gp_minimize(
        objective, space,
        x0=x0,
        n_calls=n_calls,
        n_initial_points=len(x0),
        acq_func="EI",
        random_state=42,
        verbose=False,
    )

    best_weights, best_group_alpha, best_group_bias = _decode(result.x)

    best_metrics = evaluate(
        records, targets, feature_cols, feature_directions,
        best_weights, top_k, match_warheads,
        best_group_alpha, best_group_bias,
    )
    cv_metrics = cv_evaluate(
        folds, targets, feature_cols, feature_directions,
        best_weights, top_k, match_warheads,
        best_group_alpha, best_group_bias,
    )
    for k in cv_metrics:
        best_metrics[f"cv_{k}"] = cv_metrics[k]

    return best_weights, best_metrics, best_group_alpha, best_group_bias


# ─────────────────────────────────────────────────────────────
# Main
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


def _fmt_pct(v: float) -> str:
    """Format a 0–1 fraction as percentage string, or 'n/a' if NaN."""
    if np.isnan(v):
        return "  n/a  "
    return f"{v*100:.1f}%"


def run(
    training_path: str,
    labels_path: str,
    feature_cols: list[str],
    feature_directions: list[bool],
    top_k: int,
    n_calls: int,
    n_folds: int,
    match_warheads: bool,
    residue_types: set[str],
    manual_weights: Optional[list[float]],
    verbose: bool,
    export_scores: Optional[str],
    prefilters: list[tuple[str, str, float]],
    export_breakdown: Optional[str] = None,
    manual_group_alpha: Optional[dict] = None,
    manual_group_bias: Optional[dict] = None,
) -> None:

    n_feat = len(feature_cols)
    has_pf = bool(prefilters)

    print("=" * 70)
    print("  REACTIVITY RANK OPTIMIZER")
    print("=" * 70)
    print(f"  Features ({n_feat})     : {', '.join(feature_cols)}")
    dir_str = ', '.join("↑" if d else "↓" for d in feature_directions)
    print(f"  Directions          : {dir_str}  (↑ higher better, ↓ lower better)")
    print(f"  Scaling             : within-group min-max per (Name × Warhead)")
    print(f"  Group modulation    : α·score + b  per nucleophile group (S/O/N)")
    print(f"  Objective           : macro-Hit@K (S=CYS / O=SER,THR,TYR / N=LYS,HIS)")
    if has_pf:
        pf_str = "  +  ".join(f"{c} {o} {v}" for c, o, v in prefilters)
        print(f"  Prefilters          : {pf_str}")
    else:
        print(f"  Prefilters          : none")
    print(f"  Residue types       : {', '.join(sorted(residue_types))}")
    print(f"  Top-K               : {top_k}")
    print(f"  CV folds            : {n_folds}")
    print(f"  Warhead matching    : {'enabled' if match_warheads else 'disabled'}")
    print(f"  Epsilon             : {EPSILON}")
    mode = "MANUAL WEIGHTS" if manual_weights else "WEIGHT OPTIMISATION (BO)"
    print(f"  Mode                : {mode}")

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

    # Count targets per nucleophile group
    for g, label in [("S", "CYS"), ("O", "SER/THR/TYR"), ("N", "LYS/HIS")]:
        n_g = (targets["_group"] == g).sum()
        print(f"[INFO]   Group {g} ({label:<12})          : {n_g:,} targets")

    default_alpha = {"S": 1.0, "O": 1.0, "N": 1.0}
    default_bias  = {"S": 0.0, "O": 0.0, "N": 0.0}

    if manual_weights is not None:
        if len(manual_weights) != n_feat:
            sys.exit(f"[ERROR] --weights expects {n_feat} values, "
                     f"got {len(manual_weights)}")
        best_weights     = manual_weights
        best_group_alpha = manual_group_alpha or default_alpha
        best_group_bias  = manual_group_bias  or default_bias
        best_metrics = evaluate(
            records, targets, feature_cols, feature_directions,
            best_weights, top_k, match_warheads,
            best_group_alpha, best_group_bias,
        )
        folds = make_cv_folds(records, targets, n_folds)
        cv_m  = cv_evaluate(
            folds, targets, feature_cols, feature_directions,
            best_weights, top_k, match_warheads,
            best_group_alpha, best_group_bias,
        )
        for k in cv_m:
            best_metrics[f"cv_{k}"] = cv_m[k]
    else:
        best_weights, best_metrics, best_group_alpha, best_group_bias = run_bo(
            records, targets, feature_cols, feature_directions,
            top_k, n_calls, n_folds, match_warheads,
        )

    records["_R"]    = compute_R(
        records, feature_cols, feature_directions, best_weights,
        best_group_alpha, best_group_bias,
    )
    records["_rank"] = rank_within_group(records)

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'MANUAL' if manual_weights else 'OPTIMISED'} WEIGHTS & GROUP PARAMS")
    print("=" * 70)
    max_col_len = max(len(c) for c in feature_cols)
    for col, w, hib in zip(feature_cols, best_weights, feature_directions):
        dir_tag = "↑ higher better" if hib else "↓ lower better"
        print(f"  {col:<{max_col_len}}  w={w:.4f}  [{dir_tag}]")

    print()
    print(f"  Group modulation (R = α·score + b):")
    print(f"  {'Group':<6}  {'Residues':<20}  {'alpha':>7}  {'bias':>7}")
    print(f"  {'-'*46}")
    group_residues = {"S": "CYS", "O": "SER,THR,TYR", "N": "LYS,HIS"}
    for g in ["S", "O", "N"]:
        print(f"  {g:<6}  {group_residues[g]:<20}  "
              f"{best_group_alpha.get(g, 1.0):>7.4f}  "
              f"{best_group_bias.get(g, 0.0):>7.4f}")

    def _fv(key, m=None):
        """Format a value from metrics dict as 0–1 → percent string."""
        d = m if m is not None else best_metrics
        v = d.get(key, float("nan"))
        return _fmt_pct(v)

    def _fvr(key, m=None):
        """Format a raw float value."""
        d = m if m is not None else best_metrics
        v = d.get(key, float("nan"))
        return f"{v:.4f}" if not np.isnan(v) else "  n/a  "

    # ── full dataset ──────────────────────────────────────────────────────
    print()
    print(f"  {'─'*66}")
    print(f"  FULL DATASET")
    print(f"  {'─'*66}")
    print(f"  Hit@{top_k} macro (S/O/N equal weight)  : {_fv('hitk_macro')}"
          f"  ← BO objective")
    print(f"  Hit@{top_k} micro (all targets, flat)   : {_fv('hitk_all')}"
          f"  [random: {_fv('mean_random_hitk_all')}]")
    if has_pf:
        pf    = best_metrics['pct_filtered']
        nfilt = best_metrics['n_filtered_out']
        nrank = best_metrics['n_rankable']
        ntot  = best_metrics['n_targets']
        print(f"  Hit@{top_k} micro (rankable only)        : {_fv('hitk_rankable')}"
              f"  [random: {_fv('mean_random_hitk_rankable')}]")
        print(f"  Targets prefiltered out             : "
              f"{nfilt}/{ntot}  ({pf:.1f}%)  — {nrank} rankable")

    print()
    print(f"  Per-group Hit@{top_k} (rankable, full dataset):")
    print(f"  {'Group':<6}  {'Residues':<18}  {'N':>5}  {'Hit@K':>8}  {'random':>8}")
    print(f"  {'-'*52}")
    for g, label in [("S", "CYS"), ("O", "SER/THR/TYR"), ("N", "LYS/HIS")]:
        n_g  = best_metrics.get(f"n_{g}", 0)
        h_g  = best_metrics.get(f"hitk_{g}", float("nan"))
        r_g  = best_metrics.get(f"rand_{g}", float("nan"))
        print(f"  {g:<6}  {label:<18}  {n_g:>5}  "
              f"{_fmt_pct(h_g):>8}  {_fmt_pct(r_g):>8}")
    print()
    print(f"  Rank bonus                          : {_fvr('rank_bonus')}  (ε={EPSILON})")
    print(f"  Objective (macro + ε·bonus)         : {_fvr('objective')}")

    # ── CV ────────────────────────────────────────────────────────────────
    print()
    print(f"  {'─'*66}")
    print(f"  {n_folds}-FOLD CV HELD-OUT")
    print(f"  {'─'*66}")
    cv = {k.replace("cv_", ""): v for k, v in best_metrics.items() if k.startswith("cv_")}
    print(f"  CV Hit@{top_k} macro (S/O/N equal weight)  : {_fmt_pct(cv.get('hitk_macro', float('nan')))}"
          f"  ← BO objective")
    print(f"  CV Hit@{top_k} micro (all targets, flat)   : {_fmt_pct(cv.get('hitk_all', float('nan')))}"
          f"  [random: {_fmt_pct(cv.get('mean_random_hitk_all', float('nan')))}]")
    if has_pf:
        cv_pf    = cv.get('pct_filtered', 0.0)
        cv_nfilt = cv.get('n_filtered_out', 0)
        cv_nrank = cv.get('n_rankable', 0)
        cv_ntot  = cv.get('n_targets', 0)
        print(f"  CV Hit@{top_k} micro (rankable only)        : {_fmt_pct(cv.get('hitk_rankable', float('nan')))}"
              f"  [random: {_fmt_pct(cv.get('mean_random_hitk_rankable', float('nan')))}]")
        print(f"  CV Targets prefiltered out             : "
              f"{cv_nfilt}/{cv_ntot}  ({cv_pf:.1f}%)  — {cv_nrank} rankable")

    print()
    print(f"  Per-group CV Hit@{top_k} (rankable, macro-averaged across folds):")
    print(f"  {'Group':<6}  {'Residues':<18}  {'N':>5}  {'Hit@K':>8}  {'random':>8}")
    print(f"  {'-'*52}")
    for g, label in [("S", "CYS"), ("O", "SER/THR/TYR"), ("N", "LYS/HIS")]:
        n_g  = cv.get(f"n_{g}", 0)
        h_g  = cv.get(f"hitk_{g}", float("nan"))
        r_g  = cv.get(f"rand_{g}", float("nan"))
        print(f"  {g:<6}  {label:<18}  {n_g:>5}  "
              f"{_fmt_pct(h_g):>8}  {_fmt_pct(r_g):>8}")
    print()
    print(f"  CV Rank bonus                       : {cv.get('rank_bonus', float('nan')):.4f}")
    print(f"  CV Objective (macro + ε·bonus)      : {cv.get('objective', float('nan')):.4f}")
    print(f"  N target entries                    : {cv.get('n_targets', 0)}")
    print("=" * 70)

    # ── per-residue and per-warhead breakdown ─────────────────────────────
    breakdown_analysis(records, targets, top_k, match_warheads, has_pf, export_breakdown)

    # ── verbose: per-target ranks ─────────────────────────────────────────
    if verbose:
        print(f"\n[VERBOSE] Per-target ranks (full dataset):")
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
                    "Group": trow.get("_group", "?"),
                    "Score_R": np.nan, "Rank": "filtered",
                    f"Hit@{top_k}": False,
                    "N_Candidates": 0, "Random_Hit%": 0.0,
                    "Prefiltered": True,
                })
                continue

            mask  = records["_name_upper"] == name
            mask &= records["_res_upper"]  == res
            if match_warheads and wh_set:
                mask &= records["_warhead_lower"].isin(wh_set)
            matched = records[mask]
            if matched.empty:
                rank, score, n_cands = np.inf, np.nan, 0
            else:
                rank    = float(matched["_rank"].min())
                score   = float(matched.loc[matched["_rank"].idxmin(), "_R"])
                n_cands = int(matched["_n_candidates"].iloc[0])
            rows.append({
                "Name":           name,
                "Residue":        res,
                "Warheads":       ",".join(sorted(wh_set)) if wh_set else "any",
                "Group":          trow.get("_group", "?"),
                "Score_R":        round(score, 4) if not np.isnan(score) else np.nan,
                "Rank":           rank,
                f"Hit@{top_k}":   rank <= top_k,
                "N_Candidates":   n_cands,
                "Random_Hit%":    round(random_hitk(n_cands, top_k) * 100, 1),
                "Prefiltered":    False,
            })
        vdf = pd.DataFrame(rows).sort_values(
            "Rank",
            key=lambda s: pd.to_numeric(s, errors="coerce").fillna(9999)
        )
        print(vdf.to_string(index=False))

    # ── export ────────────────────────────────────────────────────────────
    if export_scores:
        export_cols = ["Name", "_res_upper", "Warhead", "_R", "_rank",
                       "_labeled", "_n_candidates", "_group"]
        for col in feature_cols:
            export_cols.append(col)
        out = records[[c for c in export_cols if c in records.columns]].copy()
        out["_random_hitk_pct"] = out["_n_candidates"].apply(
            lambda n: round(random_hitk(n, top_k) * 100, 1)
        )
        out = out.rename(columns={
            "_res_upper":       "Residue",
            "_R":               "Score_R",
            "_rank":            "Rank",
            "_labeled":         "Labeled",
            "_n_candidates":    "N_Candidates_After_Filter",
            "_random_hitk_pct": f"Random_Hit@{top_k}_Pct",
            "_group":           "Nucleophile_Group",
        })
        out.to_csv(export_scores, index=False)
        print(f"\n[INFO] Scores exported to: {export_scores}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BO-optimised reactivity ranker for nucleophilic residue types.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
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
    parser.add_argument("--n-calls",        type=int,   default=60)
    parser.add_argument("--n-folds",        type=int,   default=5)
    parser.add_argument("--match-warheads", action="store_true")
    parser.add_argument(
        "--residue-types", nargs="+",
        default=list(DEFAULT_RESIDUE_TYPES), metavar="RES",
    )
    parser.add_argument("--weights",           nargs="+", type=float, default=None)
    parser.add_argument("--verbose",           action="store_true")
    parser.add_argument("--export-scores",     default=None, metavar="PATH")
    parser.add_argument("--export-breakdown",  default=None, metavar="PATH",
                        help="Export Hit@K breakdown by residue type and warhead "
                             "to a CSV file instead of printing to terminal.")

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
        training_path      = args.training,
        labels_path        = args.labels,
        feature_cols       = args.features,
        feature_directions = directions,
        top_k              = args.top_k,
        n_calls            = args.n_calls,
        n_folds            = args.n_folds,
        match_warheads     = args.match_warheads,
        residue_types      = {r.strip().upper() for r in args.residue_types},
        manual_weights     = args.weights,
        verbose            = args.verbose,
        export_scores      = args.export_scores,
        prefilters         = prefilters,
        export_breakdown   = args.export_breakdown,
    )


if __name__ == "__main__":
    main()