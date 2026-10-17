"""
prefilter_analysis.py

Evaluates a sequential set of pre-filters (and an optional reactivity
score filter) on a training CSV against a labels CSV to report:

  - Overall hit rate (target site still present after all filters)
  - Hit rate by residue type
  - Warhead mismatch counts
  - Per-filter failure counts
  - Per-filter average absolute and relative search-space reduction
  - Average total search-space reduction
  - Per-label row-level output CSV (one hit max per electrophile target site)
  - Summary CSV with overall, per-residue hit rates, per-filter stats (nucleophilic
    residues only: CYS, HIS, SER, LYS, TYR, THR), and per-residue per-filter stats
    (Filter_Residue_Type group; same residue restriction). Filter rows include
    Total/Avg/N_Count and *_If_Match variants (latter exclude warhead evals whose
    target was already eliminated upstream).
  - Optional --warhead_comparison for per-(Name × Warhead × Target) analysis

FILTER TYPES
────────────
1. Simple column filter   : COL  gt|lt|gte|lte|eq  VALUE
2. Reactivity score filter: --reactivity-features, --reactivity-directions,
                            --reactivity-weights, --reactivity-groups,
                            --reactivity-top-k, --reactivity-position
3. LightGBM ranker filter : --lgbm-model  (path to .pkl produced by
                            reactivity_lgbm_ranker.py).  Mutually exclusive
                            with option 2.  Use --lgbm-position and
                            --lgbm-top-k to control placement and cutoff
                            (--lgbm-top-k overrides the value stored in the
                            pkl when provided).

LABEL MATCHING
──────────────
Each label row has a Name, Residue, and Frankenstein_Warhead (comma-separated).
A label matches a training row if:
    - Name matches exactly (case-insensitive, stripped)
    - At least one warhead in Frankenstein_Warhead matches the training Warhead
      (case-insensitive, stripped)
If no warhead matches, that label is counted as a mismatch (not a miss).

HIT COUNTING (default)
──────────────────────
By default, hit rate is computed per label (Name + Residue + ResNum + Chain).
An electrophile may have multiple matching warheads in training, but it can
contribute at most one hit: the label is a hit if ANY matching warhead's filter
pipeline keeps the exact labeled site (Residue + ResNum + Chain) alive. Detail-row search-space reduction
values are averaged across all matching warhead evaluations for that label.

Summary search-space reduction (per-filter and total averages) and per-filter
failure counts are always computed from warhead-level evaluations. Hit rate
uses per label by default, per (label × warhead) with --warhead-comparison.

Use --warhead-comparison to restore per-(label × warhead) hit counting and
per-warhead detail output.

Use --perfect-match when a label lists multiple Frankenstein warheads: every
warhead present in both the label and training must keep the target residue
alive for the label to count as a hit. Label warheads missing from training
are ignored (e.g. label 4/4, training 3/4 → only those 3 must hit).
Default counts a hit if any matched warhead succeeds.

SEARCH SPACE REDUCTION
──────────────────────
The true starting search space denominator is the total number of amino acid
residues in the protein PDB file (all chains, ATOM records only).

  Filter 0  : nucleophilic residue selection (implicit in training pipeline)
              abs_red = rel_red = (n_pdb_residues - n_nucleophilic) / n_pdb_residues

  Filter N  : abs_red = (n_before - n_after) / n_pdb_residues   ← PDB total denom
              rel_red = (n_before - n_after) / n_before          ← local step denom

PDB files are located in --pdb-dir by matching the `protein_pdb` column
(case-insensitive). If not found, they are automatically downloaded from
the RCSB PDB API and saved to --pdb-dir.

Usage
─────
    python prefilter_analysis.py \
    --training training.csv \
    --labels batch_pdbs_bo_fixed.csv \
    --pdb-dir ./pdbs \
    --filters Rel_Side_SASA gte 15 deprotonation_prob gte 0.12 \
    --reactivity-position 2 \
    --reactivity-features Nucleophilicity_Index_Deprotonated Fukui_Deprotonated HOMO_LUMO_Gap_Deprotonated \
    --reactivity-directions higher higher lower \
    --reactivity-weights 5.0 1.2132 0.013 \
    --reactivity-groups S:CYS:0.5:1 O:SER,THR,TYR:1.8418:-1.0 N:LYS,HIS:0.5:1 \
    --reactivity-top-k 3 \
    --output-detail detail_baby_frank.csv \
    --output-summary summary_baby_frank.csv


    python baby_frank.py     --training training_eval.csv     --labels batch_pdbs_deprot_with_name.csv     --pdb-dir ../../Existing_Structures     --filters Rel_Side_SASA gte 12 deprotonation_prob gte 0.14     --lgbm-model lgbm_model_TOP2.pkl     --lgbm-position 3     --lgbm-top-k 3     --output-detail detail_eval_frank_moreinfo.csv     --output-summary summary_eval_frank_moreinfo.csv --perfect-match
    """

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_RESIDUE_TYPES = {"CYS", "SER", "THR", "TYR", "LYS", "HIS"}

FILTER_OPS = {
    "gt":  lambda a, b: a >  b,
    "lt":  lambda a, b: a <  b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "eq":  lambda a, b: a == b,
}

RESIDUE_GROUPS = {
    "S": {"CYS"},
    "O": {"SER", "THR", "TYR"},
    "N": {"LYS", "HIS"},
}

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class SimpleFilter:
    col:   str
    op:    str
    value: float
    label: str = ""       # auto-set after init

    def __post_init__(self):
        self.label = f"{self.col} {self.op} {self.value}"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.col not in df.columns:
            sys.exit(f"[ERROR] Filter column '{self.col}' not in training CSV.")
        numeric = pd.to_numeric(df[self.col], errors="coerce")
        return FILTER_OPS[self.op](numeric, self.value)


@dataclass
class GroupParams:
    alpha: float
    bias:  float


@dataclass
class ReactivityFilter:
    features:    list[str]
    directions:  list[bool]   # True = higher better
    weights:     list[float]
    group_params: dict[str, GroupParams]   # group_name -> GroupParams
    group_map:   dict[str, str]            # residue_upper -> group_name
    top_k:       int
    label:       str = "Reactivity(top-K)"

    def __post_init__(self):
        self.label = f"Reactivity top-{self.top_k}"


@dataclass
class LGBMRankerFilter:
    """
    Wraps a pkl produced by reactivity_lgbm_ranker.py.

    Expected pkl structure (dict):
        model            : LGBMRanker  (sklearn API)
        feature_cols     : list[str]   features in the order the model expects
        feature_directions: list[bool] (stored but not used at inference time)
        normalize        : bool        whether to min-max scale features before predict
        residue_types    : set[str]    residue types the model was trained on
        top_k            : int         default top-K stored at training time
    """
    model:         object          # LGBMRanker instance
    feature_cols:  list[str]
    normalize:     bool
    residue_types: set[str]
    top_k:         int             # may be overridden by --lgbm-top-k
    label:         str = ""

    def __post_init__(self):
        self.label = f"LGBM ranker top-{self.top_k}"


def load_lgbm_filter(pkl_path: str, top_k_override: Optional[int]) -> "LGBMRankerFilter":
    """Load a LGBMRankerFilter from a pkl file, with optional top-K override."""
    try:
        with open(pkl_path, "rb") as fh:
            bundle = pickle.load(fh)
    except FileNotFoundError:
        sys.exit(f"[ERROR] LGBM model pkl not found: {pkl_path}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load LGBM pkl: {e}")

    required_keys = {"model", "feature_cols", "normalize", "residue_types", "top_k"}
    missing = required_keys - set(bundle.keys())
    if missing:
        sys.exit(f"[ERROR] LGBM pkl is missing expected keys: {missing}")

    top_k = top_k_override if top_k_override is not None else bundle["top_k"]
    return LGBMRankerFilter(
        model         = bundle["model"],
        feature_cols  = bundle["feature_cols"],
        normalize     = bundle["normalize"],
        residue_types = bundle["residue_types"],
        top_k         = top_k,
    )


# ─────────────────────────────────────────────────────────────
# LightGBM scoring
# ─────────────────────────────────────────────────────────────

def compute_lgbm_scores(
    group_df: pd.DataFrame,
    full_df:  pd.DataFrame,
    lf:       LGBMRankerFilter,
) -> pd.Series:
    """
    Predict relevance scores for rows in group_df using the stored LGBMRanker.

    If lf.normalize is True, features are min-max scaled using full_df
    (the unfiltered Name×Warhead pool) as the reference range — matching
    the training-time normalisation convention.

    Rows whose residue type is not in lf.residue_types receive a score of
    -inf so they are never selected by the top-K cutoff.
    """
    feat_cols = lf.feature_cols

    # Check that all required feature columns exist
    missing = [c for c in feat_cols if c not in group_df.columns]
    if missing:
        sys.exit(
            f"[ERROR] LGBM model requires columns missing from training CSV: {missing}"
        )

    # Reset index so positional assignment is safe regardless of upstream filtering
    gdf = group_df.reset_index(drop=False)   # keeps original index in a column
    orig_index = group_df.index

    X = np.empty((len(gdf), len(feat_cols)), dtype=np.float64)
    for j, col in enumerate(feat_cols):
        X[:, j] = pd.to_numeric(gdf[col], errors="coerce").to_numpy(dtype=np.float64)

    if lf.normalize:
        for j, col in enumerate(feat_cols):
            lo = pd.to_numeric(full_df[col], errors="coerce").min()
            hi = pd.to_numeric(full_df[col], errors="coerce").max()
            rng = hi - lo
            if rng == 0:
                X[:, j] = 0.5
            else:
                X[:, j] = (X[:, j] - lo) / rng

    # Fill any remaining NaN with column median (safe fallback)
    for j in range(X.shape[1]):
        col_vals = X[:, j]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(col_vals)
            col_vals[nan_mask] = 0.0 if np.isnan(median) else median


    scores = pd.Series(
        lf.model.predict(X),
        index=orig_index,
        dtype=float,
    )

    # Zero out rows with residue types the model wasn't trained on
    for idx in orig_index:
        res = str(group_df.loc[idx, "_res_upper"])
        if res not in lf.residue_types:
            scores[idx] = -float("inf")

    return scores


# ─────────────────────────────────────────────────────────────
# PDB utilities
# ─────────────────────────────────────────────────────────────

def find_pdb_file(pdb_id: str, pdb_dir: str) -> Optional[str]:
    """
    Search pdb_dir for a file matching pdb_id (case-insensitive).
    Returns the full path if found, else None.
    """
    pdb_id_lower = pdb_id.strip().lower()
    try:
        for fname in os.listdir(pdb_dir):
            if fname.lower() == f"{pdb_id_lower}.pdb":
                return os.path.join(pdb_dir, fname)
    except FileNotFoundError:
        os.makedirs(pdb_dir, exist_ok=True)
    return None


def download_pdb(pdb_id: str, pdb_dir: str) -> str:
    """
    Download pdb_id from RCSB and save as {pdb_id}.pdb in pdb_dir.
    Returns the saved file path.
    """
    os.makedirs(pdb_dir, exist_ok=True)
    url      = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    out_path = os.path.join(pdb_dir, f"{pdb_id.upper()}.pdb")
    #print(f"[INFO] Downloading PDB {pdb_id.upper()} from {url} ...")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to download PDB {pdb_id}: {e}")
    #print(f"[INFO] Saved to {out_path}")
    return out_path


def count_pdb_residues(pdb_path: str) -> int:
    """
    Count unique (chain, residue_number, insertion_code) tuples from
    ATOM records in a PDB file — i.e. total amino acid residues across
    all chains (HETATM excluded).
    """
    residues = set()
    try:
        with open(pdb_path, "r") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                chain    = line[21]          # column 22 (0-indexed 21)
                res_seq  = line[22:26].strip()  # residue sequence number
                i_code   = line[26]          # insertion code
                residues.add((chain, res_seq, i_code))
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read PDB file {pdb_path}: {e}")
    return len(residues)


def get_pdb_residue_count(pdb_id: str, pdb_dir: str, _cache: dict = {}) -> int:
    """
    Return total residue count for pdb_id, using pdb_dir for storage.
    Results are cached in-memory so each PDB is parsed at most once.
    """
    key = pdb_id.strip().upper()
    if key in _cache:
        return _cache[key]

    pdb_path = find_pdb_file(pdb_id, pdb_dir)
    if pdb_path is None:
        pdb_path = download_pdb(pdb_id, pdb_dir)

    count = count_pdb_residues(pdb_path)
    print(f"[INFO] PDB {key}: {count:,} total amino acid residues (all chains)")
    _cache[key] = count
    return count


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


def norm(s: str) -> str:
    return str(s).strip().lower()


def parse_frankenstein_warheads(raw) -> set[str]:
    if pd.isna(raw):
        return set()
    return {norm(w) for w in str(raw).split(",") if w.strip()}


def warhead_matches(label_wh_set: set[str], train_wh: str) -> bool:
    return norm(train_wh) in label_wh_set


def label_key(name, residue, res_num, chain) -> tuple[str, str, str, str]:
    return (
        str(name).strip().upper(),
        str(residue).strip().upper(),
        str(res_num).strip(),
        str(chain).strip(),
    )


def target_site_mask(
    df: pd.DataFrame,
    residue: str,
    res_num,
    chain,
) -> pd.Series:
    """Boolean mask for the exact labeled site (Residue + ResNum + Chain)."""
    return (
        (df["_res_upper"] == str(residue).strip().upper())
        & (df["ResNum"].astype(str) == str(res_num).strip())
        & (df["Chain"].astype(str) == str(chain).strip())
    )


def target_site_in_df(
    df: pd.DataFrame,
    residue: str,
    res_num,
    chain,
) -> bool:
    if df.empty:
        return False
    return bool(target_site_mask(df, residue, res_num, chain).any())


def _average_reduction_fields(rows: list[dict]) -> dict[str, object]:
    """Average Abs/Rel reduction columns across warhead-level rows."""
    if not rows:
        return {}

    reduction_keys = {
        key
        for row in rows
        for key in row
        if key.endswith("_AbsReduction")
        or key.endswith("_RelReduction")
        or key in {"Total_AbsReduction", "Total_RelReduction"}
    }
    passed_keys = {
        key
        for row in rows
        for key in row
        if key.endswith("_Passed")
    }
    count_keys = {
        key
        for row in rows
        for key in row
        if key.endswith("_ResidueCount") or key == "Starting_Residue_Count"
        or key == "Filter0_Starting_Residue_Count"
    }
    rank_keys = {
        key
        for row in rows
        for key in row
        if key.endswith("_Rank")
    }

    averaged: dict[str, object] = {}
    for key in reduction_keys:
        vals = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and not pd.isna(row.get(key))
        ]
        averaged[key] = round(float(np.mean(vals)), 4) if vals else None

    for key in passed_keys:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        averaged[key] = any(vals) if vals else None

    for key in count_keys:
        vals = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and not pd.isna(row.get(key))
        ]
        averaged[key] = int(round(float(np.mean(vals)))) if vals else None

    for key in rank_keys:
        vals = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and not pd.isna(row.get(key))
        ]
        averaged[key] = round(float(np.mean(vals)), 4) if vals else None

    return averaged


def aggregate_label_detail_rows(
    detail_rows: list[dict],
    perfect_match: bool = False,
) -> list[dict]:
    """Collapse warhead-level rows to one row per label (electrophile target site).

    By default a label counts as a hit if any matching warhead's filter pipeline
    keeps the target residue alive. With perfect_match, labels that list multiple
    Frankenstein warheads require every training-matched warhead to hit.
    Search-space reduction columns are averaged across all matching warhead
    evaluations for that label.
    """
    from collections import defaultdict

    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in detail_rows:
        key = label_key(row["Name"], row["Residue"], row["ResNum"], row["Chain"])
        groups[key].append(row)

    aggregated: list[dict] = []
    for key in sorted(groups):
        rows = groups[key]
        if any(r.get("Is_Mismatch") for r in rows):
            rep = rows[0]
            agg = dict(rep)
            agg["Matched_Warheads"] = ""
            agg["Hitting_Warheads"] = ""
            aggregated.append(agg)
            continue

        hitting = [r["Warhead"] for r in rows if r.get("Hit")]
        matched = sorted({r["Warhead"] for r in rows})
        label_wh_count = int(rows[0].get("Label_Warhead_Count", 1) or 1)
        if perfect_match and label_wh_count > 1:
            label_hit = all(r.get("Hit") for r in rows)
        else:
            label_hit = bool(hitting)

        miss_rows = [r for r in rows if not r.get("Hit")]
        rep = miss_rows[0] if miss_rows else rows[0]
        agg = dict(rep)
        agg.update(_average_reduction_fields(rows))
        agg["Warhead"] = ",".join(matched)
        agg["Matched_Warheads"] = ",".join(matched)
        agg["Hitting_Warheads"] = ",".join(sorted(hitting))
        agg["Hit"] = label_hit
        if label_hit:
            agg["Miss_Reason"] = None
            agg["Failing_Filter"] = None
        elif perfect_match and label_wh_count > 1 and hitting:
            agg["Miss_Reason"] = "partial_warhead_miss"
            agg["Failing_Filter"] = rep.get("Failing_Filter")
        else:
            agg["Miss_Reason"] = rep.get("Miss_Reason")
            agg["Failing_Filter"] = rep.get("Failing_Filter")
        aggregated.append(agg)

    return aggregated


def annotate_warhead_detail_rows(detail_rows: list[dict]) -> list[dict]:
    """Add Matched/Hitting warhead columns for per-warhead comparison mode."""
    annotated = []
    for row in detail_rows:
        copy = dict(row)
        wh = copy.get("Warhead", "")
        copy["Matched_Warheads"] = wh
        copy["Hitting_Warheads"] = wh if copy.get("Hit") else ""
        annotated.append(copy)
    return annotated


def compute_filter_fail_counts_by_residue(
    failure_rows: list[dict],
    n_filters: int,
) -> dict[str, list[int]]:
    """Per-residue filter failure counts (one per warhead evaluation)."""
    from collections import defaultdict

    by_residue: dict[str, list[int]] = defaultdict(lambda: [0] * n_filters)
    for row in failure_rows:
        if row.get("Hit"):
            continue
        failing_filter = row.get("Failing_Filter")
        if failing_filter is None or pd.isna(failing_filter):
            continue
        residue = str(row["Residue"])
        by_residue[residue][int(failing_filter) - 1] += 1
    return dict(by_residue)


def _mean_round(vals: list[float], ndigits: int = 1) -> float | None:
    if not vals:
        return None
    return round(float(np.mean(vals)), ndigits)


def _avg_sum_count(
    vals: list[float],
    ndigits: int = 1,
) -> tuple[float | None, float | None, int | None]:
    """Return (avg, total, n_count) for summary filter-level N statistics."""
    if not vals:
        return None, None, None
    total = float(sum(vals))
    n = len(vals)
    return round(total / n, ndigits), round(total, 1), n


def _flatten_residue_vals(
    by_residue: dict[str, list[float]],
    residues: set[str] | frozenset[str],
) -> list[float]:
    """Concatenate per-residue value lists for the given residue types."""
    vals: list[float] = []
    for residue in sorted(residues):
        vals.extend(by_residue.get(residue, []))
    return vals


def _filter_n_summary_columns(
    before_vals: list[float],
    after_vals: list[float],
    *,
    if_match: bool = False,
) -> dict:
    """Build Total/Avg/N_Count summary fields (optionally the If_Match variants)."""
    suffix = "_If_Match" if if_match else ""
    avg_b, tot_b, n = _avg_sum_count(before_vals)
    avg_a, tot_a, _ = _avg_sum_count(after_vals)
    return {
        f"Total_N_Before{suffix}": tot_b,
        f"Total_N_After{suffix}":  tot_a,
        f"N_Count{suffix}":        n,
        f"Avg_N_Before{suffix}":   avg_b,
        f"Avg_N_After{suffix}":    avg_a,
    }


def build_filter_residue_summary_rows(
    residue_order: list[str],
    res_totals: dict[str, int],
    filter0_label: str,
    filter_labels: list[str],
    filter_fail_by_residue: dict[str, list[int]],
    filter0_abs_by_residue: dict[str, list[float]],
    filter0_rel_by_residue: dict[str, list[float]],
    filter0_n_before_by_residue: dict[str, list[float]],
    filter0_n_after_by_residue: dict[str, list[float]],
    filter_abs_by_residue: list[dict[str, list[float]]],
    filter_rel_by_residue: list[dict[str, list[float]]],
    filter_n_before_by_residue: list[dict[str, list[float]]],
    filter_n_after_by_residue: list[dict[str, list[float]]],
    filter0_n_before_if_match_by_residue: dict[str, list[float]],
    filter0_n_after_if_match_by_residue: dict[str, list[float]],
    filter_n_before_if_match_by_residue: list[dict[str, list[float]]],
    filter_n_after_if_match_by_residue: list[dict[str, list[float]]],
) -> list[dict]:
    """One summary row per (residue, filter) with the same stats as overall Filter rows."""
    rows: list[dict] = []
    n_filters = len(filter_labels)

    for residue in residue_order:
        n_matchable = res_totals.get(residue, 0)
        fail_counts = filter_fail_by_residue.get(residue, [0] * n_filters)

        f0_n_before = filter0_n_before_by_residue.get(residue, [])
        f0_n_after  = filter0_n_after_by_residue.get(residue, [])
        f0_im_before = filter0_n_before_if_match_by_residue.get(residue, [])
        f0_im_after  = filter0_n_after_if_match_by_residue.get(residue, [])

        rows.append({
            "Group":                 "Filter_Residue_Type",
            "Category":              residue,
            "Filter":                filter0_label,
            "N_Matchable":           n_matchable,
            "N_Failures":            0,
            **_filter_n_summary_columns(f0_n_before, f0_n_after),
            **_filter_n_summary_columns(f0_im_before, f0_im_after, if_match=True),
            "Avg_Abs_Reduction_Pct": _mean_round(
                [v * 100.0 for v in filter0_abs_by_residue.get(residue, [])]
            ),
            "Avg_Rel_Reduction_Pct": _mean_round(
                [v * 100.0 for v in filter0_rel_by_residue.get(residue, [])]
            ),
        })

        for f_idx, fl in enumerate(filter_labels):
            abs_vals = filter_abs_by_residue[f_idx].get(residue, [])
            rel_vals = filter_rel_by_residue[f_idx].get(residue, [])
            n_before = filter_n_before_by_residue[f_idx].get(residue, [])
            n_after  = filter_n_after_by_residue[f_idx].get(residue, [])
            im_before = filter_n_before_if_match_by_residue[f_idx].get(residue, [])
            im_after  = filter_n_after_if_match_by_residue[f_idx].get(residue, [])
            rows.append({
                "Group":                 "Filter_Residue_Type",
                "Category":              residue,
                "Filter":                fl,
                "N_Matchable":           n_matchable,
                "N_Failures":            fail_counts[f_idx],
                **_filter_n_summary_columns(n_before, n_after),
                **_filter_n_summary_columns(im_before, im_after, if_match=True),
                "Avg_Abs_Reduction_Pct": _mean_round([v * 100.0 for v in abs_vals]),
                "Avg_Rel_Reduction_Pct": _mean_round([v * 100.0 for v in rel_vals]),
            })

    return rows


# ─────────────────────────────────────────────────────────────
# Reactivity score
# ─────────────────────────────────────────────────────────────

def compute_reactivity_scores(
    group_df: pd.DataFrame,
    full_df: pd.DataFrame,
    rf: ReactivityFilter,
) -> pd.Series:
    """
    Compute R scores for rows in group_df using min-max scaling
    derived from full_df (all candidates for that Name×Warhead group,
    before any filtering).

    R_i = alpha_g * (weighted_avg of scaled features) + bias_g
    """
    scaled = pd.DataFrame(index=full_df.index)
    for col, hib in zip(rf.features, rf.directions):
        lo = pd.to_numeric(full_df[col], errors="coerce").min()
        hi = pd.to_numeric(full_df[col], errors="coerce").max()
        rng = hi - lo
        vals = pd.to_numeric(group_df[col], errors="coerce").reindex(group_df.index)
        if rng == 0:
            scaled.loc[group_df.index, col] = 0.5
        else:
            s = (vals - lo) / rng
            scaled.loc[group_df.index, col] = s if hib else 1.0 - s

    w_total = sum(rf.weights)
    if w_total == 0:
        base_score = pd.Series(0.0, index=group_df.index)
    else:
        base_score = sum(
            w * scaled.loc[group_df.index, col]
            for w, col in zip(rf.weights, rf.features)
        ) / w_total

    r_scores = base_score.copy().astype(float)
    for idx in group_df.index:
        res = str(group_df.loc[idx, "_res_upper"])
        grp = rf.group_map.get(res, None)
        if grp and grp in rf.group_params:
            gp = rf.group_params[grp]
            r_scores[idx] = gp.alpha * base_score[idx] + gp.bias

    return r_scores


# ─────────────────────────────────────────────────────────────
# Parse CLI filter specs
# ─────────────────────────────────────────────────────────────

def parse_simple_filters(tokens: list[str]) -> list[SimpleFilter]:
    if len(tokens) % 3 != 0:
        sys.exit(
            f"[ERROR] --filters requires groups of 3: COL OP VALUE. "
            f"Got {len(tokens)} tokens."
        )
    filters = []
    for i in range(0, len(tokens), 3):
        col, op, val_str = tokens[i], tokens[i+1], tokens[i+2]
        if op not in FILTER_OPS:
            sys.exit(f"[ERROR] Unknown operator '{op}'. Valid: {', '.join(FILTER_OPS)}")
        try:
            val = float(val_str)
        except ValueError:
            sys.exit(f"[ERROR] Filter value must be numeric, got '{val_str}'")
        filters.append(SimpleFilter(col=col, op=op, value=val))
    return filters


def parse_reactivity_groups(specs: list[str]) -> tuple[dict[str, GroupParams], dict[str, str]]:
    """
    Parse --reactivity-groups specs of form  GROUPNAME:RES1,RES2:ALPHA:BIAS
    e.g. S:CYS:2.0:-0.1  O:SER,THR,TYR:0.5:-1.0  N:LYS,HIS:0.5:0.3
    Returns (group_params dict, residue->group_name map).
    """
    group_params: dict[str, GroupParams] = {}
    group_map:    dict[str, str]         = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 4:
            sys.exit(
                f"[ERROR] --reactivity-groups entry '{spec}' must be "
                f"GROUPNAME:RES1,RES2:ALPHA:BIAS"
            )
        gname, residues_str, alpha_str, bias_str = parts
        try:
            alpha = float(alpha_str)
            bias  = float(bias_str)
        except ValueError:
            sys.exit(f"[ERROR] Alpha/bias must be numeric in '{spec}'")
        group_params[gname] = GroupParams(alpha=alpha, bias=bias)
        for res in residues_str.split(","):
            group_map[res.strip().upper()] = gname
    return group_params, group_map


# ─────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────

def run_analysis(
    training_path:       str,
    labels_path:         str,
    simple_filters:      list[SimpleFilter],
    reactivity_filter:   Optional[ReactivityFilter],
    reactivity_position: Optional[int],
    lgbm_filter:         Optional[LGBMRankerFilter],
    lgbm_position:       Optional[int],
    pdb_dir:             Optional[str],
    output_detail:       Optional[str],
    output_summary:      Optional[str],
    warhead_comparison:  bool = False,
    perfect_match:       bool = False,
) -> None:

    # ── Build ordered filter list ─────────────────────────────────────────
    # Exactly one of reactivity_filter / lgbm_filter may be set (enforced in main).
    scored_filter = reactivity_filter if reactivity_filter is not None else lgbm_filter
    scored_position = (
        reactivity_position if reactivity_filter is not None else lgbm_position
    )

    all_filters: list[tuple[int, object]] = []
    for i, f in enumerate(simple_filters):
        all_filters.append((i + 1, f))

    if scored_filter is not None:
        pos = scored_position if scored_position else len(simple_filters) + 1
        react_pos = pos

        ordered: list[object] = []
        si = 0
        for p in range(1, len(simple_filters) + 2):
            if p == react_pos:
                ordered.append(scored_filter)
            else:
                if si < len(simple_filters):
                    ordered.append(simple_filters[si])
                    si += 1
    else:
        ordered = [f for _, f in all_filters]

    n_filters = len(ordered)
    filter_labels = [
        f.label if hasattr(f, "label") else str(f)
        for f in ordered
    ]
    lgbm_filter_indices = {
        i for i, f in enumerate(ordered) if isinstance(f, LGBMRankerFilter)
    }

    # Filter 0 is always present: nucleophilic residue selection
    filter0_label = "Nucleophilic residue selection (Filter 0)"

    # ── Load data ─────────────────────────────────────────────────────────
    train  = load_csv(training_path, "training")
    labels = load_csv(labels_path,   "labels")

    train.columns  = train.columns.str.strip()
    labels.columns = labels.columns.str.strip()

    # Validate that labels has protein pdb column when pdb_dir is given.
    # Accept either "protein_pdb" or "protein pdb" (space variant).
    pdb_col_name = None
    for candidate in ("protein_pdb", "protein pdb"):
        if candidate in labels.columns:
            pdb_col_name = candidate
            break
    has_pdb_col = pdb_col_name is not None
    if pdb_dir and not has_pdb_col:
        sys.exit(
            "[ERROR] --pdb-dir specified but labels CSV has no 'protein_pdb' "
            "or 'protein pdb' column."
        )
    if not pdb_dir and has_pdb_col:
        print("[WARN] Labels has a protein pdb column but --pdb-dir not specified. "
              "Absolute reduction will use nucleophilic candidate count as denominator "
              "(original behaviour).")

    use_pdb_denom = pdb_dir is not None and has_pdb_col

    # Normalise key columns
    train["_name_upper"]    = train["Name"].astype(str).str.strip().str.upper()
    train["_res_upper"]     = train["Residue"].astype(str).str.strip().str.upper()
    train["_warhead_lower"] = train["Warhead"].astype(str).str.strip().str.lower()

    labels["_name_upper"]   = labels["Name"].astype(str).str.strip().str.upper()
    labels["_res_upper"]    = labels["Residue"].astype(str).str.strip().str.upper()
    if has_pdb_col:
        labels["_pdb_id"] = labels[pdb_col_name].astype(str).str.strip()

    # Dedup training by (Name, Residue, Warhead)
    key_cols = ["_name_upper", "_res_upper", "_warhead_lower", "ResNum", "Chain"]
    before_dedup = len(train)
    train = train.drop_duplicates(subset=key_cols, keep="first").copy()
    print(f"[INFO] Training rows after dedup (Name×Residue×Warhead): "
          f"{before_dedup:,} → {len(train):,}")

    # Pre-compute full candidate pools per (Name, Warhead) for score normalisation
    full_pools: dict[tuple, pd.DataFrame] = {}
    if reactivity_filter is not None or lgbm_filter is not None:
        for (name, wh), grp in train.groupby(["_name_upper", "_warhead_lower"]):
            full_pools[(name, wh)] = grp.copy()

    # ── Build label entries ───────────────────────────────────────────────
    label_entries = []
    for _, lrow in labels.iterrows():
        name     = lrow["_name_upper"]
        res      = lrow["_res_upper"]
        res_num  = lrow.get("ResNum", "")
        chain    = lrow.get("Chain", "")
        elec     = lrow.get("electrophile_smiles", "")
        raw_wh   = lrow.get("Frankenstein_Warhead", "")
        wh_set   = parse_frankenstein_warheads(raw_wh)
        pdb_id   = lrow["_pdb_id"] if has_pdb_col else None

        train_whs = set(
            train.loc[train["_name_upper"] == name, "_warhead_lower"].unique()
        )

        if not wh_set:
            plain = lrow.get("Warhead", "")
            if plain and not pd.isna(plain):
                wh_set = {norm(plain)}

        matched_whs   = wh_set & train_whs
        unmatched_whs = wh_set - train_whs

        label_wh_count = len(wh_set)

        if not matched_whs:
            label_entries.append({
                "name":             name,
                "residue":          res,
                "res_num":          res_num,
                "chain":            chain,
                "electrophile":     elec,
                "warhead":          ",".join(sorted(wh_set)) if wh_set else "unknown",
                "matched_wh":       None,
                "is_mismatch":      True,
                "pdb_id":           pdb_id,
                "label_wh_count":   label_wh_count,
            })
        else:
            for wh in sorted(matched_whs):
                label_entries.append({
                    "name":             name,
                    "residue":          res,
                    "res_num":          res_num,
                    "chain":            chain,
                    "electrophile":     elec,
                    "warhead":          wh,
                    "matched_wh":       wh,
                    "is_mismatch":      False,
                    "pdb_id":           pdb_id,
                    "label_wh_count":   label_wh_count,
                })

    print(f"[INFO] Label entries total               : {len(label_entries):,}")
    n_mismatch  = sum(1 for e in label_entries if e["is_mismatch"])
    n_matchable = len(label_entries) - n_mismatch
    print(f"[INFO] Warhead-matched (rankable)         : {n_matchable:,}")
    print(f"[INFO] Warhead mismatches                 : {n_mismatch:,}")
    if warhead_comparison:
        print("[INFO] Hit counting mode                : label × warhead pair")
    elif perfect_match:
        print("[INFO] Hit counting mode                : one hit max per label "
              "(perfect-match: all training-matched warheads must hit when label "
              "lists multiple Frankenstein warheads)")
    else:
        print("[INFO] Hit counting mode                : one hit max per label "
              "(any matched warhead hit counts; use --warhead_comparison for "
              "per-warhead breakdown)")

    # ── Pre-fetch PDB residue counts ──────────────────────────────────────
    # Do this upfront so downloads are batched before the main loop.
    pdb_residue_counts: dict[str, int] = {}   # pdb_id_upper -> count
    if use_pdb_denom:
        unique_pdb_ids = {
            e["pdb_id"] for e in label_entries
            if e["pdb_id"] and not pd.isna(e["pdb_id"]) and e["pdb_id"] != "nan"
        }
        print(f"[INFO] Resolving {len(unique_pdb_ids)} unique PDB ID(s)...")
        for pid in sorted(unique_pdb_ids):
            pdb_residue_counts[pid.upper()] = get_pdb_residue_count(pid, pdb_dir)

    # ── Per-entry analysis ────────────────────────────────────────────────
    # Accumulators for Filter 0 (nucleophilic selection) and Filters 1..N
    from collections import defaultdict

    filter0_abs_reductions: list[float] = []   # one per matchable entry
    filter0_rel_reductions: list[float] = []

    filter_abs_reductions  = [[] for _ in range(n_filters)]
    filter_rel_reductions  = [[] for _ in range(n_filters)]

    filter0_n_before_counts: list[float] = []
    filter0_n_after_counts:  list[float] = []
    filter_n_before_counts = [[] for _ in range(n_filters)]
    filter_n_after_counts  = [[] for _ in range(n_filters)]

    filter0_abs_by_residue: dict[str, list[float]] = defaultdict(list)
    filter0_rel_by_residue: dict[str, list[float]] = defaultdict(list)
    filter0_n_before_by_residue: dict[str, list[float]] = defaultdict(list)
    filter0_n_after_by_residue: dict[str, list[float]] = defaultdict(list)
    filter_abs_by_residue = [defaultdict(list) for _ in range(n_filters)]
    filter_rel_by_residue = [defaultdict(list) for _ in range(n_filters)]
    filter_n_before_by_residue = [defaultdict(list) for _ in range(n_filters)]
    filter_n_after_by_residue = [defaultdict(list) for _ in range(n_filters)]
    filter_n_before_if_match = [[] for _ in range(n_filters)]
    filter_n_after_if_match  = [[] for _ in range(n_filters)]
    filter_n_before_if_match_by_residue = [defaultdict(list) for _ in range(n_filters)]
    filter_n_after_if_match_by_residue  = [defaultdict(list) for _ in range(n_filters)]

    detail_rows = []

    for entry in label_entries:
        name    = entry["name"]
        res     = entry["residue"]
        wh      = entry["matched_wh"]
        is_mis  = entry["is_mismatch"]
        pdb_id  = entry["pdb_id"]

        row = {
            "Name":                 name,
            "Residue":              res,
            "ResNum":               entry["res_num"],
            "Chain":                entry["chain"],
            "Electrophile":         entry["electrophile"],
            "Warhead":              entry["warhead"],
            "Is_Mismatch":          is_mis,
            "Label_Warhead_Count":  entry.get("label_wh_count", 1),
        }

        if use_pdb_denom and pdb_id and pdb_id.upper() in pdb_residue_counts:
            n_pdb = pdb_residue_counts[pdb_id.upper()]
        else:
            n_pdb = None   # fallback: will use n_initial as before

        if is_mis:
            row["Hit"]            = False
            row["Miss_Reason"]    = "warhead_mismatch"
            row["Failing_Filter"] = None
            row["Filter0_Starting_Residue_Count"] = None
            row["Starting_Residue_Count"] = None
            row["Deprotonation_Target_Rank"] = None
            row["LGBM_Target_Rank"] = None
            row["Combo_Target_Rank"] = None
            row[f"Filter0_{filter0_label}_AbsReduction"] = None
            row[f"Filter0_{filter0_label}_RelReduction"] = None
            for i, fl in enumerate(filter_labels):
                row[f"Filter{i+1}_{fl}_AbsReduction"] = None
                row[f"Filter{i+1}_{fl}_RelReduction"] = None
                row[f"Filter{i+1}_{fl}_Passed"]       = None
                row[f"Filter{i+1}_{fl}_ResidueCount"] = None
                if i in lgbm_filter_indices:
                    row[f"Filter{i+1}_{fl}_LGBM_Ranked_Residues"] = None
            row["Total_AbsReduction"] = None
            row["Total_RelReduction"] = None
            detail_rows.append(row)
            continue

        # Get candidates for this (Name, Warhead) group
        group_mask = (
            (train["_name_upper"]    == name) &
            (train["_warhead_lower"] == wh)
        )
        group_df = train[group_mask].copy()

        if group_df.empty:
            row["Hit"]            = False
            row["Miss_Reason"]    = "no_training_rows"
            row["Failing_Filter"] = None
            row["Filter0_Starting_Residue_Count"] = None
            row["Starting_Residue_Count"] = None
            row["Deprotonation_Target_Rank"] = None
            row["LGBM_Target_Rank"] = None
            row["Combo_Target_Rank"] = None
            row[f"Filter0_{filter0_label}_AbsReduction"] = None
            row[f"Filter0_{filter0_label}_RelReduction"] = None
            for i, fl in enumerate(filter_labels):
                row[f"Filter{i+1}_{fl}_AbsReduction"] = None
                row[f"Filter{i+1}_{fl}_RelReduction"] = None
                row[f"Filter{i+1}_{fl}_Passed"]       = None
                row[f"Filter{i+1}_{fl}_ResidueCount"] = None
                if i in lgbm_filter_indices:
                    row[f"Filter{i+1}_{fl}_LGBM_Ranked_Residues"] = None
            row["Total_AbsReduction"] = None
            row["Total_RelReduction"] = None
            detail_rows.append(row)
            continue

        n_initial = len(group_df)   # nucleophilic candidates in training

        # Denominator for absolute reductions:
        # Use PDB total if available, else fall back to n_initial (original behaviour)
        n_denom = n_pdb if (n_pdb is not None and n_pdb > 0) else n_initial

        # ── Filter 0: nucleophilic residue selection ──────────────────────
        # n_before = n_denom (full PDB), n_after = n_initial (nucleophilic candidates)
        f0_abs = (n_denom - n_initial) / n_denom if n_denom > 0 else 0.0
        f0_rel = f0_abs   # same denominator for both when n_before = n_denom
        filter0_abs_reductions.append(f0_abs)
        filter0_rel_reductions.append(f0_rel)
        filter0_n_before_counts.append(float(n_denom))
        filter0_n_after_counts.append(float(n_initial))
        filter0_abs_by_residue[res].append(f0_abs)
        filter0_rel_by_residue[res].append(f0_rel)
        filter0_n_before_by_residue[res].append(float(n_denom))
        filter0_n_after_by_residue[res].append(float(n_initial))
        row[f"Filter0_{filter0_label}_AbsReduction"] = round(f0_abs, 4)
        row[f"Filter0_{filter0_label}_RelReduction"] = round(f0_rel, 4)

        # ── Filters 1..N ──────────────────────────────────────────────────
        full_pool    = full_pools.get((name, wh), group_df)
        current_df   = group_df.copy()
        target_alive = target_site_in_df(
            current_df, res, entry["res_num"], entry["chain"]
        )

        failing_filter = None
        filter_results: list[dict] = []
        lgbm_type_scores: dict[str, float] | None = None

        for f_idx, filt in enumerate(ordered):
            target_alive_at_entry = target_alive
            n_before = len(current_df)
            lgbm_ranked_residues = None

            if isinstance(filt, SimpleFilter):
                pass_mask = filt.apply(current_df)
                next_df   = current_df[pass_mask].copy()

            elif isinstance(filt, (ReactivityFilter, LGBMRankerFilter)):
                if len(current_df) == 0:
                    next_df = current_df.copy()
                else:
                    type_df = current_df.drop_duplicates(subset=["_res_upper"]).copy()
                    if isinstance(filt, ReactivityFilter):
                        scores = compute_reactivity_scores(type_df, full_pool, filt)
                    else:
                        scores = compute_lgbm_scores(type_df, full_pool, filt)
                    type_df["_score"] = scores
                    type_df["_rank"] = type_df["_score"].rank(
                        method="min", ascending=False
                    )
                    if isinstance(filt, LGBMRankerFilter):
                        lgbm_type_scores = dict(
                            zip(
                                type_df["_res_upper"].astype(str),
                                type_df["_score"].astype(float),
                            )
                        )
                        type_df_sorted = type_df.sort_values(
                            "_score", ascending=False
                        )
                        lgbm_ranked_residues = ",".join(
                            type_df_sorted["_res_upper"].astype(str).tolist()
                        )
                    surviving_types = type_df[
                        type_df["_rank"] <= filt.top_k
                    ]["_res_upper"].values
                    next_df = current_df[
                        current_df["_res_upper"].isin(surviving_types)
                    ].copy()
            else:
                next_df = current_df.copy()

            n_after = len(next_df)

            filter_n_before_counts[f_idx].append(n_before)
            filter_n_after_counts[f_idx].append(n_after)

            # Absolute reduction: denominator is always n_denom (PDB total or n_initial)
            abs_red = (n_before - n_after) / n_denom if n_denom > 0 else 0.0
            # Relative reduction: local step denominator (unchanged)
            rel_red = (n_before - n_after) / n_before if n_before > 0 else 0.0

            filter_abs_reductions[f_idx].append(abs_red)
            filter_rel_reductions[f_idx].append(rel_red)
            filter_n_before_by_residue[f_idx][res].append(n_before)
            filter_n_after_by_residue[f_idx][res].append(n_after)
            filter_abs_by_residue[f_idx][res].append(abs_red)
            filter_rel_by_residue[f_idx][res].append(rel_red)

            if target_alive_at_entry:
                nb, na = float(n_before), float(n_after)
                filter_n_before_if_match[f_idx].append(nb)
                filter_n_after_if_match[f_idx].append(na)
                filter_n_before_if_match_by_residue[f_idx][res].append(nb)
                filter_n_after_if_match_by_residue[f_idx][res].append(na)

            target_survived = target_site_in_df(
                next_df, res, entry["res_num"], entry["chain"]
            )
            filter_passed   = target_survived or not target_alive

            if target_alive and not target_survived:
                if failing_filter is None:
                    failing_filter = f_idx + 1
                target_alive = False

            filter_results.append({
                "abs_red": abs_red,
                "rel_red": rel_red,
                "passed":  filter_passed,
                "residue_count": n_after,
                "lgbm_ranked_residues": lgbm_ranked_residues,
            })

            current_df = next_df

        # Overall outcome
        hit = target_alive

        # Total abs reduction: (n_denom → final survivors) / n_denom
        total_abs = (n_denom - len(current_df)) / n_denom if n_denom > 0 else 0.0
        # Include Filter 0 in total: (n_denom → final survivors) already covers it
        # since n_denom is the PDB total (Filter 0 input).

        row["Hit"]            = hit
        row["Miss_Reason"]    = None if hit else "filtered_out"
        row["Failing_Filter"] = failing_filter
        row["Filter0_Starting_Residue_Count"] = n_denom
        row["Starting_Residue_Count"] = n_initial
        row["Deprotonation_Target_Rank"] = None
        row["LGBM_Target_Rank"] = None
        row["Combo_Target_Rank"] = None

        if hit and len(current_df) > 0:
            target_mask = target_site_mask(
                current_df, res, entry["res_num"], entry["chain"]
            )

            if "deprotonation_prob" in current_df.columns:
                deprot_scores = pd.to_numeric(
                    current_df["deprotonation_prob"], errors="coerce"
                )
                deprot_ranks = deprot_scores.rank(method="min", ascending=False)
                if target_mask.any():
                    row["Deprotonation_Target_Rank"] = int(
                        deprot_ranks[target_mask].min()
                    )

            if lgbm_filter is not None and lgbm_type_scores:
                surviving_types = current_df["_res_upper"].astype(str).unique()
                type_scores = pd.Series(
                    {
                        t: lgbm_type_scores.get(t, float("-inf"))
                        for t in surviving_types
                    }
                )
                survivor_type_ranks = type_scores.rank(method="min", ascending=False)
                if res in survivor_type_ranks.index:
                    row["LGBM_Target_Rank"] = int(survivor_type_ranks[res])

                k = max(1, int(lgbm_filter.top_k))
                reactivity_by_type = (
                    1.0 - (survivor_type_ranks - 1.0) / float(k)
                ).clip(lower=0.0)
                reactivity_component = current_df["_res_upper"].astype(str).map(
                    reactivity_by_type
                )
                if "deprotonation_prob" in current_df.columns:
                    combo_scores = deprot_scores.fillna(0.0) + reactivity_component
                    combo_ranks = combo_scores.rank(method="min", ascending=False)
                    if target_mask.any():
                        row["Combo_Target_Rank"] = int(combo_ranks[target_mask].min())

        for f_idx, (fl, fr) in enumerate(zip(filter_labels, filter_results)):
            row[f"Filter{f_idx+1}_{fl}_AbsReduction"] = round(fr["abs_red"], 4)
            row[f"Filter{f_idx+1}_{fl}_RelReduction"] = round(fr["rel_red"], 4)
            row[f"Filter{f_idx+1}_{fl}_Passed"]       = fr["passed"]
            row[f"Filter{f_idx+1}_{fl}_ResidueCount"] = fr["residue_count"]
            if fr.get("lgbm_ranked_residues") is not None:
                row[f"Filter{f_idx+1}_{fl}_LGBM_Ranked_Residues"] = (
                    fr["lgbm_ranked_residues"]
                )

        row["Total_AbsReduction"] = round(total_abs, 4)
        row["Total_RelReduction"] = round(total_abs, 4)  # same denom for total

        detail_rows.append(row)

    # ── Summary statistics ────────────────────────────────────────────────
    warhead_matchable_rows = [r for r in detail_rows if not r["Is_Mismatch"]]

    if warhead_comparison:
        final_detail_rows = annotate_warhead_detail_rows(detail_rows)
    else:
        final_detail_rows = aggregate_label_detail_rows(
            detail_rows,
            perfect_match=perfect_match,
        )

    # Hit rate uses label- or warhead-level rows depending on mode.
    matchable_rows = [r for r in final_detail_rows if not r["Is_Mismatch"]]
    hit_rows       = [r for r in matchable_rows if r["Hit"]]

    n_matchable_total = len(matchable_rows)
    n_hits            = len(hit_rows)
    overall_hit_rate  = n_hits / n_matchable_total if n_matchable_total > 0 else 0.0
    deprot_hit_ranks = [
        float(r["Deprotonation_Target_Rank"])
        for r in hit_rows
        if r.get("Deprotonation_Target_Rank") is not None
        and not pd.isna(r.get("Deprotonation_Target_Rank"))
    ]
    lgbm_hit_ranks = [
        float(r["LGBM_Target_Rank"])
        for r in hit_rows
        if r.get("LGBM_Target_Rank") is not None
        and not pd.isna(r.get("LGBM_Target_Rank"))
    ]
    combo_hit_ranks = [
        float(r["Combo_Target_Rank"])
        for r in hit_rows
        if r.get("Combo_Target_Rank") is not None
        and not pd.isna(r.get("Combo_Target_Rank"))
    ]

    from collections import defaultdict
    res_hits   = defaultdict(int)
    res_totals = defaultdict(int)
    for r in matchable_rows:
        res_totals[r["Residue"]] += 1
        if r["Hit"]:
            res_hits[r["Residue"]] += 1

    # Per-filter failure counts: one per warhead evaluation (matches Total_N stats).
    filter_analysis_residues = DEFAULT_RESIDUE_TYPES
    failure_count_rows = warhead_matchable_rows
    filter_fail_counts_nuc = [0] * n_filters
    for r in failure_count_rows:
        if r["Residue"] not in filter_analysis_residues:
            continue
        if r["Hit"]:
            continue
        failing_filter = r.get("Failing_Filter")
        if failing_filter is not None and not pd.isna(failing_filter):
            filter_fail_counts_nuc[int(failing_filter) - 1] += 1
    filter_fail_by_residue = compute_filter_fail_counts_by_residue(
        failure_count_rows,
        n_filters,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PREFILTER ANALYSIS SUMMARY")
    print("=" * 70)

    if warhead_comparison:
        hit_mode = "label × warhead pairs"
    elif perfect_match:
        hit_mode = "labels (perfect-match: all matched warheads must hit)"
    else:
        hit_mode = "labels (one hit max per label, any matched warhead)"
    print(f"\n  Overall hit rate (target survives all filters) [{hit_mode}]:")
    print(f"    {n_hits} / {n_matchable_total}  =  {overall_hit_rate*100:.1f}%")
    print(f"    (excludes {n_mismatch} warhead-mismatched entries)")

    print(f"\n  Hit rate by residue type:")
    print(f"  {'Residue':<10} {'Hits':>6} {'Total':>6} {'Rate':>8}")
    print("  " + "-" * 34)
    for res in sorted(res_totals):
        h = res_hits[res]
        t = res_totals[res]
        print(f"  {res:<10} {h:>6} {t:>6} {h/t*100:>7.1f}%")

    print(f"\n  Warhead mismatches: {n_mismatch:,}")

    nuc_note = (
        f" (nucleophilic residues only: {', '.join(sorted(filter_analysis_residues))})"
    )
    print(f"\n  Per-filter failure counts (target removed at this filter)"
          f" [per warhead evaluation; hit rate above: {hit_mode}]{nuc_note}:")
    print(f"  {'#':<4} {'Filter':<55} {'Failures':>8}")
    print("  " + "-" * 70)
    for i, fl in enumerate(filter_labels):
        print(f"  {i+1:<4} {fl:<55} {filter_fail_counts_nuc[i]:>8}")

    print(f"\n  Per-filter average search-space reduction{nuc_note}:")
    denom_note = "(denom = PDB total residues)" if use_pdb_denom else "(denom = nucleophilic candidates)"
    print(f"  Absolute reduction {denom_note}")
    print(f"  {'#':<4} {'Filter':<45} {'Abs Reduction':>15} {'Rel Reduction':>15}")
    print("  " + "-" * 82)

    # Filter 0
    f0_abs_vals = _flatten_residue_vals(filter0_abs_by_residue, filter_analysis_residues)
    f0_rel_vals = _flatten_residue_vals(filter0_rel_by_residue, filter_analysis_residues)
    avg_f0_abs = np.mean(f0_abs_vals) if f0_abs_vals else 0.0
    avg_f0_rel = np.mean(f0_rel_vals) if f0_rel_vals else 0.0
    print(f"  {'0':<4} {'Nucleophilic residue selection':<45} "
          f"{avg_f0_abs*100:>14.1f}%  {avg_f0_rel*100:>14.1f}%")

    for i, fl in enumerate(filter_labels):
        abs_vals = _flatten_residue_vals(filter_abs_by_residue[i], filter_analysis_residues)
        rel_vals = _flatten_residue_vals(filter_rel_by_residue[i], filter_analysis_residues)
        avg_abs  = np.mean(abs_vals) if abs_vals else 0.0
        avg_rel  = np.mean(rel_vals) if rel_vals else 0.0
        print(f"  {i+1:<4} {fl:<45} {avg_abs*100:>14.1f}%  {avg_rel*100:>14.1f}%")

    total_abs_all = [r["Total_AbsReduction"] for r in warhead_matchable_rows
                     if r["Total_AbsReduction"] is not None]
    avg_total_abs = np.mean(total_abs_all) if total_abs_all else 0.0
    print(f"\n  Average total search-space reduction (Filter 0 through final): "
          f"{avg_total_abs*100:.1f}%")
    print("=" * 70)

    # ── Export detail CSV ─────────────────────────────────────────────────
    detail_df = pd.DataFrame(final_detail_rows)
    if output_detail:
        detail_df.to_csv(output_detail, index=False)
        print(f"\n[INFO] Detail output written to: {output_detail}")
    else:
        print(f"\n[Detail] {len(detail_df)} rows (use --output-detail to save CSV)")

    # ── Export summary CSV ────────────────────────────────────────────────
    summary_rows = []
    summary_rows.append({
        "Group":    "Overall",
        "Category": "All",
        "N_Matchable":       n_matchable_total,
        "N_Hits":            n_hits,
        "N_Misses":          n_matchable_total - n_hits,
        "Hit_Rate_Pct":      round(overall_hit_rate * 100, 1),
        "N_Warhead_Mismatch": n_mismatch,
        "Avg_Deprotonation_Target_Rank_Hit_Only": (
            round(float(np.mean(deprot_hit_ranks)), 4) if deprot_hit_ranks else None
        ),
        "Avg_LGBM_Target_Rank_Hit_Only": (
            round(float(np.mean(lgbm_hit_ranks)), 4) if lgbm_hit_ranks else None
        ),
        "Avg_Combo_Target_Rank_Hit_Only": (
            round(float(np.mean(combo_hit_ranks)), 4) if combo_hit_ranks else None
        ),
    })
    for res in sorted(res_totals):
        h = res_hits[res]
        t = res_totals[res]
        summary_rows.append({
            "Group":    "Residue_Type",
            "Category": res,
            "N_Matchable":       t,
            "N_Hits":            h,
            "N_Misses":          t - h,
            "Hit_Rate_Pct":      round(h / t * 100, 1) if t else None,
            "N_Warhead_Mismatch": None,
        })
    # Filter 0 (nucleophilic residue types only)
    f0_n_before_vals = _flatten_residue_vals(
        filter0_n_before_by_residue, filter_analysis_residues
    )
    f0_n_after_vals = _flatten_residue_vals(
        filter0_n_after_by_residue, filter_analysis_residues
    )
    avg_f0_n_before, total_f0_n_before, f0_n_count = _avg_sum_count(f0_n_before_vals)
    avg_f0_n_after,  total_f0_n_after,  _          = _avg_sum_count(f0_n_after_vals)
    f0_im_before_vals = f0_n_before_vals
    f0_im_after_vals  = f0_n_after_vals
    summary_rows.append({
        "Group":    "Filter",
        "Category": filter0_label,
        "Filter":   filter0_label,
        "N_Failures":            0,
        **_filter_n_summary_columns(f0_n_before_vals, f0_n_after_vals),
        **_filter_n_summary_columns(f0_im_before_vals, f0_im_after_vals, if_match=True),
        "Avg_Abs_Reduction_Pct": round(avg_f0_abs * 100, 1) if f0_abs_vals else None,
        "Avg_Rel_Reduction_Pct": round(avg_f0_rel * 100, 1) if f0_rel_vals else None,
    })
    for i, fl in enumerate(filter_labels):
        abs_vals = _flatten_residue_vals(filter_abs_by_residue[i], filter_analysis_residues)
        rel_vals = _flatten_residue_vals(filter_rel_by_residue[i], filter_analysis_residues)
        n_before_vals = _flatten_residue_vals(
            filter_n_before_by_residue[i], filter_analysis_residues
        )
        n_after_vals = _flatten_residue_vals(
            filter_n_after_by_residue[i], filter_analysis_residues
        )
        im_before_vals = _flatten_residue_vals(
            filter_n_before_if_match_by_residue[i], filter_analysis_residues
        )
        im_after_vals = _flatten_residue_vals(
            filter_n_after_if_match_by_residue[i], filter_analysis_residues
        )
        summary_rows.append({
            "Group":    "Filter",
            "Category": fl,
            "Filter":   fl,
            "N_Failures":            filter_fail_counts_nuc[i],
            **_filter_n_summary_columns(n_before_vals, n_after_vals),
            **_filter_n_summary_columns(im_before_vals, im_after_vals, if_match=True),
            "Avg_Abs_Reduction_Pct": round(np.mean(abs_vals) * 100, 1) if abs_vals else None,
            "Avg_Rel_Reduction_Pct": round(np.mean(rel_vals) * 100, 1) if rel_vals else None,
        })
    filter_residue_order = sorted(
        r for r in res_totals if r in filter_analysis_residues
    )
    summary_rows.extend(
        build_filter_residue_summary_rows(
            residue_order=filter_residue_order,
            res_totals=res_totals,
            filter0_label=filter0_label,
            filter_labels=filter_labels,
            filter_fail_by_residue=filter_fail_by_residue,
            filter0_abs_by_residue=dict(filter0_abs_by_residue),
            filter0_rel_by_residue=dict(filter0_rel_by_residue),
            filter0_n_before_by_residue=dict(filter0_n_before_by_residue),
            filter0_n_after_by_residue=dict(filter0_n_after_by_residue),
            filter_abs_by_residue=[dict(d) for d in filter_abs_by_residue],
            filter_rel_by_residue=[dict(d) for d in filter_rel_by_residue],
            filter_n_before_by_residue=[dict(d) for d in filter_n_before_by_residue],
            filter_n_after_by_residue=[dict(d) for d in filter_n_after_by_residue],
            filter0_n_before_if_match_by_residue=dict(filter0_n_before_by_residue),
            filter0_n_after_if_match_by_residue=dict(filter0_n_after_by_residue),
            filter_n_before_if_match_by_residue=[dict(d) for d in filter_n_before_if_match_by_residue],
            filter_n_after_if_match_by_residue=[dict(d) for d in filter_n_after_if_match_by_residue],
        )
    )
    summary_df = pd.DataFrame(summary_rows)
    if output_summary:
        summary_df.to_csv(output_summary, index=False)
        print(f"[INFO] Summary output written to: {output_summary}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sequential prefilter analysis for covalent reactivity pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--training",  required=True, help="Training CSV path.")
    parser.add_argument("--labels",    required=True, help="Labels CSV path.")
    parser.add_argument(
        "--pdb-dir", default=None, metavar="DIR",
        help=(
            "Directory to find/store PDB files. If a PDB is missing it will be "
            "downloaded from RCSB automatically. Required to use PDB total residue "
            "count as the absolute-reduction denominator (recommended)."
        ),
    )
    parser.add_argument(
        "--filters", nargs="*", default=[], metavar="TOKEN",
        help="Simple filters as groups of 3: COL  gt|lt|gte|lte|eq  VALUE  ..."
    )

    # Reactivity filter (Bayesian / manual weighted scorer)
    parser.add_argument("--reactivity-position",   type=int,   default=None)
    parser.add_argument("--reactivity-features",   nargs="+",  default=None)
    parser.add_argument("--reactivity-directions", nargs="+",  default=None,
                        choices=["higher", "lower"], metavar="higher|lower")
    parser.add_argument("--reactivity-weights",    nargs="+",  type=float, default=None)
    parser.add_argument(
        "--reactivity-groups", nargs="+", default=None,
        metavar="GROUPNAME:RES1,RES2:ALPHA:BIAS",
    )
    parser.add_argument("--reactivity-top-k", type=int, default=3)

    # LightGBM ranker filter (mutually exclusive with --reactivity-features)
    parser.add_argument(
        "--lgbm-model", default=None, metavar="PATH",
        help="Path to LGBMRanker pkl bundle. Mutually exclusive with --reactivity-features.",
    )
    parser.add_argument(
        "--lgbm-position", type=int, default=None,
        help="1-indexed position in filter sequence for the LGBM ranker filter.",
    )
    parser.add_argument(
        "--lgbm-top-k", type=int, default=None,
        help="Keep top-K residues by LGBM score. Overrides the value stored in the pkl.",
    )

    parser.add_argument("--output-detail",  default=None, metavar="PATH")
    parser.add_argument("--output-summary", default=None, metavar="PATH")
    parser.add_argument(
        "--warhead-comparison", action="store_true",
        help=(
            "Count hits per label × warhead pair instead of one hit max per label. "
            "Default mode deduplicates by electrophile target site (Name, Residue, "
            "ResNum, Chain) and counts a hit if any matching warhead survives all filters."
        ),
    )
    parser.add_argument(
        "--perfect-match", action="store_true",
        help=(
            "For labels with multiple comma-separated Frankenstein warheads, require "
            "every warhead present in both the label and training to keep the target "
            "residue alive. Label warheads absent from training are not required. "
            "Only applies in default label mode (not with --warhead-comparison)."
        ),
    )

    args = parser.parse_args()

    if args.perfect_match and args.warhead_comparison:
        sys.exit(
            "[ERROR] --perfect-match cannot be used with --warhead-comparison."
        )

    # ── Mutual exclusivity: reactivity vs lgbm ────────────────────────────
    if args.reactivity_features and args.lgbm_model:
        sys.exit(
            "[ERROR] --reactivity-features and --lgbm-model are mutually exclusive. "
            "Use one or the other."
        )

    simple_filters = parse_simple_filters(args.filters) if args.filters else []

    # ── Build reactivity filter (if chosen) ───────────────────────────────
    reactivity_filter = None
    if args.reactivity_features:
        feats = args.reactivity_features
        n_f   = len(feats)

        if args.reactivity_directions is None:
            directions = [True] * n_f
        else:
            if len(args.reactivity_directions) != n_f:
                sys.exit(
                    f"[ERROR] --reactivity-directions has {len(args.reactivity_directions)} "
                    f"values but --reactivity-features has {n_f}."
                )
            directions = [d == "higher" for d in args.reactivity_directions]

        if args.reactivity_weights is None:
            weights = [1.0] * n_f
        else:
            if len(args.reactivity_weights) != n_f:
                sys.exit(
                    f"[ERROR] --reactivity-weights has {len(args.reactivity_weights)} "
                    f"values but --reactivity-features has {n_f}."
                )
            weights = args.reactivity_weights

        if args.reactivity_groups:
            group_params, group_map = parse_reactivity_groups(args.reactivity_groups)
        else:
            group_params = {
                "S": GroupParams(alpha=1.0, bias=0.0),
                "O": GroupParams(alpha=1.0, bias=0.0),
                "N": GroupParams(alpha=1.0, bias=0.0),
            }
            group_map = {
                "CYS": "S", "SER": "O", "THR": "O", "TYR": "O",
                "LYS": "N", "HIS": "N",
            }

        reactivity_filter = ReactivityFilter(
            features=feats,
            directions=directions,
            weights=weights,
            group_params=group_params,
            group_map=group_map,
            top_k=args.reactivity_top_k,
        )

    # ── Build LGBM filter (if chosen) ─────────────────────────────────────
    lgbm_filter = None
    if args.lgbm_model:
        lgbm_filter = load_lgbm_filter(args.lgbm_model, args.lgbm_top_k)
        print(
            f"[INFO] Loaded LGBM ranker from '{args.lgbm_model}' | "
            f"features: {lgbm_filter.feature_cols} | "
            f"normalize: {lgbm_filter.normalize} | "
            f"top-K: {lgbm_filter.top_k} | "
            f"residue types: {sorted(lgbm_filter.residue_types)}"
        )

    if not simple_filters and reactivity_filter is None and lgbm_filter is None:
        sys.exit(
            "[ERROR] No filters specified. Use --filters and/or "
            "--reactivity-features or --lgbm-model."
        )

    run_analysis(
        training_path        = args.training,
        labels_path          = args.labels,
        simple_filters       = simple_filters,
        reactivity_filter    = reactivity_filter,
        reactivity_position  = args.reactivity_position,
        lgbm_filter          = lgbm_filter,
        lgbm_position        = args.lgbm_position,
        pdb_dir              = args.pdb_dir,
        output_detail        = args.output_detail,
        output_summary       = args.output_summary,
        warhead_comparison   = args.warhead_comparison,
        perfect_match        = args.perfect_match,
    )


if __name__ == "__main__":
    main()