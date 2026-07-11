"""
Cov_Screen.py — evaluate a saved LGBMRanker pkl on training + labels CSVs.

Uses the same (Name × Warhead) query groups, warhead matching, and metrics as
Training_Cov_Screen.py, but skips training and only runs inference + reporting.

Hit counting is one row per label (Name + Residue + ResNum + Chain). By default
a label is a hit if ANY training-matched warhead succeeds. With --perfect-match,
labels that list multiple Frankenstein warheads require EVERY training-matched
warhead query group to hit.

Usage:
    python Cov_Screen.py \\
        --model ./cov_lgbm_output/lgbm_ranker.pkl \\
        --training training_bo.csv \\
        --labels labels_bo.csv \\
        [--topk 15] \\
        [--perfect-match] \\
        [--reward-mode hit_at_k | hit_at_top_pct] \\
        [--top-pct 10] \\
        [--export-results ./screen_label_results.csv] \\
        [--export-warhead-accuracy ./screen_warhead_accuracy.csv] \\
        [--export-residue-accuracy ./screen_residue_accuracy.csv] \\
        [--export-shap ./screen_shap_per_candidate.csv] \\
        [--export-query-groups ./screen_query_group_results.csv] \\
        [--export-scores ./screen_scores.csv]

    python Cov_Screen.py   --model lgbm_ranker_nc.pkl   --training training_bo_extended_full.csv   --labels labels_figures.csv   --reward-mode hit_at_top_pct--top-pct 10   --export-results ./figures_screen_results.csv   --export-scores ./figures_screen_results.csv   --export-summary ./figures_summary.json --perfect-match

    # Virtual screening: rank unique ligand Names across all training sites of --residue.
    # Labels provide ligand Names and warhead matching only (Residue/ResNum/Chain ignored).
    python Cov_Screen.py \\
        --VS \\
        --model ./cov_lgbm_output/lgbm_ranker.pkl \\
        --training vs_training.csv \\
        --labels vs_labels.csv \\
        --residue CYS \\
        [--chain A] [--resnum 797] \\
        [--export-vs-results ./vs_results.csv]

    # VS by mean site pred_score (no intra/inter ranks; higher avg_pred_score = better):
    python Cov_Screen.py \\
        --VS --pred-score \\
        --model ./cov_lgbm_output/lgbm_ranker.pkl \\
        --training vs_training.csv \\
        --labels vs_labels.csv \\
        --residue CYS \\
        [--export-vs-results ./vs_pred_score_results.csv]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Training_Cov_Screen import (
    DEFAULT_RANK_BONUS_EPSILON,
    REWARD_MODES,
    VALID_RESIDUES,
    _format_rank_when_hit,
    _format_ss_when_hit,
    _format_ss_when_miss,
    _matching_training_warheads,
    _normalize_resnum,
    evaluate_predictions,
    load_and_merge,
    summarize_eval_metrics,
    summarize_screen_metrics,
    analyze_residue_composition,
    print_residue_composition,
    export_shap_csvs,
)


def make_label_key(name, residue, resnum, chain) -> tuple[str, str, str, str]:
    return (
        str(name).strip().upper(),
        str(residue).strip().upper(),
        _normalize_resnum(resnum),
        str(chain).strip().upper(),
    )


def vs_site_column(residue: str, resnum, chain: str) -> str:
    """Column key for a residue site, e.g. cys797A."""
    return (
        f"{str(residue).strip().lower()}"
        f"{_normalize_resnum(resnum)}"
        f"{str(chain).strip().upper()}"
    )


def _active_hit_column(reward_mode: str) -> str:
    return "hit_at_top_pct" if reward_mode == "hit_at_top_pct" else "hit_at_k"


def _hits_from_mean_rank(
    mean_rank: float,
    n_residues: float,
    k: int,
    top_pct_threshold: float,
) -> tuple[int, int, float]:
    """
    Derive Hit@K and Hit@top-% from a mean intra-protein rank (VS warhead aggregate).
    """
    p_frac = top_pct_threshold / 100.0
    hit_at_k = int(mean_rank <= k)
    if n_residues <= 1:
        rank_frac = 0.0
    else:
        rank_frac = (mean_rank - 1) / (n_residues - 1)
    hit_at_top_pct = int(rank_frac <= p_frac) if p_frac > 0 else hit_at_k
    return hit_at_k, hit_at_top_pct, float(rank_frac)


def filter_vs_site_eval(
    site_eval_df: pd.DataFrame,
    residue: str,
    chain: str | None = None,
    resnum: str | int | None = None,
) -> pd.DataFrame:
    """Keep site-eval rows for VS ranking (residue type, optional chain/resnum)."""
    if site_eval_df.empty:
        return site_eval_df.copy()

    residue_upper = str(residue).strip().upper()
    if residue_upper not in VALID_RESIDUES:
        sys.exit(
            f"[ERROR] --residue must be one of {sorted(VALID_RESIDUES)}, got {residue!r}"
        )

    out = site_eval_df[
        site_eval_df["Residue"].astype(str).str.strip().str.upper() == residue_upper
    ].copy()
    if chain is not None:
        chain_upper = str(chain).strip().upper()
        out = out[
            out["Chain"].astype(str).str.strip().str.upper() == chain_upper
        ]
    if resnum is not None:
        target_resnum = _normalize_resnum(resnum)
        out = out[out["ResNum"].map(_normalize_resnum) == target_resnum]

    return out.reset_index(drop=True)


def build_name_warhead_counts(labels_csv: str) -> dict[str, int]:
    """Frankenstein warhead count per ligand Name (for perfect-match in VS)."""
    labels = pd.read_csv(labels_csv, sep=",", engine="c", low_memory=False)
    counts: dict[str, int] = {}

    for _, row in labels.iterrows():
        name = str(row["Name"]).strip().upper()
        frank = row.get("Frankenstein_Warhead", row.get("Warhead", ""))
        if pd.isna(frank) or not str(frank).strip():
            wh_set = (
                {str(row.get("Warhead", "")).strip().lower()}
                if pd.notna(row.get("Warhead")) else set()
            )
            wh_set.discard("")
            n = len(wh_set) if wh_set else 1
        else:
            n = len([w for w in str(frank).split(",") if w.strip()])
            n = max(n, 1)
        counts[name] = max(counts.get(name, 0), n)

    return counts


def _format_site_label(residue, resnum, chain) -> str:
    res = str(residue).strip().upper()
    num = _normalize_resnum(resnum)
    ch = str(chain).strip().upper()
    return f"{res} {num}/{ch}" if ch else f"{res} {num}"


def _check_warhead_coverage(
    *,
    perfect_match: bool,
    expected_count: int,
    n_matched: int,
    context: str,
    strict: bool,
) -> bool:
    """
    Warn or exit when --perfect-match expects all listed warheads but the
    training join returned fewer query groups than expected.
    """
    if not perfect_match or expected_count <= 1 or n_matched >= expected_count:
        return False

    msg = (
        f"[WARN] {context}: only {n_matched}/{expected_count} warheads matched "
        f"in training data — perfect-match runs .all() on this partial set only"
    )
    if strict:
        sys.exit(f"[ERROR] {msg}")
    print(msg)
    return True


def load_merged_for_vs(training_csv: str, labels_csv: str) -> pd.DataFrame:
    """
    Load all training candidates for ligand Names in labels.

    Labels supply Name + warhead matching only. Residue/ResNum/Chain in the labels
    CSV are ignored. Every screened residue in each matched (Name × Warhead) group
    is retained.
    """
    print("\n[INFO] VS load: all training candidates per ligand query group...")
    train_df = pd.read_csv(training_csv, sep=",", engine="c", low_memory=False)
    label_df = pd.read_csv(labels_csv, sep=",", engine="c", low_memory=False)

    train_df["_name_upper"] = train_df["Name"].str.strip().str.upper()
    train_df["_warhead_lower"] = train_df["Warhead"].str.strip().str.lower()
    train_df["Residue"] = train_df["Residue"].str.strip().str.upper()
    train_df["_resnum_str"] = train_df["ResNum"].map(_normalize_resnum)
    train_df["_chain_upper"] = train_df["Chain"].astype(str).str.strip().str.upper()

    label_df["_name_upper"] = label_df["Name"].str.strip().str.upper()

    train_df = train_df[train_df["Residue"].isin(VALID_RESIDUES)].copy()
    train_df = train_df.drop_duplicates(
        subset=["_name_upper", "Residue", "_resnum_str", "_chain_upper", "_warhead_lower"],
        keep="first",
    )

    warheads_by_name: dict[str, list[str]] = (
        train_df.groupby("_name_upper")["Warhead"]
        .apply(lambda s: list(s.unique()))
        .to_dict()
    )

    name_warheads: dict[str, set[str]] = {}
    unmatched_names: set[str] = set()

    for _, lrow in label_df.iterrows():
        name = lrow["_name_upper"]
        frank_wh = lrow.get("Frankenstein_Warhead", lrow.get("Warhead", ""))

        matched_whs = _matching_training_warheads(name, frank_wh, warheads_by_name)
        if not matched_whs:
            unmatched_names.add(name)
            continue

        name_warheads.setdefault(name, set()).update(matched_whs)

    if not name_warheads:
        print("ERROR: No training rows matched any label Name. Check name/warhead formats.")
        sys.exit(1)

    pair_rows = [
        {"_name_upper": name, "_warhead_lower": wh}
        for name, whs in sorted(name_warheads.items())
        for wh in sorted(whs)
    ]
    pairs_df = pd.DataFrame(pair_rows)

    merged = train_df.merge(pairs_df, on=["_name_upper", "_warhead_lower"], how="inner")
    merged["query_group"] = merged["_name_upper"] + "__" + merged["_warhead_lower"]
    merged["target_residue_type"] = merged["Residue"]

    print(f"  Training rows (VS) : {len(merged):,}")
    print(f"  Ligand Names       : {merged['_name_upper'].nunique():,}")
    print(f"  Query groups       : {merged['query_group'].nunique():,}")
    if unmatched_names:
        print(f"  Unmatched Names    : {len(unmatched_names):,}")

    return merged


def compute_vs_candidate_ranks(
    merged: pd.DataFrame,
    score_col: str,
    k: int,
    top_pct_threshold: float,
) -> pd.DataFrame:
    """Per training row: rank within its (Name × Warhead) query group."""
    p_frac = top_pct_threshold / 100.0
    rows: list[dict] = []

    for qg, grp in merged.groupby("query_group"):
        grp_sorted = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        n_residues = int(len(grp_sorted))
        name = str(grp_sorted["_name_upper"].iloc[0])

        for idx, row in grp_sorted.iterrows():
            target_rank = int(idx + 1)
            if n_residues <= 1:
                rank_frac = 0.0
            else:
                rank_frac = (target_rank - 1) / (n_residues - 1)
            hit_at_k = int(target_rank <= k)
            hit_at_top_pct = (
                int(rank_frac <= p_frac) if p_frac > 0 else hit_at_k
            )
            rows.append({
                "query_group": qg,
                "Name": name,
                "Warhead": row["Warhead"],
                "Residue": row["Residue"],
                "ResNum": row["ResNum"],
                "Chain": row["Chain"],
                "target_rank": target_rank,
                "hit_at_k": hit_at_k,
                "hit_at_top_pct": hit_at_top_pct,
                "n_residues": n_residues,
            })

    return pd.DataFrame(rows)


def aggregate_vs_site_rows(
    candidate_df: pd.DataFrame,
    k: int,
    top_pct_threshold: float,
    perfect_match: bool,
    name_wh_counts: dict[str, int],
    strict_warhead_coverage: bool = False,
) -> pd.DataFrame:
    """
    Collapse warhead-level candidate rows to one row per (Name, Residue, ResNum, Chain).

    Intra-protein rank is the mean target_rank across matched warheads. Hit@K and
    Hit@top-% are derived from that mean rank and the active reward thresholds
    (not per-warhead .all() / .any()).
    """
    if candidate_df.empty:
        return candidate_df.copy()

    rows_out: list[dict] = []
    incomplete_sites = 0
    group_cols = ["Name", "Residue", "ResNum", "Chain"]

    for key, grp in candidate_df.groupby(group_cols, sort=True):
        name = str(key[0]).strip().upper()
        name_wh_count = name_wh_counts.get(name, 1)
        n_matched = int(len(grp))

        if _check_warhead_coverage(
            perfect_match=perfect_match,
            expected_count=name_wh_count,
            n_matched=n_matched,
            context=(
                f"{key[0]} at site {_format_site_label(key[1], key[2], key[3])}"
            ),
            strict=strict_warhead_coverage,
        ):
            incomplete_sites += 1

        mean_rank = float(grp["target_rank"].mean()) if n_matched else float("nan")
        mean_n_residues = float(grp["n_residues"].mean()) if n_matched else float("nan")
        if n_matched and not pd.isna(mean_rank):
            label_hit_k, label_hit_top, rank_frac = _hits_from_mean_rank(
                mean_rank, mean_n_residues, k, top_pct_threshold,
            )
        else:
            label_hit_k, label_hit_top, rank_frac = 0, 0, float("nan")

        rows_out.append({
            "Name": key[0],
            "Residue": key[1],
            "ResNum": key[2],
            "Chain": key[3],
            "name_warhead_count": name_wh_count,
            "n_matched_warheads": n_matched,
            "mean_target_rank": mean_rank,
            "target_rank": mean_rank,
            "mean_n_residues": mean_n_residues,
            "rank_frac": rank_frac,
            "hit_at_k": label_hit_k,
            "hit_at_top_pct": label_hit_top,
        })

    if incomplete_sites:
        print(
            f"\n[WARN] Incomplete warhead coverage: {incomplete_sites} "
            f"(Name × site) group(s) had fewer warheads than expected "
            f"while --perfect-match was enabled."
        )

    return pd.DataFrame(rows_out)


def aggregate_vs_site_pred_scores(
    merged: pd.DataFrame,
    perfect_match: bool,
    name_wh_counts: dict[str, int],
    strict_warhead_coverage: bool = False,
) -> pd.DataFrame:
    """
    One row per (Name, Residue, ResNum, Chain) with mean pred_score across
    label-matched warheads (training rows already filtered to matched warheads).
    """
    if merged.empty:
        return merged.copy()

    rows_out: list[dict] = []
    incomplete_sites = 0
    group_cols = ["Name", "Residue", "ResNum", "Chain"]

    for key, grp in merged.groupby(group_cols, sort=True):
        name = str(key[0]).strip().upper()
        name_wh_count = name_wh_counts.get(name, 1)
        n_matched = int(len(grp))

        if _check_warhead_coverage(
            perfect_match=perfect_match,
            expected_count=name_wh_count,
            n_matched=n_matched,
            context=(
                f"{key[0]} at site {_format_site_label(key[1], key[2], key[3])}"
            ),
            strict=strict_warhead_coverage,
        ):
            incomplete_sites += 1

        mean_score = float(grp["pred_score"].mean()) if n_matched else float("nan")
        rows_out.append({
            "Name": key[0],
            "Residue": key[1],
            "ResNum": key[2],
            "Chain": key[3],
            "name_warhead_count": name_wh_count,
            "n_matched_warheads": n_matched,
            "mean_pred_score": mean_score,
            "pred_score": mean_score,
        })

    if incomplete_sites:
        print(
            f"\n[WARN] Incomplete warhead coverage: {incomplete_sites} "
            f"(Name × site) group(s) had fewer warheads than expected "
            f"while --perfect-match was enabled."
        )

    return pd.DataFrame(rows_out)


def build_vs_site_eval_from_merged(
    merged: pd.DataFrame,
    score_col: str,
    residue: str,
    k: int,
    top_pct_threshold: float,
    perfect_match: bool,
    name_wh_counts: dict[str, int],
    chain: str | None = None,
    resnum: str | int | None = None,
    strict_warhead_coverage: bool = False,
) -> pd.DataFrame:
    """Rank all training sites of *residue* per ligand (not label targets only)."""
    candidate_df = compute_vs_candidate_ranks(
        merged, score_col, k=k, top_pct_threshold=top_pct_threshold,
    )
    site_eval = aggregate_vs_site_rows(
        candidate_df,
        k=k,
        top_pct_threshold=top_pct_threshold,
        perfect_match=perfect_match,
        name_wh_counts=name_wh_counts,
        strict_warhead_coverage=strict_warhead_coverage,
    )
    return filter_vs_site_eval(site_eval, residue, chain=chain, resnum=resnum)


def _compute_intra_protein_ranks(
    site_rows: pd.DataFrame,
    hit_col: str,
) -> pd.DataFrame:
    """
    Per Name × site: use mean warhead target_rank (already aggregated in site_eval).

    Hits and misses alike keep that mean intra-protein rank so inter-ligand
    comparison distinguishes a near-miss from a poor rank.
    """
    rows: list[dict] = []
    for name, grp in site_rows.groupby("Name", sort=True):
        for _, row in grp.iterrows():
            site = str(row["site"])
            is_hit = int(row[hit_col]) == 1
            mean_rank = float(row["target_rank"])
            rows.append({
                "Name": name,
                "site": site,
                "Residue": row["Residue"],
                "ResNum": row["ResNum"],
                "Chain": row["Chain"],
                "target_rank": mean_rank,
                "mean_target_rank": mean_rank,
                "is_hit": int(is_hit),
                "intra_rank": mean_rank,
            })

    return pd.DataFrame(rows)


def _add_inter_protein_ranks(intra_df: pd.DataFrame) -> pd.DataFrame:
    """Per site column, rank Names by intra_rank (lower is better)."""
    if intra_df.empty:
        return intra_df.copy()

    out = intra_df.copy()
    out["inter_rank"] = 0
    for site, grp in out.groupby("site", sort=True):
        idx = grp.index
        out.loc[idx, "inter_rank"] = (
            grp["intra_rank"].rank(method="min", ascending=True).astype(int)
        )
    return out


def build_vs_results(
    site_eval_df: pd.DataFrame,
    residue: str,
    reward_mode: str,
    chain: str | None = None,
    resnum: str | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build virtual-screening rankings from per-site training ranks.

    Returns (wide_summary, long_detail) where wide_summary has one row per Name
    with per-site intra/inter ranks, avg inter rank, and overall rank.
    """
    hit_col = _active_hit_column(reward_mode)
    filtered = filter_vs_site_eval(site_eval_df, residue, chain=chain, resnum=resnum)
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    site_rows = filtered.copy()
    site_rows["site"] = site_rows.apply(
        lambda r: vs_site_column(r["Residue"], r["ResNum"], r["Chain"]),
        axis=1,
    )
    site_rows = site_rows.drop_duplicates(subset=["Name", "site"], keep="first")

    intra_df = _compute_intra_protein_ranks(site_rows, hit_col)
    ranked_df = _add_inter_protein_ranks(intra_df)

    sites = sorted(ranked_df["site"].unique())
    names = sorted(ranked_df["Name"].unique())

    wide: dict[str, dict[str, object]] = {
        name: {"Name": name} for name in names
    }
    for _, row in ranked_df.iterrows():
        name = row["Name"]
        site = row["site"]
        wide[name][f"{site}_intra"] = float(row["intra_rank"])
        wide[name][f"{site}_inter"] = int(row["inter_rank"])
        wide[name][f"{site}_hit"] = int(row["is_hit"])
        wide[name][f"{site}_target_rank"] = float(row["target_rank"])

    summary_rows: list[dict] = []
    for name in names:
        row_data = wide[name]
        inter_vals = [
            int(row_data[f"{site}_inter"])
            for site in sites
            if f"{site}_inter" in row_data
        ]
        hit_vals = [
            int(row_data[f"{site}_hit"])
            for site in sites
            if f"{site}_hit" in row_data
        ]
        n_sites = len(inter_vals)
        n_hits = int(sum(hit_vals))
        n_misses = int(n_sites - n_hits)
        avg_inter = float(np.mean(inter_vals)) if inter_vals else float("nan")

        summary_rows.append({
            "Name": name,
            "n_sites": n_sites,
            "n_hits": n_hits,
            "n_misses": n_misses,
            "avg_inter_rank": avg_inter,
            **row_data,
        })

    wide_df = pd.DataFrame(summary_rows)
    wide_df = wide_df.sort_values(
        ["n_misses", "avg_inter_rank", "Name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    wide_df["overall_rank"] = np.arange(1, len(wide_df) + 1)

    meta_cols = [
        "Name", "overall_rank", "avg_inter_rank", "n_hits", "n_misses", "n_sites",
    ]
    site_cols: list[str] = []
    for site in sites:
        site_cols.extend([
            f"{site}_intra",
            f"{site}_inter",
            f"{site}_hit",
            f"{site}_target_rank",
        ])
    ordered = meta_cols + [c for c in site_cols if c in wide_df.columns]
    wide_df = wide_df[ordered]

    return wide_df, ranked_df


def print_vs_results(
    vs_wide_df: pd.DataFrame,
    residue: str,
    reward_mode: str,
    chain: str | None = None,
    resnum: str | int | None = None,
) -> None:
    hit_col = _active_hit_column(reward_mode)
    filt = f"residue={residue.upper()}"
    if chain is not None:
        filt += f", chain={str(chain).upper()}"
    if resnum is not None:
        filt += f", resnum={_normalize_resnum(resnum)}"

    print(f"\n{'='*60}")
    print("  Virtual Screening Results")
    print(f"{'='*60}")
    print(f"  Filter            : {filt}")
    print(f"  Hit criterion     : {hit_col}")
    print(f"  Site source       : all training candidates (not label targets only)")
    print(f"  Ligands (Names)   : {len(vs_wide_df):,}")
    if vs_wide_df.empty:
        print("  [WARN] No VS results to display.")
        print(f"{'='*60}\n")
        return

    site_cols = sorted({
        c.rsplit("_", 1)[0]
        for c in vs_wide_df.columns
        if c.endswith("_intra")
    })
    print(f"  Residue sites     : {len(site_cols):,}")
    print()
    show_cols = ["Name", "overall_rank", "avg_inter_rank", "n_hits", "n_misses"]
    if len(site_cols) <= 6:
        for site in site_cols:
            show_cols.extend([f"{site}_intra", f"{site}_inter"])
    print(vs_wide_df[show_cols].head(20).to_string(index=False))
    if len(vs_wide_df) > 20:
        print(f"  ... ({len(vs_wide_df) - 20:,} more ligands)")
    print(f"{'='*60}\n")


def build_vs_pred_score_results(
    site_df: pd.DataFrame,
    residue: str,
    chain: str | None = None,
    resnum: str | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank ligands by mean site pred_score (higher is better).

    Per site: mean pred_score across matched warheads for that ligand.
    Overall: average of site pred_scores; overall_rank 1 = highest avg_pred_score.
    """
    filtered = filter_vs_site_eval(site_df, residue, chain=chain, resnum=resnum)
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    long_df = filtered.copy()
    long_df["site"] = long_df.apply(
        lambda r: vs_site_column(r["Residue"], r["ResNum"], r["Chain"]),
        axis=1,
    )
    long_df = long_df.drop_duplicates(subset=["Name", "site"], keep="first")
    long_df = long_df[
        [
            "Name", "site", "Residue", "ResNum", "Chain",
            "pred_score", "mean_pred_score", "n_matched_warheads",
            "name_warhead_count",
        ]
    ].sort_values(["Name", "site"]).reset_index(drop=True)

    sites = sorted(long_df["site"].unique())
    names = sorted(long_df["Name"].unique())

    wide: dict[str, dict[str, object]] = {
        name: {"Name": name} for name in names
    }
    for _, row in long_df.iterrows():
        name = row["Name"]
        site = row["site"]
        wide[name][f"{site}_pred_score"] = float(row["pred_score"])
        wide[name][f"{site}_n_warheads"] = int(row["n_matched_warheads"])

    summary_rows: list[dict] = []
    for name in names:
        row_data = wide[name]
        score_vals = [
            float(row_data[f"{site}_pred_score"])
            for site in sites
            if f"{site}_pred_score" in row_data
        ]
        avg_score = float(np.mean(score_vals)) if score_vals else float("nan")
        summary_rows.append({
            "Name": name,
            "n_sites": len(score_vals),
            "avg_pred_score": avg_score,
            **row_data,
        })

    wide_df = pd.DataFrame(summary_rows)
    wide_df = wide_df.sort_values(
        ["avg_pred_score", "Name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    wide_df["overall_rank"] = np.arange(1, len(wide_df) + 1)

    meta_cols = ["Name", "overall_rank", "avg_pred_score", "n_sites"]
    site_cols: list[str] = []
    for site in sites:
        site_cols.extend([f"{site}_pred_score", f"{site}_n_warheads"])
    ordered = meta_cols + [c for c in site_cols if c in wide_df.columns]
    wide_df = wide_df[ordered]

    return wide_df, long_df


def print_vs_pred_score_results(
    vs_wide_df: pd.DataFrame,
    residue: str,
    chain: str | None = None,
    resnum: str | int | None = None,
) -> None:
    filt = f"residue={residue.upper()}"
    if chain is not None:
        filt += f", chain={str(chain).upper()}"
    if resnum is not None:
        filt += f", resnum={_normalize_resnum(resnum)}"

    print(f"\n{'='*60}")
    print("  Virtual Screening Results (pred_score)")
    print(f"{'='*60}")
    print(f"  Filter            : {filt}")
    print(f"  Ranking           : mean site pred_score (warhead-mean); higher = better")
    print(f"  Overall order     : avg_pred_score across sites (descending)")
    print(f"  Ligands (Names)   : {len(vs_wide_df):,}")
    if vs_wide_df.empty:
        print("  [WARN] No VS results to display.")
        print(f"{'='*60}\n")
        return

    site_cols = sorted({
        c[: -len("_pred_score")]
        for c in vs_wide_df.columns
        if c.endswith("_pred_score")
    })
    print(f"  Residue sites     : {len(site_cols):,}")
    print()
    show_cols = ["Name", "overall_rank", "avg_pred_score", "n_sites"]
    if len(site_cols) <= 6:
        for site in site_cols:
            show_cols.append(f"{site}_pred_score")
    print(vs_wide_df[show_cols].head(20).to_string(index=False))
    if len(vs_wide_df) > 20:
        print(f"  ... ({len(vs_wide_df) - 20:,} more ligands)")
    print(f"{'='*60}\n")


def build_label_warhead_counts(labels_csv: str) -> dict[tuple[str, str, str, str], int]:
    """
    Count Frankenstein warheads listed per label (same convention as baby_frank).
    """
    labels = pd.read_csv(labels_csv, sep=",", engine="c", low_memory=False)
    counts: dict[tuple[str, str, str, str], int] = {}

    for _, row in labels.iterrows():
        key = make_label_key(row["Name"], row["Residue"], row["ResNum"], row["Chain"])
        frank = row.get("Frankenstein_Warhead", row.get("Warhead", ""))
        if pd.isna(frank) or not str(frank).strip():
            wh_set = {str(row.get("Warhead", "")).strip().lower()} if pd.notna(row.get("Warhead")) else set()
            wh_set.discard("")
            n = len(wh_set) if wh_set else 1
        else:
            n = len([w for w in str(frank).split(",") if w.strip()])
            n = max(n, 1)
        counts[key] = n

    return counts


def enrich_query_group_eval(
    eval_df: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """Attach label site + warhead metadata to per-query-group metrics."""
    if {"tgt_residue", "tgt_resnum", "tgt_chain"}.issubset(merged.columns):
        meta = (
            merged.groupby("query_group")
            .agg(
                Name=("_name_upper", "first"),
                Residue=("tgt_residue", "first"),
                ResNum=("tgt_resnum", "first"),
                Chain=("tgt_chain", "first"),
                Warhead=("Warhead", "first"),
            )
            .reset_index()
        )
    else:
        targets = merged.loc[merged["relevance"] == 1].copy()
        if targets.empty:
            targets = merged
        meta = (
            targets.groupby("query_group")
            .agg(
                Name=("_name_upper", "first"),
                Residue=("Residue", "first"),
                ResNum=("ResNum", "first"),
                Chain=("Chain", "first"),
                Warhead=("Warhead", "first"),
            )
            .reset_index()
        )
    return eval_df.merge(meta, on="query_group", how="left")


def aggregate_label_eval_rows(
    eval_df: pd.DataFrame,
    perfect_match: bool,
    label_wh_counts: dict[tuple[str, str, str, str], int],
    strict_warhead_coverage: bool = False,
) -> pd.DataFrame:
    """
    Collapse warhead-level (query-group) rows to one row per label site.

    Default: label hits if any matched warhead query group hits.
    perfect_match + label lists multiple Frankenstein warheads: all matched
    warhead query groups must hit (same as baby_frank).
    """
    if eval_df.empty:
        return eval_df.copy()

    rows_out: list[dict] = []
    incomplete_labels = 0

    group_cols = ["Name", "Residue", "ResNum", "Chain"]
    for key, grp in eval_df.groupby(group_cols, sort=True):
        label_key = make_label_key(key[0], key[1], key[2], key[3])
        label_wh_count = label_wh_counts.get(label_key, 1)
        n_matched = len(grp)

        if _check_warhead_coverage(
            perfect_match=perfect_match,
            expected_count=label_wh_count,
            n_matched=n_matched,
            context=(
                f"{key[0]} at site {_format_site_label(key[1], key[2], key[3])}"
            ),
            strict=strict_warhead_coverage,
        ):
            incomplete_labels += 1

        matched_whs = sorted(grp["Warhead"].astype(str).unique())
        hitting_k = sorted(grp.loc[grp["hit_at_k"] == 1, "Warhead"].astype(str).unique())
        hitting_top = sorted(
            grp.loc[grp["hit_at_top_pct"] == 1, "Warhead"].astype(str).unique()
        )

        if perfect_match and label_wh_count > 1:
            label_hit_k = int(grp["hit_at_k"].all()) if n_matched else 0
            label_hit_top = int(grp["hit_at_top_pct"].all()) if n_matched else 0
            if hitting_k and not label_hit_k:
                miss_reason = "partial_warhead_miss"
            else:
                miss_reason = None
        else:
            label_hit_k = int(grp["hit_at_k"].any()) if n_matched else 0
            label_hit_top = int(grp["hit_at_top_pct"].any()) if n_matched else 0
            miss_reason = None

        rows_out.append({
            "Name":                   key[0],
            "Residue":                key[1],
            "ResNum":                 key[2],
            "Chain":                  key[3],
            "target_residue_type":    grp["target_residue_type"].iloc[0],
            "label_warhead_count":    label_wh_count,
            "n_matched_warheads":     n_matched,
            "matched_warheads":       ",".join(matched_whs),
            "hitting_warheads":       ",".join(hitting_k),
            "hit_at_k":               label_hit_k,
            "hit_at_top_pct":         label_hit_top,
            "target_rank":            int(grp["target_rank"].max()),
            "mean_target_rank":       float(grp["target_rank"].mean()),
            "ndcg_at_k":              float(grp["ndcg_at_k"].mean()),
            "n_residues":             float(grp["n_residues"].mean()),
            "rank_frac":              float(grp["rank_frac"].max()),
            "top_pct":                float(grp["top_pct"].min()),
            "search_space_reduction": float(grp["search_space_reduction"].mean()),
            "rank_bonus":             float(grp["rank_bonus"].mean()),
            "objective":              float(grp["objective"].mean()),
            "miss_reason":            miss_reason,
        })

    if incomplete_labels:
        print(
            f"\n[WARN] Incomplete warhead coverage: {incomplete_labels} "
            f"label site(s) had fewer warheads than expected while "
            f"--perfect-match was enabled."
        )

    return pd.DataFrame(rows_out)


def per_residue_screen_breakdown(
    eval_label_df: pd.DataFrame,
    eval_qg_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-residue: hits from labels; rank/SS from all matched query groups."""
    label_agg = (
        eval_label_df.groupby("target_residue_type")
        .agg(
            N_labels=("hit_at_k", "count"),
            hit_rate=("hit_at_k", "mean"),
            hit_top_pct=("hit_at_top_pct", "mean"),
            objective=("objective", "mean"),
        )
    )
    qg_agg = (
        eval_qg_df.groupby("target_residue_type")
        .agg(
            N_query_groups=("query_group", "count"),
            avg_rank=("target_rank", "mean"),
            median_rank=("target_rank", "median"),
            ndcg=("ndcg_at_k", "mean"),
            avg_top_pct=("top_pct", "mean"),
            avg_ss_reduction=("search_space_reduction", "mean"),
        )
    )
    return label_agg.join(qg_agg, how="outer").reset_index()


def _subset_rank_ss_stats(subset: pd.DataFrame) -> dict[str, float]:
    """Mean/median rank and search-space reduction for a query-group subset."""
    if subset.empty:
        return {
            "avg_rank": float("nan"),
            "median_rank": float("nan"),
            "avg_ss_reduction": float("nan"),
            "median_ss_reduction": float("nan"),
        }
    return {
        "avg_rank": float(subset["target_rank"].mean()),
        "median_rank": float(subset["target_rank"].median()),
        "avg_ss_reduction": float(subset["search_space_reduction"].mean()),
        "median_ss_reduction": float(subset["search_space_reduction"].median()),
    }


def _warhead_hit_split_row(grp: pd.DataFrame, hit_col: str) -> dict[str, float | int | str]:
    """Per-warhead metrics with overall / hit / miss splits for one hit criterion."""
    hits = grp.loc[grp[hit_col] == 1]
    misses = grp.loc[grp[hit_col] == 0]
    overall = _subset_rank_ss_stats(grp)
    hit_stats = _subset_rank_ss_stats(hits)
    miss_stats = _subset_rank_ss_stats(misses)
    prefix = "top_pct" if hit_col == "hit_at_top_pct" else "k"
    return {
        f"hit_rate_{prefix}": float(grp[hit_col].mean()),
        f"n_hit_{prefix}": int(len(hits)),
        f"n_miss_{prefix}": int(len(misses)),
        f"avg_rank_{prefix}": overall["avg_rank"],
        f"median_rank_{prefix}": overall["median_rank"],
        f"avg_rank_hit_{prefix}": hit_stats["avg_rank"],
        f"median_rank_hit_{prefix}": hit_stats["median_rank"],
        f"avg_rank_miss_{prefix}": miss_stats["avg_rank"],
        f"median_rank_miss_{prefix}": miss_stats["median_rank"],
        f"avg_ss_reduction_{prefix}": overall["avg_ss_reduction"],
        f"median_ss_reduction_{prefix}": overall["median_ss_reduction"],
        f"avg_ss_reduction_hit_{prefix}": hit_stats["avg_ss_reduction"],
        f"median_ss_reduction_hit_{prefix}": hit_stats["median_ss_reduction"],
        f"avg_ss_reduction_miss_{prefix}": miss_stats["avg_ss_reduction"],
        f"median_ss_reduction_miss_{prefix}": miss_stats["median_ss_reduction"],
    }


def per_warhead_screen_breakdown(
    eval_qg_df: pd.DataFrame,
    reward_mode: str,
) -> pd.DataFrame:
    """
    Per training warhead accuracy from matched (Name × Warhead) query groups.

    Each row is one warhead type seen in label-matched screening. Metrics include
    hit rate at K and at top-%, plus avg/median rank and search-space reduction
    overall and split by hit vs miss for each criterion.
    """
    if eval_qg_df.empty or "Warhead" not in eval_qg_df.columns:
        return pd.DataFrame()

    active_hit_col = (
        "hit_at_top_pct" if reward_mode == "hit_at_top_pct" else "hit_at_k"
    )
    active_prefix = "top_pct" if active_hit_col == "hit_at_top_pct" else "k"

    rows: list[dict] = []
    for warhead, grp in eval_qg_df.groupby("Warhead", sort=True):
        row: dict[str, object] = {
            "Warhead": warhead,
            "n_query_groups": int(len(grp)),
            "hit_criterion_headline": active_hit_col,
        }
        row.update(_warhead_hit_split_row(grp, "hit_at_k"))
        row.update(_warhead_hit_split_row(grp, "hit_at_top_pct"))
        active_hits = grp.loc[grp[active_hit_col] == 1]
        active_misses = grp.loc[grp[active_hit_col] == 0]
        row["avg_rank_when_hit"] = _subset_rank_ss_stats(active_hits)["avg_rank"]
        row["median_rank_when_hit"] = _subset_rank_ss_stats(active_hits)["median_rank"]
        row["avg_rank_when_miss"] = _subset_rank_ss_stats(active_misses)["avg_rank"]
        row["median_rank_when_miss"] = _subset_rank_ss_stats(active_misses)["median_rank"]
        row["avg_ss_reduction_when_hit"] = _subset_rank_ss_stats(active_hits)["avg_ss_reduction"]
        row["median_ss_reduction_when_hit"] = _subset_rank_ss_stats(active_hits)["median_ss_reduction"]
        row["avg_ss_reduction_when_miss"] = _subset_rank_ss_stats(active_misses)["avg_ss_reduction"]
        row["median_ss_reduction_when_miss"] = _subset_rank_ss_stats(active_misses)["median_ss_reduction"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        [f"hit_rate_{active_prefix}", "n_query_groups"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return out


def per_residue_accuracy_breakdown(
    eval_qg_df: pd.DataFrame,
    eval_label_df: pd.DataFrame,
    reward_mode: str,
) -> pd.DataFrame:
    """
    Per target residue type (CYS, SER, ...) from query-group metrics only.

    Each value is mean/median across (Name × Warhead) query groups for that
    residue type — never pooled over individual candidate residues.
    """
    if eval_qg_df.empty or "target_residue_type" not in eval_qg_df.columns:
        return pd.DataFrame()

    active_hit_col = (
        "hit_at_top_pct" if reward_mode == "hit_at_top_pct" else "hit_at_k"
    )

    rows: list[dict] = []
    for res_type, grp in eval_qg_df.groupby("target_residue_type", sort=True):
        overall = _subset_rank_ss_stats(grp)
        hits = grp.loc[grp[active_hit_col] == 1]
        misses = grp.loc[grp[active_hit_col] == 0]
        hit_stats = _subset_rank_ss_stats(hits)
        miss_stats = _subset_rank_ss_stats(misses)
        rows.append({
            "target_residue_type": res_type,
            "n_query_groups": int(len(grp)),
            "hit_criterion_headline": active_hit_col,
            "hit_rate_at_k": float(grp["hit_at_k"].mean()),
            "hit_rate_at_top_pct": float(grp["hit_at_top_pct"].mean()),
            "avg_rank": overall["avg_rank"],
            "median_rank": overall["median_rank"],
            "avg_rank_when_hit": hit_stats["avg_rank"],
            "avg_ss_reduction": overall["avg_ss_reduction"],
            "median_ss_reduction": overall["median_ss_reduction"],
            "avg_ss_reduction_when_hit": hit_stats["avg_ss_reduction"],
            "median_ss_reduction_when_hit": hit_stats["median_ss_reduction"],
            "avg_ss_reduction_when_miss": miss_stats["avg_ss_reduction"],
            "median_ss_reduction_when_miss": miss_stats["median_ss_reduction"],
        })

    out = pd.DataFrame(rows)
    if not eval_label_df.empty and "target_residue_type" in eval_label_df.columns:
        label_agg = (
            eval_label_df.groupby("target_residue_type", sort=True)
            .agg(
                n_labels=("hit_at_k", "count"),
                label_hit_rate_at_k=("hit_at_k", "mean"),
                label_hit_rate_at_top_pct=("hit_at_top_pct", "mean"),
            )
            .reset_index()
        )
        out = out.merge(label_agg, on="target_residue_type", how="left")

    sort_col = (
        "hit_rate_at_top_pct" if reward_mode == "hit_at_top_pct" else "hit_rate_at_k"
    )
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


def print_cov_screen_results(
    eval_label_df: pd.DataFrame,
    eval_qg_df: pd.DataFrame,
    k: int,
    top_pct_threshold: float,
    reward_mode: str,
    epsilon: float,
    split_name: str = "Evaluation",
) -> None:
    summary = summarize_screen_metrics(
        eval_label_df, eval_qg_df, k, top_pct_threshold, reward_mode, epsilon,
    )
    pct_label = f"top-{top_pct_threshold:g}%"

    print(f"\n{'='*60}")
    print(f"  {split_name} Results")
    print(f"{'='*60}")
    print(f"  Reward mode (headline) : {reward_mode}")
    print(f"  Labels (hit counting)  : {summary['n_labels']:,}")
    print(f"  Query groups (rank avg): {summary['n_query_groups']:,}")
    print(f"  Objective (per label): {summary['objective']:.3f}  (ε={epsilon})")
    print(f"  Rank bonus (per label) : {summary['rank_bonus']:.3f}")
    print()
    print(f"  Hit metrics (per label, N={summary['n_labels']}):")
    print(f"    Hit@{k}              : {summary['hit_rate']:.3f}")
    print(f"    Hit@{pct_label}      : {summary['hit_at_top_pct']:.3f}")
    print()
    print(f"  Rank / search-space (all matched warheads, "
          f"N={summary['n_query_groups']}):")
    print(f"    NDCG@{k}             : {summary['ndcg']:.3f}")
    print(f"    Average rank         : {summary['avg_rank']:.1f}")
    print(f"    Median rank          : {summary['median_rank']:.1f}")
    print(_format_rank_when_hit(
        summary, f"Hit@{k}",
        "avg_rank_when_hit_at_k", "median_rank_when_hit_at_k", "n_hit_at_k",
        summary["n_query_groups"],
    ))
    print(_format_rank_when_hit(
        summary,
        f"Hit@{pct_label}",
        "avg_rank_when_hit_at_top_pct",
        "median_rank_when_hit_at_top_pct",
        "n_hit_at_top_pct",
        summary["n_query_groups"],
    ))
    print(f"    Avg top-%ile         : {summary['avg_top_pct']:.1f}")
    print(f"    Avg SS reduction     : "
          f"{100.0 * summary['avg_search_space_reduction']:.1f}%")
    print(f"    Median SS reduction  : "
          f"{100.0 * summary['median_search_space_reduction']:.1f}%")
    print(_format_ss_when_hit(
        summary, f"Hit@{k}",
        "avg_ss_reduction_when_hit_at_k",
        "median_ss_reduction_when_hit_at_k",
        "n_hit_at_k",
        summary["n_query_groups"],
    ))
    print(_format_ss_when_hit(
        summary,
        f"Hit@{pct_label}",
        "avg_ss_reduction_when_hit_at_top_pct",
        "median_ss_reduction_when_hit_at_top_pct",
        "n_hit_at_top_pct",
        summary["n_query_groups"],
    ))
    print(_format_ss_when_miss(
        summary, f"Hit@{k}",
        "avg_ss_reduction_when_miss_at_k",
        "median_ss_reduction_when_miss_at_k",
        "n_miss_at_k",
        summary["n_query_groups"],
    ))
    print(_format_ss_when_miss(
        summary,
        f"Hit@{pct_label}",
        "avg_ss_reduction_when_miss_at_top_pct",
        "median_ss_reduction_when_miss_at_top_pct",
        "n_miss_at_top_pct",
        summary["n_query_groups"],
    ))
    headline = (
        f"Hit@{pct_label}" if reward_mode == "hit_at_top_pct" else f"Hit@{k}"
    )
    print()
    print(f"  Headline rank quality ({reward_mode}, {headline}):")
    if pd.isna(summary["avg_rank_when_hit"]):
        print("    Avg rank when hit    : n/a")
        print("    Median rank when hit : n/a")
    else:
        print(f"    Avg rank when hit    : {summary['avg_rank_when_hit']:.2f}")
        print(f"    Median rank when hit : {summary['median_rank_when_hit']:.1f}")
    if pd.isna(summary["avg_ss_reduction_when_hit"]):
        print("    Avg SS red. when hit : n/a")
        print("    Med SS red. when hit : n/a")
    else:
        print(f"    Avg SS red. when hit : "
              f"{100.0 * summary['avg_ss_reduction_when_hit']:.1f}%")
        print(f"    Med SS red. when hit : "
              f"{100.0 * summary['median_ss_reduction_when_hit']:.1f}%")
    if pd.isna(summary.get("avg_ss_reduction_when_miss")):
        print("    Avg SS red. when miss: n/a")
        print("    Med SS red. when miss: n/a")
    else:
        print(f"    Avg SS red. when miss: "
              f"{100.0 * summary['avg_ss_reduction_when_miss']:.1f}%")
        print(f"    Med SS red. when miss: "
              f"{100.0 * summary['median_ss_reduction_when_miss']:.1f}%")

    print(f"\n  Per Residue Type:")
    per_res = per_residue_screen_breakdown(eval_label_df, eval_qg_df)
    print(per_res.to_string(index=False, float_format="%.3f"))
    print(f"{'='*60}\n")


def load_model_bundle(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Model pkl not found: {path}")
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to load model pkl: {exc}")

    if "model" not in bundle or "features" not in bundle:
        sys.exit("[ERROR] Pkl must contain 'model' and 'features' keys.")

    return bundle


def prepare_inference_features(
    df: pd.DataFrame,
    model_features: list[str],
    normalize_within_protein: bool = False,
) -> pd.DataFrame:
    """
    Build the feature matrix expected by the saved model.

    If the model was trained with --normalize_within_protein, pass the same
    flag here (or rely on auto-detect when any model feature ends with _norm).
    """
    out = df.copy()

    for res in VALID_RESIDUES:
        col = f"res_{res}"
        if col in model_features:
            out[col] = (out["Residue"] == res).astype(int)

    resnum_feats = [c for c in model_features if c.startswith("resnum_eq_")]
    if resnum_feats:
        if "_resnum_str" not in out.columns:
            out["_resnum_str"] = out["ResNum"].map(_normalize_resnum)
        for col in resnum_feats:
            key = col[len("resnum_eq_"):]
            out[col] = (out["_resnum_str"] == key).astype(int)

    norm_cols_in_model = [c for c in model_features if c.endswith("_norm")]
    if norm_cols_in_model and not normalize_within_protein:
        print("[INFO] Model uses *_norm features — enabling within-group "
              "normalization.")
        normalize_within_protein = True

    if normalize_within_protein:
        base_cols = sorted({c[:-5] for c in norm_cols_in_model})
        for col in base_cols:
            if col not in out.columns:
                continue
            out[f"{col}_norm"] = out.groupby("query_group")[col].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )

    missing = [c for c in model_features if c not in out.columns]
    if missing:
        sys.exit(
            "[ERROR] Eval CSV is missing columns required by the model:\n  "
            + ", ".join(missing)
        )

    return out


def predict_scores(
    df: pd.DataFrame,
    model,
    model_features: list[str],
    normalize_within_protein: bool = False,
    model_type: str = "ranker",
) -> pd.Series:
    """
    Score candidates for ranking / VS.

    Ranker: model.predict(X)
    Classifier: model.predict_proba(X)[:, 1]  (hit probability)
    Downstream code always consumes the Series as pred_score.
    """
    prepared = prepare_inference_features(
        df, model_features, normalize_within_protein=normalize_within_protein
    )
    X = prepared[model_features].values.astype(np.float32)
    mtype = (model_type or "ranker").strip().lower()
    if mtype == "classifier":
        if not hasattr(model, "predict_proba"):
            sys.exit(
                "[ERROR] Bundle model_type=classifier but model has no predict_proba."
            )
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)
    return pd.Series(scores, index=df.index, name="pred_score")


def export_scores_csv(df: pd.DataFrame, path: str) -> None:
    cols = [
        c for c in [
            "Name", "Residue", "ResNum", "Chain", "Warhead",
            "query_group", "relevance", "pred_score",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()
    out = out.sort_values(["query_group", "pred_score"], ascending=[True, False])
    out.to_csv(path, index=False)
    print(f"[INFO] Per-row scores exported → {path}")


def export_results_json(summary: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[INFO] Summary JSON exported → {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Screen/evaluate a saved LGBMRanker on training + labels CSVs",
    )
    p.add_argument("--model", required=True,
                   help="Path to lgbm_ranker.pkl or lgbm_classifier.pkl "
                        "(bundle may include model_type=ranker|classifier)")
    p.add_argument("--training", required=True,
                   help="Path to candidate/training CSV")
    p.add_argument("--labels", required=True,
                   help="Path to labels CSV")
    p.add_argument("--topk", type=int, default=None,
                   help="K for Hit@K / NDCG@K (default: value stored in pkl)")
    p.add_argument("--reward-mode", choices=REWARD_MODES, default=None,
                   help="Headline objective mode (default: value stored in pkl)")
    p.add_argument("--top-pct", type=float, default=None,
                   help="Top-X%% threshold (default: value stored in pkl)")
    p.add_argument("--rank-bonus-epsilon", type=float, default=None,
                   help="Rank bonus weight (default: value stored in pkl)")
    p.add_argument("--normalize-within-protein", action="store_true",
                   help="Apply within-query-group min-max norm (must match training)")
    p.add_argument("--perfect-match", action="store_true",
                   help="When a label lists multiple Frankenstein warheads, require "
                        "every training-matched warhead query group to hit. "
                        "Default: any matched warhead hit counts.")
    p.add_argument("--strict-warhead-coverage", action="store_true",
                   help="With --perfect-match, exit with an error if any label site "
                        "has fewer training-matched warheads than listed in labels. "
                        "Default: print warnings only.")
    p.add_argument("--export-results", default=None,
                   help="Write per-label metrics CSV (one row per label site)")
    p.add_argument("--export-warhead-accuracy", default=None,
                   help="Write per-warhead accuracy CSV (default: "
                        "<export-results stem>_warhead_accuracy.csv when "
                        "--export-results is set)")
    p.add_argument("--export-residue-accuracy", default=None,
                   help="Write per-residue-type accuracy CSV (default: "
                        "<export-results stem>_residue_accuracy.csv when "
                        "--export-results is set)")
    p.add_argument("--export-shap", default=None,
                   help="Write SHAP per-candidate CSV (default: "
                        "<export-results stem>_shap_per_candidate.csv when "
                        "--export-results is set; also writes *_shap_global.csv)")
    p.add_argument("--no-shap", action="store_true",
                   help="Disable SHAP export even when --export-results is set")
    p.add_argument("--shap-max-rows", type=int, default=None,
                   help="Optional cap on rows for SHAP (random subsample if exceeded)")
    p.add_argument("--export-query-groups", default=None,
                   help="Write per-(Name × Warhead) query-group metrics CSV")
    p.add_argument("--export-scores", default=None,
                   help="Write per-row prediction scores CSV")
    p.add_argument("--export-summary", default=None,
                   help="Write summary JSON (overall + per-residue)")
    p.add_argument("--VS", action="store_true",
                   help="Virtual screening mode: rank ligands over all training sites "
                        "of --residue. Labels supply Name + warhead only.")
    p.add_argument("--pred-score", action="store_true",
                   help="With --VS: rank ligands by mean site pred_score (warhead-mean) "
                        "and overall avg_pred_score across sites. No intra/inter ranks.")
    p.add_argument("--residue", default=None,
                   help="Residue type for --VS ranking (e.g. CYS, SER). Required with --VS.")
    p.add_argument("--chain", default=None,
                   help="Optional chain filter for --VS (e.g. A)")
    p.add_argument("--resnum", default=None,
                   help="Optional ResNum filter for --VS (e.g. 797)")
    p.add_argument("--export-vs-results", default=None,
                   help="Write VS ranking CSV (default: ./vs_results.csv when --VS)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_model_bundle(args.model)
    model = bundle["model"]
    model_features: list[str] = list(bundle["features"])
    model_type = str(bundle.get("model_type", "")).strip().lower()
    if model_type not in ("ranker", "classifier"):
        cls_name = type(model).__name__
        if (
            hasattr(model, "calibrated_classifiers_")
            or "Classifier" in cls_name
            or cls_name.startswith("Calibrated")
        ):
            model_type = "classifier"
        else:
            model_type = "ranker"

    k = args.topk if args.topk is not None else int(bundle.get("k", 10))
    reward_mode = args.reward_mode or bundle.get("reward_mode", "hit_at_k")
    top_pct = (
        args.top_pct if args.top_pct is not None
        else float(bundle.get("top_pct_threshold", 10.0))
    )
    epsilon = (
        args.rank_bonus_epsilon if args.rank_bonus_epsilon is not None
        else float(bundle.get("rank_bonus_epsilon", DEFAULT_RANK_BONUS_EPSILON))
    )

    if top_pct <= 0 or top_pct > 100:
        sys.exit("[ERROR] --top-pct must be in (0, 100].")
    if k < 1:
        sys.exit("[ERROR] --topk must be >= 1.")
    if args.VS and not args.residue:
        sys.exit("[ERROR] --VS requires --residue (e.g. --residue CYS).")
    if args.pred_score and not args.VS:
        sys.exit("[ERROR] --pred-score requires --VS.")

    vs_mode = bool(args.VS)
    vs_pred_score_mode = bool(args.pred_score)

    print("=" * 60)
    print("  Cov_Screen — evaluation"
          + (" (virtual screening)" if vs_mode else "")
          + (" [pred_score]" if vs_pred_score_mode else ""))
    print("=" * 60)
    print(f"  Model           : {args.model}")
    print(f"  Model type      : {model_type}")
    print(f"  Features        : {len(model_features)}")
    print(f"  Reward mode     : {reward_mode}")
    print(f"  Top-K           : {k}")
    print(f"  Top-% threshold : {top_pct:g}%")
    print(f"  Epsilon         : {epsilon}")
    if vs_mode:
        vs_filt = f"residue={args.residue.upper()}"
        if args.chain:
            vs_filt += f", chain={args.chain.upper()}"
        if args.resnum is not None:
            vs_filt += f", resnum={_normalize_resnum(args.resnum)}"
        print(f"  VS mode         : {vs_filt}")
        if vs_pred_score_mode:
            print(
                "  VS ranking      : mean site pred_score per warhead set; "
                "overall by avg_pred_score (higher = better)"
            )
        else:
            print(
                "  VS warheads     : mean intra-protein rank across matched warheads; "
                "hit from reward threshold on that mean"
            )
        if args.perfect_match:
            if args.strict_warhead_coverage:
                print("  Warhead coverage: strict (exit if any site is missing warheads)")
            else:
                print("  Warhead coverage: warn if any site is missing warheads")
        print("  VS labels use   : Name + warhead only (target site columns ignored)")
    elif args.perfect_match:
        print("  Hit counting    : one per label (perfect-match: all matched "
              "warheads must hit when label lists multiple Frankenstein warheads)")
        if args.strict_warhead_coverage:
            print("  Warhead coverage: strict (exit if any site is missing warheads)")
        else:
            print("  Warhead coverage: warn if any site is missing warheads")
    else:
        print("  Hit counting    : one per label (any matched warhead hit counts)")

    merged = (
        load_merged_for_vs(args.training, args.labels)
        if vs_mode else
        load_and_merge(args.training, args.labels)
    )
    n_groups = merged["query_group"].nunique()
    print(f"\n[INFO] Query groups to score: {n_groups:,}")

    label_wh_counts = build_label_warhead_counts(args.labels) if not vs_mode else {}
    name_wh_counts = build_name_warhead_counts(args.labels) if vs_mode else {}

    merged = merged.sort_values("query_group").reset_index(drop=True)
    merged["pred_score"] = predict_scores(
        merged, model, model_features,
        normalize_within_protein=args.normalize_within_protein,
        model_type=model_type,
    )

    eval_qg_df = pd.DataFrame()
    eval_label_df = pd.DataFrame()
    composition: dict = {}
    overall: dict = {}
    per_res: list = []
    per_warhead_df = pd.DataFrame()
    per_residue_accuracy_df = pd.DataFrame()

    vs_wide_df = pd.DataFrame()
    vs_long_df = pd.DataFrame()

    if vs_mode:
        if vs_pred_score_mode:
            vs_site_eval = aggregate_vs_site_pred_scores(
                merged,
                perfect_match=args.perfect_match,
                name_wh_counts=name_wh_counts,
                strict_warhead_coverage=args.strict_warhead_coverage,
            )
            vs_site_eval = filter_vs_site_eval(
                vs_site_eval,
                args.residue,
                chain=args.chain,
                resnum=args.resnum,
            )
            n_vs_sites = (
                vs_site_eval.apply(
                    lambda r: vs_site_column(r["Residue"], r["ResNum"], r["Chain"]),
                    axis=1,
                ).nunique()
                if not vs_site_eval.empty else 0
            )
            print(f"[INFO] VS sites scored ({args.residue.upper()}): {n_vs_sites:,}")
            vs_wide_df, vs_long_df = build_vs_pred_score_results(
                vs_site_eval,
                residue=args.residue,
                chain=args.chain,
                resnum=args.resnum,
            )
            print_vs_pred_score_results(
                vs_wide_df,
                residue=args.residue,
                chain=args.chain,
                resnum=args.resnum,
            )
        else:
            vs_site_eval = build_vs_site_eval_from_merged(
                merged,
                score_col="pred_score",
                residue=args.residue,
                k=k,
                top_pct_threshold=top_pct,
                perfect_match=args.perfect_match,
                name_wh_counts=name_wh_counts,
                chain=args.chain,
                resnum=args.resnum,
                strict_warhead_coverage=args.strict_warhead_coverage,
            )
            n_vs_sites = (
                vs_site_eval.apply(
                    lambda r: vs_site_column(r["Residue"], r["ResNum"], r["Chain"]),
                    axis=1,
                ).nunique()
                if not vs_site_eval.empty else 0
            )
            print(f"[INFO] VS sites ranked ({args.residue.upper()}): {n_vs_sites:,}")
            vs_wide_df, vs_long_df = build_vs_results(
                vs_site_eval,
                residue=args.residue,
                reward_mode=reward_mode,
                chain=args.chain,
                resnum=args.resnum,
            )
            print_vs_results(
                vs_wide_df,
                residue=args.residue,
                reward_mode=reward_mode,
                chain=args.chain,
                resnum=args.resnum,
            )
        if vs_wide_df.empty:
            print("[WARN] VS ranking produced no rows — check --residue/--chain/--resnum.")
    else:
        eval_qg_df = evaluate_predictions(
            merged, "pred_score",
            k=k,
            top_pct_threshold=top_pct,
            reward_mode=reward_mode,
            epsilon=epsilon,
        )

        if eval_qg_df.empty:
            sys.exit("[ERROR] No query groups could be evaluated.")

        eval_qg_df = enrich_query_group_eval(eval_qg_df, merged)
        eval_label_df = aggregate_label_eval_rows(
            eval_qg_df,
            args.perfect_match,
            label_wh_counts,
            strict_warhead_coverage=args.strict_warhead_coverage,
        )

        if eval_label_df.empty:
            sys.exit("[ERROR] No labels could be aggregated for evaluation.")

        print(f"[INFO] Labels evaluated (one row per site): {len(eval_label_df):,}")
        print(f"[INFO] Query groups in rank/SS averages : {len(eval_qg_df):,}")

        print_cov_screen_results(
            eval_label_df, eval_qg_df,
            k=k, top_pct_threshold=top_pct,
            reward_mode=reward_mode, epsilon=epsilon,
            split_name="Evaluation",
        )

        composition = analyze_residue_composition(
            merged, "pred_score", k=k, top_pct_threshold=top_pct,
        )
        print_residue_composition(composition, k=k, top_pct_threshold=top_pct)

        overall = summarize_screen_metrics(
            eval_label_df, eval_qg_df, k, top_pct, reward_mode, epsilon,
        )
        per_res = per_residue_screen_breakdown(eval_label_df, eval_qg_df).to_dict(
            orient="records"
        )
        per_warhead_df = per_warhead_screen_breakdown(eval_qg_df, reward_mode)
        per_residue_accuracy_df = per_residue_accuracy_breakdown(
            eval_qg_df, eval_label_df, reward_mode,
        )

    shap_global_df = pd.DataFrame()
    run_shap = not vs_mode and not args.no_shap and (
        args.export_shap is not None or args.export_results is not None
    )
    if run_shap:
        prepared = prepare_inference_features(
            merged, model_features,
            normalize_within_protein=args.normalize_within_protein,
        )
        shap_per_path = args.export_shap
        shap_global_path = None
        if shap_per_path is None and args.export_results:
            stem = Path(args.export_results).stem
            parent = Path(args.export_results).parent
            shap_per_path = str(parent / f"{stem}_shap_per_candidate.csv")
            shap_global_path = str(parent / f"{stem}_shap_global.csv")
        _, shap_global_df = export_shap_csvs(
            model=model,
            df=prepared,
            feature_cols=model_features,
            feature_matrix=prepared[model_features].values.astype(np.float32),
            per_candidate_path=shap_per_path,
            global_path=shap_global_path,
            max_rows=args.shap_max_rows,
        )

    summary = {
        "model": str(Path(args.model).resolve()),
        "model_type": model_type,
        "training": str(Path(args.training).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "k": k,
        "top_pct_threshold": top_pct,
        "reward_mode": reward_mode,
        "rank_bonus_epsilon": epsilon,
        "perfect_match": args.perfect_match,
        "mode": "vs_pred_score" if vs_pred_score_mode else ("vs" if vs_mode else "eval"),
        "n_query_groups": int(n_groups if vs_mode else len(eval_qg_df)),
        "n_labels": int(len(eval_label_df)),
        "overall": overall,
        "per_residue": per_res,
        "per_residue_accuracy": per_residue_accuracy_df.to_dict(orient="records"),
        "per_warhead": per_warhead_df.to_dict(orient="records"),
    }
    if not shap_global_df.empty:
        summary["shap_global"] = shap_global_df.to_dict(orient="records")
    if composition.get("n_query_groups"):
        summary["residue_composition"] = {
            k_: v for k_, v in composition.items() if k_ != "detail"
        }
    if vs_mode and not vs_wide_df.empty:
        vs_summary: dict = {
            "residue": str(args.residue).upper(),
            "chain": str(args.chain).upper() if args.chain else None,
            "resnum": _normalize_resnum(args.resnum) if args.resnum is not None else None,
            "n_ligands": int(len(vs_wide_df)),
            "rankings": vs_wide_df.to_dict(orient="records"),
        }
        if vs_pred_score_mode:
            vs_summary["ranking_metric"] = "avg_pred_score"
            vs_summary["site_metric"] = "mean_pred_score_per_warhead"
        else:
            vs_summary["hit_criterion"] = _active_hit_column(reward_mode)
        summary["vs"] = vs_summary

    vs_export_path = args.export_vs_results
    if vs_mode and vs_export_path is None:
        vs_export_path = (
            "./vs_pred_score_results.csv" if vs_pred_score_mode else "./vs_results.csv"
        )
    if vs_mode and vs_export_path:
        if vs_wide_df.empty:
            print("[WARN] No VS results to export.")
        else:
            vs_wide_df.to_csv(vs_export_path, index=False)
            print(f"[INFO] VS rankings exported → {vs_export_path}")

    if args.export_results:
        if vs_mode:
            print("[WARN] --export-results is label-eval output; skipped in --VS mode.")
        else:
            eval_label_df.to_csv(args.export_results, index=False)
            print(f"[INFO] Per-label metrics exported → {args.export_results}")

    warhead_path = args.export_warhead_accuracy
    if warhead_path is None and args.export_results:
        warhead_path = str(Path(args.export_results).with_name(
            Path(args.export_results).stem + "_warhead_accuracy.csv"
        ))
    if warhead_path:
        if per_warhead_df.empty:
            print("[WARN] No warhead breakdown to export.")
        else:
            per_warhead_df.to_csv(warhead_path, index=False)
            print(f"[INFO] Per-warhead accuracy exported → {warhead_path}")

    residue_path = args.export_residue_accuracy
    if residue_path is None and args.export_results:
        residue_path = str(Path(args.export_results).with_name(
            Path(args.export_results).stem + "_residue_accuracy.csv"
        ))
    if residue_path:
        if per_residue_accuracy_df.empty:
            print("[WARN] No residue accuracy breakdown to export.")
        else:
            per_residue_accuracy_df.to_csv(residue_path, index=False)
            print(f"[INFO] Per-residue accuracy exported → {residue_path}")

    if args.export_query_groups:
        eval_qg_df.to_csv(args.export_query_groups, index=False)
        print(f"[INFO] Per-query-group metrics exported → {args.export_query_groups}")

    if composition.get("n_query_groups") and args.export_summary:
        comp_path = str(Path(args.export_summary).with_name(
            Path(args.export_summary).stem + "_residue_composition.csv"
        ))
        composition["detail"].to_csv(comp_path, index=False)
        print(f"[INFO] Residue composition detail → {comp_path}")

    if args.export_scores:
        if vs_mode:
            print("[INFO] Exporting scores (--export-scores) in --VS mode.")
        export_scores_csv(merged, args.export_scores)

    if args.export_summary:
        export_results_json(summary, args.export_summary)

    print("Done.")


if __name__ == "__main__":
    main()
