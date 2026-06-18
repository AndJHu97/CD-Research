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
        [--export-query-groups ./screen_query_group_results.csv] \\
        [--export-scores ./screen_scores.csv]
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
    _normalize_resnum,
    evaluate_predictions,
    load_and_merge,
    summarize_eval_metrics,
    analyze_residue_composition,
    print_residue_composition,
)


def make_label_key(name, residue, resnum, chain) -> tuple[str, str, str, str]:
    return (
        str(name).strip().upper(),
        str(residue).strip().upper(),
        _normalize_resnum(resnum),
        str(chain).strip().upper(),
    )


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
    return eval_df.merge(meta, on="query_group", how="left")


def aggregate_label_eval_rows(
    eval_df: pd.DataFrame,
    perfect_match: bool,
    label_wh_counts: dict[tuple[str, str, str, str], int],
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

    group_cols = ["Name", "Residue", "ResNum", "Chain"]
    for key, grp in eval_df.groupby(group_cols, sort=True):
        label_key = make_label_key(key[0], key[1], key[2], key[3])
        label_wh_count = label_wh_counts.get(label_key, 1)
        n_matched = len(grp)

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

    return pd.DataFrame(rows_out)


def summarize_screen_metrics(
    eval_label_df: pd.DataFrame,
    eval_qg_df: pd.DataFrame,
    k: int,
    top_pct_threshold: float,
    reward_mode: str,
    epsilon: float,
) -> dict:
    """
    Hit/objective at label level; rank / NDCG / search-space stats over all
    matched (Name × Warhead) query groups (may exceed label count).
    """
    label_stats = summarize_eval_metrics(
        eval_label_df, k, top_pct_threshold, reward_mode, epsilon,
    )
    qg_stats = summarize_eval_metrics(
        eval_qg_df, k, top_pct_threshold, reward_mode, epsilon,
    )
    return {
        "reward_mode": reward_mode,
        "top_pct_threshold": top_pct_threshold,
        "k": k,
        "epsilon": epsilon,
        "n_labels": int(len(eval_label_df)),
        "n_query_groups": int(len(eval_qg_df)),
        "hit_rate": label_stats["hit_rate"],
        "hit_at_top_pct": label_stats["hit_at_top_pct"],
        "objective": label_stats["objective"],
        "rank_bonus": label_stats["rank_bonus"],
        "avg_rank": qg_stats["avg_rank"],
        "median_rank": float(eval_qg_df["target_rank"].median()),
        "ndcg": qg_stats["ndcg"],
        "avg_top_pct": qg_stats["avg_top_pct"],
        "avg_search_space_reduction": qg_stats["avg_search_space_reduction"],
    }


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
    print(f"    Avg top-%ile         : {summary['avg_top_pct']:.1f}")
    print(f"    Avg SS reduction     : "
          f"{100.0 * summary['avg_search_space_reduction']:.1f}%")

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
) -> pd.Series:
    prepared = prepare_inference_features(
        df, model_features, normalize_within_protein=normalize_within_protein
    )
    X = prepared[model_features].values.astype(np.float32)
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
                   help="Path to lgbm_ranker.pkl from Training_Cov_Screen.py")
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
    p.add_argument("--export-results", default=None,
                   help="Write per-label metrics CSV (one row per label site)")
    p.add_argument("--export-query-groups", default=None,
                   help="Write per-(Name × Warhead) query-group metrics CSV")
    p.add_argument("--export-scores", default=None,
                   help="Write per-row prediction scores CSV")
    p.add_argument("--export-summary", default=None,
                   help="Write summary JSON (overall + per-residue)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_model_bundle(args.model)
    model = bundle["model"]
    model_features: list[str] = list(bundle["features"])

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

    print("=" * 60)
    print("  Cov_Screen — LGBMRanker evaluation")
    print("=" * 60)
    print(f"  Model           : {args.model}")
    print(f"  Features        : {len(model_features)}")
    print(f"  Reward mode     : {reward_mode}")
    print(f"  Top-K           : {k}")
    print(f"  Top-% threshold : {top_pct:g}%")
    print(f"  Epsilon         : {epsilon}")
    if args.perfect_match:
        print("  Hit counting    : one per label (perfect-match: all matched "
              "warheads must hit when label lists multiple Frankenstein warheads)")
    else:
        print("  Hit counting    : one per label (any matched warhead hit counts)")

    merged = load_and_merge(args.training, args.labels)
    n_groups = merged["query_group"].nunique()
    print(f"\n[INFO] Query groups to score: {n_groups:,}")

    label_wh_counts = build_label_warhead_counts(args.labels)

    merged = merged.sort_values("query_group").reset_index(drop=True)
    merged["pred_score"] = predict_scores(
        merged, model, model_features,
        normalize_within_protein=args.normalize_within_protein,
    )

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
        eval_qg_df, args.perfect_match, label_wh_counts,
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

    summary = {
        "model": str(Path(args.model).resolve()),
        "training": str(Path(args.training).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "k": k,
        "top_pct_threshold": top_pct,
        "reward_mode": reward_mode,
        "rank_bonus_epsilon": epsilon,
        "perfect_match": args.perfect_match,
        "n_query_groups": int(len(eval_qg_df)),
        "n_labels": int(len(eval_label_df)),
        "overall": overall,
        "per_residue": per_res,
    }
    if composition.get("n_query_groups"):
        summary["residue_composition"] = {
            k_: v for k_, v in composition.items() if k_ != "detail"
        }

    if args.export_results:
        eval_label_df.to_csv(args.export_results, index=False)
        print(f"[INFO] Per-label metrics exported → {args.export_results}")

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
        export_scores_csv(merged, args.export_scores)

    if args.export_summary:
        export_results_json(summary, args.export_summary)

    print("Done.")


if __name__ == "__main__":
    main()
