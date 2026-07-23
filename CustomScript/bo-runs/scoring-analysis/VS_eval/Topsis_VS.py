"""
TOPSIS virtual screening over matched ligand-warhead variants.

Labels are used like Cov_Screen.py --VS:
  * select ligand Names;
  * match label Frankenstein_Warhead values to training Warhead values;
  * ignore label Residue/ResNum/Chain.

Training rows are selected with one or more --residue values and optional
--chain/--resnum. Features are never averaged across sites. Each
Residue/ResNum/Chain x Name x Warhead row is retained as a candidate.

Two analyses are produced:
  * per-site: TOPSIS is run independently for every selected residue site;
  * pooled: all selected residue-site x ligand x warhead rows are ranked
    together as one collective screen.

Outputs:
  topsis_per_site_ranking.csv         independent rankings for every site
  topsis_pooled_pair_ranking.csv      collective ranking of all site pairs
  enrichment_factors.csv              all candidates ranked; any hit detected
  feature_statistics.csv              per-site and pooled single-feature stats
  feature_enrichment_factors.csv      all candidates ranked; any hit detected
  matched_site_warhead_rows.csv       exact source rows used

Example:
  python TOPSIS_VS.py \
      --training ../training_usp8_extended_full.csv \
      --labels ../labels_usp8.csv \
      --hits usp8_hits.csv \
      --residue CYS \
      --benefit Nucleophilicity_Index_Deprotonated Hydrophobic_Fit Geo_Fit \
      --cost HOMO_LUMO_Gap_Deprotonated \
      --weights 1 1 1 1 \
      --output-dir usp8_topsis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SCORING_DIR = Path(__file__).resolve().parent.parent
if str(SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_DIR))

from Cov_Screen import filter_vs_site_eval, load_merged_for_vs  # noqa: E402


EF_FRACTIONS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TOPSIS ranking of VS-matched Name x Warhead candidates"
    )
    parser.add_argument("--training", required=True, help="Training feature CSV")
    parser.add_argument("--labels", required=True, help="Labels CSV for Names/warheads")
    parser.add_argument(
        "--hits",
        "--input",
        dest="hits",
        required=True,
        help="CSV containing positive ligand Names",
    )
    parser.add_argument(
        "--name-hit-column",
        default="Name_Hit",
        help="Positive-name column in --hits (default: Name_Hit)",
    )
    parser.add_argument(
        "--residue",
        nargs="+",
        required=True,
        help="One or more training-site residue types (e.g. CYS or CYS SER)",
    )
    parser.add_argument("--chain", default=None, help="Optional chain filter")
    parser.add_argument("--resnum", default=None, help="Optional residue-number filter")
    parser.add_argument(
        "--benefit",
        nargs="*",
        default=[],
        metavar="FEATURE",
        help="Features where larger values are preferred",
    )
    parser.add_argument(
        "--cost",
        nargs="*",
        default=[],
        metavar="FEATURE",
        help="Features where smaller values are preferred",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional non-negative weights in feature order: all --benefit "
            "features followed by all --cost features. Default: equal."
        ),
    )
    parser.add_argument(
        "--missing-policy",
        choices=("median", "drop"),
        default="median",
        help=(
            "Handle missing feature values by criterion median "
            "(default) or drop the candidate"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="topsis_results",
        help="Output directory (default: topsis_results)",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def load_hit_names(path: str, column: str) -> set[str]:
    hit_df = pd.read_csv(path)
    hit_df.columns = hit_df.columns.str.strip()
    if column not in hit_df.columns:
        sys.exit(
            f"[ERROR] Hit CSV is missing {column!r}. "
            f"Available: {', '.join(hit_df.columns)}"
        )
    names = {normalize_name(value) for value in hit_df[column]}
    names.discard("")
    if not names:
        sys.exit(f"[ERROR] No non-empty hit Names found in {column!r}.")
    return names


def validate_feature_args(args: argparse.Namespace) -> tuple[list[str], np.ndarray]:
    benefit = list(dict.fromkeys(args.benefit))
    cost = list(dict.fromkeys(args.cost))
    overlap = sorted(set(benefit) & set(cost))
    if overlap:
        sys.exit(
            "[ERROR] Features cannot be both benefit and cost:\n  "
            + ", ".join(overlap)
        )

    features = benefit + cost
    if not features:
        sys.exit("[ERROR] Supply at least one --benefit or --cost feature.")

    if args.weights is None:
        weights = np.ones(len(features), dtype=float)
    else:
        if len(args.weights) != len(features):
            sys.exit(
                f"[ERROR] --weights has {len(args.weights)} values but "
                f"{len(features)} features were specified. Weight order is "
                "all --benefit features followed by all --cost features."
            )
        weights = np.asarray(args.weights, dtype=float)

    if not np.isfinite(weights).all() or (weights < 0).any():
        sys.exit("[ERROR] --weights must be finite and non-negative.")
    if float(weights.sum()) <= 0:
        sys.exit("[ERROR] At least one TOPSIS weight must be positive.")
    weights = weights / weights.sum()
    return features, weights


def load_selected_rows(
    args: argparse.Namespace,
    features: list[str],
    hit_names: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    merged = load_merged_for_vs(args.training, args.labels)
    residues = list(dict.fromkeys(str(value).strip().upper() for value in args.residue))
    selected_parts = [
        filter_vs_site_eval(
            merged,
            residue=residue,
            chain=args.chain,
            resnum=args.resnum,
        )
        for residue in residues
    ]
    selected = pd.concat(selected_parts, ignore_index=True).copy()
    if selected.empty:
        sys.exit(
            "[ERROR] No matched training rows remained after the requested "
            "residue/chain/resnum filter."
        )

    missing = [feature for feature in features if feature not in selected.columns]
    if missing:
        sys.exit(
            "[ERROR] Features missing from training CSV:\n  " + ", ".join(missing)
        )

    selected["_analysis_name"] = selected["_name_upper"].map(normalize_name)
    selected["_analysis_warhead"] = (
        selected["Warhead"].astype(str).str.strip()
    )
    selected["_site_key"] = (
        selected["Residue"].astype(str).str.strip().str.upper()
        + ":"
        + selected["ResNum"].astype(str).str.strip()
        + ":"
        + selected["Chain"].astype(str).str.strip().str.upper()
    )
    selected["is_hit"] = selected["_analysis_name"].isin(hit_names).astype(int)
    for feature in features:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")

    matched_names = set(selected["_analysis_name"])
    unmatched_hits = sorted(hit_names - matched_names)
    if selected["is_hit"].sum() == 0:
        sys.exit("[ERROR] None of the Name_Hit values matched selected rows.")
    return selected, unmatched_hits


def build_site_candidates(
    selected: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Keep every site x Name x Warhead pair separate; never average sites.
    """
    columns = [
        "_analysis_name",
        "_analysis_warhead",
        "Residue",
        "ResNum",
        "Chain",
        "_site_key",
        "is_hit",
        *features,
    ]
    candidates = selected[columns].rename(
        columns={
            "_analysis_name": "Name",
            "_analysis_warhead": "Warhead",
            "_site_key": "Site",
        }
    ).copy()
    key_cols = ["Name", "Warhead", "Residue", "ResNum", "Chain"]
    duplicate_count = int(candidates.duplicated(key_cols).sum())
    if duplicate_count:
        print(
            f"[WARN] Dropping {duplicate_count:,} duplicate site x Name x "
            "Warhead rows; no feature values are averaged."
        )
        candidates = candidates.drop_duplicates(key_cols, keep="first").copy()

    candidates["Name_Warhead"] = (
        candidates["Name"] + "_" + candidates["Warhead"]
    )
    candidates["Site_Name_Warhead"] = (
        candidates["Site"] + "_" + candidates["Name_Warhead"]
    )
    return candidates.reset_index(drop=True)


def handle_missing(
    candidates: pd.DataFrame,
    features: list[str],
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_counts = pd.DataFrame(
        {
            "feature": features,
            "n_missing_before": [
                int(candidates[feature].isna().sum()) for feature in features
            ],
        }
    )

    result = candidates.copy()
    if policy == "drop":
        before = len(result)
        result = result.dropna(subset=features).copy()
        print(f"  Missing policy drop: removed {before - len(result):,} candidates")
    else:
        for feature in features:
            median = result[feature].median()
            if pd.isna(median):
                sys.exit(
                    f"[ERROR] Feature {feature!r} is entirely missing after aggregation."
                )
            result[feature] = result[feature].fillna(float(median))

    if result.empty:
        sys.exit("[ERROR] No candidates remain after missing-value handling.")
    return result, missing_counts


def topsis_rank(
    candidates: pd.DataFrame,
    benefit_features: list[str],
    cost_features: list[str],
    weights: np.ndarray,
    rank_column: str = "overall_rank",
) -> pd.DataFrame:
    features = benefit_features + cost_features
    matrix = candidates[features].to_numpy(dtype=float)

    # Standard vector normalization followed by normalized criterion weights.
    denominators = np.sqrt(np.square(matrix).sum(axis=0))
    zero_norm = denominators == 0
    if zero_norm.any():
        constant = [features[i] for i in np.flatnonzero(zero_norm)]
        print(
            "[WARN] Zero vector norm for criterion/criteria; normalized values "
            "set to zero: " + ", ".join(constant)
        )
        denominators[zero_norm] = 1.0
    weighted = (matrix / denominators) * weights

    n_benefit = len(benefit_features)
    ideal_best = np.empty(len(features), dtype=float)
    ideal_worst = np.empty(len(features), dtype=float)
    for idx in range(len(features)):
        if idx < n_benefit:
            ideal_best[idx] = weighted[:, idx].max()
            ideal_worst[idx] = weighted[:, idx].min()
        else:
            ideal_best[idx] = weighted[:, idx].min()
            ideal_worst[idx] = weighted[:, idx].max()

    distance_best = np.sqrt(np.square(weighted - ideal_best).sum(axis=1))
    distance_worst = np.sqrt(np.square(weighted - ideal_worst).sum(axis=1))
    distance_sum = distance_best + distance_worst
    score = np.divide(
        distance_worst,
        distance_sum,
        out=np.full_like(distance_worst, 0.5),
        where=distance_sum > 0,
    )

    ranked = candidates.copy()
    ranked["distance_to_ideal_best"] = distance_best
    ranked["distance_to_ideal_worst"] = distance_worst
    ranked["topsis_score"] = score
    ranked = ranked.sort_values(
        ["topsis_score", "Name", "Warhead"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranked[rank_column] = np.arange(1, len(ranked) + 1)

    leading = [
        rank_column,
        "Site",
        "Residue",
        "ResNum",
        "Chain",
        "Name",
        "Warhead",
        "Name_Warhead",
        "Site_Name_Warhead",
        "is_hit",
        "topsis_score",
        "distance_to_ideal_best",
        "distance_to_ideal_worst",
    ]
    leading = [column for column in leading if column in ranked.columns]
    remaining = [column for column in ranked.columns if column not in leading]
    return ranked[leading + remaining]


def enrichment_factors(
    ranking: pd.DataFrame,
    level: str,
    rank_column: str = "overall_rank",
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    EF_x = (hits in top x / top x size) / (all hits / all candidates).
    """
    n_total = len(ranking)
    n_hits = int(ranking["is_hit"].sum())
    if n_total == 0 or n_hits == 0:
        return pd.DataFrame()

    prevalence = n_hits / n_total
    records: list[dict] = []
    for fraction in EF_FRACTIONS:
        top_n = max(1, int(np.ceil(fraction * n_total)))
        top = ranking.nsmallest(top_n, rank_column)
        hits_top = int(top["is_hit"].sum())
        ef = (hits_top / top_n) / prevalence
        max_possible_hits = min(top_n, n_hits)
        max_ef = (max_possible_hits / top_n) / prevalence
        record = {
                "ranking_level": level,
                "cutoff_percent": 100.0 * fraction,
                "top_n": top_n,
                "n_candidates": n_total,
                "n_total_hits": n_hits,
                "n_hits_in_top": hits_top,
                "hit_prevalence": prevalence,
                "enrichment_factor": float(ef),
                "maximum_possible_ef": float(max_ef),
                "fraction_of_hits_recovered": hits_top / n_hits,
        }
        if metadata:
            record = {**metadata, **record}
        records.append(record)
    return pd.DataFrame(records)


def enrichment_any_candidate_per_ligand(
    ranking: pd.DataFrame,
    level: str,
    rank_column: str,
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    Rank all candidate rows, but count each hit ligand at most once per cutoff.

    The top-X% cutoff is based on all warhead/site candidate rows. A known-hit
    ligand is recovered when any of its candidates occurs within that cutoff.
    EF is hit-ligand recovery divided by the actual fraction of candidate rows
    screened.
    """
    n_rows = len(ranking)
    unique_names = ranking["Name"].nunique()
    hit_names = set(ranking.loc[ranking["is_hit"] == 1, "Name"])
    if n_rows == 0 or not hit_names:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for fraction in EF_FRACTIONS:
        top_n = max(1, int(np.ceil(fraction * n_rows)))
        top = ranking.nsmallest(top_n, rank_column)
        recovered_names = set(top.loc[top["is_hit"] == 1, "Name"])
        n_recovered = len(recovered_names)
        screened_fraction = top_n / n_rows
        recovery_fraction = n_recovered / len(hit_names)
        max_recovered = min(top_n, len(hit_names))
        max_recovery_fraction = max_recovered / len(hit_names)
        record: dict[str, object] = {
            "ranking_level": level,
            "cutoff_percent": 100.0 * fraction,
            "top_n_candidate_rows": top_n,
            "n_candidate_rows": n_rows,
            "n_unique_ligands": unique_names,
            "n_total_hit_ligands": len(hit_names),
            "n_hit_ligands_recovered": n_recovered,
            "candidate_fraction_screened": screened_fraction,
            "fraction_of_hit_ligands_recovered": recovery_fraction,
            "enrichment_factor": recovery_fraction / screened_fraction,
            "maximum_possible_ef": (
                max_recovery_fraction / screened_fraction
            ),
        }
        if metadata:
            record = {**metadata, **record}
        records.append(record)
    return pd.DataFrame(records)


def rank_each_site(
    candidates: pd.DataFrame,
    features: list[str],
    benefit_features: list[str],
    cost_features: list[str],
    weights: np.ndarray,
    missing_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rankings: list[pd.DataFrame] = []
    ef_tables: list[pd.DataFrame] = []
    missing_tables: list[pd.DataFrame] = []
    for site, group in candidates.groupby("Site", sort=True):
        handled, missing = handle_missing(group, features, missing_policy)
        site_ranked = topsis_rank(
            handled,
            benefit_features,
            cost_features,
            weights,
            rank_column="site_rank",
        )
        rankings.append(site_ranked)
        site_metadata = {
            "analysis_scope": "per_site",
            "Site": site,
            "Residue": group["Residue"].iloc[0],
            "ResNum": group["ResNum"].iloc[0],
            "Chain": group["Chain"].iloc[0],
        }
        ef_tables.append(
            enrichment_any_candidate_per_ligand(
                site_ranked,
                "topsis_site_all_warheads_any_hit",
                rank_column="site_rank",
                metadata=site_metadata,
            )
        )
        for key, value in site_metadata.items():
            missing[key] = value
        missing_tables.append(missing)

    if not rankings:
        sys.exit("[ERROR] No per-site candidates could be ranked.")
    return (
        pd.concat(rankings, ignore_index=True),
        pd.concat(ef_tables, ignore_index=True),
        pd.concat(missing_tables, ignore_index=True),
    )


def feature_statistics_and_enrichment(
    candidates: pd.DataFrame,
    features: list[str],
    benefit_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate row-level feature statistics and EF without averaging sites."""
    directions = {
        feature: ("benefit" if feature in benefit_features else "cost")
        for feature in features
    }
    scopes: list[tuple[str, str, pd.DataFrame]] = [
        ("pooled", "ALL", candidates)
    ]
    scopes.extend(
        ("per_site", str(site), group)
        for site, group in candidates.groupby("Site", sort=True)
    )

    statistic_records: list[dict[str, object]] = []
    ef_tables: list[pd.DataFrame] = []
    for scope, site, group in scopes:
        site_metadata: dict[str, object] = {
            "analysis_scope": scope,
            "Site": site,
            "Residue": "ALL" if scope == "pooled" else group["Residue"].iloc[0],
            "ResNum": "ALL" if scope == "pooled" else group["ResNum"].iloc[0],
            "Chain": "ALL" if scope == "pooled" else group["Chain"].iloc[0],
        }
        for feature in features:
            valid = group.loc[
                group[feature].notna(), ["Name", feature, "is_hit"]
            ].copy()
            hit_values = valid.loc[valid["is_hit"] == 1, feature]
            non_hit_values = valid.loc[valid["is_hit"] == 0, feature]
            oriented = (
                valid[feature]
                if directions[feature] == "benefit"
                else -valid[feature]
            )
            has_two_classes = valid["is_hit"].nunique() == 2
            auc = (
                float(roc_auc_score(valid["is_hit"], oriented))
                if has_two_classes
                else np.nan
            )
            pr_auc = (
                float(average_precision_score(valid["is_hit"], oriented))
                if has_two_classes
                else np.nan
            )
            statistic_records.append(
                {
                    **site_metadata,
                    "feature": feature,
                    "direction": directions[feature],
                    "n_rows": len(valid),
                    "n_hit_rows": int((valid["is_hit"] == 1).sum()),
                    "n_non_hit_rows": int((valid["is_hit"] == 0).sum()),
                    "hit_mean": float(hit_values.mean()) if len(hit_values) else np.nan,
                    "non_hit_mean": (
                        float(non_hit_values.mean()) if len(non_hit_values) else np.nan
                    ),
                    "hit_median": (
                        float(hit_values.median()) if len(hit_values) else np.nan
                    ),
                    "non_hit_median": (
                        float(non_hit_values.median())
                        if len(non_hit_values)
                        else np.nan
                    ),
                    "roc_auc_oriented": auc,
                    "pr_auc_oriented": pr_auc,
                }
            )
            if valid.empty:
                continue
            feature_ranking = valid.assign(_oriented_score=oriented).sort_values(
                "_oriented_score", ascending=False
            )
            feature_ranking["overall_rank"] = np.arange(
                1, len(feature_ranking) + 1
            )
            ef_tables.append(
                enrichment_any_candidate_per_ligand(
                    feature_ranking,
                    f"single_feature_all_candidates_any_hit:{feature}",
                    rank_column="overall_rank",
                    metadata={
                        **site_metadata,
                        "feature": feature,
                        "direction": directions[feature],
                    },
                )
            )

    statistics = pd.DataFrame(statistic_records)
    enrichment = (
        pd.concat(ef_tables, ignore_index=True) if ef_tables else pd.DataFrame()
    )
    return statistics, enrichment


def export_results(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    per_site_ranking: pd.DataFrame,
    pooled_ranking: pd.DataFrame,
    topsis_enrichment: pd.DataFrame,
    feature_statistics: pd.DataFrame,
    feature_enrichment: pd.DataFrame,
    missing_counts: pd.DataFrame,
    unmatched_hits: list[str],
    features: list[str],
    weights: np.ndarray,
) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_site_ranking.to_csv(
        out_dir / "topsis_per_site_ranking.csv", index=False
    )
    per_site_ranking.loc[per_site_ranking["is_hit"] == 1].to_csv(
        out_dir / "hit_per_site_ranks.csv", index=False
    )
    pooled_ranking.to_csv(
        out_dir / "topsis_pooled_pair_ranking.csv", index=False
    )
    pooled_ranking.loc[pooled_ranking["is_hit"] == 1].to_csv(
        out_dir / "hit_pooled_pair_ranks.csv", index=False
    )
    topsis_enrichment.to_csv(out_dir / "enrichment_factors.csv", index=False)
    feature_statistics.to_csv(out_dir / "feature_statistics.csv", index=False)
    feature_enrichment.to_csv(
        out_dir / "feature_enrichment_factors.csv", index=False
    )

    audit_cols = [
        column
        for column in [
            "Name",
            "Residue",
            "ResNum",
            "Chain",
            "Warhead",
            "query_group",
            "_analysis_name",
            "_analysis_warhead",
            "is_hit",
        ]
        if column in selected.columns
    ] + features
    selected[audit_cols].to_csv(
        out_dir / "matched_site_warhead_rows.csv", index=False
    )
    candidates.to_csv(out_dir / "site_name_warhead_candidates.csv", index=False)
    missing_counts.to_csv(out_dir / "missing_feature_counts.csv", index=False)
    pd.DataFrame({"Name_Hit": unmatched_hits}).to_csv(
        out_dir / "unmatched_hit_names.csv", index=False
    )

    direction = {
        feature: "benefit" for feature in args.benefit
    } | {feature: "cost" for feature in args.cost}
    pd.DataFrame(
        {
            "feature": features,
            "direction": [direction[feature] for feature in features],
            "normalized_weight": weights,
        }
    ).to_csv(out_dir / "topsis_criteria.csv", index=False)

    print("\nTOPSIS virtual screening complete")
    print(f"  Selected residue sites:    {candidates['Site'].nunique():,}")
    print(f"  Site × ligand × warhead:   {len(candidates):,}")
    print(f"  Unique ligand Names:       {candidates['Name'].nunique():,}")
    print(
        "  Positive Names matched:    "
        f"{candidates.loc[candidates['is_hit'] == 1, 'Name'].nunique():,}"
    )
    print(f"  Unmatched positive Names:  {len(unmatched_hits):,}")
    print(
        f"  Per-site ranking:          "
        f"{out_dir / 'topsis_per_site_ranking.csv'}"
    )
    print(
        f"  Pooled pair ranking:       "
        f"{out_dir / 'topsis_pooled_pair_ranking.csv'}"
    )
    print(f"  TOPSIS enrichment:         {out_dir / 'enrichment_factors.csv'}")
    print(
        f"  Feature enrichment:        "
        f"{out_dir / 'feature_enrichment_factors.csv'}"
    )


def main() -> None:
    args = parse_args()
    features, weights = validate_feature_args(args)
    hit_names = load_hit_names(args.hits, args.name_hit_column)
    selected, unmatched_hits = load_selected_rows(args, features, hit_names)
    candidates = build_site_candidates(selected, features)
    benefit_features = list(dict.fromkeys(args.benefit))
    cost_features = list(dict.fromkeys(args.cost))

    per_site_ranking, per_site_ef, per_site_missing = rank_each_site(
        candidates,
        features,
        benefit_features,
        cost_features,
        weights,
        args.missing_policy,
    )
    pooled_candidates, pooled_missing = handle_missing(
        candidates, features, args.missing_policy
    )
    pooled_missing["analysis_scope"] = "pooled"
    pooled_missing["Site"] = "ALL"
    pooled_missing["Residue"] = "ALL"
    pooled_missing["ResNum"] = "ALL"
    pooled_missing["Chain"] = "ALL"
    missing_counts = pd.concat(
        [per_site_missing, pooled_missing], ignore_index=True
    )
    pooled_ranking = topsis_rank(
        pooled_candidates,
        benefit_features,
        cost_features,
        weights,
    )
    pooled_ef = enrichment_any_candidate_per_ligand(
        pooled_ranking,
        "topsis_pooled_all_pairs_any_hit",
        rank_column="overall_rank",
        metadata={
            "analysis_scope": "pooled",
            "Site": "ALL",
            "Residue": "ALL",
            "ResNum": "ALL",
            "Chain": "ALL",
        },
    )
    topsis_enrichment = pd.concat([per_site_ef, pooled_ef], ignore_index=True)
    feature_statistics, feature_enrichment = feature_statistics_and_enrichment(
        candidates,
        features,
        benefit_features,
    )
    export_results(
        args,
        selected,
        candidates,
        per_site_ranking,
        pooled_ranking,
        topsis_enrichment,
        feature_statistics,
        feature_enrichment,
        missing_counts,
        unmatched_hits,
        features,
        weights,
    )


if __name__ == "__main__":
    main()
