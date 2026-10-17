"""
Compare features between known ligand hits and non-hits.

The matching logic mirrors Cov_Screen.py --VS:
  * Labels select ligand Name and matched Frankenstein_Warhead values.
  * Residue/ResNum/Chain columns in labels are ignored.
  * --residue selects all sites of that residue type from training.
  * Optional --chain and --resnum narrow the selected training sites.

Each matched site-warhead row is an observation. A ligand is positive when its
normalized Name occurs in the input CSV's Name_Hit column; all other matched
ligands are negative.

Outputs:
  feature_statistics.csv
  spearman_correlation.csv
  matched_feature_rows.csv
  unmatched_hit_names.csv

Example:
  python feature_importance.py \
      --training ../training_bo_extended_full.csv \
      --labels ../labels_bo.csv \
      --hits hits.csv \
      --residue CYS \
      --features deprotonation_prob HOMO_LUMO_Gap_Deprotonated \
                 Fukui_Deprotonated Nucleophilicity_Index_Deprotonated \
      --output-dir feature_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


# Allow execution from VS_eval/ while importing the sibling training module.
SCORING_DIR = Path(__file__).resolve().parent.parent
if str(SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_DIR))

from Cov_Screen import filter_vs_site_eval, load_merged_for_vs  # noqa: E402


IDENTIFIER_COLUMNS = [
    "Name",
    "Residue",
    "ResNum",
    "Chain",
    "Warhead",
    "query_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected features between Name_Hit ligands and all other "
            "training/label-matched ligands."
        )
    )
    parser.add_argument("--training", required=True, help="Training CSV")
    parser.add_argument("--labels", required=True, help="Labels CSV")
    parser.add_argument(
        "--hits",
        "--input",
        dest="hits",
        required=True,
        help="CSV containing the positive ligand names",
    )
    parser.add_argument(
        "--name-hit-column",
        default="Name_Hit",
        help="Positive-name column in --hits (default: Name_Hit)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Training CSV feature columns to analyze",
    )
    parser.add_argument(
        "--residue",
        required=True,
        help="Training-site residue type to analyze (e.g. CYS)",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="Optional training-site chain filter (e.g. A)",
    )
    parser.add_argument(
        "--resnum",
        default=None,
        help="Optional training-site residue-number filter (e.g. 809)",
    )
    parser.add_argument(
        "--output-dir",
        default="feature_analysis",
        help="Output directory (default: feature_analysis)",
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
            f"[ERROR] Hit CSV is missing column {column!r}. "
            f"Available columns: {', '.join(hit_df.columns)}"
        )

    names = {normalize_name(value) for value in hit_df[column]}
    names.discard("")
    if not names:
        sys.exit(f"[ERROR] No non-empty names found in {column!r}.")
    return names


def prepare_matched_rows(
    training_csv: str,
    labels_csv: str,
    hit_names: set[str],
    features: list[str],
    residue: str,
    chain: str | None,
    resnum: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return one row per VS-matched training site-warhead observation.

    Labels supply only ligand Name and matched warheads. Training rows are then
    filtered by --residue and optional --chain/--resnum, exactly as in VS mode.
    """
    merged = load_merged_for_vs(training_csv, labels_csv)
    target_rows = filter_vs_site_eval(
        merged,
        residue=residue,
        chain=chain,
        resnum=resnum,
    ).copy()
    if target_rows.empty:
        filter_description = f"residue={str(residue).upper()}"
        if chain is not None:
            filter_description += f", chain={str(chain).upper()}"
        if resnum is not None:
            filter_description += f", resnum={resnum}"
        sys.exit(
            "[ERROR] No VS-matched training rows remained after filtering "
            f"{filter_description}."
        )

    missing_features = [feature for feature in features if feature not in target_rows]
    if missing_features:
        sys.exit(
            "[ERROR] Feature columns missing from matched training data:\n  "
            + ", ".join(missing_features)
        )

    target_rows["_analysis_name"] = target_rows["_name_upper"].map(normalize_name)
    target_rows["is_hit"] = target_rows["_analysis_name"].isin(hit_names).astype(int)
    target_rows["class_label"] = target_rows["is_hit"].map(
        {1: "hit", 0: "non_hit"}
    )

    for feature in features:
        target_rows[feature] = pd.to_numeric(target_rows[feature], errors="coerce")

    if target_rows["is_hit"].sum() == 0:
        sys.exit(
            "[ERROR] None of the Name_Hit values matched labeled training rows."
        )
    if (target_rows["is_hit"] == 0).sum() == 0:
        sys.exit(
            "[ERROR] No negative observations remain: every matched Name is in Name_Hit."
        )

    matched_names = set(target_rows["_analysis_name"])
    unmatched_hits = sorted(hit_names - matched_names)
    return target_rows, unmatched_hits


def rank_biserial_from_u(u_statistic: float, n_hit: int, n_non_hit: int) -> float:
    """
    Rank-biserial correlation oriented so positive means larger values in hits.

    scipy's U is computed for the first sample (hits):
        r_rb = 2U / (n_hit * n_non_hit) - 1
    """
    return float(2.0 * u_statistic / (n_hit * n_non_hit) - 1.0)


def feature_statistics(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records: list[dict] = []

    for feature in features:
        valid = rows.loc[rows[feature].notna(), ["is_hit", feature]]
        hit_values = valid.loc[valid["is_hit"] == 1, feature].to_numpy(dtype=float)
        non_hit_values = valid.loc[
            valid["is_hit"] == 0, feature
        ].to_numpy(dtype=float)

        n_hit = len(hit_values)
        n_non_hit = len(non_hit_values)
        if n_hit == 0 or n_non_hit == 0:
            records.append(
                {
                    "feature": feature,
                    "n_hit": n_hit,
                    "n_non_hit": n_non_hit,
                    "n_missing": int(rows[feature].isna().sum()),
                    "hit_mean": float(np.mean(hit_values)) if n_hit else np.nan,
                    "hit_median": float(np.median(hit_values)) if n_hit else np.nan,
                    "non_hit_mean": (
                        float(np.mean(non_hit_values)) if n_non_hit else np.nan
                    ),
                    "non_hit_median": (
                        float(np.median(non_hit_values)) if n_non_hit else np.nan
                    ),
                    "mann_whitney_u": np.nan,
                    "mann_whitney_p_two_sided": np.nan,
                    "rank_biserial_correlation": np.nan,
                    "roc_auc_raw": np.nan,
                    "pr_auc_raw": np.nan,
                    "higher_in": "insufficient_data",
                }
            )
            continue

        u_stat, p_value = mannwhitneyu(
            hit_values,
            non_hit_values,
            alternative="two-sided",
            method="auto",
        )
        y_true = valid["is_hit"].to_numpy(dtype=int)
        scores = valid[feature].to_numpy(dtype=float)
        auc = float(roc_auc_score(y_true, scores))
        pr_auc = float(average_precision_score(y_true, scores))

        records.append(
            {
                "feature": feature,
                "n_hit": n_hit,
                "n_non_hit": n_non_hit,
                "n_missing": int(rows[feature].isna().sum()),
                "hit_mean": float(np.mean(hit_values)),
                "hit_median": float(np.median(hit_values)),
                "non_hit_mean": float(np.mean(non_hit_values)),
                "non_hit_median": float(np.median(non_hit_values)),
                "mann_whitney_u": float(u_stat),
                "mann_whitney_p_two_sided": float(p_value),
                "rank_biserial_correlation": rank_biserial_from_u(
                    float(u_stat), n_hit, n_non_hit
                ),
                "roc_auc_raw": auc,
                "pr_auc_raw": pr_auc,
                "higher_in": (
                    "hits"
                    if auc > 0.5
                    else "non_hits"
                    if auc < 0.5
                    else "equal"
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        ["mann_whitney_p_two_sided", "feature"],
        na_position="last",
    )


def export_outputs(
    rows: pd.DataFrame,
    unmatched_hits: list[str],
    features: list[str],
    output_dir: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = feature_statistics(rows, features)
    stats_path = out_dir / "feature_statistics.csv"
    stats.to_csv(stats_path, index=False)

    # Spearman across all matched site-warhead observations (pairwise complete).
    corr = rows[features].corr(method="spearman", min_periods=2)
    corr.index.name = "feature"
    corr_path = out_dir / "spearman_correlation.csv"
    corr.to_csv(corr_path)

    audit_columns = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in rows.columns
    ] + ["_analysis_name", "is_hit", "class_label"] + features
    audit_path = out_dir / "matched_feature_rows.csv"
    rows[audit_columns].to_csv(audit_path, index=False)

    unmatched_path = out_dir / "unmatched_hit_names.csv"
    pd.DataFrame({"Name_Hit": unmatched_hits}).to_csv(unmatched_path, index=False)

    n_hit_rows = int((rows["is_hit"] == 1).sum())
    n_non_hit_rows = int((rows["is_hit"] == 0).sum())
    n_hit_names = int(rows.loc[rows["is_hit"] == 1, "_analysis_name"].nunique())
    n_non_hit_names = int(
        rows.loc[rows["is_hit"] == 0, "_analysis_name"].nunique()
    )

    print("\nFeature comparison complete")
    print(f"  Hit observations:     {n_hit_rows:,} ({n_hit_names:,} unique Names)")
    print(
        f"  Non-hit observations: {n_non_hit_rows:,} "
        f"({n_non_hit_names:,} unique Names)"
    )
    print(f"  Unmatched hit Names:  {len(unmatched_hits):,}")
    print(f"  Feature statistics:  {stats_path}")
    print(f"  Spearman matrix:     {corr_path}")
    print(f"  Matched audit rows:  {audit_path}")
    print(f"  Unmatched hit names: {unmatched_path}")
    print(
        "\n[NOTE] Each matched site-warhead row is treated as an observation, "
        "as requested. Rows sharing a Name are not statistically independent; "
        "interpret Mann-Whitney p-values accordingly."
    )


def main() -> None:
    args = parse_args()
    features = list(dict.fromkeys(args.features))
    hit_names = load_hit_names(args.hits, args.name_hit_column)
    rows, unmatched_hits = prepare_matched_rows(
        args.training,
        args.labels,
        hit_names,
        features,
        args.residue,
        args.chain,
        args.resnum,
    )
    export_outputs(rows, unmatched_hits, features, args.output_dir)


if __name__ == "__main__":
    main()
