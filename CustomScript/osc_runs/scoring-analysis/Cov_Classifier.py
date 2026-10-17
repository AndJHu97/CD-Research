#!/usr/bin/env python3
"""Score CovSite ligand–site candidates with a Training_Cov_Classifier model.

Give a hits/labels CSV (all_hits or labels_covdb style). Each positive is an
electrophile at one target site: PDB_ID × Residue × ResNum × Chain. Other
candidate ligands at that site are negatives.

Pass --candidates to reuse a CovSite feature table. Omit it and the script
runs CovSite --VS on --library (or on the labels file when no library is
given) and then scores those rows. --keep-warhead limits every ligand to the
named warhead before scoring.

Outputs match the training classifier: AUROC, AUPRC, accuracy, balanced
accuracy, precision, recall, F1, and MCC at the cutoff stored in the pickle,
plus per-site ranking metrics (EF@1/5/10%, LogAUC, adjusted LogAUC, BEDROC,
median search-space reduction, Hit@K, and Hit@top-%).

Example:
    python Cov_Classifier.py \\
        --model covsite_classifier_output/lgbm_classifier.pkl \\
        --labels all_hits.csv \\
        --candidates covsite_candidates_combined.csv \\
        --keep-warhead "Acrylamide warhead" \\
        --output-dir cov_classifier_eval
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Training_Cov_Classifier import (
    DEFAULT_COVSITE_VS,
    DEFAULT_TOPSIS_VS,
    VALID_RESIDUES,
    covsite_classifier_metric_rows,
    covsite_ligand_rankings,
    load_python_module,
    predict_proba_positive,
    residue_ranking_metric_rows,
)


RESIDUE_ONEHOT = ("CYS", "SER", "THR", "TYR", "LYS", "HIS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a Training_Cov_Classifier pickle to CovSite candidates "
            "and report classification plus ranking metrics."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="lgbm_classifier.pkl written by Training_Cov_Classifier",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help=(
            "Hits CSV. Positives are electrophiles at PDB_ID × Residue × "
            "ResNum × Chain. Accepts all_hits (electrophile_hit) or "
            "labels_covdb (electrophile_smiles, Name)."
        ),
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Existing CovSite candidates CSV. Skips CovSite featurization.",
    )
    parser.add_argument(
        "--library",
        default=None,
        help=(
            "Ligand library for CovSite --VS when --candidates is omitted. "
            "Needs pdb, smiles, residue, resnum, and chain. Defaults to "
            "--labels, which scores only the labeled complexes."
        ),
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="PDB directory for CovSite --VS (required without --candidates)",
    )
    parser.add_argument(
        "--output-dir",
        default="cov_classifier_eval",
        help="Output directory (default: cov_classifier_eval)",
    )
    parser.add_argument(
        "--keep-warhead",
        nargs="+",
        default=None,
        metavar="WARHEAD",
        help=(
            "Score only these warhead types for every ligand, "
            "e.g. --keep-warhead \"Acrylamide warhead\""
        ),
    )
    parser.add_argument(
        "--residue",
        nargs="+",
        default=None,
        metavar="RES",
        help="Score only these residue types, e.g. --residue CYS",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Classification cutoff. Default: the cutoff stored in the pickle."
        ),
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--top-pct", type=float, default=1.0)
    parser.add_argument("--bedroc-alpha", type=float, default=20.0)
    parser.add_argument("--best-warhead", action="store_true")
    parser.add_argument("--collapse-active-protomers", action="store_true")
    parser.add_argument("--multiple-hits", action="store_true")
    parser.add_argument("--pdb-hits", action="store_true")
    parser.add_argument("--topsis-vs", default=None)
    parser.add_argument("--pdb-column", default=None)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--residue-column", default=None)
    parser.add_argument("--resnum-column", default=None)
    parser.add_argument("--chain-column", default=None)
    parser.add_argument("--hit-residue-column", default=None)
    parser.add_argument("--hit-resnum-column", default=None)
    parser.add_argument("--hit-chain-column", default=None)
    parser.add_argument("--smiles-hit-column", default="electrophile_hit")
    parser.add_argument("--name-hit-column", default="Name_Hit")
    parser.add_argument("--name-miss-column", default="Name_Miss")
    parser.add_argument("--smiles-miss-column", default="electrophile_miss")
    parser.add_argument("--warhead-type-column", default="warhead_type")
    parser.add_argument("--sasa-deprot", action="store_true")
    parser.add_argument("--ions", action="store_true")
    parser.add_argument("--no-warhead", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--scripts-dir", default=None)
    args = parser.parse_args()
    if args.candidates is None and not args.pdb_dir:
        parser.error("--pdb-dir is required unless --candidates is set")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.topk < 1:
        parser.error("--topk must be >= 1")
    if not 0 < args.top_pct <= 100:
        parser.error("--top-pct must be in (0, 100]")
    if args.bedroc_alpha <= 0:
        parser.error("--bedroc-alpha must be > 0")
    return args


def helper_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
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


def load_model(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"[ERROR] Model pickle not found: {path}")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if "model" not in bundle or "features" not in bundle:
        sys.exit(
            "[ERROR] Pickle is missing 'model' or 'features'. "
            "Use lgbm_classifier.pkl from Training_Cov_Classifier."
        )
    features = [str(feature) for feature in bundle["features"]]
    if not features:
        sys.exit("[ERROR] Model pickle has an empty feature list.")
    bundle["features"] = features
    return bundle


def resolve_candidates(args: argparse.Namespace, vs_module) -> Path:
    if args.candidates:
        path = Path(args.candidates).resolve()
        if not path.is_file():
            sys.exit(f"[ERROR] --candidates not found: {path}")
        print(f"[INFO] Reusing candidates: {path}")
        return path

    library = Path(args.library).resolve() if args.library else Path(args.labels).resolve()
    if not library.is_file():
        sys.exit(f"[ERROR] CovSite library not found: {library}")
    if args.library is None:
        print(
            "[WARN] No --library and no --candidates. CovSite will featurize "
            "only the labeled complexes, so sites may have no decoys."
        )
    feat_dir = Path(args.output_dir).resolve() / "covsite_features"
    print(f"[INFO] Running CovSite --VS on {library}")
    return vs_module.run_covsite_vs(
        library,
        Path(args.pdb_dir).resolve(),
        feat_dir,
        args,
    )


def filter_residue_rows(
    candidates: pd.DataFrame,
    residues: list[str],
) -> pd.DataFrame:
    wanted: list[str] = []
    for residue in residues:
        token = str(residue).strip().upper()
        if token not in VALID_RESIDUES:
            allowed = ", ".join(sorted(VALID_RESIDUES))
            sys.exit(f"[ERROR] --residue {residue!r} is not one of {allowed}.")
        if token not in wanted:
            wanted.append(token)
    if "Residue" not in candidates.columns:
        sys.exit("[ERROR] --residue needs a Residue column in the candidates.")
    names = candidates["Residue"].astype(str).str.strip().str.upper()
    kept = candidates.loc[names.isin(wanted)].copy()
    print(
        f"[INFO] --residue {' '.join(wanted)}: kept {len(kept):,} / "
        f"{len(candidates):,} candidate row(s)"
    )
    if kept.empty:
        sys.exit("[ERROR] No candidate rows remain after --residue.")
    return kept


def add_model_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build one-hot and within-site normalized columns the pickle expects."""
    work = frame.copy()
    residue = work["Residue"].astype(str).str.strip().str.upper()
    resnum = work["ResNum"].astype(str).str.strip()
    # label_candidates names the site column _site_key; build_site_candidates
    # renames it to Site only later.
    site_key = "Site" if "Site" in work.columns else "_site_key"
    for feature in feature_cols:
        if feature in work.columns:
            continue
        if feature.startswith("resnum_eq_"):
            number = feature[len("resnum_eq_"):]
            work[feature] = (resnum == number).astype(int)
            continue
        if feature.startswith("res_"):
            token = feature[4:].upper()
            if token in RESIDUE_ONEHOT:
                work[feature] = (residue == token).astype(int)
                continue
        if feature.endswith("_norm"):
            base = feature[: -len("_norm")]
            if base not in work.columns:
                sys.exit(
                    f"[ERROR] Model feature {feature!r} needs source column "
                    f"{base!r}, which is not in the candidates."
                )
            if site_key not in work.columns:
                sys.exit(
                    f"[ERROR] Model feature {feature!r} needs a site column "
                    "to normalize within, which is not in the candidates."
                )
            values = pd.to_numeric(work[base], errors="coerce")
            work[feature] = values.groupby(work[site_key]).transform(
                lambda column: (column - column.min())
                / (column.max() - column.min() + 1e-8)
            )
    missing = [feature for feature in feature_cols if feature not in work.columns]
    if missing:
        sys.exit(
            "[ERROR] Model features missing from candidates:\n  "
            + ", ".join(missing)
        )
    for feature in feature_cols:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
    return work


def prepare_scored_frame(
    args: argparse.Namespace,
    candidates_path: Path,
    feature_cols: list[str],
    vs_module,
    tvs_module,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    helpers = helper_namespace(args)
    labels_path = Path(args.labels).resolve()
    if not labels_path.is_file():
        sys.exit(f"[ERROR] --labels not found: {labels_path}")

    candidates = pd.read_csv(candidates_path, low_memory=False)
    if candidates.empty:
        sys.exit("[ERROR] Candidates CSV has no rows.")
    library = vs_module.load_library(candidates_path, helpers)
    if args.residue:
        candidates = filter_residue_rows(candidates, args.residue)
    if args.keep_warhead:
        candidates = vs_module.filter_candidates_to_warheads(
            candidates, tvs_module, args.keep_warhead
        )

    hits_by_key, hit_key_level = vs_module.load_hits(
        labels_path, tvs_module, helpers
    )
    selected, unmatched = vs_module.label_candidates(
        candidates,
        library,
        hits_by_key,
        tvs_module,
        hit_key_level=hit_key_level,
    )
    if args.collapse_active_protomers:
        selected["_parent_key"] = selected["electrophile_smiles"].map(
            lambda smiles: vs_module.parent_compound_key(smiles, tvs_module)
        )
    selected = add_model_features(selected, feature_cols)
    data = tvs_module.build_site_candidates(selected, feature_cols)
    if "_parent_key" in selected.columns:
        data = vs_module.attach_parent_keys(data, selected)
    data["PDB_ID"] = data["Site"].astype(str).str.split("|").str[0]
    data["relevance"] = data["is_hit"].astype(int)
    data["query_group"] = data["Site"]
    n_hits = int(data["relevance"].sum())
    print(
        f"[INFO] Labeled {len(data):,} candidate row(s) at "
        f"{data['Site'].nunique():,} site(s); {n_hits:,} hit row(s)"
    )
    if n_hits == 0:
        sys.exit(
            "[ERROR] No candidate row matched a label. Check SMILES, "
            "PDB_ID, Residue, ResNum, and Chain."
        )
    return data.reset_index(drop=True), unmatched


def print_classifier_summary(metrics: pd.DataFrame) -> None:
    pooled = metrics.loc[metrics["analysis_scope"].eq("pooled")]
    if pooled.empty:
        return
    row = pooled.iloc[0]
    print(
        "\nClassification (pooled):\n"
        f"  threshold={row['classification_threshold']:.6g}  "
        f"AUROC={row['AUROC']:.4f}  AUPRC={row['AUPRC']:.4f}  "
        f"ACC={row['ACC']:.4f}\n"
        f"  precision={row['precision']:.4f}  recall={row['recall']:.4f}  "
        f"F1={row['F1']:.4f}  MCC={row['MCC']:.4f}\n"
        f"  TP={int(row['TP'])}  FP={int(row['FP'])}  "
        f"FN={int(row['FN'])}  TN={int(row['TN'])}"
    )


def print_ranking_summary(ranking: pd.DataFrame) -> None:
    summary = ranking.loc[ranking["analysis_scope"].eq("site_summary")]
    if summary.empty:
        return
    row = summary.iloc[0]
    print(
        "\nRanking (mean across sites; median rank):\n"
        f"  EF@1%={row['EF_1pct']:.2f}  EF@5%={row['EF_5pct']:.2f}  "
        f"EF@10%={row['EF_10pct']:.2f}\n"
        f"  logAUC={row['logAUC']:.4f}  "
        f"adjusted_logAUC={row['adjusted_logAUC']:.4f}  "
        f"BEDROC={row['BEDROC']:.4f}\n"
        f"  median hit rank={row['median_hit_rank']:.3g}  "
        f"Hit@top{row['top_pct']:g}%={row['hit_at_top_pct']:.3f}  "
        f"Hit@{int(row['topk'])}={row['hit_at_topk']:.3f}"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model(Path(args.model).resolve())
    feature_cols = bundle["features"]
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(bundle.get("classification_threshold", 0.5))
    )
    threshold_source = (
        "cli" if args.threshold is not None
        else str(bundle.get("classification_threshold_source", "model_pkl"))
    )
    print(
        f"[INFO] Model: {Path(args.model).resolve()}\n"
        f"[INFO] Features: {len(feature_cols)}\n"
        f"[INFO] Cutoff: {threshold:.6g} ({threshold_source})"
    )

    vs_module = load_python_module("training_covsite_vs_helpers", DEFAULT_COVSITE_VS)
    topsis_path = (
        Path(args.topsis_vs).resolve()
        if args.topsis_vs
        else DEFAULT_TOPSIS_VS
    )
    tvs_module = vs_module.load_topsis_vs(topsis_path)
    candidates_path = resolve_candidates(args, vs_module)
    scored, unmatched = prepare_scored_frame(
        args, candidates_path, feature_cols, vs_module, tvs_module
    )

    scored["pred_prob"] = predict_proba_positive(
        bundle["model"], scored[feature_cols].to_numpy()
    )
    scored["pred_score"] = scored["pred_prob"]
    scored["pred_class"] = (scored["pred_prob"] >= threshold).astype(int)

    per_site, _pooled, enrichment = covsite_ligand_rankings(
        scored, args, vs_module, tvs_module
    )
    per_site["relevance"] = per_site["is_hit"].astype(int)
    per_site["query_group"] = per_site["Site"]
    per_site["pred_class"] = (per_site["pred_prob"] >= threshold).astype(int)

    classifier_metrics = covsite_classifier_metric_rows(
        per_site,
        threshold,
        data_split="inference",
        threshold_source=threshold_source,
    )
    ranking_metrics = residue_ranking_metric_rows(
        per_site,
        data_split="inference",
        topk=args.topk,
        top_pct=args.top_pct,
        bedroc_alpha=args.bedroc_alpha,
    )
    ranking_metrics = ranking_metrics.rename(columns={"query_group": "Site"})
    ranking_metrics["analysis_scope"] = ranking_metrics["analysis_scope"].replace(
        {
            "per_query_group": "per_site",
            "query_group_summary": "site_summary",
        }
    )

    classified = per_site.loc[per_site["pred_class"] == 1].sort_values(
        ["Site", "pred_prob"], ascending=[True, False], kind="mergesort"
    )
    export_columns = [
        column
        for column in [
            "Site", "PDB_ID", "Residue", "ResNum", "Chain", "Name", "Warhead",
            "electrophile_smiles", "pred_prob", "pred_class", "is_hit",
            "site_rank", "relevance",
        ]
        if column in per_site.columns
    ]
    per_site[export_columns].to_csv(output_dir / "scored_candidates.csv", index=False)
    classified[export_columns].to_csv(
        output_dir / "classified_hits.csv", index=False
    )
    classifier_metrics.to_csv(output_dir / "classifier_metrics.csv", index=False)
    ranking_metrics.to_csv(output_dir / "ranking_metrics.csv", index=False)
    enrichment.to_csv(output_dir / "enrichment_factors.csv", index=False)
    if unmatched:
        pd.DataFrame(unmatched).to_csv(output_dir / "unmatched_hits.csv", index=False)
        print(f"[WARN] Unmatched labels: {len(unmatched):,} (unmatched_hits.csv)")

    summary = {
        "model": str(Path(args.model).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "candidates": str(candidates_path),
        "n_rows": int(len(per_site)),
        "n_sites": int(per_site["Site"].nunique()),
        "n_hits": int(per_site["relevance"].sum()),
        "n_predicted_positive": int(per_site["pred_class"].sum()),
        "classification_threshold": threshold,
        "classification_threshold_source": threshold_source,
        "keep_warhead": args.keep_warhead,
        "residue": args.residue,
        "classifier_metrics": classifier_metrics.to_dict(orient="records"),
        "ranking_summary": ranking_metrics.loc[
            ranking_metrics["analysis_scope"].eq("site_summary")
        ].to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print_classifier_summary(classifier_metrics)
    print_ranking_summary(ranking_metrics)
    print(f"\n[INFO] Wrote {output_dir}")
    print(f"  Classified hits: {output_dir / 'classified_hits.csv'}")
    print(f"  Class metrics:   {output_dir / 'classifier_metrics.csv'}")
    print(f"  Rank metrics:    {output_dir / 'ranking_metrics.csv'}")


if __name__ == "__main__":
    main()
