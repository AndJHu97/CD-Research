#!/usr/bin/env python3
"""Score CovSite candidates with a Training_Cov_Classifier model.

Two modes, mirroring the two task modes in Training_Cov_Classifier.

Default (residue mode) — "which residue does this ligand hit?"
Give a CovSite-style input sheet. CovSite enumerates every nucleophilic
residue (CYS/SER/THR/TYR/LYS/HIS) in the whole protein; the Residue/ResNum/
Chain on each input row is the only positive and every other nucleophile is a
negative. Query groups are ligand base Name × Warhead, so the ranking answers
which site that ligand is predicted to modify.

    protein pdb            electrophile_smiles  Residue  ResNum  Chain
    3L00_CYS_MODIFIED      Nc3nc(OCc1ccccc1)... Cys      145     A

Ligand mode (--VS) — "which ligand hits this site?"
Give a hits CSV with --hits (all_hits or labels_covdb style). Each positive is
an electrophile at one target site: PDB_ID × Residue × ResNum × Chain, and
other candidate ligands at that site are negatives. Query groups are sites.

    pdb_id  Electrophile_Hit         Warhead_Type         Residue ResNum Chain
    6O8I    C=CC(=O)N1CCC2(CC1)...   Acrylamide warhead   Cys     481    A

Both modes report the same metric families: AUROC, AUPRC, accuracy, balanced
accuracy, precision, recall, F1, and MCC at the cutoff stored in the pickle,
plus ranking metrics (EF@1/5/10%, LogAUC, adjusted LogAUC, BEDROC, median
search-space reduction, Hit@K, and Hit@top-%). Residue mode adds the CovSite
rank report (Hit@K, NDCG@K, rank percentiles, residue composition).

Pass --candidates to reuse a CovSite feature table instead of featurizing.
Residue mode also needs the matching --labels; --VS needs --hits either way.

Examples:
    # Residue mode: featurize the whole protein, then score
    python Cov_Classifier.py VS/snaptag_inputs.csv \\
        --model covsite_classifier_output/lgbm_classifier.pkl \\
        --pdb-dir Existing_Structures \\
        --output-dir cov_classifier_sites

    # Ligand mode: today's behavior
    python Cov_Classifier.py --VS \\
        --model covsite_classifier_output/lgbm_classifier.pkl \\
        --hits all_hits.csv \\
        --candidates covsite_candidates_combined.csv \\
        --keep-warhead "Acrylamide warhead" \\
        --output-dir cov_classifier_eval
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
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
    filter_to_residues,
    load_python_module,
    per_residue_type_clf_metrics,
    predict_proba_positive,
    residue_classifier_metric_rows,
    residue_ranking_metric_rows,
    secondary_rank_diagnostics,
)
from Training_Cov_Screen import (
    DEFAULT_RANK_BONUS_EPSILON,
    REWARD_MODES,
    load_and_merge,
    print_residue_composition,
    print_results,
)


RESIDUE_ONEHOT = ("CYS", "SER", "THR", "TYR", "LYS", "HIS")
HERE = Path(__file__).resolve().parent
# Written by CovSite.write_final_feature_cache (see FINAL_*_GLOB in CovSite.py).
COVSITE_OUTPUTS = {
    "candidates": ("covsite_candidates.csv", "covsite_candidates_*.csv"),
    "labels": ("covsite_labels.csv", "covsite_labels_*.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a Training_Cov_Classifier pickle to CovSite candidates "
            "and report classification plus ranking metrics. Default mode "
            "ranks nucleophilic residues for each ligand; --VS ranks ligands "
            "at each fixed site."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help=(
            "CovSite-style input sheet for residue mode: pdb, smiles, and "
            "the target Residue/ResNum/Chain. Omit when passing "
            "--candidates/--labels, and never use it with --VS."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="lgbm_classifier.pkl written by Training_Cov_Classifier",
    )
    parser.add_argument(
        "--VS",
        dest="vs",
        action="store_true",
        help=(
            "Ligand-screening mode: rank ligands within each fixed CovSite "
            "site using a --hits CSV. This was the original behavior of this "
            "script and is now opt-in."
        ),
    )
    parser.add_argument(
        "--hits",
        default=None,
        help=(
            "--VS only. Hits CSV; positives are electrophiles at PDB_ID × "
            "Residue × ResNum × Chain. Accepts all_hits (electrophile_hit) "
            "or labels_covdb (electrophile_smiles, Name)."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Residue mode only. CovSite labels CSV (covsite_labels_*.csv) "
            "naming the target site per ligand. Required with --candidates; "
            "generated automatically when featurizing an input sheet."
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
            "--VS only. Ligand library for CovSite --VS when --candidates is "
            "omitted. Needs pdb, smiles, residue, resnum, and chain. "
            "Defaults to --hits, which scores only the labeled complexes."
        ),
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help=(
            "PDB directory for CovSite featurization "
            "(required without --candidates)"
        ),
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
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="hit_at_k",
        help="Residue mode: headline reward for the CovSite rank report",
    )
    parser.add_argument(
        "--rank-bonus-epsilon",
        type=float,
        default=DEFAULT_RANK_BONUS_EPSILON,
        help="Residue mode: rank bonus weight in the CovSite rank report",
    )
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
    vs_only = (
        ("--hits", args.hits),
        ("--library", args.library),
        ("--pdb-hits", args.pdb_hits),
        ("--best-warhead", args.best_warhead),
        ("--multiple-hits", args.multiple_hits),
        ("--collapse-active-protomers", args.collapse_active_protomers),
    )
    if args.vs:
        if not args.hits:
            parser.error("--VS requires --hits")
        if args.input_csv:
            parser.error(
                "--VS does not take a positional input CSV; pass the ligand "
                "library with --library"
            )
        if args.labels:
            parser.error(
                "--labels is the residue-mode CovSite labels CSV; "
                "--VS takes its positives from --hits"
            )
    else:
        used = [flag for flag, value in vs_only if value]
        if used:
            parser.error(
                f"{', '.join(used)} require --VS; residue mode takes its "
                "positive site from the input sheet or --labels"
            )
        if args.candidates:
            if not args.labels:
                parser.error(
                    "--candidates requires --labels in residue mode "
                    "(the covsite_labels_*.csv from the same CovSite run)"
                )
            if args.input_csv:
                parser.error(
                    "Pass either an input CSV or --candidates/--labels, "
                    "not both"
                )
        elif not args.input_csv:
            parser.error(
                "Residue mode requires an input CSV to featurize, or "
                "--candidates with --labels"
            )
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

    library = Path(args.library).resolve() if args.library else Path(args.hits).resolve()
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


def locate_covsite_script() -> Path:
    """CovSite.py sits at the CustomScript root, above this analysis folder."""
    for root in (HERE, HERE.parent, HERE.parent.parent):
        candidate = root / "CovSite.py"
        if candidate.is_file():
            return candidate.resolve()
    sys.exit(
        "[ERROR] Could not find CovSite.py in "
        + ", ".join(str(root) for root in (HERE, HERE.parent, HERE.parent.parent))
    )


def find_covsite_output(output_dir: Path, kind: str) -> Path:
    """Newest CovSite candidates/labels CSV, preferring the stable alias."""
    alias_name, glob_pattern = COVSITE_OUTPUTS[kind]
    alias = output_dir / alias_name
    if alias.is_file():
        return alias
    matches = sorted(output_dir.glob(glob_pattern))
    if not matches:
        sys.exit(f"[ERROR] No CovSite {kind} CSV found in {output_dir}")
    return matches[-1]


def run_covsite_features_only(
    input_csv: Path,
    pdb_dir: Path,
    feat_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Featurize every nucleophile in each protein, skipping Cov_Screen.

    Residue mode needs the full nucleophile pool as the candidate list, so
    this deliberately omits --VS (which narrows to the row's own site) and
    --skip-sasa-deprot (whose NaN columns the classifier would consume).
    """
    feat_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(locate_covsite_script()),
        str(input_csv),
        "--features-only",
        "--pdb-dir",
        str(pdb_dir),
        "--output-dir",
        str(feat_dir),
    ]
    if args.ions:
        cmd.append("--ions")
    if args.no_warhead:
        cmd.append("--no-warhead")
    if args.no_cache:
        cmd.append("--no-cache")
    if args.scripts_dir:
        cmd.extend(["--scripts-dir", str(args.scripts_dir)])
    for flag, value in (
        ("--pdb-column", args.pdb_column),
        ("--smiles-column", args.smiles_column),
        ("--name-column", args.name_column),
        ("--residue-column", args.residue_column),
        ("--resnum-column", args.resnum_column),
        ("--chain-column", args.chain_column),
    ):
        if value:
            cmd.extend([flag, str(value)])
    print("[INFO] Running CovSite --features-only:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(
            f"[ERROR] CovSite --features-only failed with exit code "
            f"{result.returncode}"
        )
    return (
        find_covsite_output(feat_dir, "candidates"),
        find_covsite_output(feat_dir, "labels"),
    )


def validate_residue_labels(labels_path: Path) -> None:
    """Reject a --VS hits CSV handed to residue mode's --labels.

    The two modes take different label files, so catch the mix-up here rather
    than letting load_and_merge fail on a missing Name column.
    """
    columns = pd.read_csv(labels_path, nrows=0).columns
    lowered = {str(column).strip().lower() for column in columns}
    required = {"name", "residue", "resnum", "chain"}
    missing = sorted(required - lowered)
    if not missing:
        return
    hit_markers = {"electrophile_hit", "warhead_type", "name_hit"}
    if lowered & hit_markers:
        sys.exit(
            f"[ERROR] {labels_path.name} looks like a --VS hits CSV "
            f"(found {', '.join(sorted(lowered & hit_markers))}), not a "
            "CovSite labels CSV. Use --VS --hits for ligand ranking, or pass "
            "a covsite_labels_*.csv for residue ranking."
        )
    sys.exit(
        f"[ERROR] --labels {labels_path.name} is missing required column(s): "
        f"{', '.join(missing)}. Residue mode expects a covsite_labels_*.csv."
    )


def resolve_residue_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Candidates + labels for residue mode, featurizing when needed."""
    if args.candidates:
        candidates_path = Path(args.candidates).resolve()
        labels_path = Path(args.labels).resolve()
        for label, path in (
            ("--candidates", candidates_path),
            ("--labels", labels_path),
        ):
            if not path.is_file():
                sys.exit(f"[ERROR] {label} not found: {path}")
        validate_residue_labels(labels_path)
        print(f"[INFO] Reusing candidates: {candidates_path}")
        print(f"[INFO] Reusing labels:     {labels_path}")
        return candidates_path, labels_path

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.is_file():
        sys.exit(f"[ERROR] Input CSV not found: {input_csv}")
    feat_dir = Path(args.output_dir).resolve() / "covsite_features"
    return run_covsite_features_only(
        input_csv,
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
    group_col: str | None = None,
) -> pd.DataFrame:
    """Build one-hot and within-group normalized columns the pickle expects.

    *_norm features are min-max scaled inside group_col, matching how
    Training_Cov_Screen.build_features derived them: within Site for ligand
    mode, within query_group (ligand × warhead) for residue mode.
    """
    work = frame.copy()
    residue = work["Residue"].astype(str).str.strip().str.upper()
    resnum = work["ResNum"].astype(str).str.strip()
    # label_candidates names the site column _site_key; build_site_candidates
    # renames it to Site only later.
    if group_col is None:
        group_col = "Site" if "Site" in work.columns else "_site_key"
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
            if group_col not in work.columns:
                sys.exit(
                    f"[ERROR] Model feature {feature!r} needs column "
                    f"{group_col!r} to normalize within, which is not in "
                    "the candidates."
                )
            values = pd.to_numeric(work[base], errors="coerce")
            work[feature] = values.groupby(work[group_col]).transform(
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
    labels_path = Path(args.hits).resolve()
    if not labels_path.is_file():
        sys.exit(f"[ERROR] --hits not found: {labels_path}")

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


def print_ranking_summary(
    ranking: pd.DataFrame,
    scope: str = "site_summary",
    unit: str = "sites",
) -> None:
    summary = ranking.loc[ranking["analysis_scope"].eq(scope)]
    if summary.empty:
        return
    row = summary.iloc[0]
    print(
        f"\nRanking (mean across {unit}; median rank):\n"
        f"  EF@1%={row['EF_1pct']:.2f}  EF@5%={row['EF_5pct']:.2f}  "
        f"EF@10%={row['EF_10pct']:.2f}\n"
        f"  logAUC={row['logAUC']:.4f}  "
        f"adjusted_logAUC={row['adjusted_logAUC']:.4f}  "
        f"BEDROC={row['BEDROC']:.4f}\n"
        f"  median hit rank={row['median_hit_rank']:.3g}  "
        f"Hit@top{row['top_pct']:g}%={row['hit_at_top_pct']:.3f}  "
        f"Hit@{int(row['topk'])}={row['hit_at_topk']:.3f}\n"
        f"  median search-space reduction="
        f"{row['median_search_space_reduction']:.4f}"
    )


def load_vs_helpers(args: argparse.Namespace):
    """Load the CovSite VS helper module and its TOPSIS metric dependency."""
    vs_module = load_python_module("training_covsite_vs_helpers", DEFAULT_COVSITE_VS)
    topsis_path = (
        Path(args.topsis_vs).resolve()
        if args.topsis_vs
        else DEFAULT_TOPSIS_VS
    )
    return vs_module, vs_module.load_topsis_vs(topsis_path)


def main_residue(
    args: argparse.Namespace,
    bundle: dict,
    threshold: float,
    threshold_source: str,
    output_dir: Path,
) -> None:
    """Rank nucleophilic residues within each ligand × warhead query group.

    The labeled target site is the positive; every other nucleophile CovSite
    enumerated in that protein is a negative.
    """
    feature_cols = bundle["features"]
    candidates_path, labels_path = resolve_residue_inputs(args)

    merged = load_and_merge(str(candidates_path), str(labels_path))
    # DataFrame.attrs does not survive every filtering step below.
    unmatched = list(merged.attrs.get("unmatched_warhead_labels", []))
    if args.keep_warhead:
        vs_module, tvs_module = load_vs_helpers(args)
        merged = filter_candidates_to_warheads_frame(
            merged, vs_module, tvs_module, args.keep_warhead
        )
    if args.residue:
        merged = filter_to_residues(merged, args.residue)

    merged = add_model_features(merged, feature_cols, group_col="query_group")
    merged["pred_prob"] = predict_proba_positive(
        bundle["model"], merged[feature_cols].to_numpy()
    )
    merged["pred_score"] = merged["pred_prob"]
    merged["pred_class"] = (merged["pred_prob"] >= threshold).astype(int)
    merged["group_rank"] = (
        merged.groupby("query_group")["pred_prob"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    merged["group_size"] = merged.groupby("query_group")["pred_prob"].transform(
        "size"
    ).astype(int)

    n_hits = int(merged["relevance"].sum())
    print(
        f"[INFO] Scored {len(merged):,} nucleophile row(s) across "
        f"{merged['query_group'].nunique():,} ligand group(s); "
        f"{n_hits:,} labeled target site(s)"
    )

    classifier_metrics = residue_classifier_metric_rows(
        merged,
        threshold,
        data_split="inference",
        threshold_source=threshold_source,
    )
    ranking_metrics = residue_ranking_metric_rows(
        merged,
        data_split="inference",
        topk=args.topk,
        top_pct=args.top_pct,
        bedroc_alpha=args.bedroc_alpha,
    )
    per_res_clf = per_residue_type_clf_metrics(merged, "pred_prob")
    rank_eval, _rank_summary, composition = secondary_rank_diagnostics(
        merged,
        "pred_prob",
        k=args.topk,
        top_pct=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
    )

    export_columns = [
        column
        for column in [
            "Name", "PDB_ID", "Residue", "ResNum", "Chain", "Warhead",
            "query_group", "electrophile_smiles", "target_residue_type",
            "pred_prob", "pred_class", "relevance", "group_rank", "group_size",
        ]
        if column in merged.columns
    ]
    scored_out = merged.sort_values(
        ["query_group", "group_rank"], kind="mergesort"
    )
    scored_out[export_columns].to_csv(
        output_dir / "scored_candidates.csv", index=False
    )
    scored_out.loc[scored_out["pred_class"] == 1, export_columns].to_csv(
        output_dir / "classified_hits.csv", index=False
    )
    classifier_metrics.to_csv(output_dir / "classifier_metrics.csv", index=False)
    ranking_metrics.to_csv(output_dir / "ranking_metrics.csv", index=False)
    rank_eval.to_csv(output_dir / "rank_report.csv", index=False)
    if not per_res_clf.empty:
        per_res_clf.to_csv(
            output_dir / "residue_type_metrics.csv", index=False
        )
    if unmatched:
        pd.DataFrame(unmatched).to_csv(
            output_dir / "unmatched_labels.csv", index=False
        )
        print(
            f"[WARN] Unmatched warhead labels: {len(unmatched):,} "
            "(unmatched_labels.csv)"
        )

    summary = {
        "mode": "residue",
        "model": str(Path(args.model).resolve()),
        "candidates": str(candidates_path),
        "labels": str(labels_path),
        "n_rows": int(len(merged)),
        "n_query_groups": int(merged["query_group"].nunique()),
        "n_hits": n_hits,
        "n_predicted_positive": int(merged["pred_class"].sum()),
        "classification_threshold": threshold,
        "classification_threshold_source": threshold_source,
        "keep_warhead": args.keep_warhead,
        "residue": args.residue,
        "classifier_metrics": classifier_metrics.to_dict(orient="records"),
        "ranking_summary": ranking_metrics.loc[
            ranking_metrics["analysis_scope"].eq("query_group_summary")
        ].to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print_results(
        rank_eval,
        k=args.topk,
        top_pct_threshold=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
        split_name="Inference",
    )
    print_residue_composition(
        composition, k=args.topk, top_pct_threshold=args.top_pct
    )
    if not per_res_clf.empty:
        print("\nPer target-residue-type:")
        print(per_res_clf.to_string(index=False, float_format="%.4f"))
    print_classifier_summary(classifier_metrics)
    print_ranking_summary(
        ranking_metrics, scope="query_group_summary", unit="ligand groups"
    )
    print(f"\n[INFO] Wrote {output_dir}")
    print(f"  Scored sites:    {output_dir / 'scored_candidates.csv'}")
    print(f"  Class metrics:   {output_dir / 'classifier_metrics.csv'}")
    print(f"  Rank metrics:    {output_dir / 'ranking_metrics.csv'}")
    print(f"  Rank report:     {output_dir / 'rank_report.csv'}")


def filter_candidates_to_warheads_frame(
    frame: pd.DataFrame,
    vs_module,
    tvs_module,
    warheads: list[str],
) -> pd.DataFrame:
    """--keep-warhead on an already merged frame, then drop emptied groups."""
    kept = vs_module.filter_candidates_to_warheads(frame, tvs_module, warheads)
    if "relevance" in kept.columns:
        positive_groups = set(
            kept.loc[kept["relevance"] == 1, "query_group"].astype(str)
        )
        dropped = kept.loc[
            ~kept["query_group"].astype(str).isin(positive_groups)
        ]
        if not dropped.empty:
            print(
                f"[INFO] --keep-warhead left {dropped['query_group'].nunique():,} "
                "query group(s) with no labeled target; dropping them"
            )
            kept = kept.loc[
                kept["query_group"].astype(str).isin(positive_groups)
            ].copy()
        if kept.empty:
            sys.exit("[ERROR] No labeled rows remain after --keep-warhead")
    return kept


def main_vs(
    args: argparse.Namespace,
    bundle: dict,
    threshold: float,
    threshold_source: str,
    output_dir: Path,
) -> None:
    """Rank ligands within each fixed CovSite site (original behavior)."""
    feature_cols = bundle["features"]
    vs_module, tvs_module = load_vs_helpers(args)
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
        "mode": "vs",
        "model": str(Path(args.model).resolve()),
        "hits": str(Path(args.hits).resolve()),
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model(Path(args.model).resolve())
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(bundle.get("classification_threshold", 0.5))
    )
    threshold_source = (
        "cli" if args.threshold is not None
        else str(bundle.get("classification_threshold_source", "model_pkl"))
    )
    mode = "ligand ranking per site (--VS)" if args.vs else "residue ranking per ligand"
    print(
        f"[INFO] Mode: {mode}\n"
        f"[INFO] Model: {Path(args.model).resolve()}\n"
        f"[INFO] Features: {len(bundle['features'])}\n"
        f"[INFO] Cutoff: {threshold:.6g} ({threshold_source})"
    )
    task_mode = str(bundle.get("task_mode", "")).strip()
    expected = "covsite_ligand_classifier" if args.vs else "residue_site_classifier"
    if task_mode and task_mode != expected:
        print(
            f"[WARN] Model was trained as {task_mode!r} but this run is "
            f"{expected!r}. Metrics will compute, but the query groups differ "
            "from training."
        )

    runner = main_vs if args.vs else main_residue
    runner(args, bundle, threshold, threshold_source, output_dir)


if __name__ == "__main__":
    main()
