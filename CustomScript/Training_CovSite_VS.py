#!/usr/bin/env python3
"""LightGBM ligand ranker for a fixed covalent site.

Calls CovSite --VS for features (or reuses --candidates). Hits are matched by
RDKit canonical SMILES plus pdb_id, and by Residue/ResNum/Chain when those
columns are present so the same PDB with two sites stays two label groups.
Ranking groups are PDB × Residue × ResNum × Chain. Default cross-validation
is leave-one-site-out. --leave-one-cluster-out clusters PDBs with MMseqs2 and
holds out every site from proteins in one sequence cluster at a time.
Metrics (EF, LogAUC, univariate feature stats) are imported from TOPSIS_VS.py;
there is no TOPSIS solver. --keep-warhead filters every ligand (hits and decoys).
--collapse-active-protomers matches COValid SI eval: one best hit protomer
per parent compound × site, all decoy protomers kept.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import CovSite as covsite


HERE = Path(__file__).resolve().parent
DEFAULT_TOPSIS_VS = (
    HERE / "bo-runs" / "scoring-analysis" / "VS_eval" / "TOPSIS_VS.py"
)
# LightGBM Lambdarank rejects query groups larger than this.
LGBM_MAX_QUERY_ROWS = 10000


def load_topsis_vs(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TOPSIS_VS.py not found: {path}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("TOPSIS_VS", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LightGBM ligand-screening ranker. Default CV is "
            "leave-one-site-out. --leave-one-cluster-out holds out MMseqs2 "
            "sequence clusters instead. Features come from CovSite --VS "
            "(or --candidates). Hits are matched by canonical SMILES and "
            "pdb_id, and by residue location when Residue/ResNum/Chain are "
            "in the hits CSV."
        )
    )
    parser.add_argument(
        "library_csv",
        help="Library CSV: pdb/pdb_id, smiles, residue, resnum, chain",
    )
    parser.add_argument(
        "hits_csv",
        help=(
            "Hits CSV: pdb_id plus Electrophile_Hit / electrophile_hit. "
            "Include residue, resnum, and chain to label hits per site "
            "(same PDB, different residues stay separate). Optional "
            "Name_Hit, warhead_type"
        ),
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Local PDB directory (required unless --candidates is set)",
    )
    parser.add_argument(
        "--output-dir",
        default="covsite_vs_output",
        help="Output directory (default: covsite_vs_output)",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Existing CovSite candidates CSV; skip CovSite featurization",
    )
    parser.add_argument(
        "--covsite-output-dir",
        default=None,
        help=(
            "Directory for CovSite --VS outputs (default: "
            "<output-dir>/covsite_features)"
        ),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        metavar="FEATURE",
        help="CovSite candidate column names used by the ranker",
    )
    parser.add_argument(
        "--pdb-column", default=None, help="Library PDB column override"
    )
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--residue-column", default=None)
    parser.add_argument("--resnum-column", default=None)
    parser.add_argument("--chain-column", default=None)
    parser.add_argument(
        "--hit-residue-column",
        default=None,
        help="Hits residue column override (default: auto-detect residue/resname/aa)",
    )
    parser.add_argument(
        "--hit-resnum-column",
        default=None,
        help="Hits residue-number column override",
    )
    parser.add_argument(
        "--hit-chain-column",
        default=None,
        help="Hits chain column override",
    )
    parser.add_argument(
        "--pdb-hits",
        action="store_true",
        help=(
            "Ignore residue/resnum/chain on hits and label a ligand as a hit "
            "at every site of that pdb_id (legacy matching)"
        ),
    )
    parser.add_argument(
        "--smiles-hit-column",
        default="electrophile_hit",
        help="Positive-SMILES column in hits (default: electrophile_hit)",
    )
    parser.add_argument(
        "--name-hit-column",
        default="Name_Hit",
        help="Optional positive-name column in hits (default: Name_Hit)",
    )
    parser.add_argument(
        "--name-miss-column",
        default="Name_Miss",
    )
    parser.add_argument(
        "--smiles-miss-column",
        default="electrophile_miss",
    )
    parser.add_argument(
        "--warhead-type-column",
        default="warhead_type",
    )
    parser.add_argument(
        "--keep-warhead",
        nargs="+",
        default=None,
        metavar="WARHEAD",
        help=(
            "Keep only these Warhead_Base values for every ligand (hits and "
            "decoys), e.g. 'Acrylamide warhead'. Applied before labeling and "
            "training. Distinct from hits-CSV warhead_type, which only "
            "restricts listed actives"
        ),
    )
    parser.add_argument(
        "--best-warhead",
        action="store_true",
        help=(
            "Keep the highest pred_score warhead per electrophile_smiles × "
            "site before ranking exports and EF/LogAUC"
        ),
    )
    parser.add_argument(
        "--collapse-active-protomers",
        action="store_true",
        help=(
            "COValid SI eval: among hit rows that share a parent compound "
            "(neutralized, stereo-stripped SMILES) at the same site, keep "
            "the highest pred_score protomer; keep every decoy protomer. "
            "Training still uses all protomer rows. Apply after "
            "--best-warhead if both are set"
        ),
    )
    parser.add_argument(
        "--multiple-hits",
        action="store_true",
        help=(
            "Count each Name × Warhead × SMILES variant as its own hit for "
            "EF and LogAUC (default: one hit per ligand Name)"
        ),
    )
    parser.add_argument(
        "--sasa-deprot",
        action="store_true",
        help=(
            "Compute FreeSASA and deprotonation in CovSite --VS (default: "
            "pass --skip-sasa-deprot)"
        ),
    )
    parser.add_argument("--ions", action="store_true")
    parser.add_argument("--no-warhead", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--scripts-dir", default=None)
    parser.add_argument(
        "--topsis-vs",
        default=None,
        help=(
            "Path to TOPSIS_VS.py (hit matching, EF/LogAUC helpers). "
            "Default: bo-runs/scoring-analysis/VS_eval/TOPSIS_VS.py "
            "beside this script"
        ),
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        default=True,
        help="Write models/fold_*.pkl (default)",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not write fold model pickles",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-query-size",
        type=int,
        default=LGBM_MAX_QUERY_ROWS,
        help=(
            "Max rows per LightGBM ranking query (hard cap "
            f"{LGBM_MAX_QUERY_ROWS}). Larger sites are split for training "
            "only; held-out scoring still uses every candidate"
        ),
    )
    parser.add_argument(
        "--leave-one-cluster-out",
        action="store_true",
        help=(
            "Hold out one MMseqs2 sequence cluster per fold. Every site "
            "from every PDB in that cluster is test; ranking queries stay "
            "per site. Default clustering is per chain, then PDBs that "
            "share a chain cluster are merged."
        ),
    )
    parser.add_argument(
        "--no-deep-cluster",
        action="store_true",
        help=(
            "With --leave-one-cluster-out, cluster one concatenated "
            "sequence per PDB instead of merging PDBs that share a "
            "homologous chain"
        ),
    )
    parser.add_argument(
        "--seq-identity",
        type=float,
        default=0.5,
        help="MMseqs2 minimum sequence identity (default: 0.5)",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="MMseqs2 alignment coverage, cov-mode 0 (default: 0.8)",
    )
    parser.add_argument(
        "--min-chain-length",
        type=int,
        default=30,
        help=(
            "Minimum chain length included in chain-level clustering "
            "(default: 30)"
        ),
    )
    parser.add_argument(
        "--clusters-tsv",
        default=None,
        help=(
            "Reuse an MMseqs2 createtsv file (representative, member) "
            "instead of running MMseqs2. Chain-level TSVs use PDB__CHAIN "
            "headers and are merged to PDBs. A TSV without those headers "
            "is treated as PDB-level."
        ),
    )
    parser.add_argument(
        "--mmseqs-dir",
        default=None,
        help=(
            "Directory for MMseqs2 files (default: <output-dir>/mmseqs)"
        ),
    )
    args = parser.parse_args()
    if args.no_save_models:
        args.save_models = False
    if args.candidates is None and not args.pdb_dir:
        parser.error("--pdb-dir is required unless --candidates is set")
    if args.max_query_size < 2 or args.max_query_size > LGBM_MAX_QUERY_ROWS:
        parser.error(
            "--max-query-size must be between 2 and "
            f"{LGBM_MAX_QUERY_ROWS} (LightGBM Lambdarank limit)"
        )
    if args.leave_one_cluster_out:
        if args.clusters_tsv is None and not args.pdb_dir:
            parser.error(
                "--leave-one-cluster-out needs --pdb-dir or --clusters-tsv"
            )
        if not 0 < args.seq_identity <= 1:
            parser.error("--seq-identity must be in (0, 1]")
        if not 0 < args.coverage <= 1:
            parser.error("--coverage must be in (0, 1]")
        if args.min_chain_length < 1:
            parser.error("--min-chain-length must be at least 1")
    elif args.no_deep_cluster or args.clusters_tsv or args.mmseqs_dir:
        parser.error(
            "--no-deep-cluster, --clusters-tsv, and --mmseqs-dir require "
            "--leave-one-cluster-out"
        )
    return args


def lgbm_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "boosting_type": "gbdt",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": args.random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def find_candidates_csv(output_dir: Path) -> Path:
    alias = output_dir / "covsite_candidates.csv"
    if alias.is_file():
        return alias
    matches = sorted(output_dir.glob(covsite.FINAL_CANDIDATES_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No CovSite candidates CSV found in {output_dir}"
        )
    return matches[-1]


def run_covsite_vs(
    library_path: Path,
    pdb_dir: Path,
    feat_dir: Path,
    args: argparse.Namespace,
) -> Path:
    feat_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HERE / "CovSite.py"),
        str(library_path),
        "--VS",
        "--pdb-dir",
        str(pdb_dir),
        "--output-dir",
        str(feat_dir),
    ]
    if not args.sasa_deprot:
        cmd.append("--skip-sasa-deprot")
    if args.ions:
        cmd.append("--ions")
    if args.no_warhead:
        cmd.append("--no-warhead")
    if args.no_cache:
        cmd.append("--no-cache")
    if args.scripts_dir:
        cmd.extend(["--scripts-dir", str(args.scripts_dir)])
    print("[Training_CovSite_VS] Running CovSite:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"[ERROR] CovSite --VS failed with exit code {result.returncode}"
        )
    return find_candidates_csv(feat_dir)


def load_library(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False).reset_index(drop=True)
    if raw.empty:
        raise SystemExit("[ERROR] Library CSV has no rows")
    pdb_col = covsite.detect_column(
        raw, args.pdb_column, ("pdb", "pdb_id", "protein pdb", "protein_pdb"), True
    )
    smiles_col = covsite.detect_column(
        raw,
        args.smiles_column,
        ("smiles", "electrophile_smiles", "electrophile smiles"),
        True,
    )
    residue_col = covsite.detect_column(
        raw, args.residue_column, ("residue", "resname", "aa"), True
    )
    resnum_col = covsite.detect_column(
        raw,
        args.resnum_column,
        ("resnum", "res_num", "residue_number", "residue_num"),
        True,
    )
    chain_col = covsite.detect_column(
        raw, args.chain_column, ("chain",), True
    )
    name_col = covsite.detect_column(raw, args.name_column, ("name",), False)
    work = pd.DataFrame(
        {
            "PDB_ID": raw[pdb_col].map(covsite.pdb_id_from_value),
            "electrophile_smiles": raw[smiles_col].map(covsite.clean_text),
            "Residue": raw[residue_col].map(covsite.clean_text).str.upper(),
            "ResNum": raw[resnum_col].map(covsite.normalized_resnum),
            "Chain": raw[chain_col].map(covsite.clean_text),
            "library_name": (
                raw[name_col].map(covsite.clean_text) if name_col else ""
            ),
        }
    )
    missing = (
        (work["PDB_ID"] == "")
        | (work["electrophile_smiles"] == "")
        | (work["Residue"] == "")
        | (work["ResNum"] == "")
        | (work["Chain"] == "")
    )
    if missing.any():
        bad = work.index[missing].tolist()
        raise SystemExit(
            "[ERROR] Library rows missing pdb/smiles/residue/resnum/chain: "
            f"{bad}"
        )
    return work


def resolve_named_column(
    tvs: ModuleType,
    columns: pd.Index,
    explicit: str | None,
    aliases: tuple[str, ...],
) -> str | None:
    if explicit:
        found = tvs.resolve_column(columns, explicit)
        if found is None:
            sys.exit(
                f"[ERROR] Hits column {explicit!r} is not in the hits CSV "
                f"(available: {', '.join(map(str, columns))})"
            )
        return found
    for alias in aliases:
        found = tvs.resolve_column(columns, alias)
        if found is not None:
            return found
    return None


def load_hits(
    path: Path,
    tvs: ModuleType,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    """Load positives keyed by site (default) or pdb_id.

    When residue, resnum, and chain are present (and --pdb-hits is not set),
    each hit belongs to one PDB × Residue × ResNum × Chain group. Otherwise
    a ligand listed for a pdb_id is a hit at every site of that PDB.
    """
    hit_df = pd.read_csv(path)
    hit_df.columns = hit_df.columns.str.strip()
    pdb_col = tvs.resolve_column(hit_df.columns, "pdb_id") or tvs.resolve_column(
        hit_df.columns, "pdb"
    )
    if pdb_col is None:
        sys.exit(
            "[ERROR] Hits CSV must contain pdb_id or pdb "
            f"(available: {', '.join(map(str, hit_df.columns))})"
        )

    name_col = tvs.resolve_column(hit_df.columns, args.name_hit_column)
    smiles_col = tvs.resolve_column(hit_df.columns, args.smiles_hit_column)
    if smiles_col is None:
        for alias in ("Electrophile_Hit", "electrophile_smiles", "smiles"):
            smiles_col = tvs.resolve_column(hit_df.columns, alias)
            if smiles_col is not None:
                break
    name_miss_col = tvs.resolve_column(hit_df.columns, args.name_miss_column)
    smiles_miss_col = tvs.resolve_column(hit_df.columns, args.smiles_miss_column)
    hit_wh_col = tvs.resolve_paired_warhead_column(
        hit_df.columns,
        smiles_col or name_col,
        args.warhead_type_column,
    )
    miss_entity = smiles_miss_col or name_miss_col
    miss_wh_col = tvs.resolve_paired_warhead_column(
        hit_df.columns, miss_entity, args.warhead_type_column
    )
    if (
        hit_wh_col is not None
        and miss_wh_col is not None
        and hit_wh_col == miss_wh_col
        and miss_entity is not None
    ):
        miss_wh_col = None
        for col in hit_df.columns:
            col_s = str(col)
            if col_s == hit_wh_col:
                continue
            stem = col_s.split(".")[0].strip().casefold()
            if stem == args.warhead_type_column.strip().casefold():
                miss_wh_col = col_s
                break

    if name_col is None and smiles_col is None:
        sys.exit(
            f"[ERROR] Hits CSV must contain {args.name_hit_column!r} and/or "
            f"{args.smiles_hit_column!r} (case-insensitive). "
            f"Available: {', '.join(map(str, hit_df.columns))}"
        )

    residue_col = resolve_named_column(
        tvs,
        hit_df.columns,
        args.hit_residue_column,
        ("residue", "resname", "aa"),
    )
    resnum_col = resolve_named_column(
        tvs,
        hit_df.columns,
        args.hit_resnum_column,
        ("resnum", "res_num", "residue_number", "residue_num"),
    )
    chain_col = resolve_named_column(
        tvs,
        hit_df.columns,
        args.hit_chain_column,
        ("chain",),
    )
    n_site_cols = sum(
        col is not None for col in (residue_col, resnum_col, chain_col)
    )
    if args.pdb_hits:
        key_level = "pdb"
        residue_col = resnum_col = chain_col = None
        if n_site_cols:
            print(
                "[INFO] --pdb-hits: ignoring residue/resnum/chain on hits; "
                "a ligand is a hit at every site of that pdb_id"
            )
    elif n_site_cols == 0:
        key_level = "pdb"
        print(
            "[WARN] Hits CSV has no residue/resnum/chain; a ligand listed "
            "for a pdb_id is a hit at every site of that PDB. Add those "
            "columns to label hits per site."
        )
    elif n_site_cols != 3:
        sys.exit(
            "[ERROR] Hits site labels need residue, resnum, and chain "
            "together (or omit all three / use --pdb-hits)."
        )
    else:
        key_level = "site"

    by_key: dict[str, Any] = {}
    skipped_key = 0
    incomplete_site = 0
    for _, row in hit_df.iterrows():
        pdb_id = covsite.pdb_id_from_value(row[pdb_col])
        if not pdb_id:
            skipped_key += 1
            continue
        if key_level == "site":
            residue = covsite.clean_text(row[residue_col]).upper()
            resnum = covsite.normalized_resnum(row[resnum_col])
            chain = covsite.clean_text(row[chain_col])
            if not (residue and resnum and chain):
                incomplete_site += 1
                continue
            group_key = make_site_key(pdb_id, residue, resnum, chain)
        else:
            group_key = pdb_id
        spec = by_key.setdefault(group_key, tvs.HitMissSpec())
        name = tvs.normalize_name(row[name_col]) if name_col else ""
        smiles = tvs.normalize_smiles(row[smiles_col]) if smiles_col else ""
        hit_wh = (
            tvs.parse_warhead_type_cell(row[hit_wh_col]) if hit_wh_col else None
        )
        if name:
            spec.hit_names.add(name)
            if hit_wh_col is not None:
                spec.hit_name_warheads[name] = tvs._merge_warhead_allowance(
                    spec.hit_name_warheads.get(name, set()),
                    hit_wh,
                )
        if smiles:
            spec.hit_smiles.add(smiles)
            if hit_wh_col is not None:
                spec.hit_smiles_warheads[smiles] = tvs._merge_warhead_allowance(
                    spec.hit_smiles_warheads.get(smiles, set()),
                    hit_wh,
                )

        miss_name = tvs.normalize_name(row[name_miss_col]) if name_miss_col else ""
        miss_smiles = (
            tvs.normalize_smiles(row[smiles_miss_col]) if smiles_miss_col else ""
        )
        miss_wh = (
            tvs.parse_warhead_type_cell(row[miss_wh_col]) if miss_wh_col else None
        )
        if miss_name:
            spec.miss_names.add(miss_name)
            if miss_wh_col is not None:
                spec.miss_name_warheads[miss_name] = tvs._merge_warhead_allowance(
                    spec.miss_name_warheads.get(miss_name, set()),
                    miss_wh,
                )
        if miss_smiles:
            spec.miss_smiles.add(miss_smiles)
            if miss_wh_col is not None:
                spec.miss_smiles_warheads[miss_smiles] = (
                    tvs._merge_warhead_allowance(
                        spec.miss_smiles_warheads.get(miss_smiles, set()),
                        miss_wh,
                    )
                )

    if skipped_key:
        print(f"[WARN] Skipped {skipped_key:,} hit row(s) with empty pdb_id")
    if incomplete_site:
        print(
            f"[WARN] Skipped {incomplete_site:,} hit row(s) with incomplete "
            "residue/resnum/chain"
        )
    n_hits = sum(
        1
        for spec in by_key.values()
        if spec.hit_names or spec.hit_smiles
    )
    if n_hits == 0:
        sys.exit(
            f"[ERROR] No non-empty positives found in {args.name_hit_column!r} "
            f"/ {args.smiles_hit_column!r}."
        )
    return by_key, key_level


def make_site_key(pdb_id: str, residue: str, resnum: str, chain: str) -> str:
    return (
        f"{pdb_id}|{covsite.clean_text(residue).upper()}|"
        f"{covsite.normalized_resnum(resnum)}|{covsite.clean_text(chain)}"
    )


def library_lookup_keys(library: pd.DataFrame, tvs: ModuleType) -> pd.DataFrame:
    out = library.copy()
    out["_canon_smiles"] = out["electrophile_smiles"].map(tvs.normalize_smiles)
    out["_site_key"] = [
        make_site_key(pdb, res, num, chain)
        for pdb, res, num, chain in zip(
            out["PDB_ID"], out["Residue"], out["ResNum"], out["Chain"]
        )
    ]
    return out


def drop_constant_within_site(
    df: pd.DataFrame, features: list[str], site_col: str = "Site"
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for feature in features:
        varies = False
        for _, group in df.groupby(site_col, sort=False):
            if int(group[feature].nunique(dropna=True)) > 1:
                varies = True
                break
        if varies:
            kept.append(feature)
        else:
            dropped.append(feature)
    return kept, dropped


def label_candidates(
    candidates: pd.DataFrame,
    library: pd.DataFrame,
    hits_by_key: dict[str, Any],
    tvs: ModuleType,
    *,
    hit_key_level: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    work = candidates.copy()
    if "PDB_ID" not in work.columns:
        sys.exit("[ERROR] Candidates CSV has no PDB_ID column")
    if "electrophile_smiles" not in work.columns:
        sys.exit("[ERROR] Candidates CSV has no electrophile_smiles column")
    work["PDB_ID"] = work["PDB_ID"].map(covsite.pdb_id_from_value)
    work["Residue"] = work["Residue"].map(covsite.clean_text).str.upper()
    work["ResNum"] = work["ResNum"].map(covsite.normalized_resnum)
    work["Chain"] = work["Chain"].map(covsite.clean_text)
    work["_analysis_smiles"] = work["electrophile_smiles"].map(
        tvs.normalize_smiles
    )
    if "Warhead_Base" in work.columns:
        work["_analysis_warhead_base"] = work["Warhead_Base"].map(
            tvs.normalize_warhead_base
        )
    else:
        work["_analysis_warhead_base"] = work["Warhead"].map(
            tvs.normalize_warhead_base
        )
    work["_site_key"] = [
        make_site_key(pdb, res, num, chain)
        for pdb, res, num, chain in zip(
            work["PDB_ID"], work["Residue"], work["ResNum"], work["Chain"]
        )
    ]

    lib = library_lookup_keys(library, tvs)
    lib_site_keys = set(lib["_site_key"])
    lib_pairs = set(zip(lib["_site_key"], lib["_canon_smiles"]))
    before = len(work)
    work = work.loc[work["_site_key"].isin(lib_site_keys)].copy()
    pair_mask = [
        (site, smi) in lib_pairs
        for site, smi in zip(work["_site_key"], work["_analysis_smiles"])
    ]
    work = work.loc[pair_mask].copy()
    dropped = before - len(work)
    if dropped:
        print(
            f"[INFO] Filtered {dropped:,} candidate row(s) that are not in "
            "the library site × SMILES set"
        )
    if work.empty:
        sys.exit("[ERROR] No candidate rows remain after library filtering")

    name_lookup: dict[tuple[str, str], str] = {}
    for _, row in lib.iterrows():
        key = (row["_site_key"], row["_canon_smiles"])
        supplied = covsite.clean_text(row["library_name"])
        if key not in name_lookup:
            name_lookup[key] = supplied

    ligand_names: list[str] = []
    match_names: list[str] = []
    keep: list[bool] = []
    is_hit: list[int] = []
    for _, row in work.iterrows():
        pdb_id = str(row["PDB_ID"])
        smiles = str(row["_analysis_smiles"])
        warhead = str(row["_analysis_warhead_base"])
        supplied = name_lookup.get((row["_site_key"], smiles), "")
        ident = supplied or smiles
        name = tvs.normalize_name(f"{row['_site_key']}|{ident}")
        match_name = tvs.normalize_name(supplied) if supplied else name
        ligand_names.append(name)
        match_names.append(match_name)
        hit_key = str(row["_site_key"]) if hit_key_level == "site" else pdb_id
        spec = hits_by_key.get(hit_key)
        if spec is None:
            keep.append(True)
            is_hit.append(0)
            continue
        allowed = spec.allowed_warheads_for_row(match_name, smiles)
        keep.append(allowed is None or warhead in allowed)
        is_hit.append(int(spec.row_is_hit(match_name, smiles, warhead)))

    work["_analysis_name"] = ligand_names
    work["_match_name"] = match_names
    work["_analysis_warhead"] = work["Warhead"].astype(str).str.strip()
    keep_arr = np.asarray(keep, dtype=bool)
    hit_arr = np.asarray(is_hit, dtype=int)
    n_before_wh = len(work)
    work = work.iloc[np.flatnonzero(keep_arr)].copy()
    n_dropped_wh = n_before_wh - len(work)
    if n_dropped_wh:
        print(
            f"[INFO] Per-ligand warhead_type filter removed {n_dropped_wh:,} / "
            f"{n_before_wh:,} rows"
        )
    if work.empty:
        sys.exit("[ERROR] No candidates remain after warhead_type filter")
    work["is_hit"] = hit_arr[keep_arr]
    work["Site"] = work["_site_key"]
    work["Name"] = work["_analysis_name"]
    work["query_group"] = work["Site"]

    unmatched = collect_unmatched_hits(
        hits_by_key, work, hit_key_level=hit_key_level
    )
    return work.reset_index(drop=True), unmatched


def collect_unmatched_hits(
    hits_by_key: dict[str, Any],
    candidates: pd.DataFrame,
    *,
    hit_key_level: str,
) -> list[dict[str, str]]:
    group_col = "Site" if hit_key_level == "site" else "PDB_ID"
    missing_reason = (
        "site_not_in_candidates"
        if hit_key_level == "site"
        else "pdb_not_in_candidates"
    )
    missing_detail = (
        "No featurized rows for this site"
        if hit_key_level == "site"
        else "No featurized rows for this pdb_id"
    )
    present_smiles: dict[str, set[str]] = {}
    present_names: dict[str, set[str]] = {}
    for key, group in candidates.groupby(group_col, sort=False):
        present_smiles[str(key)] = set(group["_analysis_smiles"].astype(str))
        names = set(group["Name"].astype(str))
        if "_match_name" in group.columns:
            names |= set(group["_match_name"].astype(str))
        present_names[str(key)] = names
    records: list[dict[str, str]] = []
    for group_key, spec in hits_by_key.items():
        smiles_here = present_smiles.get(group_key, set())
        names_here = present_names.get(group_key, set())
        if group_key not in present_smiles:
            records.append(
                {
                    "Hit": group_key,
                    "Reason": missing_reason,
                    "Detail": missing_detail,
                }
            )
        for smiles in sorted(spec.hit_smiles):
            if smiles and smiles not in smiles_here:
                records.append(
                    {
                        "Hit": smiles,
                        "Reason": "smiles_not_in_candidates",
                        "Detail": f"{group_col}={group_key}",
                    }
                )
        if not spec.hit_smiles:
            for name in sorted(spec.hit_names):
                if name and name not in names_here:
                    records.append(
                        {
                            "Hit": name,
                            "Reason": "name_not_in_candidates",
                            "Detail": f"{group_col}={group_key}",
                        }
                    )
    return records


def valid_sites(df: pd.DataFrame) -> list[str]:
    kept: list[str] = []
    for site, group in df.groupby("Site", sort=True):
        n = len(group)
        n_hits = int((group["is_hit"] == 1).sum())
        if n < 2:
            print(
                f"[WARN] Skipping site {site}: {n} candidate(s) "
                "(need at least 2; EF undefined)"
            )
            continue
        if n_hits < 1:
            print(
                f"[WARN] Skipping site {site}: 0 hits (EF undefined)"
            )
            continue
        kept.append(str(site))
    return kept


def filter_candidates_to_warheads(
    candidates: pd.DataFrame,
    tvs: ModuleType,
    warheads: list[str],
) -> pd.DataFrame:
    """Keep only listed Warhead_Base values for every ligand row."""
    allowed = {
        tvs.normalize_warhead_base(name) for name in warheads if name
    }
    allowed.discard("")
    if not allowed:
        sys.exit("[ERROR] --keep-warhead produced no usable warhead names")
    if "Warhead_Base" in candidates.columns:
        col = "Warhead_Base"
    elif "Warhead" in candidates.columns:
        col = "Warhead"
    else:
        sys.exit(
            "[ERROR] --keep-warhead needs a Warhead_Base or Warhead column "
            "in the candidates CSV"
        )
    keys = candidates[col].map(tvs.normalize_warhead_base)
    kept = candidates.loc[keys.isin(allowed)].copy()
    dropped = len(candidates) - len(kept)
    labels = ", ".join(sorted(allowed))
    print(
        f"[INFO] --keep-warhead {labels}: kept {len(kept):,} / "
        f"{len(candidates):,} candidate row(s) "
        f"(dropped {dropped:,})"
    )
    if kept.empty:
        sys.exit(
            "[ERROR] No candidate rows remain after --keep-warhead "
            f"({labels})"
        )
    present = set(keys.loc[keys.isin(allowed)].unique())
    missing = sorted(allowed - present)
    if missing:
        print(
            "[WARN] --keep-warhead names with zero matching rows: "
            + ", ".join(missing)
        )
    return kept


def parent_compound_key(smiles: object, tvs: ModuleType) -> str:
    """Group COValid-style protomers: neutralize charges, drop stereo.

    Parse the original (not casefolded) SMILES. Re-parsing casefolded
    strings (O → o) makes RDKit emit SMILES parse errors.
    """
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return ""
    text = str(smiles).strip()
    if not text or text.lower() == "nan":
        return ""
    chem = getattr(tvs, "Chem", None)
    if chem is None:
        return tvs.smiles_match_key(text)
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")
        mol = chem.MolFromSmiles(text)
    except Exception:
        mol = None
    finally:
        try:
            from rdkit import RDLogger

            RDLogger.EnableLog("rdApp.error")
            RDLogger.EnableLog("rdApp.warning")
        except Exception:
            pass
    if mol is None:
        return tvs.smiles_match_key(text)
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        uncharger = getattr(parent_compound_key, "_uncharger", None)
        if uncharger is None:
            uncharger = rdMolStandardize.Uncharger()
            parent_compound_key._uncharger = uncharger
        mol = uncharger.uncharge(mol)
    except Exception:
        pass
    work = chem.Mol(mol)
    chem.RemoveStereochemistry(work)
    return chem.MolToSmiles(work).casefold()


def attach_parent_keys(
    site_candidates: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    """Copy parent keys computed from original SMILES onto ranked rows."""
    if "_parent_key" not in selected.columns:
        return site_candidates
    needed = ["_analysis_name", "_analysis_warhead", "_site_key", "_parent_key"]
    if "_analysis_smiles" in selected.columns:
        needed.append("_analysis_smiles")
    missing = [col for col in needed if col not in selected.columns]
    if missing:
        return site_candidates
    lookup = selected[needed].drop_duplicates(
        [col for col in needed if col != "_parent_key"],
        keep="first",
    )
    rename = {
        "_analysis_name": "Name",
        "_analysis_warhead": "Warhead",
        "_site_key": "Site",
        "_analysis_smiles": "electrophile_smiles",
    }
    lookup = lookup.rename(columns=rename)
    on = [col for col in ("Name", "Warhead", "Site", "electrophile_smiles") if col in lookup.columns]
    out = site_candidates.merge(lookup, on=on, how="left")
    if "_parent_key" in out.columns:
        out["_parent_key"] = out["_parent_key"].fillna("").astype(str)
    return out


def collapse_active_protomers(
    ranking: pd.DataFrame,
    tvs: ModuleType,
    score_column: str,
    rank_column: str | None = None,
    *,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Keep the best-scoring hit protomer per parent compound × site.

    Decoy protomers are left unchanged (COValid SI LogAUC protocol).
    """
    if ranking.empty or "is_hit" not in ranking.columns:
        return ranking
    if "electrophile_smiles" not in ranking.columns:
        print(
            "[WARN] --collapse-active-protomers ignored: ranking has no "
            "electrophile_smiles column"
        )
        return ranking
    work = ranking.copy()
    if "_parent_key" not in work.columns:
        print(
            "[WARN] --collapse-active-protomers ignored: parent keys were "
            "not attached from original SMILES (refusing to re-parse "
            "casefolded ranking SMILES)"
        )
        return ranking
    hits = work.loc[work["is_hit"] == 1]
    decoys = work.loc[work["is_hit"] != 1]
    if hits.empty:
        return ranking
    empty_mask = hits["_parent_key"].eq("")
    keyed = hits.loc[~empty_mask]
    unmatched = hits.loc[empty_mask]
    group_cols = ["_parent_key"]
    if "Site" in keyed.columns:
        group_cols.append("Site")
    keyed = keyed.sort_values(
        score_column,
        ascending=not higher_is_better,
        kind="mergesort",
    )
    kept_hits = keyed.drop_duplicates(subset=group_cols, keep="first")
    dropped = len(keyed) - len(kept_hits)
    if dropped:
        print(
            "[INFO] --collapse-active-protomers: kept best-scoring hit "
            "protomer per parent compound × site; dropped "
            f"{dropped:,} extra hit protomer row(s) "
            f"({len(keyed):,} → {len(kept_hits):,})"
        )
    result = pd.concat(
        [kept_hits, unmatched, decoys], ignore_index=True
    )
    result = result.drop(columns=["_parent_key"], errors="ignore")
    result = result.sort_values(
        score_column,
        ascending=not higher_is_better,
        kind="mergesort",
    ).reset_index(drop=True)
    if rank_column is not None:
        result[rank_column] = np.arange(1, len(result) + 1)
    return result


def ranking_level_names(args: argparse.Namespace) -> tuple[str, str]:
    parts = ["lgbm", "loco" if args.leave_one_cluster_out else "loso"]
    if args.best_warhead:
        parts.append("best_warhead")
    else:
        parts.append("all_warheads")
    if args.keep_warhead:
        parts.append("keep_wh")
    if args.collapse_active_protomers:
        parts.append("collapse_actives")
    if args.multiple_hits:
        parts.append("multi_hit")
    else:
        parts.append("any_hit")
    stem = "_".join(parts)
    return f"{stem}_site", f"{stem}_pooled"


def rank_frame(
    df: pd.DataFrame,
    score_column: str,
    rank_column: str,
) -> pd.DataFrame:
    ranked = df.sort_values(
        [score_column, "Name", "Warhead"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked[rank_column] = np.arange(1, len(ranked) + 1)
    leading = [
        rank_column,
        "Site",
        "PDB_ID",
        "cluster_id",
        "Residue",
        "ResNum",
        "Chain",
        "Name",
        "Warhead",
        "electrophile_smiles",
        "is_hit",
        score_column,
    ]
    leading = [col for col in leading if col in ranked.columns]
    remaining = [col for col in ranked.columns if col not in leading]
    return ranked[leading + remaining]


def lgbm_query_count(n_rows: int, n_pos: int, max_rows: int) -> int:
    """How many LightGBM queries a site group needs under the row cap."""
    if n_rows <= max_rows:
        return 1
    if n_pos <= 0 or n_pos >= max_rows:
        return int(np.ceil(n_rows / max_rows))
    n_neg = n_rows - n_pos
    cap_neg = max_rows - n_pos
    return int(np.ceil(n_neg / cap_neg)) if n_neg else 1


def split_lgbm_rank_queries(
    train: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split site groups so each Lambdarank query has at most max_rows rows.

    Hits stay in every sub-query; negatives are partitioned without
    replacement. Hit sample weights are 1/K so a large site does not
    dominate because its actives are repeated across K queries.
    """
    pieces: list[pd.DataFrame] = []
    weights: list[np.ndarray] = []
    qid = 0
    rng = np.random.RandomState(seed)
    for _, group in train.groupby("query_group", sort=False):
        group = group.reset_index(drop=True)
        n_rows = len(group)
        pos = group.loc[group["is_hit"] == 1]
        neg = group.loc[group["is_hit"] != 1]
        n_pos = len(pos)
        if n_rows <= max_rows:
            out = group.copy()
            out["_lgb_query"] = qid
            pieces.append(out)
            weights.append(np.ones(n_rows, dtype=float))
            qid += 1
            continue
        if n_pos <= 0 or n_pos >= max_rows:
            shuffled = group.sample(frac=1, random_state=rng.randint(0, 2**31 - 1))
            for start in range(0, n_rows, max_rows):
                chunk = shuffled.iloc[start : start + max_rows].copy()
                chunk["_lgb_query"] = qid
                pieces.append(chunk)
                weights.append(np.ones(len(chunk), dtype=float))
                qid += 1
            continue
        cap_neg = max_rows - n_pos
        shuffled_neg = neg.sample(
            frac=1, random_state=rng.randint(0, 2**31 - 1)
        ).reset_index(drop=True)
        n_chunks = int(np.ceil(len(shuffled_neg) / cap_neg))
        hit_weight = 1.0 / n_chunks
        for i in range(n_chunks):
            chunk_neg = shuffled_neg.iloc[i * cap_neg : (i + 1) * cap_neg]
            chunk = pd.concat([pos, chunk_neg], ignore_index=True)
            chunk["_lgb_query"] = qid
            pieces.append(chunk)
            weights.append(
                np.concatenate(
                    [
                        np.full(n_pos, hit_weight, dtype=float),
                        np.ones(len(chunk_neg), dtype=float),
                    ]
                )
            )
            qid += 1
    out = pd.concat(pieces, ignore_index=True)
    sample_weight = np.concatenate(weights)
    group_sizes = out.groupby("_lgb_query", sort=False).size().to_numpy()
    return out, group_sizes, sample_weight


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node

    def find(self, node: str) -> str:
        self.add(node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def merge_chain_clusters_to_pdb_map(
    pdb_ids: list[str],
    chain_cluster_map: dict[str, str],
) -> dict[str, str]:
    """Lift per-chain MMseqs2 clusters to PDB clusters.

    Two PDBs share a cluster when any of their chains belong to the same
    sequence cluster. Same rule as Training_Cov_Screen.py.
    """
    pdb_ids_upper = [str(pid).strip().upper() for pid in pdb_ids]
    rep_to_pdbs: dict[str, list[str]] = defaultdict(list)
    for member, rep in chain_cluster_map.items():
        if "__" not in member:
            continue
        pdb_id, _chain = member.rsplit("__", 1)
        pdb_id = pdb_id.strip().upper()
        if pdb_id:
            rep_to_pdbs[rep.upper()].append(pdb_id)

    uf = _UnionFind()
    for pdb_id in pdb_ids_upper:
        uf.add(pdb_id)
    for pdbs in rep_to_pdbs.values():
        unique_pdbs = list(dict.fromkeys(pdbs))
        if len(unique_pdbs) < 2:
            continue
        anchor = unique_pdbs[0]
        for other in unique_pdbs[1:]:
            uf.union(anchor, other)

    components: dict[str, list[str]] = defaultdict(list)
    for pdb_id in pdb_ids_upper:
        components[uf.find(pdb_id)].append(pdb_id)
    cluster_map: dict[str, str] = {}
    for members in components.values():
        cluster_rep = min(members)
        for pdb_id in members:
            cluster_map[pdb_id] = cluster_rep
    return cluster_map


def load_clusters_tsv(path: Path) -> dict[str, str]:
    """MMseqs2 createtsv rows are representative, member."""
    cluster_map: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            cluster_map[member.upper()] = rep.upper()
    return cluster_map


def index_pdb_files(pdb_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not pdb_dir.is_dir():
        sys.exit(f"[ERROR] PDB directory not found: {pdb_dir}")
    for path in pdb_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pdb", ".ent"}:
            continue
        stem = path.stem.upper()
        pdb_id = stem[3:] if stem.startswith("PDB") and len(stem) == 7 else stem
        found.setdefault(pdb_id, path)
    return found


def resolve_pdb_file(
    pdb_id: str,
    pdb_dir: Path,
    index: dict[str, Path],
) -> Path | None:
    pdb_id = pdb_id.upper()
    if pdb_id in index:
        return index[pdb_id]
    dest = pdb_dir / f"{pdb_id}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[INFO] Downloading {pdb_id} for MMseqs2 clustering")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"[WARN] Could not download {pdb_id}: {exc}")
        return None
    index[pdb_id] = dest
    return dest


def write_clustering_fasta(
    pdb_ids: list[str],
    pdb_dir: Path,
    fasta_path: Path,
    *,
    deep: bool,
    min_chain_length: int,
) -> int:
    try:
        from Bio.PDB import PDBParser, PPBuilder
    except ImportError:
        sys.exit("[ERROR] Biopython is required for MMseqs2 clustering")

    parser = PDBParser(QUIET=True)
    builder = PPBuilder()
    index = index_pdb_files(pdb_dir)
    written = 0
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for pdb_id in pdb_ids:
            path = resolve_pdb_file(pdb_id, pdb_dir, index)
            if path is None:
                print(f"[WARN] No structure for {pdb_id}; it will be its own cluster")
                continue
            try:
                structure = parser.get_structure(pdb_id, str(path))
            except Exception as exc:
                print(f"[WARN] Could not parse {path.name}: {exc}")
                continue
            model = next(structure.get_models(), None)
            if model is None:
                continue
            if deep:
                for chain in model:
                    sequence = "".join(
                        str(peptide.get_sequence())
                        for peptide in builder.build_peptides(chain)
                    )
                    if len(sequence) < min_chain_length:
                        continue
                    chain_id = str(chain.id).strip() or "_"
                    handle.write(f">{pdb_id}__{chain_id}\n{sequence}\n")
                    written += 1
            else:
                sequence = "".join(
                    str(peptide.get_sequence())
                    for peptide in builder.build_peptides(model)
                )
                if not sequence:
                    print(f"[WARN] No polymer sequence in {pdb_id}")
                    continue
                handle.write(f">{pdb_id}\n{sequence}\n")
                written += 1
    print(f"[INFO] Wrote {written} sequence(s) to {fasta_path}")
    return written


def run_mmseqs2(
    fasta_path: Path,
    output_dir: Path,
    seq_identity: float,
    coverage: float,
) -> dict[str, str]:
    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        sys.exit(
            "[ERROR] mmseqs was not found on PATH. Install MMseqs2 or pass "
            "--clusters-tsv from a previous clustering run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    db = output_dir / "seqdb"
    clustered = output_dir / "clusters"
    tmp = output_dir / "tmp"
    tsv_path = output_dir / "clusters.tsv"
    # createdb writes one file per thread (seqdb.0 ... seqdb.N). A high
    # thread count on a parallel filesystem fails with "Could not open
    # seqdb.N for writing". One thread is enough for these PDB sets.
    threads = "1"
    for path in list(output_dir.iterdir()):
        name = path.name
        if name == fasta_path.name:
            continue
        if name == "tmp" or name.startswith(("seqdb", "clusters")):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    commands = [
        [mmseqs, "createdb", str(fasta_path), str(db), "--threads", threads],
        [
            mmseqs, "cluster", str(db), str(clustered), str(tmp),
            "--min-seq-id", str(seq_identity),
            "-c", str(coverage),
            "--cov-mode", "0",
            "--threads", threads,
        ],
        [
            mmseqs, "createtsv", str(db), str(db), str(clustered), str(tsv_path),
            "--threads", threads,
        ],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part and part.strip()
            )
            sys.exit(
                "[ERROR] MMseqs2 failed:\n  "
                + " ".join(cmd)
                + (f"\n{detail}" if detail else "")
            )
    cluster_map = load_clusters_tsv(tsv_path)
    if not cluster_map:
        sys.exit(f"[ERROR] MMseqs2 wrote an empty cluster table: {tsv_path}")
    n_clusters = len(set(cluster_map.values()))
    print(
        f"[INFO] MMseqs2: {n_clusters} cluster(s) from "
        f"{len(cluster_map)} sequence(s) "
        f"(identity {seq_identity:g}, coverage {coverage:g})"
    )
    return cluster_map


def assign_mmseqs_clusters(
    df: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    """Add cluster_id. PDBs that share sequence similarity share a cluster."""
    if "PDB_ID" not in df.columns:
        sys.exit("[ERROR] Cluster CV needs a PDB_ID column")
    pdb_ids = sorted({str(pdb).strip().upper() for pdb in df["PDB_ID"] if str(pdb).strip()})
    if len(pdb_ids) < 2:
        sys.exit(
            "[ERROR] Leave-one-cluster-out needs at least 2 PDB ids "
            f"(found {len(pdb_ids)})"
        )
    deep = not args.no_deep_cluster
    if args.clusters_tsv:
        tsv_path = Path(args.clusters_tsv).resolve()
        if not tsv_path.is_file():
            sys.exit(f"[ERROR] --clusters-tsv not found: {tsv_path}")
        raw_map = load_clusters_tsv(tsv_path)
        if not raw_map:
            sys.exit(f"[ERROR] --clusters-tsv is empty: {tsv_path}")
        chain_level = any("__" in member for member in raw_map)
        if chain_level:
            cluster_map = merge_chain_clusters_to_pdb_map(pdb_ids, raw_map)
            print(
                "[INFO] Chain-level clusters.tsv: "
                f"{len(set(raw_map.values()))} chain cluster(s) -> "
                f"{len(set(cluster_map.values()))} PDB cluster(s)"
            )
        else:
            cluster_map = dict(raw_map)
            print(
                "[INFO] PDB-level clusters.tsv: "
                f"{len(set(cluster_map.values()))} cluster(s), "
                f"{len(cluster_map)} member(s)"
            )
            if deep:
                print(
                    "[WARN] clusters.tsv has no PDB__CHAIN ids, so PDBs were "
                    "not merged through shared chains"
                )
    else:
        mmseqs_dir = (
            Path(args.mmseqs_dir).resolve()
            if args.mmseqs_dir
            else out_dir / "mmseqs"
        )
        fasta_name = "chains.fasta" if deep else "proteins.fasta"
        n_written = write_clustering_fasta(
            pdb_ids,
            Path(args.pdb_dir).resolve(),
            mmseqs_dir / fasta_name,
            deep=deep,
            min_chain_length=args.min_chain_length,
        )
        if n_written == 0:
            sys.exit("[ERROR] No sequences were extracted for MMseqs2")
        raw_map = run_mmseqs2(
            mmseqs_dir / fasta_name,
            mmseqs_dir,
            args.seq_identity,
            args.coverage,
        )
        if deep:
            cluster_map = merge_chain_clusters_to_pdb_map(pdb_ids, raw_map)
            print(
                "[INFO] Deep cluster: "
                f"{len(set(raw_map.values()))} chain cluster(s) -> "
                f"{len(set(cluster_map.values()))} PDB cluster(s) "
                f"from {len(pdb_ids)} protein(s)"
            )
        else:
            cluster_map = raw_map
            print(
                "[INFO] PDB cluster: "
                f"{len(set(cluster_map.values()))} cluster(s) "
                f"from {len(pdb_ids)} protein(s)"
            )

    missing = [pdb_id for pdb_id in pdb_ids if pdb_id not in cluster_map]
    for pdb_id in missing:
        cluster_map[pdb_id] = pdb_id
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        print(
            f"[WARN] {len(missing)} PDB(s) were not in the MMseqs2 table "
            f"and are singleton clusters: {preview}{suffix}"
        )

    out = df.copy()
    out["cluster_id"] = out["PDB_ID"].astype(str).str.upper().map(cluster_map)
    out["cluster_id"] = out["cluster_id"].fillna(out["PDB_ID"].astype(str).str.upper())
    assignment = (
        out.groupby(["PDB_ID", "cluster_id"], sort=True)
        .agg(
            n_sites=("Site", "nunique"),
            n_rows=("Site", "size"),
            n_hit_rows=("is_hit", "sum"),
        )
        .reset_index()
    )
    assignment_path = out_dir / "pdb_clusters.csv"
    assignment.to_csv(assignment_path, index=False)
    n_clusters = int(out["cluster_id"].nunique())
    print(
        f"[INFO] Leave-one-cluster-out: {n_clusters} cluster(s) "
        f"over {out['PDB_ID'].nunique()} PDB(s). Wrote {assignment_path.name}"
    )
    return out


def leave_one_group_out(
    df: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
    model_dir: Path | None,
    *,
    group_col: str,
) -> pd.DataFrame:
    """Hold out one site or one sequence cluster per fold.

    LightGBM queries stay per site in both modes. A cluster fold removes
    every site whose PDB belongs to that cluster.
    """
    label = "cluster" if group_col == "cluster_id" else "site"
    sites = valid_sites(df)
    usable = df.loc[df["Site"].isin(sites)].copy()
    if group_col == "Site":
        groups = sites
    else:
        if group_col not in usable.columns:
            sys.exit(f"[ERROR] Missing {group_col} column for cluster CV")
        groups = sorted(usable[group_col].astype(str).unique())
    if len(groups) < 2:
        sys.exit(
            f"[ERROR] Leave-one-{label}-out needs at least 2 {label}s with "
            f">=2 candidates and >=1 hit (found {len(groups)})"
        )
    print(
        f"[Training_CovSite_VS] Leave-one-{label}-out: {len(groups)} {label}(s), "
        f"{usable['Site'].nunique()} site(s), {len(usable):,} candidate row(s)"
    )
    max_rows = args.max_query_size
    oversized = []
    for site, group in usable.groupby("Site", sort=True):
        n_rows = len(group)
        if n_rows <= max_rows:
            continue
        n_pos = int((group["is_hit"] == 1).sum())
        n_queries = lgbm_query_count(n_rows, n_pos, max_rows)
        oversized.append(
            f"{site}: {n_rows:,} rows, {n_pos} hit row(s) -> "
            f"{n_queries} training queries of <= {max_rows}"
        )
    if oversized:
        print(
            "[INFO] LightGBM Lambdarank allows at most "
            f"{LGBM_MAX_QUERY_ROWS} rows per query; these sites are split "
            "for training only (held-out scoring still uses every row):"
        )
        for line in oversized:
            print(f"  {line}")
    oof_frames: list[pd.DataFrame] = []
    params = lgbm_params(args)
    group_values = usable[group_col].astype(str)
    for fold, held_out in enumerate(groups, start=1):
        train = usable.loc[group_values != held_out].copy()
        test = usable.loc[group_values == held_out].copy()
        train = train.sort_values("query_group").reset_index(drop=True)
        train, group_sizes, sample_weight = split_lgbm_rank_queries(
            train,
            max_rows=max_rows,
            seed=args.random_state + fold,
        )
        if int(group_sizes.max()) > LGBM_MAX_QUERY_ROWS:
            sys.exit(
                "[ERROR] A LightGBM query still exceeds "
                f"{LGBM_MAX_QUERY_ROWS} rows (max {int(group_sizes.max())})"
            )
        X_tr = train[features].apply(pd.to_numeric, errors="coerce")
        y_tr = train["is_hit"].astype(int).to_numpy()
        X_te = test[features].apply(pd.to_numeric, errors="coerce")
        model = lgb.LGBMRanker(**params)
        model.fit(
            X_tr, y_tr, group=group_sizes, sample_weight=sample_weight
        )
        test = test.copy()
        test["pred_score"] = model.predict(X_te)
        test["fold"] = fold
        if group_col == "cluster_id":
            test["held_out_cluster"] = held_out
        else:
            test["held_out_site"] = held_out
        oof_frames.append(test)
        print(
            f"  Fold {fold}/{len(groups)} held-out {label} {held_out}: "
            f"{test['PDB_ID'].nunique()} PDB(s), {test['Site'].nunique()} site(s), "
            f"{len(test)} rows, {int((test['is_hit'] == 1).sum())} hit row(s)"
        )
        if model_dir is not None:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", held_out).strip("_")
            joblib.dump(model, model_dir / f"fold_{fold:03d}_{safe}.pkl")
    return pd.concat(oof_frames, ignore_index=True)


def collapse_ligand_ranking(
    tvs: ModuleType, pooled: pd.DataFrame, multiple_hits: bool
) -> pd.DataFrame:
    work = pooled.copy()
    added_alias = False
    if "topsis_score" not in work.columns and "pred_score" in work.columns:
        work["topsis_score"] = work["pred_score"]
        added_alias = True
    out = tvs.collapse_pooled_to_ligand_ranking(
        work, multiple_hits=multiple_hits
    )
    if added_alias and "topsis_score" in out.columns:
        out = out.drop(columns=["topsis_score"])
    return out


def export_results(
    out_dir: Path,
    selected: pd.DataFrame,
    per_site: pd.DataFrame,
    pooled: pd.DataFrame,
    enrichment: pd.DataFrame,
    feature_statistics: pd.DataFrame,
    feature_enrichment: pd.DataFrame,
    unmatched_hits: list[dict[str, str]],
    features: list[str],
    tvs: ModuleType,
    multiple_hits: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_site.to_csv(out_dir / "lgbm_per_site_ranking.csv", index=False)
    per_site.loc[per_site["is_hit"] == 1].to_csv(
        out_dir / "hit_per_site_ranks.csv", index=False
    )
    pooled.to_csv(out_dir / "lgbm_pooled_pair_ranking.csv", index=False)
    pooled.loc[pooled["is_hit"] == 1].to_csv(
        out_dir / "hit_pooled_pair_ranks.csv", index=False
    )
    ligand = collapse_ligand_ranking(tvs, pooled, multiple_hits)
    ligand.to_csv(out_dir / "full_pooled_ligand_ranking.csv", index=False)
    enrichment.to_csv(out_dir / "enrichment_factors.csv", index=False)
    feature_statistics.to_csv(out_dir / "feature_statistics.csv", index=False)
    feature_enrichment.to_csv(
        out_dir / "feature_enrichment_factors.csv", index=False
    )
    audit_cols = [
        col
        for col in [
            "Name",
            "PDB_ID",
            "Residue",
            "ResNum",
            "Chain",
            "Warhead",
            "electrophile_smiles",
            "Site",
            "is_hit",
        ]
        if col in selected.columns
    ] + [f for f in features if f in selected.columns]
    selected[audit_cols].to_csv(
        out_dir / "matched_site_warhead_rows.csv", index=False
    )
    unmatched_df = pd.DataFrame(
        unmatched_hits, columns=["Hit", "Reason", "Detail"]
    )
    unmatched_df.to_csv(out_dir / "unmatched_hit_names.csv", index=False)

    print("\nLigand VS ranker complete")
    if "held_out_cluster" in per_site.columns:
        print(f"  Held-out clusters:         {per_site['held_out_cluster'].nunique():,}")
    print(f"  Held-out sites:            {per_site['Site'].nunique():,}")
    print(f"  Site × ligand × warhead:   {len(per_site):,}")
    print(
        "  Positive Names matched:    "
        f"{per_site.loc[per_site['is_hit'] == 1, 'Name'].nunique():,}"
    )
    print(f"  Unmatched positives:       {len(unmatched_hits):,}")
    print("  Full rankings:")
    print(f"    Per-site pairs:          {out_dir / 'lgbm_per_site_ranking.csv'}")
    print(f"    Pooled pairs:            {out_dir / 'lgbm_pooled_pair_ranking.csv'}")
    print(f"    Pooled ligands:          {out_dir / 'full_pooled_ligand_ranking.csv'}")
    print("  Hit-only subsets:")
    print(f"    Per-site:                {out_dir / 'hit_per_site_ranks.csv'}")
    print(f"    Pooled pairs:            {out_dir / 'hit_pooled_pair_ranks.csv'}")
    print(f"  Enrichment:                {out_dir / 'enrichment_factors.csv'}")
    print(
        f"  Feature enrichment:        {out_dir / 'feature_enrichment_factors.csv'}"
    )


def main() -> None:
    args = parse_args()
    topsis_path = (
        Path(args.topsis_vs).resolve()
        if args.topsis_vs
        else DEFAULT_TOPSIS_VS
    )
    tvs = load_topsis_vs(topsis_path)
    print(f"[INFO] VS eval helpers: {topsis_path}")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    library_path = Path(args.library_csv).resolve()
    library = load_library(library_path, args)
    hits_by_key, hit_key_level = load_hits(
        Path(args.hits_csv).resolve(), tvs, args
    )
    n_hit_smiles = sum(len(spec.hit_smiles) for spec in hits_by_key.values())
    n_hit_names = sum(len(spec.hit_names) for spec in hits_by_key.values())
    if hit_key_level == "site":
        print(
            f"[INFO] Hits for {len(hits_by_key):,} site(s) "
            f"(PDB × Residue × ResNum × Chain): "
            f"{n_hit_smiles:,} SMILES, {n_hit_names:,} names"
        )
    else:
        print(
            f"[INFO] Hits for {len(hits_by_key):,} pdb_id(s): "
            f"{n_hit_smiles:,} SMILES, {n_hit_names:,} names"
        )
    if args.multiple_hits:
        print("[INFO] Hit counting: Name × Warhead × SMILES (--multiple-hits)")
    else:
        print("[INFO] Hit counting: one hit per ligand Name")
    if args.best_warhead:
        print(
            "[INFO] --best-warhead: keep highest pred_score warhead per "
            "electrophile_smiles × site"
        )
    if args.keep_warhead:
        print(
            "[INFO] --keep-warhead: restrict every ligand to "
            + ", ".join(args.keep_warhead)
        )
    if args.collapse_active_protomers:
        print(
            "[INFO] --collapse-active-protomers: EF/LogAUC keep the best "
            "hit protomer per parent compound × site; keep all decoys"
        )

    if args.candidates:
        candidates_path = Path(args.candidates).resolve()
        print(f"[INFO] Reusing candidates: {candidates_path}")
    else:
        feat_dir = (
            Path(args.covsite_output_dir).resolve()
            if args.covsite_output_dir
            else out_dir / "covsite_features"
        )
        candidates_path = run_covsite_vs(
            library_path, Path(args.pdb_dir).resolve(), feat_dir, args
        )
        print(f"[INFO] CovSite candidates: {candidates_path}")

    candidates = pd.read_csv(candidates_path, low_memory=False)
    if candidates.empty:
        sys.exit("[ERROR] Candidates CSV has no rows")
    if args.keep_warhead:
        candidates = filter_candidates_to_warheads(
            candidates, tvs, args.keep_warhead
        )

    features = list(dict.fromkeys(args.features))
    missing = [f for f in features if f not in candidates.columns]
    if missing:
        sys.exit(
            "[ERROR] Features missing from candidates CSV:\n  "
            + ", ".join(missing)
        )

    selected, unmatched = label_candidates(
        candidates,
        library,
        hits_by_key,
        tvs,
        hit_key_level=hit_key_level,
    )
    if args.collapse_active_protomers:
        selected["_parent_key"] = selected["electrophile_smiles"].map(
            lambda smi: parent_compound_key(smi, tvs)
        )
        n_blank = int(selected["_parent_key"].eq("").sum())
        n_parents = selected.loc[selected["is_hit"] == 1, "_parent_key"].nunique()
        print(
            "[INFO] --collapse-active-protomers: "
            f"{n_parents:,} hit parent compound(s) "
            f"({int((selected['is_hit'] == 1).sum()):,} hit protomer row(s))"
        )
        if n_blank:
            print(
                f"[WARN] {n_blank:,} row(s) had empty parent keys "
                "(unparseable SMILES); those hits are not merged"
            )
    site_candidates = tvs.build_site_candidates(selected, features)
    if "_parent_key" in selected.columns:
        site_candidates = attach_parent_keys(site_candidates, selected)
    site_candidates["PDB_ID"] = (
        site_candidates["Site"].astype(str).str.split("|").str[0]
    )
    site_candidates["query_group"] = site_candidates["Site"]

    kept, dropped = drop_constant_within_site(site_candidates, features)
    if dropped:
        print(
            "[WARN] Dropping features that are constant inside every site "
            "group:\n  " + ", ".join(dropped)
        )
    features = kept
    if not features:
        sys.exit(
            "[ERROR] No --features remain after dropping constant-within-site "
            "columns"
        )

    model_dir = out_dir / "models" if args.save_models else None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    if args.leave_one_cluster_out:
        site_candidates = assign_mmseqs_clusters(
            site_candidates, args, out_dir
        )
        oof = leave_one_group_out(
            site_candidates,
            features,
            args,
            model_dir,
            group_col="cluster_id",
        )
    else:
        oof = leave_one_group_out(
            site_candidates,
            features,
            args,
            model_dir,
            group_col="Site",
        )

    per_site_parts: list[pd.DataFrame] = []
    per_site_ef: list[pd.DataFrame] = []
    site_level, pooled_level = ranking_level_names(args)

    for site, group in oof.groupby("Site", sort=True):
        ranked = rank_frame(group, "pred_score", "site_rank")
        if args.best_warhead:
            ranked = tvs.keep_best_warhead_per_electrophile(
                ranked,
                score_column="pred_score",
                rank_column="site_rank",
                higher_is_better=True,
            )
        if args.collapse_active_protomers:
            ranked = collapse_active_protomers(
                ranked,
                tvs,
                score_column="pred_score",
                rank_column="site_rank",
                higher_is_better=True,
            )
        per_site_parts.append(ranked)
        per_site_ef.append(
            tvs.enrichment_any_candidate_per_ligand(
                ranked,
                site_level,
                rank_column="site_rank",
                metadata={
                    "analysis_scope": "per_site",
                    "Site": site,
                    "Residue": group["Residue"].iloc[0],
                    "ResNum": group["ResNum"].iloc[0],
                    "Chain": group["Chain"].iloc[0],
                    "PDB_ID": (
                        group["PDB_ID"].iloc[0]
                        if "PDB_ID" in group.columns
                        else ""
                    ),
                },
                multiple_hits=args.multiple_hits,
            )
        )
    per_site = pd.concat(per_site_parts, ignore_index=True)

    pooled = rank_frame(oof, "pred_score", "overall_rank")
    if args.best_warhead:
        pooled = tvs.keep_best_warhead_per_electrophile(
            pooled,
            score_column="pred_score",
            rank_column="overall_rank",
            higher_is_better=True,
        )
    if args.collapse_active_protomers:
        pooled = collapse_active_protomers(
            pooled,
            tvs,
            score_column="pred_score",
            rank_column="overall_rank",
            higher_is_better=True,
        )
    pooled_ef = tvs.enrichment_any_candidate_per_ligand(
        pooled,
        pooled_level,
        rank_column="overall_rank",
        metadata={
            "analysis_scope": "pooled",
            "Site": "ALL",
            "Residue": "ALL",
            "ResNum": "ALL",
            "Chain": "ALL",
            "PDB_ID": "ALL",
        },
        multiple_hits=args.multiple_hits,
    )
    enrichment = pd.concat([*per_site_ef, pooled_ef], ignore_index=True)

    stats_pool = site_candidates.loc[
        site_candidates["Site"].isin(oof["Site"].unique())
    ].copy()
    feature_statistics, feature_enrichment = tvs.feature_statistics_and_enrichment(
        stats_pool,
        features,
        benefit_features=features,
        multiple_hits=args.multiple_hits,
        best_warhead=args.best_warhead,
    )

    export_results(
        out_dir,
        selected,
        per_site,
        pooled,
        enrichment,
        feature_statistics,
        feature_enrichment,
        unmatched,
        features,
        tvs,
        args.multiple_hits,
    )


if __name__ == "__main__":
    main()
