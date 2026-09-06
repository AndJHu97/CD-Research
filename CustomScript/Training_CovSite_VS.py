#!/usr/bin/env python3
"""Leave-one-site-out LightGBM ligand ranker for a fixed covalent site.

Calls CovSite --VS for features (or reuses --candidates). Hits are matched by
RDKit canonical SMILES and pdb_id. Ranking groups are PDB × Residue × ResNum
× Chain. Metrics (EF, LogAUC, univariate feature stats) are imported from
TOPSIS_VS.py; there is no TOPSIS solver.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import CovSite as covsite


HERE = Path(__file__).resolve().parent
TOPSIS_VS_PATH = HERE / "bo-runs" / "scoring-analysis" / "VS_eval" / "TOPSIS_VS.py"


def load_topsis_vs() -> ModuleType:
    spec = importlib.util.spec_from_file_location("TOPSIS_VS", TOPSIS_VS_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load {TOPSIS_VS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LightGBM ligand-screening ranker with leave-one-site-out "
            "CV. Features come from CovSite --VS (or --candidates). Hits are "
            "matched by canonical SMILES and pdb_id."
        )
    )
    parser.add_argument(
        "library_csv",
        help="Library CSV: pdb/pdb_id, smiles, residue, resnum, chain",
    )
    parser.add_argument(
        "hits_csv",
        help=(
            "Hits CSV: pdb_id plus Electrophile_Hit / electrophile_hit "
            "(optional Name_Hit, warhead_type)"
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
        "--best-warhead",
        action="store_true",
        help=(
            "Keep the highest pred_score warhead per electrophile_smiles × "
            "site before ranking exports and EF/LogAUC"
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
    args = parser.parse_args()
    if args.no_save_models:
        args.save_models = False
    if args.candidates is None and not args.pdb_dir:
        parser.error("--pdb-dir is required unless --candidates is set")
    return args


def lgbm_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
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


def load_hits_by_pdb(
    path: Path,
    tvs: ModuleType,
    args: argparse.Namespace,
) -> dict[str, Any]:
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

    by_pdb: dict[str, Any] = {}
    skipped_pdb = 0
    for _, row in hit_df.iterrows():
        pdb_id = covsite.pdb_id_from_value(row[pdb_col])
        if not pdb_id:
            skipped_pdb += 1
            continue
        spec = by_pdb.setdefault(pdb_id, tvs.HitMissSpec())
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

    if skipped_pdb:
        print(
            f"[WARN] Skipped {skipped_pdb:,} hit row(s) with empty pdb_id"
        )
    n_hits = sum(
        1
        for spec in by_pdb.values()
        if spec.hit_names or spec.hit_smiles
    )
    if n_hits == 0:
        sys.exit(
            f"[ERROR] No non-empty positives found in {args.name_hit_column!r} "
            f"/ {args.smiles_hit_column!r}."
        )
    return by_pdb


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
    hits_by_pdb: dict[str, Any],
    tvs: ModuleType,
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
        name = tvs.normalize_name(f"{pdb_id}|{ident}")
        match_name = tvs.normalize_name(supplied) if supplied else name
        ligand_names.append(name)
        match_names.append(match_name)
        spec = hits_by_pdb.get(pdb_id)
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

    unmatched = collect_unmatched_hits(hits_by_pdb, work)
    return work.reset_index(drop=True), unmatched


def collect_unmatched_hits(
    hits_by_pdb: dict[str, Any], candidates: pd.DataFrame
) -> list[dict[str, str]]:
    present_smiles: dict[str, set[str]] = {}
    present_names: dict[str, set[str]] = {}
    for pdb_id, group in candidates.groupby("PDB_ID", sort=False):
        present_smiles[str(pdb_id)] = set(group["_analysis_smiles"].astype(str))
        names = set(group["Name"].astype(str))
        if "_match_name" in group.columns:
            names |= set(group["_match_name"].astype(str))
        present_names[str(pdb_id)] = names
    records: list[dict[str, str]] = []
    for pdb_id, spec in hits_by_pdb.items():
        smiles_here = present_smiles.get(pdb_id, set())
        names_here = present_names.get(pdb_id, set())
        if pdb_id not in present_smiles:
            records.append(
                {
                    "Hit": pdb_id,
                    "Reason": "pdb_not_in_candidates",
                    "Detail": "No featurized rows for this pdb_id",
                }
            )
        for smiles in sorted(spec.hit_smiles):
            if smiles and smiles not in smiles_here:
                records.append(
                    {
                        "Hit": smiles,
                        "Reason": "smiles_not_in_candidates",
                        "Detail": f"pdb_id={pdb_id}",
                    }
                )
        if not spec.hit_smiles:
            for name in sorted(spec.hit_names):
                if name and name not in names_here:
                    records.append(
                        {
                            "Hit": name,
                            "Reason": "name_not_in_candidates",
                            "Detail": f"pdb_id={pdb_id}",
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


def leave_one_site_out(
    df: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
    model_dir: Path | None,
) -> pd.DataFrame:
    sites = valid_sites(df)
    if len(sites) < 2:
        sys.exit(
            "[ERROR] Leave-one-site-out needs at least 2 sites with "
            f">=2 candidates and >=1 hit (found {len(sites)})"
        )
    usable = df.loc[df["Site"].isin(sites)].copy()
    print(
        f"[Training_CovSite_VS] LOSO: {len(sites)} site(s), "
        f"{len(usable):,} candidate row(s)"
    )
    oof_frames: list[pd.DataFrame] = []
    params = lgbm_params(args)
    for fold, held_out in enumerate(sites, start=1):
        train = usable.loc[usable["Site"] != held_out].copy()
        test = usable.loc[usable["Site"] == held_out].copy()
        train = train.sort_values("query_group").reset_index(drop=True)
        group_sizes = (
            train.groupby("query_group", sort=False).size().to_numpy()
        )
        X_tr = train[features].apply(pd.to_numeric, errors="coerce")
        y_tr = train["is_hit"].astype(int).to_numpy()
        X_te = test[features].apply(pd.to_numeric, errors="coerce")
        model = lgb.LGBMRanker(**params)
        model.fit(X_tr, y_tr, group=group_sizes)
        test = test.copy()
        test["pred_score"] = model.predict(X_te)
        test["fold"] = fold
        test["held_out_site"] = held_out
        oof_frames.append(test)
        print(
            f"  Fold {fold}/{len(sites)} held-out {held_out}: "
            f"{len(test)} ligands, "
            f"{int((test['is_hit'] == 1).sum())} hit row(s)"
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
    tvs = load_topsis_vs()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    library_path = Path(args.library_csv).resolve()
    library = load_library(library_path, args)
    hits_by_pdb = load_hits_by_pdb(Path(args.hits_csv).resolve(), tvs, args)
    n_hit_smiles = sum(len(spec.hit_smiles) for spec in hits_by_pdb.values())
    n_hit_names = sum(len(spec.hit_names) for spec in hits_by_pdb.values())
    print(
        f"[INFO] Hits for {len(hits_by_pdb):,} pdb_id(s): "
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

    features = list(dict.fromkeys(args.features))
    missing = [f for f in features if f not in candidates.columns]
    if missing:
        sys.exit(
            "[ERROR] Features missing from candidates CSV:\n  "
            + ", ".join(missing)
        )

    selected, unmatched = label_candidates(
        candidates, library, hits_by_pdb, tvs
    )
    site_candidates = tvs.build_site_candidates(selected, features)
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

    oof = leave_one_site_out(site_candidates, features, args, model_dir)

    per_site_parts: list[pd.DataFrame] = []
    per_site_ef: list[pd.DataFrame] = []
    if args.best_warhead and args.multiple_hits:
        site_level = "lgbm_site_best_warhead_multi_hit"
        pooled_level = "lgbm_pooled_best_warhead_multi_hit"
    elif args.best_warhead:
        site_level = "lgbm_site_best_warhead"
        pooled_level = "lgbm_pooled_best_warhead"
    elif args.multiple_hits:
        site_level = "lgbm_site_all_warheads_multi_hit"
        pooled_level = "lgbm_pooled_all_warheads_multi_hit"
    else:
        site_level = "lgbm_site_all_warheads_any_hit"
        pooled_level = "lgbm_pooled_all_warheads_any_hit"

    for site, group in oof.groupby("Site", sort=True):
        ranked = rank_frame(group, "pred_score", "site_rank")
        if args.best_warhead:
            ranked = tvs.keep_best_warhead_per_electrophile(
                ranked,
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
