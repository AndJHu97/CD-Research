#!/usr/bin/env python3
"""
Run fpocket on PDBs referenced in a CovSite label-results CSV and compare
pocket detection against CovSite hits (hit_at_top_pct).

Outputs:
  - detail CSV: per-label fpocket pocket rank, target-in-pocket flags, nucleophilic
    residue counts, and fpocket descriptors (score, druggability, volume, etc.)
  - summary CSV: hit rates (top-1 / any pocket), nucleophilic stats (when_hit and
    all_labels scope), per-residue-type accuracy, and fpocket-vs-CovSite disagreement
  - fpocket_found CSVs (top-1 and overall): labels fpocket finds but CovSite misses
  - covsite_found CSVs (top-1 and overall): labels CovSite finds but fpocket misses

  --top-1-analysis compares fpocket rank-1 pockets (hits) vs target-containing
  pockets when fpocket misses rank-1 but CovSite hits and fpocket finds the site
  elsewhere (overall hit).

Usage:
    python check_f_pocket.py test_label_results.csv --pdb-dir /path/to/pdbs
    python check_f_pocket.py test_label_results.csv --pdb-dir ./pdbs \\
        --output-dir ./fpocket_analysis --prefix test --top-3-flag
    python check_f_pocket.py --fix \\
        --detail-csv test_label_results_fpocket_detail.csv \\
        --summary-csv test_label_results_fpocket_summary.csv \\
        test_label_results.csv --work-dir fpocket_runs
    python check_f_pocket.py --top-1-analysis \\
        --detail-csv test_label_results_fpocket_detail.csv \\
        --covsite-found-top1-csv test_label_results_covsite_found_top1.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

NUCLEOPHILIC_RESIDUES = frozenset({"CYS", "SER", "HIS", "LYS", "TYR", "THR"})
HIS_VARIANTS = frozenset({"HIE", "HID", "HIP"})
NUCLEOPHILIC_ORDER = ("CYS", "SER", "HIS", "LYS", "TYR", "THR")

INFO_KEY_MAP = {
    "score": "Score",
    "druggability_score": "Druggability Score",
    "n_alpha_spheres": "Number of Alpha Spheres",
    "total_sasa": "Total SASA",
    "polar_sasa": "Polar SASA",
    "apolar_sasa": "Apolar SASA",
    "volume": "Volume",
    "mean_local_hydrophobic_density": "Mean local hydrophobic density",
    "mean_alpha_sphere_radius": "Mean alpha sphere radius",
    "mean_alpha_sphere_solvent_access": "Mean alp. sph. solvent access",
    "apolar_alpha_sphere_proportion": "Apolar alpha sphere proportion",
    "hydrophobicity_score": "Hydrophobicity score",
    "volume_score": "Volume score",
    "polarity_score": "Polarity score",
    "charge_score": "Charge score",
    "flexibility": "Flexibility",
    "prop_polar_atm": "Proportion of polar atoms",
}


@dataclass
class PocketData:
    rank: int
    residues: set[tuple[str, int, str]]
    nucleophilic_count: int
    nucleophilic_by_type: dict[str, int]
    descriptors: dict[str, float] = field(default_factory=dict)


@dataclass
class FpocketResult:
    pdb_id: str
    out_dir: Path
    pockets: list[PocketData]
    error: Optional[str] = None


def normalize_resname(resname: str) -> str:
    name = str(resname).strip().upper()[:3]
    if name in HIS_VARIANTS:
        return "HIS"
    return name


def normalize_resnum(resnum) -> int:
    if pd.isna(resnum):
        return 0
    text = str(resnum).strip()
    if not text:
        return 0
    return int(float(text))


def normalize_chain(chain) -> str:
    if pd.isna(chain):
        return ""
    return str(chain).strip().upper()


def pdb_id_from_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return ""
    return text.split("-")[0].replace(".pdb", "").split(".")[0].upper()


def find_pdb_file(pdb_id: str, pdb_dir: str, recursive: bool = False) -> Optional[str]:
    pdb_id = str(pdb_id).strip().replace(".pdb", "").split(".")[0]
    if not pdb_id:
        return None

    pdb_id_lower = pdb_id.lower()
    for candidate in (
        os.path.join(pdb_dir, f"{pdb_id_lower}.pdb"),
        os.path.join(pdb_dir, f"{pdb_id.upper()}.pdb"),
    ):
        if os.path.isfile(candidate):
            return candidate

    search_roots = [pdb_dir]
    if recursive:
        search_roots = [root for root, _, _ in os.walk(pdb_dir)]

    for root in search_roots:
        try:
            for fname in os.listdir(root):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in {".pdb", ".ent"}:
                    continue
                if stem.lower() == pdb_id_lower:
                    return os.path.join(root, fname)
        except FileNotFoundError:
            continue
    return None


def parse_fpocket_info(info_path: Path) -> dict[int, dict[str, float]]:
    if not info_path.is_file():
        return {}

    text = info_path.read_text(encoding="utf-8", errors="replace")
    pockets: dict[int, dict[str, float]] = {}
    current_rank: Optional[int] = None
    current: dict[str, float] = {}

    pocket_re = re.compile(r"^\s*Pocket\s+(\d+)\s*:", re.IGNORECASE)
    kv_re = re.compile(r"^\s*([^:]+?)\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$")

    for line in text.splitlines():
        m_pocket = pocket_re.match(line)
        if m_pocket:
            if current_rank is not None:
                pockets[current_rank] = current
            current_rank = int(m_pocket.group(1))
            current = {}
            continue

        m_kv = kv_re.match(line)
        if m_kv and current_rank is not None:
            key = m_kv.group(1).strip()
            value = float(m_kv.group(2))
            for norm_key, label in INFO_KEY_MAP.items():
                if key.lower() == label.lower():
                    current[norm_key] = value
                    break

    if current_rank is not None:
        pockets[current_rank] = current
    return pockets


def parse_pocket_atm(atm_path: Path) -> set[tuple[str, int, str]]:
    residues: set[tuple[str, int, str]] = set()
    if not atm_path.is_file():
        return residues

    with atm_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if len(line) < 26:
                continue
            resname = normalize_resname(line[17:20])
            chain = normalize_chain(line[21])
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            residues.add((chain, resnum, resname))
    return residues


def count_nucleophilic(residues: set[tuple[str, int, str]]) -> tuple[int, dict[str, int]]:
    by_type = {aa: 0 for aa in NUCLEOPHILIC_ORDER}
    for _, _, resname in residues:
        if resname in NUCLEOPHILIC_RESIDUES:
            by_type[resname] += 1
    return sum(by_type.values()), by_type


def union_residues_in_pockets(pockets: list[PocketData]) -> set[tuple[str, int, str]]:
    merged: set[tuple[str, int, str]] = set()
    for pocket in pockets:
        merged |= pocket.residues
    return merged


def union_nucleophilic_in_pockets(pockets: list[PocketData]) -> tuple[int, dict[str, int]]:
    return count_nucleophilic(union_residues_in_pockets(pockets))


def target_in_residues(
    residues: set[tuple[str, int, str]],
    chain: str,
    resnum: int,
    residue: str,
) -> bool:
    target_name = normalize_resname(residue)
    return (chain, resnum, target_name) in residues


def run_fpocket(
    pdb_path: str,
    pdb_id: str,
    work_dir: Path,
    fpocket_bin: str,
    force: bool,
) -> FpocketResult:
    out_dir = work_dir / f"{pdb_id.lower()}_out"
    info_path = out_dir / f"{pdb_id.lower()}_info.txt"
    pockets_dir = out_dir / "pockets"

    if not force and info_path.is_file() and pockets_dir.is_dir():
        return load_fpocket_result(pdb_id, out_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    local_pdb = work_dir / f"{pdb_id.lower()}.pdb"
    if not local_pdb.exists() or force:
        shutil.copy2(pdb_path, local_pdb)

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)

    try:
        subprocess.run(
            [fpocket_bin, "-f", str(local_pdb.resolve())],
            cwd=str(work_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return FpocketResult(pdb_id, out_dir, [], error=f"fpocket not found: {fpocket_bin}")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        return FpocketResult(
            pdb_id,
            out_dir,
            [],
            error=stderr or f"fpocket failed with exit code {exc.returncode}",
        )

    if not out_dir.is_dir():
        alt = work_dir / f"{pdb_id.upper()}_out"
        if alt.is_dir():
            out_dir = alt
        else:
            return FpocketResult(pdb_id, out_dir, [], error="fpocket output directory missing")

    return load_fpocket_result(pdb_id, out_dir)


def load_fpocket_result(pdb_id: str, out_dir: Path) -> FpocketResult:
    stem = out_dir.name.replace("_out", "")
    info_path = out_dir / f"{stem}_info.txt"
    if not info_path.is_file():
        candidates = list(out_dir.glob("*_info.txt"))
        info_path = candidates[0] if candidates else info_path

    descriptors_by_rank = parse_fpocket_info(info_path)
    pockets_dir = out_dir / "pockets"
    pockets: list[PocketData] = []

    if pockets_dir.is_dir():
        atm_files = sorted(
            pockets_dir.glob("pocket*_atm.pdb"),
            key=lambda p: int(re.search(r"pocket(\d+)_atm", p.name).group(1)),
        )
        for atm_file in atm_files:
            file_idx = int(re.search(r"pocket(\d+)_atm", atm_file.name).group(1))
            rank = file_idx + 1
            residues = parse_pocket_atm(atm_file)
            nucl_count, nucl_by_type = count_nucleophilic(residues)
            pockets.append(
                PocketData(
                    rank=rank,
                    residues=residues,
                    nucleophilic_count=nucl_count,
                    nucleophilic_by_type=nucl_by_type,
                    descriptors=descriptors_by_rank.get(rank, {}),
                )
            )
    else:
        for rank in sorted(descriptors_by_rank):
            pockets.append(
                PocketData(
                    rank=rank,
                    residues=set(),
                    nucleophilic_count=0,
                    nucleophilic_by_type={aa: 0 for aa in NUCLEOPHILIC_ORDER},
                    descriptors=descriptors_by_rank[rank],
                )
            )

    return FpocketResult(pdb_id, out_dir, pockets)


def pocket_descriptor_columns(prefix: str) -> list[str]:
    cols = []
    for key in INFO_KEY_MAP:
        cols.append(f"{prefix}_{key}")
    cols.extend(
        [
            f"{prefix}_nucleophilic_count",
            f"{prefix}_nucleophilic_cys",
            f"{prefix}_nucleophilic_ser",
            f"{prefix}_nucleophilic_his",
            f"{prefix}_nucleophilic_lys",
            f"{prefix}_nucleophilic_tyr",
            f"{prefix}_nucleophilic_thr",
        ]
    )
    return cols


def pocket_to_columns(pocket: Optional[PocketData], prefix: str) -> dict:
    out: dict = {}
    for key in INFO_KEY_MAP:
        out[f"{prefix}_{key}"] = np.nan
    for aa in NUCLEOPHILIC_ORDER:
        out[f"{prefix}_nucleophilic_{aa.lower()}"] = 0
    out[f"{prefix}_nucleophilic_count"] = 0

    if pocket is None:
        return out

    for key in INFO_KEY_MAP:
        out[f"{prefix}_{key}"] = pocket.descriptors.get(key, np.nan)
    out[f"{prefix}_nucleophilic_count"] = pocket.nucleophilic_count
    for aa in NUCLEOPHILIC_ORDER:
        out[f"{prefix}_nucleophilic_{aa.lower()}"] = pocket.nucleophilic_by_type.get(aa, 0)
    return out


def analyze_label(
    fpocket: FpocketResult,
    chain: str,
    resnum: int,
    residue: str,
    include_top3: bool = False,
    include_fix: bool = False,
) -> dict:
    top1 = fpocket.pockets[0] if fpocket.pockets else None
    top1_hit = bool(
        top1 and target_in_residues(top1.residues, chain, resnum, residue)
    )

    best_pocket: Optional[PocketData] = None
    best_rank = np.nan
    for pocket in fpocket.pockets:
        if target_in_residues(pocket.residues, chain, resnum, residue):
            best_pocket = pocket
            best_rank = pocket.rank
            break

    overall_hit = best_pocket is not None
    row = {
        "fpocket_n_pockets": len(fpocket.pockets),
        "fpocket_error": fpocket.error or "",
        "fpocket_top1_hit": int(top1_hit),
        "fpocket_overall_hit": int(overall_hit),
        "fpocket_best_rank": best_rank,
        "fpocket_top1_target_found": int(top1_hit),
        "fpocket_overall_target_found": int(overall_hit),
    }
    row.update(pocket_to_columns(top1, "fpocket_top1"))
    row.update(pocket_to_columns(best_pocket, "fpocket_best"))

    if include_top3:
        top3_pockets = fpocket.pockets[:3]
        top3_hit = any(
            target_in_residues(pocket.residues, chain, resnum, residue)
            for pocket in top3_pockets
        )
        nucl_count, nucl_by_type = union_nucleophilic_in_pockets(top3_pockets)
        row["fpocket_top3_hit"] = int(top3_hit)
        row["fpocket_top3_nucleophilic_count"] = nucl_count
        for aa in NUCLEOPHILIC_ORDER:
            row[f"fpocket_top3_nucleophilic_{aa.lower()}"] = nucl_by_type.get(aa, 0)

    if include_fix:
        nucl_all, nucl_by_type_all = union_nucleophilic_in_pockets(fpocket.pockets)
        row["fpocket_all_pockets_nucleophilic_count"] = nucl_all
        for aa in NUCLEOPHILIC_ORDER:
            row[f"fpocket_all_pockets_nucleophilic_{aa.lower()}"] = nucl_by_type_all.get(aa, 0)

    return row


def compute_top3_from_fpocket(
    fpocket: Optional[FpocketResult],
    chain: str,
    resnum: int,
    residue: str,
) -> dict:
    """Top-3 hit + union nucleophilic count across fpocket ranks 1-3."""
    empty = {
        "fpocket_top3_hit": 0,
        "fpocket_top3_nucleophilic_count": 0,
        **{f"fpocket_top3_nucleophilic_{aa.lower()}": 0 for aa in NUCLEOPHILIC_ORDER},
    }
    if fpocket is None or not fpocket.pockets:
        return empty

    top3_pockets = fpocket.pockets[:3]
    top3_hit = any(
        target_in_residues(pocket.residues, chain, resnum, residue)
        for pocket in top3_pockets
    )
    nucl_count, nucl_by_type = union_nucleophilic_in_pockets(top3_pockets)
    return {
        "fpocket_top3_hit": int(top3_hit),
        "fpocket_top3_nucleophilic_count": nucl_count,
        **{f"fpocket_top3_nucleophilic_{aa.lower()}": nucl_by_type.get(aa, 0) for aa in NUCLEOPHILIC_ORDER},
    }


def load_fpocket_from_work_dir(pdb_id: str, work_dir: Path) -> Optional[FpocketResult]:
    for stem in (pdb_id.lower(), pdb_id.upper()):
        out_dir = work_dir / f"{stem}_out"
        if out_dir.is_dir() and (out_dir / "pockets").is_dir():
            return load_fpocket_result(pdb_id, out_dir)
    return None


def enrich_detail_with_top3(detail: pd.DataFrame, work_dir: Path) -> pd.DataFrame:
    """Add top-3 columns using cached fpocket pocket*_atm.pdb files."""
    out = detail.copy()
    fpocket_cache: dict[str, Optional[FpocketResult]] = {}

    top3_cols = ["fpocket_top3_hit", "fpocket_top3_nucleophilic_count"] + [
        f"fpocket_top3_nucleophilic_{aa.lower()}" for aa in NUCLEOPHILIC_ORDER
    ]
    for col in top3_cols:
        if col not in out.columns:
            out[col] = 0

    for row_idx, row in out.iterrows():
        pdb_id = pdb_id_from_name(row["Name"])
        if pdb_id not in fpocket_cache:
            fpocket_cache[pdb_id] = load_fpocket_from_work_dir(pdb_id, work_dir)

        metrics = compute_top3_from_fpocket(
            fpocket_cache[pdb_id],
            normalize_chain(row["Chain"]),
            normalize_resnum(row["ResNum"]),
            normalize_resname(row["Residue"]),
        )
        for col, value in metrics.items():
            out.at[row_idx, col] = value

    return out


def compute_all_pockets_nucleophilic(fpocket: Optional[FpocketResult]) -> dict:
    empty = {
        "fpocket_all_pockets_nucleophilic_count": 0,
        **{f"fpocket_all_pockets_nucleophilic_{aa.lower()}": 0 for aa in NUCLEOPHILIC_ORDER},
    }
    if fpocket is None or not fpocket.pockets:
        return empty
    nucl_count, nucl_by_type = union_nucleophilic_in_pockets(fpocket.pockets)
    return {
        "fpocket_all_pockets_nucleophilic_count": nucl_count,
        **{f"fpocket_all_pockets_nucleophilic_{aa.lower()}": nucl_by_type.get(aa, 0) for aa in NUCLEOPHILIC_ORDER},
    }


def merge_n_residues(detail: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv, low_memory=False)
    if "n_residues" not in labels.columns:
        sys.exit(f"[ERROR] Column 'n_residues' not found in {labels_csv}")

    key_cols = ["Name", "Residue", "ResNum", "Chain"]
    for col in key_cols:
        if col not in labels.columns:
            sys.exit(f"[ERROR] Column '{col}' not found in {labels_csv}")
        if col not in detail.columns:
            sys.exit(f"[ERROR] Column '{col}' not found in detail CSV")

    merge_df = labels[key_cols + ["n_residues"]].copy()
    merge_df["Name"] = merge_df["Name"].astype(str).str.strip().str.upper()
    merge_df["Residue"] = merge_df["Residue"].astype(str).str.strip().str.upper()
    merge_df["ResNum"] = merge_df["ResNum"].map(normalize_resnum)
    merge_df["Chain"] = merge_df["Chain"].map(normalize_chain)
    merge_df = merge_df.drop_duplicates(
        subset=["Name", "Residue", "ResNum", "Chain"], keep="first"
    )

    out = detail.copy()
    lookup = {
        (
            str(r.Name).strip().upper(),
            str(r.Residue).strip().upper(),
            normalize_resnum(r.ResNum),
            normalize_chain(r.Chain),
        ): r.n_residues
        for r in merge_df.itertuples(index=False)
    }
    out["n_residues"] = [
        lookup.get(
            (
                str(row["Name"]).strip().upper(),
                str(row["Residue"]).strip().upper(),
                normalize_resnum(row["ResNum"]),
                normalize_chain(row["Chain"]),
            ),
            np.nan,
        )
        for _, row in out.iterrows()
    ]
    return out


def add_search_space_reduction_columns(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    n_res = pd.to_numeric(out["n_residues"], errors="coerce")
    denom = n_res.replace(0, np.nan)

    out["fpocket_ss_red_top1"] = pd.to_numeric(out["fpocket_top1_nucleophilic_count"], errors="coerce") / denom
    if "fpocket_top3_nucleophilic_count" in out.columns:
        out["fpocket_ss_red_top3"] = (
            pd.to_numeric(out["fpocket_top3_nucleophilic_count"], errors="coerce") / denom
        )
    if "fpocket_all_pockets_nucleophilic_count" in out.columns:
        out["fpocket_ss_red_all_pockets"] = (
            pd.to_numeric(out["fpocket_all_pockets_nucleophilic_count"], errors="coerce") / denom
        )
    return out


def enrich_detail_with_fix(
    detail: pd.DataFrame,
    work_dir: Path,
    labels_csv: str,
) -> pd.DataFrame:
    """Add all-pocket nucleophilic union, n_residues, and search-space reduction."""
    out = merge_n_residues(detail, labels_csv)

    if "fpocket_top3_hit" not in out.columns:
        out = enrich_detail_with_top3(out, work_dir)

    fpocket_cache: dict[str, Optional[FpocketResult]] = {}
    all_cols = ["fpocket_all_pockets_nucleophilic_count"] + [
        f"fpocket_all_pockets_nucleophilic_{aa.lower()}" for aa in NUCLEOPHILIC_ORDER
    ]
    for col in all_cols:
        if col not in out.columns:
            out[col] = 0

    for row_idx, row in out.iterrows():
        pdb_id = pdb_id_from_name(row["Name"])
        if pdb_id not in fpocket_cache:
            fpocket_cache[pdb_id] = load_fpocket_from_work_dir(pdb_id, work_dir)

        metrics = compute_all_pockets_nucleophilic(fpocket_cache[pdb_id])
        for col, value in metrics.items():
            out.at[row_idx, col] = value

    return add_search_space_reduction_columns(out)


def _safe_stats(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna()
    if clean.empty:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.median())


def _overall_nucleophilic_col(detail: pd.DataFrame, fix_mode: bool) -> str:
    if fix_mode and "fpocket_all_pockets_nucleophilic_count" in detail.columns:
        return "fpocket_all_pockets_nucleophilic_count"
    return "fpocket_best_nucleophilic_count"


def build_summary(
    detail: pd.DataFrame,
    include_top3: bool = False,
    fix_mode: bool = False,
) -> pd.DataFrame:
    n = len(detail)
    cov_hit = detail["covsite_hit_at_top_pct"].astype(int)
    has_top3 = include_top3 and "fpocket_top3_hit" in detail.columns
    overall_nucl_col = _overall_nucleophilic_col(detail, fix_mode)

    rows: list[dict] = []

    def add_row(
        section: str,
        metric: str,
        top1_val,
        overall_val=np.nan,
        note: str = "",
        scope: str = "",
        top3_val=np.nan,
    ):
        row = {
            "section": section,
            "metric": metric,
            "scope": scope,
            "top1": top1_val,
            "overall": overall_val,
            "note": note,
        }
        if has_top3:
            row["top3"] = top3_val
        rows.append(row)

    add_row("counts", "n_labels", n, n, top3_val=n if has_top3 else np.nan)
    add_row(
        "hit_rate",
        "fpocket_target_hit_rate",
        detail["fpocket_top1_hit"].mean(),
        detail["fpocket_overall_hit"].mean(),
        top3_val=detail["fpocket_top3_hit"].mean() if has_top3 else np.nan,
    )
    add_row(
        "hit_rate",
        "covsite_hit_at_top_pct_rate",
        cov_hit.mean(),
        cov_hit.mean(),
        "from input CSV hit_at_top_pct",
        top3_val=cov_hit.mean() if has_top3 else np.nan,
    )

    top1_hits = detail[detail["fpocket_top1_hit"] == 1]
    overall_hits = detail[detail["fpocket_overall_hit"] == 1]
    mean_t1, med_t1 = _safe_stats(top1_hits["fpocket_top1_nucleophilic_count"])
    mean_ov_hit, med_ov_hit = _safe_stats(overall_hits[overall_nucl_col])
    mean_all_ov, med_all_ov = _safe_stats(detail[overall_nucl_col])
    if has_top3:
        top3_hits = detail[detail["fpocket_top3_hit"] == 1]
        mean_t3, med_t3 = _safe_stats(top3_hits["fpocket_top3_nucleophilic_count"])
        mean_all_t3, med_all_t3 = _safe_stats(detail["fpocket_top3_nucleophilic_count"])
    else:
        mean_t3 = med_t3 = mean_all_t3 = med_all_t3 = np.nan

    add_row(
        "nucleophilic_in_pocket",
        "mean_count",
        mean_t1,
        mean_ov_hit,
        scope="when_hit",
        top3_val=mean_t3,
        note=(
            "overall (--fix): union across all pockets when target found in any pocket"
            if fix_mode
            else "overall: nucleophiles in target pocket"
        ),
    )
    add_row(
        "nucleophilic_in_pocket",
        "median_count",
        med_t1,
        med_ov_hit,
        scope="when_hit",
        top3_val=med_t3,
    )

    mean_all_t1, med_all_t1 = _safe_stats(detail["fpocket_top1_nucleophilic_count"])
    add_row(
        "nucleophilic_in_pocket",
        "mean_count",
        mean_all_t1,
        mean_all_ov,
        scope="all_labels",
        top3_val=mean_all_t3,
        note=(
            "overall (--fix): union across all fpocket pockets per label"
            if fix_mode
            else "overall: nucleophiles in target pocket"
        ),
    )
    add_row(
        "nucleophilic_in_pocket",
        "median_count",
        med_all_t1,
        med_all_ov,
        scope="all_labels",
        top3_val=med_all_t3,
    )

    for aa in NUCLEOPHILIC_ORDER:
        subset = detail[detail["target_residue_type"].str.upper() == aa]
        if subset.empty:
            add_row("residue_type_accuracy", aa, np.nan, np.nan, "no labels")
            continue
        add_row(
            "residue_type_accuracy",
            aa,
            subset["fpocket_top1_hit"].mean(),
            subset["fpocket_overall_hit"].mean(),
            f"n={len(subset)}",
            top3_val=subset["fpocket_top3_hit"].mean() if has_top3 else np.nan,
        )

    nucl_mask = detail["target_residue_type"].str.upper().isin(NUCLEOPHILIC_RESIDUES)
    nucl_subset = detail[nucl_mask]
    if nucl_subset.empty:
        add_row("residue_type_accuracy", "all_nucleophilic_targets", np.nan, np.nan)
    else:
        add_row(
            "residue_type_accuracy",
            "all_nucleophilic_targets",
            nucl_subset["fpocket_top1_hit"].mean(),
            nucl_subset["fpocket_overall_hit"].mean(),
            f"n={len(nucl_subset)}",
            top3_val=nucl_subset["fpocket_top3_hit"].mean() if has_top3 else np.nan,
        )

    fpocket_only_top1 = detail[(detail["fpocket_top1_hit"] == 1) & (cov_hit == 0)]
    fpocket_only_overall = detail[(detail["fpocket_overall_hit"] == 1) & (cov_hit == 0)]
    if has_top3:
        fpocket_only_top3 = detail[(detail["fpocket_top3_hit"] == 1) & (cov_hit == 0)]
        n_fpocket_only_t3 = len(fpocket_only_top3)
    else:
        fpocket_only_top3 = detail.iloc[0:0]
        n_fpocket_only_t3 = np.nan

    add_row(
        "fpocket_found_covsite_missed",
        "n_labels",
        len(fpocket_only_top1),
        len(fpocket_only_overall),
        top3_val=n_fpocket_only_t3,
    )
    m1, med1 = _safe_stats(fpocket_only_top1["fpocket_top1_nucleophilic_count"])
    mo, medo = _safe_stats(fpocket_only_overall[overall_nucl_col])
    if has_top3:
        m3, med3 = _safe_stats(fpocket_only_top3["fpocket_top3_nucleophilic_count"])
    else:
        m3 = med3 = np.nan
    add_row(
        "fpocket_found_covsite_missed",
        "mean_nucleophilic_count",
        m1,
        mo,
        top3_val=m3,
    )
    add_row(
        "fpocket_found_covsite_missed",
        "median_nucleophilic_count",
        med1,
        medo,
        top3_val=med3,
    )

    cov_only_top1 = detail[(cov_hit == 1) & (detail["fpocket_top1_hit"] == 0)]
    cov_only_overall = detail[(cov_hit == 1) & (detail["fpocket_overall_hit"] == 0)]
    if has_top3:
        cov_only_top3 = detail[(cov_hit == 1) & (detail["fpocket_top3_hit"] == 0)]
        n_cov_only_t3 = len(cov_only_top3)
    else:
        n_cov_only_t3 = np.nan

    add_row(
        "covsite_found_fpocket_missed",
        "n_labels",
        len(cov_only_top1),
        len(cov_only_overall),
        top3_val=n_cov_only_t3,
    )

    if fix_mode and "n_residues" in detail.columns:
        n_res = pd.to_numeric(detail["n_residues"], errors="coerce")
        mean_nres, med_nres = _safe_stats(n_res)
        add_row(
            "n_residues",
            "mean_total_nucleophilic_residues",
            mean_nres,
            mean_nres,
            scope="all_labels",
            top3_val=mean_nres,
            note="CovSite search-space size (n_residues from labels CSV)",
        )
        add_row(
            "n_residues",
            "median_total_nucleophilic_residues",
            med_nres,
            med_nres,
            scope="all_labels",
            top3_val=med_nres,
        )

        if "fpocket_ss_red_top1" in detail.columns:
            ss_top1_hit = detail[detail["fpocket_top1_hit"] == 1]
            ss_top3_hit = detail[detail["fpocket_top3_hit"] == 1] if has_top3 else detail.iloc[0:0]
            ss_all_hit = detail[detail["fpocket_overall_hit"] == 1]

            m_ss_t1_hit, med_ss_t1_hit = _safe_stats(ss_top1_hit["fpocket_ss_red_top1"])
            m_ss_t1_all, med_ss_t1_all = _safe_stats(detail["fpocket_ss_red_top1"])
            m_ss_t3_hit, med_ss_t3_hit = (
                _safe_stats(ss_top3_hit["fpocket_ss_red_top3"]) if has_top3 else (np.nan, np.nan)
            )
            m_ss_t3_all, med_ss_t3_all = (
                _safe_stats(detail["fpocket_ss_red_top3"]) if has_top3 else (np.nan, np.nan)
            )
            m_ss_all_hit, med_ss_all_hit = _safe_stats(ss_all_hit["fpocket_ss_red_all_pockets"])
            m_ss_all_all, med_ss_all_all = _safe_stats(detail["fpocket_ss_red_all_pockets"])

            add_row(
                "search_space_reduction",
                "mean_fraction",
                m_ss_t1_hit,
                m_ss_all_hit,
                scope="when_hit",
                top3_val=m_ss_t3_hit,
                note="fpocket nucleophilic union / n_residues",
            )
            add_row(
                "search_space_reduction",
                "median_fraction",
                med_ss_t1_hit,
                med_ss_all_hit,
                scope="when_hit",
                top3_val=med_ss_t3_hit,
            )
            add_row(
                "search_space_reduction",
                "mean_fraction",
                m_ss_t1_all,
                m_ss_all_all,
                scope="all_labels",
                top3_val=m_ss_t3_all,
                note="top1=1 pocket; top3=ranks 1-3; overall=all fpocket pockets",
            )
            add_row(
                "search_space_reduction",
                "median_fraction",
                med_ss_t1_all,
                med_ss_all_all,
                scope="all_labels",
                top3_val=med_ss_t3_all,
            )

    return pd.DataFrame(rows)


def detail_export_columns() -> list[str]:
    base = [
        "Name",
        "Residue",
        "ResNum",
        "Chain",
        "target_residue_type",
        "covsite_hit_at_top_pct",
        "fpocket_n_pockets",
        "fpocket_error",
        "fpocket_top1_hit",
        "fpocket_overall_hit",
        "fpocket_best_rank",
        "fpocket_top3_hit",
        "fpocket_top3_nucleophilic_count",
    ]
    base.extend(pocket_descriptor_columns("fpocket_top1"))
    base.extend(pocket_descriptor_columns("fpocket_best"))
    return base


def subset_detail(detail: pd.DataFrame, mask: pd.Series, export_scope: str) -> pd.DataFrame:
    cols = [c for c in detail_export_columns() if c in detail.columns]
    out = detail.loc[mask, cols].copy()
    out.insert(0, "export_scope", export_scope)
    return out


def top1_analysis_feature_names() -> list[str]:
    features = ["score", *list(INFO_KEY_MAP.keys())]
    features.append("nucleophilic_count")
    features.extend(f"nucleophilic_{aa.lower()}" for aa in NUCLEOPHILIC_ORDER)
    return list(dict.fromkeys(features))


def _infer_prefix_from_detail_path(detail_csv: str) -> str:
    stem = Path(detail_csv).stem
    if stem.endswith("_fpocket_detail"):
        return stem[: -len("_fpocket_detail")]
    return stem.replace("_detail", "")


def run_top1_analysis(
    detail_csv: str,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    covsite_found_top1_csv: Optional[str] = None,
) -> None:
    """
    Compare fpocket rank-1 pocket descriptors vs target-pocket descriptors when
    fpocket misses rank-1 but CovSite hits and fpocket finds the target overall.
    """
    detail_path = Path(detail_csv)
    if not detail_path.is_file():
        sys.exit(f"[ERROR] Detail CSV not found: {detail_path}")

    out_dir = Path(output_dir) if output_dir else detail_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = prefix or _infer_prefix_from_detail_path(str(detail_path))

    detail = pd.read_csv(detail_path, low_memory=False)
    if "covsite_hit_at_top_pct" not in detail.columns:
        sys.exit("[ERROR] Detail CSV missing covsite_hit_at_top_pct column")

    cov_hit = detail["covsite_hit_at_top_pct"].astype(int)
    group_top1_hit = detail[detail["fpocket_top1_hit"] == 1].copy()
    group_recovered = detail[
        (detail["fpocket_top1_hit"] == 0)
        & (cov_hit == 1)
        & (detail["fpocket_overall_hit"] == 1)
    ].copy()

    if covsite_found_top1_csv:
        cov_path = Path(covsite_found_top1_csv)
        if cov_path.is_file():
            cov_export = pd.read_csv(cov_path, low_memory=False)
            recovered_keys = {
                (
                    str(r.Name).strip().upper(),
                    str(r.Residue).strip().upper(),
                    normalize_resnum(r.ResNum),
                    normalize_chain(r.Chain),
                )
                for _, r in cov_export.iterrows()
                if int(r.get("fpocket_overall_hit", 0)) == 1
            }
            group_recovered = group_recovered[
                group_recovered.apply(
                    lambda row: (
                        str(row["Name"]).strip().upper(),
                        str(row["Residue"]).strip().upper(),
                        normalize_resnum(row["ResNum"]),
                        normalize_chain(row["Chain"]),
                    )
                    in recovered_keys,
                    axis=1,
                )
            ].copy()

    features = top1_analysis_feature_names()
    comparison_rows: list[dict] = []
    test_rows: list[dict] = []

    for feature in features:
        top1_col = f"fpocket_top1_{feature}"
        best_col = f"fpocket_best_{feature}"
        if top1_col not in detail.columns or best_col not in detail.columns:
            continue

        vals_hit = pd.to_numeric(group_top1_hit[top1_col], errors="coerce").dropna()
        vals_recovered = pd.to_numeric(group_recovered[best_col], errors="coerce").dropna()

        for group_name, values, pocket_source in (
            ("fpocket_top1_hit", vals_hit, "fpocket_top1"),
            (
                "fpocket_top1_miss_covsite_recovered_overall_hit",
                vals_recovered,
                "fpocket_best_target_pocket",
            ),
        ):
            comparison_rows.append(
                {
                    "feature": feature,
                    "analysis_group": group_name,
                    "pocket_source": pocket_source,
                    "n": len(values),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                }
            )

        if scipy_stats is not None and len(vals_hit) and len(vals_recovered):
            mw = scipy_stats.mannwhitneyu(
                vals_hit,
                vals_recovered,
                alternative="two-sided",
            )
            test_rows.append(
                {
                    "feature": feature,
                    "test": "mannwhitney_u",
                    "comparison": "top1_hit_vs_recovered_target_pocket",
                    "group_a": "fpocket_top1_hit",
                    "group_a_column": top1_col,
                    "group_b": "fpocket_top1_miss_covsite_recovered_overall_hit",
                    "group_b_column": best_col,
                    "n_group_a": len(vals_hit),
                    "n_group_b": len(vals_recovered),
                    "statistic": float(mw.statistic),
                    "pvalue": float(mw.pvalue),
                    "note": "Unpaired: rank-1 pocket (hits) vs target pocket (recovered misses)",
                }
            )

        paired_top1 = pd.to_numeric(group_recovered[top1_col], errors="coerce")
        paired_best = pd.to_numeric(group_recovered[best_col], errors="coerce")
        paired_mask = paired_top1.notna() & paired_best.notna()
        paired_top1 = paired_top1[paired_mask]
        paired_best = paired_best[paired_mask]
        if scipy_stats is not None and len(paired_top1) >= 2:
            try:
                wx = scipy_stats.wilcoxon(
                    paired_top1,
                    paired_best,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                test_rows.append(
                    {
                        "feature": feature,
                        "test": "wilcoxon_signed_rank",
                        "comparison": "rank1_pocket_vs_target_pocket_within_recovered",
                        "group_a": "fpocket_top1_miss_covsite_recovered_overall_hit",
                        "group_a_column": top1_col,
                        "group_b": "fpocket_top1_miss_covsite_recovered_overall_hit",
                        "group_b_column": best_col,
                        "n_group_a": len(paired_top1),
                        "n_group_b": len(paired_best),
                        "statistic": float(wx.statistic),
                        "pvalue": float(wx.pvalue),
                        "note": "Paired within recovered labels: rank-1 vs target pocket",
                    }
                )
            except ValueError:
                test_rows.append(
                    {
                        "feature": feature,
                        "test": "wilcoxon_signed_rank",
                        "comparison": "rank1_pocket_vs_target_pocket_within_recovered",
                        "group_a": "fpocket_top1_miss_covsite_recovered_overall_hit",
                        "group_a_column": top1_col,
                        "group_b": "fpocket_top1_miss_covsite_recovered_overall_hit",
                        "group_b_column": best_col,
                        "n_group_a": len(paired_top1),
                        "n_group_b": len(paired_best),
                        "statistic": np.nan,
                        "pvalue": np.nan,
                        "note": "Wilcoxon failed (likely all differences zero)",
                    }
                )

    group_export_rows: list[dict] = []
    id_cols = [
        "Name",
        "Residue",
        "ResNum",
        "Chain",
        "target_residue_type",
        "covsite_hit_at_top_pct",
        "fpocket_top1_hit",
        "fpocket_overall_hit",
        "fpocket_best_rank",
    ]

    for _, row in group_top1_hit.iterrows():
        out_row = {
            "analysis_group": "fpocket_top1_hit",
            "pocket_features_from": "fpocket_top1",
        }
        for col in id_cols:
            if col in row:
                out_row[col] = row[col]
        for feature in features:
            col = f"fpocket_top1_{feature}"
            if col in row:
                out_row[feature] = row[col]
        group_export_rows.append(out_row)

    for _, row in group_recovered.iterrows():
        out_row = {
            "analysis_group": "fpocket_top1_miss_covsite_recovered_overall_hit",
            "pocket_features_from": "fpocket_best_target_pocket",
        }
        for col in id_cols:
            if col in row:
                out_row[col] = row[col]
        for feature in features:
            best_col = f"fpocket_best_{feature}"
            top1_col = f"fpocket_top1_{feature}"
            if best_col in row:
                out_row[feature] = row[best_col]
            if top1_col in row:
                out_row[f"rank1_{feature}"] = row[top1_col]
        group_export_rows.append(out_row)

    groups_path = out_dir / f"{file_prefix}_top1_analysis_groups.csv"
    comparison_path = out_dir / f"{file_prefix}_top1_analysis_comparison.csv"
    tests_path = out_dir / f"{file_prefix}_top1_analysis_tests.csv"

    pd.DataFrame(group_export_rows).to_csv(groups_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(test_rows).to_csv(tests_path, index=False)

    print(f"fpocket top-1 hit labels: {len(group_top1_hit)}")
    print(
        "fpocket top-1 miss, CovSite hit, fpocket overall hit: "
        f"{len(group_recovered)}"
    )
    if scipy_stats is None:
        print("[WARN] scipy not installed; statistical tests skipped")
    print(f"Wrote: {groups_path}")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {tests_path}")


def run_analysis(
    labels_csv: str,
    pdb_dir: str,
    output_dir: str,
    prefix: str,
    fpocket_bin: str,
    work_dir: Optional[str],
    recursive_pdb: bool,
    force_fpocket: bool,
    limit: Optional[int],
    include_top3: bool = False,
    include_fix: bool = False,
) -> None:
    labels = pd.read_csv(labels_csv)
    required = {"Name", "Residue", "ResNum", "Chain", "hit_at_top_pct"}
    missing = required - set(labels.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in labels CSV: {sorted(missing)}")

    if "target_residue_type" not in labels.columns:
        labels["target_residue_type"] = labels["Residue"].map(normalize_resname)

    if limit is not None:
        labels = labels.head(limit).copy()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fpocket_work = Path(work_dir) if work_dir else out_path / "fpocket_runs"
    fpocket_work.mkdir(parents=True, exist_ok=True)

    fpocket_cache: dict[str, FpocketResult] = {}
    detail_rows: list[dict] = []
    missing_pdbs: list[str] = []

    for _, row in labels.iterrows():
        name = str(row["Name"])
        pdb_id = pdb_id_from_name(name)
        chain = normalize_chain(row["Chain"])
        resnum = normalize_resnum(row["ResNum"])
        residue = normalize_resname(row["Residue"])

        if pdb_id not in fpocket_cache:
            pdb_path = find_pdb_file(pdb_id, pdb_dir, recursive=recursive_pdb)
            if pdb_path is None:
                missing_pdbs.append(pdb_id)
                fpocket_cache[pdb_id] = FpocketResult(
                    pdb_id, fpocket_work / f"{pdb_id.lower()}_out", [], error="PDB not found"
                )
            else:
                fpocket_cache[pdb_id] = run_fpocket(
                    pdb_path, pdb_id, fpocket_work, fpocket_bin, force_fpocket
                )

        fpocket = fpocket_cache[pdb_id]
        analysis = analyze_label(
            fpocket,
            chain,
            resnum,
            residue,
            include_top3=include_top3 or include_fix,
            include_fix=include_fix,
        )
        detail_row = {
            "Name": name,
            "Residue": residue,
            "ResNum": resnum,
            "Chain": chain,
            "target_residue_type": str(row["target_residue_type"]).upper(),
            "covsite_hit_at_top_pct": int(row["hit_at_top_pct"]),
            **analysis,
        }
        if include_fix and "n_residues" in row:
            detail_row["n_residues"] = row["n_residues"]
        detail_rows.append(detail_row)

    detail = pd.DataFrame(detail_rows)
    if include_fix:
        detail = add_search_space_reduction_columns(detail)
    summary = build_summary(detail, include_top3=include_top3 or include_fix, fix_mode=include_fix)

    cov_hit = detail["covsite_hit_at_top_pct"].astype(int)
    fpocket_found_top1 = subset_detail(
        detail,
        (detail["fpocket_top1_hit"] == 1) & (cov_hit == 0),
        "top1",
    )
    fpocket_found_overall = subset_detail(
        detail,
        (detail["fpocket_overall_hit"] == 1) & (cov_hit == 0),
        "overall",
    )
    covsite_found_top1 = subset_detail(
        detail,
        (cov_hit == 1) & (detail["fpocket_top1_hit"] == 0),
        "top1",
    )
    covsite_found_overall = subset_detail(
        detail,
        (cov_hit == 1) & (detail["fpocket_overall_hit"] == 0),
        "overall",
    )

    detail_path = out_path / f"{prefix}_fpocket_detail.csv"
    summary_path = out_path / f"{prefix}_fpocket_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    fpocket_found_top1.to_csv(out_path / f"{prefix}_fpocket_found_top1.csv", index=False)
    fpocket_found_overall.to_csv(
        out_path / f"{prefix}_fpocket_found_overall.csv", index=False
    )
    covsite_found_top1.to_csv(out_path / f"{prefix}_covsite_found_top1.csv", index=False)
    covsite_found_overall.to_csv(
        out_path / f"{prefix}_covsite_found_overall.csv", index=False
    )

    print(f"Labels analyzed: {len(detail)}")
    print(f"Unique PDBs: {len(fpocket_cache)}")
    print(
        f"fpocket top-1 hit rate: {detail['fpocket_top1_hit'].mean():.3f}  "
        f"overall: {detail['fpocket_overall_hit'].mean():.3f}"
    )
    if include_top3 and "fpocket_top3_hit" in detail.columns:
        print(f"fpocket top-3 hit rate: {detail['fpocket_top3_hit'].mean():.3f}")
    print(
        f"CovSite hit@top-pct: {detail['covsite_hit_at_top_pct'].mean():.3f}"
    )
    print(f"fpocket-only (top1 / overall): {len(fpocket_found_top1)} / {len(fpocket_found_overall)}")
    print(f"covsite-only (top1 / overall): {len(covsite_found_top1)} / {len(covsite_found_overall)}")
    if missing_pdbs:
        print(f"Missing PDBs ({len(set(missing_pdbs))}): {', '.join(sorted(set(missing_pdbs))[:10])}"
              + (" ..." if len(set(missing_pdbs)) > 10 else ""))
    print(f"Wrote: {detail_path}")
    print(f"Wrote: {summary_path}")


def _resolve_summary_csv(detail_csv: str, summary_csv: Optional[str]) -> str:
    detail_path = Path(detail_csv)
    if summary_csv:
        return summary_csv
    if detail_path.stem.endswith("_fpocket_detail"):
        return str(
            detail_path.with_name(
                detail_path.stem.replace("_fpocket_detail", "_fpocket_summary") + ".csv"
            )
        )
    return str(detail_path.with_name(detail_path.stem + "_summary.csv"))


def postprocess_fix(
    detail_csv: str,
    summary_csv: str,
    labels_csv: str,
    work_dir: Optional[str],
) -> None:
    """Apply --fix metrics to existing detail/summary CSVs."""
    detail_path = Path(detail_csv)
    summary_path = Path(summary_csv)
    if not detail_path.is_file():
        sys.exit(f"[ERROR] Detail CSV not found: {detail_path}")

    work = Path(work_dir) if work_dir else detail_path.parent / "fpocket_runs"
    if not work.is_dir():
        sys.exit(
            f"[ERROR] fpocket cache not found: {work}\n"
            "Pass --work-dir pointing to the fpocket_runs folder."
        )

    print(f"[INFO] Loading {detail_path}")
    detail = pd.read_csv(detail_path)
    print(f"[INFO] Applying --fix enrichment from {work}")
    detail = enrich_detail_with_fix(detail, work, labels_csv)

    summary = build_summary(detail, include_top3=True, fix_mode=True)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Labels analyzed: {len(detail)}")
    print(
        f"all-pockets nucleophilic union (all labels): "
        f"mean={detail['fpocket_all_pockets_nucleophilic_count'].mean():.2f}  "
        f"median={detail['fpocket_all_pockets_nucleophilic_count'].median():.1f}"
    )
    print(
        f"search-space reduction all pockets (when_hit): "
        f"mean={detail.loc[detail['fpocket_overall_hit'] == 1, 'fpocket_ss_red_all_pockets'].mean():.3f}"
    )
    print(f"Wrote: {detail_path}")
    print(f"Wrote: {summary_path}")


def postprocess_top3(
    detail_csv: str,
    summary_csv: str,
    work_dir: Optional[str],
) -> None:
    """Enrich an existing detail CSV with top-3 metrics and refresh the summary."""
    detail_path = Path(detail_csv)
    summary_path = Path(summary_csv)
    if not detail_path.is_file():
        sys.exit(f"[ERROR] Detail CSV not found: {detail_path}")

    work = Path(work_dir) if work_dir else detail_path.parent / "fpocket_runs"
    if not work.is_dir():
        sys.exit(
            f"[ERROR] fpocket cache not found: {work}\n"
            "Pass --work-dir pointing to the fpocket_runs folder."
        )

    print(f"[INFO] Loading {detail_path}")
    detail = pd.read_csv(detail_path)
    print(f"[INFO] Enriching top-3 metrics from {work}")
    detail = enrich_detail_with_top3(detail, work)

    summary = build_summary(detail, include_top3=True, fix_mode=False)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Labels analyzed: {len(detail)}")
    print(f"fpocket top-3 hit rate: {detail['fpocket_top3_hit'].mean():.3f}")
    print(
        f"top-3 nucleophilic union (all labels): "
        f"mean={detail['fpocket_top3_nucleophilic_count'].mean():.2f}  "
        f"median={detail['fpocket_top3_nucleophilic_count'].median():.1f}"
    )
    print(f"Wrote: {detail_path}")
    print(f"Wrote: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fpocket on label PDBs and compare against CovSite results."
    )
    parser.add_argument(
        "labels_csv",
        nargs="?",
        default=None,
        help="Input CSV (e.g. test_label_results.csv); optional with --top-3-flag post-process",
    )
    parser.add_argument("--pdb-dir", default=None, help="Directory containing PDB files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same directory as labels CSV)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: labels CSV stem)",
    )
    parser.add_argument(
        "--fpocket-bin",
        default="fpocket",
        help="fpocket executable (default: fpocket)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for fpocket run artifacts (default: <output-dir>/fpocket_runs)",
    )
    parser.add_argument(
        "--recursive-pdb",
        action="store_true",
        help="Search PDB directory recursively",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run fpocket even if cached output exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N label rows (for testing)",
    )
    parser.add_argument(
        "--top-3-flag",
        action="store_true",
        help=(
            "Include top-3 pocket metrics (hit in ranks 1-3; nucleophilic union "
            "across top 3 pockets). With --detail-csv, post-process existing outputs."
        ),
    )
    parser.add_argument(
        "--detail-csv",
        default=None,
        help="Existing detail CSV to enrich (use with --top-3-flag)",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Summary CSV to update (post-process with --top-3-flag or --fix)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Use all-pocket nucleophilic union for overall counts; add n_residues "
            "baseline and search-space reduction (top1/top3/all pockets / n_residues). "
            "With --detail-csv, post-process existing outputs (requires labels CSV)."
        ),
    )
    parser.add_argument(
        "--top-1-analysis",
        action="store_true",
        help=(
            "Compare fpocket rank-1 pocket descriptors (hits) vs target-pocket "
            "descriptors when fpocket misses rank-1 but CovSite hits and fpocket "
            "finds the site overall. Requires --detail-csv."
        ),
    )
    parser.add_argument(
        "--covsite-found-top1-csv",
        default=None,
        help=(
            "Optional covsite_found_top1 export to restrict recovered group "
            "(only rows with fpocket_overall_hit==1)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fix and args.detail_csv:
        if not args.labels_csv:
            sys.exit("[ERROR] --labels-csv is required with --fix --detail-csv (for n_residues)")
        postprocess_fix(
            detail_csv=args.detail_csv,
            summary_csv=_resolve_summary_csv(args.detail_csv, args.summary_csv),
            labels_csv=args.labels_csv,
            work_dir=args.work_dir,
        )
        return

    if args.top_3_flag and args.detail_csv:
        postprocess_top3(
            detail_csv=args.detail_csv,
            summary_csv=_resolve_summary_csv(args.detail_csv, args.summary_csv),
            work_dir=args.work_dir,
        )
        return

    if args.top_1_analysis:
        if not args.detail_csv:
            sys.exit("[ERROR] --top-1-analysis requires --detail-csv")
        run_top1_analysis(
            detail_csv=args.detail_csv,
            output_dir=args.output_dir,
            prefix=args.prefix,
            covsite_found_top1_csv=args.covsite_found_top1_csv,
        )
        return

    if not args.labels_csv:
        sys.exit(
            "[ERROR] labels_csv is required unless using --fix, --top-3-flag, "
            "or --top-1-analysis with --detail-csv"
        )
    labels_path = Path(args.labels_csv)
    if not labels_path.is_file():
        sys.exit(f"[ERROR] Labels CSV not found: {labels_path}")
    if not args.pdb_dir or not os.path.isdir(args.pdb_dir):
        sys.exit(f"[ERROR] PDB directory not found: {args.pdb_dir}")

    output_dir = args.output_dir or str(labels_path.parent)
    prefix = args.prefix or labels_path.stem

    run_analysis(
        labels_csv=str(labels_path),
        pdb_dir=args.pdb_dir,
        output_dir=output_dir,
        prefix=prefix,
        fpocket_bin=args.fpocket_bin,
        work_dir=args.work_dir,
        recursive_pdb=args.recursive_pdb,
        force_fpocket=args.force,
        limit=args.limit,
        include_top3=args.top_3_flag or args.fix,
        include_fix=args.fix,
    )


if __name__ == "__main__":
    main()
