#!/usr/bin/env python3
"""
Count cysteine and total amino acid residues in PDB files and report averages.

Usage:
    python avg_cysteine.py --pdb-dir /path/to/pdbs
    python avg_cysteine.py --pdb-dir /path/to/pdbs --csv structures.csv
    python avg_cysteine.py --pdb-dir /path/to/pdbs --csv structures.csv --output cys_counts.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

import pandas as pd

CYS_RESIDUES = frozenset({"CYS"})

PDB_COLUMN_CANDIDATES = (
    "pdb",
    "pdb_id",
    "pdbid",
    "structure",
    "structure_id",
    "id",
)


def find_pdb_file(pdb_id: str, pdb_dir: str, recursive: bool = False) -> Optional[str]:
    """Return the path to a PDB file matching pdb_id (case-insensitive)."""
    pdb_id = str(pdb_id).strip().replace(".pdb", "").split(".")[0]
    if not pdb_id:
        return None

    pdb_id_lower = pdb_id.lower()
    direct = os.path.join(pdb_dir, f"{pdb_id_lower}.pdb")
    if os.path.isfile(direct):
        return direct

    direct_upper = os.path.join(pdb_dir, f"{pdb_id.upper()}.pdb")
    if os.path.isfile(direct_upper):
        return direct_upper

    search_roots = [pdb_dir]
    if recursive:
        search_roots = []
        for root, _, _ in os.walk(pdb_dir):
            search_roots.append(root)

    for root in search_roots:
        try:
            for fname in os.listdir(root):
                name, ext = os.path.splitext(fname)
                if ext.lower() not in {".pdb", ".ent"}:
                    continue
                if name.lower() == pdb_id_lower:
                    return os.path.join(root, fname)
        except FileNotFoundError:
            continue

    return None


def list_pdb_files(pdb_dir: str, recursive: bool = False) -> list[str]:
    """Return sorted PDB file paths found in pdb_dir."""
    paths: list[str] = []
    if recursive:
        for root, _, files in os.walk(pdb_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in {".pdb", ".ent"}:
                    paths.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(pdb_dir):
            if os.path.splitext(fname)[1].lower() in {".pdb", ".ent"}:
                paths.append(os.path.join(pdb_dir, fname))
    return sorted(paths)


def count_residues(pdb_path: str) -> tuple[int, int]:
    """
    Count unique residues from ATOM records.

    Returns (n_cysteine, n_total_amino_acids).
    """
    all_residues: set[tuple[str, str, str]] = set()
    cys_residues: set[tuple[str, str, str]] = set()
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21]
            res_seq = line[22:26].strip()
            i_code = line[26]
            key = (chain, res_seq, i_code)
            all_residues.add(key)
            if resname in CYS_RESIDUES:
                cys_residues.add(key)
    return len(cys_residues), len(all_residues)


def detect_pdb_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            sys.exit(
                f"[ERROR] Column '{requested}' not found in CSV.\n"
                f"Available: {list(df.columns)}"
            )
        return requested

    colmap = {c.lower(): c for c in df.columns}
    for candidate in PDB_COLUMN_CANDIDATES:
        if candidate in colmap:
            return colmap[candidate]

    sys.exit(
        "[ERROR] Could not detect a PDB column in CSV. "
        f"Pass --pdb-column explicitly. Available: {list(df.columns)}"
    )


def load_pdb_ids_from_csv(csv_path: str, pdb_column: Optional[str]) -> list[str]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to read CSV '{csv_path}': {exc}")

    column = detect_pdb_column(df, pdb_column)
    pdb_ids = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    if not pdb_ids:
        sys.exit(f"[ERROR] No PDB IDs found in column '{column}' of {csv_path}")
    return pdb_ids


def pdb_id_from_path(pdb_path: str) -> str:
    return os.path.splitext(os.path.basename(pdb_path))[0]


def analyze_pdbs(
    pdb_paths: Iterable[str],
) -> tuple[pd.DataFrame, float, float]:
    rows: list[dict[str, object]] = []
    for pdb_path in pdb_paths:
        pdb_id = pdb_id_from_path(pdb_path)
        try:
            n_cys, n_residues = count_residues(pdb_path)
            rows.append(
                {
                    "pdb": pdb_id,
                    "pdb_path": pdb_path,
                    "n_cysteine": n_cys,
                    "n_residues": n_residues,
                    "status": "ok",
                }
            )
        except OSError as exc:
            rows.append(
                {
                    "pdb": pdb_id,
                    "pdb_path": pdb_path,
                    "n_cysteine": pd.NA,
                    "n_residues": pd.NA,
                    "status": f"read_error: {exc}",
                }
            )

    result_df = pd.DataFrame(rows)
    ok_df = result_df[result_df["status"] == "ok"]
    avg_cys = float(ok_df["n_cysteine"].mean()) if not ok_df.empty else float("nan")
    avg_residues = float(ok_df["n_residues"].mean()) if not ok_df.empty else float("nan")
    return result_df, avg_cys, avg_residues


def resolve_pdb_paths(
    pdb_dir: str,
    csv_path: Optional[str],
    pdb_column: Optional[str],
    recursive: bool,
) -> tuple[list[str], list[str]]:
    missing: list[str] = []

    if csv_path:
        pdb_ids = load_pdb_ids_from_csv(csv_path, pdb_column)
        paths: list[str] = []
        for pdb_id in pdb_ids:
            path = find_pdb_file(pdb_id, pdb_dir, recursive=recursive)
            if path is None:
                missing.append(pdb_id)
            else:
                paths.append(path)
        return paths, missing

    return list_pdb_files(pdb_dir, recursive=recursive), missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count cysteine and total amino acid residues in PDB files and "
            "report averages across structures."
        )
    )
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="Directory containing PDB files",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Optional CSV listing PDB IDs to analyze. "
            "If omitted, all PDB files in --pdb-dir are used."
        ),
    )
    parser.add_argument(
        "--pdb-column",
        default=None,
        help=(
            "Column in --csv with PDB IDs (auto-detected if omitted: "
            "pdb, pdb_id, structure, id, ...)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write per-structure residue counts as CSV",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PDB files in subdirectories of --pdb-dir",
    )
    args = parser.parse_args()

    pdb_dir = os.path.abspath(args.pdb_dir)
    if not os.path.isdir(pdb_dir):
        sys.exit(f"[ERROR] PDB directory not found: {pdb_dir}")

    pdb_paths, missing = resolve_pdb_paths(
        pdb_dir=pdb_dir,
        csv_path=args.csv,
        pdb_column=args.pdb_column,
        recursive=args.recursive,
    )

    if missing:
        print(f"[WARN] {len(missing)} PDB file(s) not found in {pdb_dir}:")
        for pdb_id in missing[:20]:
            print(f"  - {pdb_id}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if not pdb_paths:
        sys.exit("[ERROR] No PDB files to analyze.")

    result_df, avg_cys, avg_residues = analyze_pdbs(pdb_paths)

    print("\n[INFO] RESIDUE SUMMARY")
    print(f"  PDB directory     : {pdb_dir}")
    print(f"  Structures parsed : {int((result_df['status'] == 'ok').sum())}")
    print(f"  Structures failed : {int((result_df['status'] != 'ok').sum())}")
    print(f"  Missing from dir  : {len(missing)}")
    if pd.notna(avg_cys):
        ok_df = result_df[result_df["status"] == "ok"]
        print(f"  Average cysteines : {avg_cys:.2f}")
        print(f"  Total cysteines   : {int(ok_df['n_cysteine'].sum())}")
        print(f"  Average residues  : {avg_residues:.2f}")
        print(f"  Total residues    : {int(ok_df['n_residues'].sum())}")
    else:
        print("  Average cysteines : n/a (no structures parsed successfully)")
        print("  Average residues  : n/a (no structures parsed successfully)")

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Per-structure counts -> {args.output}")


if __name__ == "__main__":
    main()
