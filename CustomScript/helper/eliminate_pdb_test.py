#!/usr/bin/env python3
"""Filter rows out of a CSV when the same protein pdb appears in a second CSV."""

import argparse
import os
import sys
import json
import time
import urllib.request

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = []
PDB_COLUMN_CANDIDATES = ["protein pdb", "pdb", "protein_pdb", "Proteins"]


def read_csv_with_validation(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Basic validation performed by caller depending on context
    return df


def find_pdb_column(df: pd.DataFrame) -> str:
    # Case-insensitive search for common PDB column names
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in PDB_COLUMN_CANDIDATES:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def build_pdb_series(df: pd.DataFrame, col: str = None) -> pd.Series:
    # Determine which PDB-ID column to use
    if col and col in df.columns:
        use = col
    else:
        found = find_pdb_column(df)
        if not found:
            raise ValueError(f"No PDB column found; expected one of {PDB_COLUMN_CANDIDATES}")
        use = found
    return df[use].astype(str).str.strip().str.upper()



def filter_rows(primary_csv: str, secondary_csv: str, output_csv: str) -> tuple[int, int]:
    """
    Simpler algorithm per user request:
    - Use secondary's PDB column (e.g., 'protein pdb') to find matching rows in primary by PDB.
    - For each matching primary row, take its Proteins value (UniProt id(s)).
    - Remove from primary any row whose Proteins contains any of those UniProt ids.

    Requires primary to have a Proteins column (case-insensitive).
    """
    primary_df = read_csv_with_validation(primary_csv)
    secondary_df = read_csv_with_validation(secondary_csv)

    # locate pdb columns
    sec_pdb_col = find_pdb_column(secondary_df)
    pri_pdb_col = find_pdb_column(primary_df)
    if not sec_pdb_col:
        raise ValueError(f"Secondary CSV must contain one of PDB columns: {PDB_COLUMN_CANDIDATES}")
    if not pri_pdb_col:
        raise ValueError(f"Primary CSV must contain one of PDB columns: {PDB_COLUMN_CANDIDATES}")

    # locate proteins column in primary (case-insensitive)
    proteins_col = None
    for c in primary_df.columns:
        if c.lower() == "proteins" or c.lower() in ["protein", "uniprot", "uniprot_id"]:
            proteins_col = c
            break
    if proteins_col is None:
        raise ValueError("Primary CSV must contain a Proteins column with UniProt ids")

    # Build normalized series
    sec_pdbs = set(build_pdb_series(secondary_df, col=sec_pdb_col).tolist())
    pri_pdb_series = build_pdb_series(primary_df, col=pri_pdb_col)

    # Find primary rows that match any secondary PDB
    matched_primary_idxs = primary_df[pri_pdb_series.isin(sec_pdbs)].index.tolist()

    # Collect UniProt ids from matched primary rows
    proteins_to_remove = set()
    for idx in matched_primary_idxs:
        val = str(primary_df.at[idx, proteins_col])
        if not val:
            continue
        for acc in [p.strip() for p in val.split(",") if p.strip()]:
            proteins_to_remove.add(acc)

    # If we found no proteins to remove, fall back to removing rows matching PDB directly
    if not proteins_to_remove:
        keep_mask = ~pri_pdb_series.isin(sec_pdbs)
    else:
        def row_keep_proteins(val: str) -> bool:
            if not isinstance(val, str) or not val:
                return True
            for acc in [p.strip() for p in val.split(",") if p.strip()]:
                if acc in proteins_to_remove:
                    return False
            return True

        keep_mask = primary_df[proteins_col].apply(row_keep_proteins)

    filtered_df = primary_df.loc[keep_mask].copy()
    filtered_df.to_csv(output_csv, index=False)

    removed_count = int((~keep_mask).sum())
    kept_count = int(keep_mask.sum())
    return kept_count, removed_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove rows from a CSV when the same (name, protein pdb) pair appears in a second CSV.",
    )
    parser.add_argument("primary_csv", help="CSV to filter")
    parser.add_argument("secondary_csv", help="CSV containing rows to remove from the primary CSV")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <primary>_filtered.csv)",
    )
    args = parser.parse_args()

    primary_base, primary_ext = os.path.splitext(args.primary_csv)
    output_csv = args.output or f"{primary_base}_filtered{primary_ext or '.csv'}"

    try:
        kept_count, removed_count = filter_rows(args.primary_csv, args.secondary_csv, output_csv)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote filtered CSV to: {output_csv}")
    print(f"Kept rows: {kept_count}")
    print(f"Removed rows: {removed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
