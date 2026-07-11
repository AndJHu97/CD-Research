#!/usr/bin/env python3
"""
Merge parallel add_residue_info.py batch outputs into one CSV.

Each batch file is expected to contain the full row set (same length as the
original training CSV). Only the processed slice has residue-info columns
filled; other rows are empty/NaN. This script copies filled residue columns
into a single merged output.

Usage:
    # Auto-detect filled rows per file (non-null residue columns)
    python merge_csv_runs.py -o training_extended.csv batch_0.csv batch_1.csv batch_2.csv

    # Explicit row slices (0-based, end exclusive) — matches --row-start/--row-end
    python merge_csv_runs.py -o training_extended.csv \\
        --slice 0 500000 batch_0.csv \\
        --slice 500000 1000000 batch_1.csv

    # Custom residue columns
    python merge_csv_runs.py -o out.csv --columns DCFS,Hydrophobic_Fraction_Residue batch_*.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Columns added by add_residue_info.py
DEFAULT_RESIDUE_COLUMNS = (
    "DCFS",
    "Hydrophobic_Fraction_Residue",
    "electrostatic_charge_residue",
    "H_Bond_Donor_Residue_Count",
    "H_Bond_Acceptor_Residue_Count",
)

IDENTITY_COLUMNS = ("Name", "Residue", "ResNum", "Chain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge parallel add_residue_info batch CSV outputs."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path for merged output CSV",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Batch CSV files (used in auto-detect mode if no --slice)",
    )
    parser.add_argument(
        "--slice",
        action="append",
        nargs=3,
        metavar=("START", "END", "CSV"),
        help=(
            "Explicit row slice for a batch file (0-based, end exclusive). "
            "Repeat for each batch."
        ),
    )
    parser.add_argument(
        "--columns",
        default=None,
        help=(
            "Comma-separated residue columns to merge "
            f"(default: {','.join(DEFAULT_RESIDUE_COLUMNS)})"
        ),
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base CSV for non-residue columns (default: first input/slice file)",
    )
    parser.add_argument(
        "--on-conflict",
        choices=("error", "warn", "last"),
        default="error",
        help="Behavior when two batches fill the same row (default: error)",
    )
    parser.add_argument(
        "--check-identity",
        action="store_true",
        default=True,
        help="Verify Name/Residue/ResNum/Chain match across inputs (default: on)",
    )
    parser.add_argument(
        "--no-check-identity",
        action="store_false",
        dest="check_identity",
        help="Skip identity-column checks",
    )
    return parser.parse_args()


def parse_residue_columns(text: str | None) -> list[str]:
    if not text:
        return list(DEFAULT_RESIDUE_COLUMNS)
    return [c.strip() for c in text.split(",") if c.strip()]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        sys.exit(f"[ERROR] File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def row_has_residue_data(row: pd.Series, residue_cols: list[str]) -> bool:
    for col in residue_cols:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def filled_row_indices(df: pd.DataFrame, residue_cols: list[str]) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    present_cols = [c for c in residue_cols if c in df.columns]
    if not present_cols:
        return mask
    for col in present_cols:
        col_vals = df[col]
        non_null = col_vals.notna()
        if col_vals.dtype == object:
            non_empty = col_vals.astype(str).str.strip() != ""
            non_null = non_null & non_empty
        mask |= non_null.to_numpy()
    return mask


def check_identity_columns(base: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    shared = [c for c in IDENTITY_COLUMNS if c in base.columns and c in other.columns]
    if not shared:
        return
    mismatches = 0
    for col in shared:
        left = base[col].astype(str).fillna("")
        right = other[col].astype(str).fillna("")
        mismatches += int((left != right).sum())
    if mismatches:
        sys.exit(
            f"[ERROR] Identity mismatch vs base in {label}: "
            f"{mismatches} cell(s) differ on {shared}"
        )


def apply_slice(
    merged: pd.DataFrame,
    batch: pd.DataFrame,
    start: int,
    end: int,
    residue_cols: list[str],
    source_label: str,
    filled_by: np.ndarray,
    on_conflict: str,
) -> int:
    n = len(merged)
    if len(batch) != n:
        sys.exit(
            f"[ERROR] {source_label}: row count {len(batch)} != base {n}"
        )
    if start < 0 or end > n or start >= end:
        sys.exit(
            f"[ERROR] Invalid slice [{start}:{end}) for {source_label} (n={n})"
        )

    copied = 0
    for i in range(start, end):
        if not row_has_residue_data(batch.iloc[i], residue_cols):
            continue
        if filled_by[i]:
            msg = (
                f"Row {i} already filled; {source_label} also has data "
                f"in slice [{start}:{end})"
            )
            if on_conflict == "error":
                sys.exit(f"[ERROR] {msg}")
            if on_conflict == "warn":
                print(f"[WARN] {msg}; keeping first value", file=sys.stderr)
                continue
        for col in residue_cols:
            if col in batch.columns:
                merged.at[i, col] = batch.at[i, col]
        filled_by[i] = True
        copied += 1
    return copied


def apply_auto(
    merged: pd.DataFrame,
    batch: pd.DataFrame,
    residue_cols: list[str],
    source_label: str,
    filled_by: np.ndarray,
    on_conflict: str,
) -> int:
    n = len(merged)
    if len(batch) != n:
        sys.exit(
            f"[ERROR] {source_label}: row count {len(batch)} != base {n}"
        )

    mask = filled_row_indices(batch, residue_cols)
    copied = 0
    for i in np.flatnonzero(mask):
        if filled_by[i]:
            msg = f"Row {i} already filled; {source_label} also has residue data"
            if on_conflict == "error":
                sys.exit(f"[ERROR] {msg}")
            if on_conflict == "warn":
                print(f"[WARN] {msg}; keeping first value", file=sys.stderr)
                continue
        for col in residue_cols:
            if col in batch.columns:
                merged.at[i, col] = batch.at[i, col]
        filled_by[i] = True
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    residue_cols = parse_residue_columns(args.columns)

    if args.slice:
        slices: list[tuple[int, int, Path]] = []
        for start_s, end_s, csv_path in args.slice:
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                sys.exit(f"[ERROR] Invalid slice bounds: {start_s} {end_s}")
            slices.append((start, end, Path(csv_path)))
        if not slices:
            sys.exit("[ERROR] No --slice entries provided")
        base_path = Path(args.base) if args.base else slices[0][2]
    elif args.inputs:
        slices = None
        input_paths = [Path(p) for p in args.inputs]
        base_path = Path(args.base) if args.base else input_paths[0]
    else:
        sys.exit("[ERROR] Provide batch CSV files or --slice START END CSV entries")

    base = load_csv(base_path)
    merged = base.copy()
    for col in residue_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    filled_by = np.zeros(len(merged), dtype=bool)
    total_copied = 0

    if slices is not None:
        check_identity = args.check_identity
        if check_identity:
            for _, _, path in slices[1:]:
                check_identity_columns(base, load_csv(path), str(path))
        for start, end, path in slices:
            batch = load_csv(path)
            if path != base_path and args.check_identity:
                check_identity_columns(base, batch, str(path))
            n = apply_slice(
                merged,
                batch,
                start,
                end,
                residue_cols,
                str(path),
                filled_by,
                args.on_conflict,
            )
            print(f"[INFO] {path.name} [{start}:{end}): copied {n} filled row(s)")
            total_copied += n
    else:
        for path in input_paths:
            if path == base_path:
                continue
            if args.check_identity:
                check_identity_columns(base, load_csv(path), str(path))
        for path in input_paths:
            batch = load_csv(path)
            if path != base_path and args.check_identity:
                check_identity_columns(base, batch, str(path))
            n = apply_auto(
                merged,
                batch,
                residue_cols,
                str(path),
                filled_by,
                args.on_conflict,
            )
            print(f"[INFO] {path.name}: copied {n} filled row(s)")
            total_copied += n

    out_path = Path(args.output)
    merged.to_csv(out_path, index=False)

    n_filled = int(filled_by.sum())
    n_empty = len(merged) - n_filled
    print(f"[INFO] Wrote {len(merged):,} rows to {out_path}")
    print(f"[INFO] Rows with merged residue data: {n_filled:,}")
    print(f"[INFO] Rows still empty: {n_empty:,}")
    if n_empty:
        print(
            "[WARN] Some rows have no residue data in any batch; "
            "check slices or re-run missing ranges",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
