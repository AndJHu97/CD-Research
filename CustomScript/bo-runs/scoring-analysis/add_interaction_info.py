#!/usr/bin/env python3
"""
Add protein–ligand interaction-fit columns from existing residue and ligand features.

Expects columns produced by add_residue_info.py and add_ligand_info.py:
  Residue: DCFS, Hydrophobic_Fraction_Residue, electrostatic_charge_residue,
           H_Bond_Donor_Residue_Count, H_Bond_Acceptor_Residue_Count
  Ligand:  D_L_Ligand, LogP_Ligand, H_Bond_Donor_Ligand_Count,
           H_Bond_Acceptor_Ligand_Count, Formal_Charge_Ligand

Adds:
  - Geo_Fit            = DCFS / D_L_Ligand
  - Hydrophobic_Fit    = LogP_Ligand * Hydrophobic_Fraction_Residue
  - Hydrogen_DPAL_Fit  = H_Bond_Donor_Residue_Count * H_Bond_Acceptor_Ligand_Count
  - Hydrogen_APDL_Fit  = H_Bond_Acceptor_Residue_Count * H_Bond_Donor_Ligand_Count
  - Electro_Fit        = Formal_Charge_Ligand * electrostatic_charge_residue

If any input term for a row is missing, that interaction column is left blank (NaN).

Usage:
    python add_interaction_info.py training_bo_extended_ligand.csv
    python add_interaction_info.py training.csv --output training_with_interactions.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

RESIDUE_COLUMNS = (
    "DCFS",
    "Hydrophobic_Fraction_Residue",
    "electrostatic_charge_residue",
    "H_Bond_Donor_Residue_Count",
    "H_Bond_Acceptor_Residue_Count",
)

LIGAND_COLUMNS = (
    "D_L_Ligand",
    "LogP_Ligand",
    "H_Bond_Donor_Ligand_Count",
    "H_Bond_Acceptor_Ligand_Count",
    "Formal_Charge_Ligand",
)

NEW_COLUMNS = (
    "Geo_Fit",
    "Hydrophobic_Fit",
    "Hydrogen_DPAL_Fit",
    "Hydrogen_APDL_Fit",
    "Electro_Fit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add interaction-fit columns from residue and ligand descriptor columns."
        )
    )
    parser.add_argument("training_csv", help="Input CSV with residue and ligand columns")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: overwrite input)",
    )
    return parser.parse_args()


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        sys.exit(f"[ERROR] Required column not found: {column}")
    return pd.to_numeric(df[column], errors="coerce")


def _product(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise product; NaN if either operand is missing."""
    return a * b


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio; NaN if either operand is missing or denominator is zero."""
    out = numerator / denominator
    out = out.where(denominator.notna() & (denominator != 0))
    out = out.where(numerator.notna())
    return out


def add_interaction_info(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = [c for c in (*RESIDUE_COLUMNS, *LIGAND_COLUMNS) if c not in df.columns]
    if missing:
        sys.exit(
            "[ERROR] Missing required input columns:\n  "
            + "\n  ".join(missing)
        )

    out = df.copy()

    dcfs = _numeric_series(out, "DCFS")
    d_l = _numeric_series(out, "D_L_Ligand")
    hydro_frac = _numeric_series(out, "Hydrophobic_Fraction_Residue")
    logp = _numeric_series(out, "LogP_Ligand")
    donor_res = _numeric_series(out, "H_Bond_Donor_Residue_Count")
    acceptor_res = _numeric_series(out, "H_Bond_Acceptor_Residue_Count")
    donor_lig = _numeric_series(out, "H_Bond_Donor_Ligand_Count")
    acceptor_lig = _numeric_series(out, "H_Bond_Acceptor_Ligand_Count")
    q_site = _numeric_series(out, "electrostatic_charge_residue")
    q_lig = _numeric_series(out, "Formal_Charge_Ligand")

    out["Geo_Fit"] = _ratio(dcfs, d_l)
    out["Hydrophobic_Fit"] = _product(logp, hydro_frac)
    out["Hydrogen_DPAL_Fit"] = _product(donor_res, acceptor_lig)
    out["Hydrogen_APDL_Fit"] = _product(acceptor_res, donor_lig)
    out["Electro_Fit"] = _product(q_lig, q_site)

    filled = {col: int(out[col].notna().sum()) for col in NEW_COLUMNS}
    return out, filled


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading {args.training_csv}")
    df = pd.read_csv(args.training_csv, low_memory=False)
    print(f"[INFO] {len(df):,} rows, {len(df.columns)} columns")

    out, filled = add_interaction_info(df)

    output_path = args.output or args.training_csv
    out.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {len(out):,} rows to {output_path}")

    for col in NEW_COLUMNS:
        print(f"[INFO] {col}: {filled[col]:,}/{len(out):,} rows filled")


if __name__ == "__main__":
    main()
