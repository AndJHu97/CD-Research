#!/usr/bin/env python3
"""
Replace Frankenstein_Warhead in a CovSite/Cov_Screen labels CSV using values
from a PROTACS-style input CSV keyed by protein PDB ID.

For each protacs row's protein PDB, every labels row whose PDB_ID matches
(or whose Name starts with that PDB id) gets Frankenstein_Warhead set to the
protacs Frankenstein_Warhead text exactly as written.

Example:
  python convert_warhead_labels.py \\
      --protacs ../evaluation/inputs/protacs_input.csv \\
      --labels "/path/to/labels_covdb.csv" \\
      --output labels_covdb_warheads_updated.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy Frankenstein_Warhead from a protacs input CSV onto matching "
            "rows of a labels CSV by protein PDB."
        )
    )
    parser.add_argument(
        "--protacs",
        required=True,
        help="PROTACS/input CSV with protein pdb + Frankenstein_Warhead",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Labels CSV with Name / PDB_ID / Frankenstein_Warhead",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output labels CSV path "
            "(default: <labels stem>_warheads_updated.csv beside labels)"
        ),
    )
    parser.add_argument(
        "--pdb-column",
        default=None,
        help="Protacs PDB column override (default: auto-detect)",
    )
    parser.add_argument(
        "--frankenstein-column",
        default=None,
        help="Protacs Frankenstein_Warhead column override (default: auto-detect)",
    )
    return parser.parse_args()


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: tuple[str, ...],
    required: bool,
) -> str | None:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    if explicit:
        matched = lookup.get(str(explicit).strip().lower())
        if matched is None:
            sys.exit(
                f"[ERROR] Column {explicit!r} not found. "
                f"Available: {list(df.columns)}"
            )
        return str(matched)
    for candidate in candidates:
        if candidate in lookup:
            return str(lookup[candidate])
    if required:
        sys.exit(
            f"[ERROR] Could not find any of {candidates}. "
            f"Available: {list(df.columns)}"
        )
    return None


def normalize_pdb(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return Path(text).stem.upper()


def pdb_from_label_row(row: pd.Series) -> str:
    if "PDB_ID" in row.index:
        pdb = normalize_pdb(row["PDB_ID"])
        if pdb:
            return pdb
    name = "" if pd.isna(row.get("Name")) else str(row["Name"]).strip()
    if not name:
        return ""
    return name.split("-", 1)[0].replace(".pdb", "").split(".")[0].upper()


def build_pdb_warhead_map(
    protacs: pd.DataFrame,
    pdb_col: str,
    frank_col: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    conflicts: list[str] = []
    for _, row in protacs.iterrows():
        pdb = normalize_pdb(row[pdb_col])
        frank = row[frank_col]
        if not pdb:
            continue
        if pd.isna(frank) or not str(frank).strip():
            continue
        frank_text = str(frank).strip()
        if pdb in mapping and mapping[pdb] != frank_text:
            conflicts.append(
                f"{pdb}: {mapping[pdb]!r} vs {frank_text!r}"
            )
            continue
        mapping[pdb] = frank_text
    if conflicts:
        print(
            "[WARN] Multiple distinct Frankenstein_Warhead values for the "
            "same PDB; keeping the first:"
        )
        for item in conflicts[:20]:
            print(f"  {item}")
        if len(conflicts) > 20:
            print(f"  ... ({len(conflicts) - 20} more)")
    return mapping


def main() -> None:
    args = parse_args()
    protacs_path = Path(args.protacs).resolve()
    labels_path = Path(args.labels).resolve()
    if not protacs_path.is_file():
        sys.exit(f"[ERROR] Protacs CSV not found: {protacs_path}")
    if not labels_path.is_file():
        sys.exit(f"[ERROR] Labels CSV not found: {labels_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else labels_path.with_name(f"{labels_path.stem}_warheads_updated.csv")
    )

    protacs = pd.read_csv(protacs_path, low_memory=False)
    labels = pd.read_csv(labels_path, low_memory=False)

    pdb_col = detect_column(
        protacs,
        args.pdb_column,
        ("protein pdb", "pdb", "pdb_id", "protein_pdb"),
        True,
    )
    frank_col = detect_column(
        protacs,
        args.frankenstein_column,
        ("frankenstein_warhead", "frankenstein warhead"),
        True,
    )
    if "Name" not in labels.columns and "PDB_ID" not in labels.columns:
        sys.exit("[ERROR] Labels CSV needs Name and/or PDB_ID")
    if "Frankenstein_Warhead" not in labels.columns:
        labels["Frankenstein_Warhead"] = ""

    mapping = build_pdb_warhead_map(protacs, pdb_col, frank_col)
    if not mapping:
        sys.exit("[ERROR] No usable protein pdb / Frankenstein_Warhead pairs found")

    out = labels.copy()
    label_pdbs = out.apply(pdb_from_label_row, axis=1)
    matched_mask = label_pdbs.map(lambda pdb: pdb in mapping)
    before = out.loc[matched_mask, "Frankenstein_Warhead"].astype(str)
    out.loc[matched_mask, "Frankenstein_Warhead"] = label_pdbs[matched_mask].map(
        mapping
    )
    after = out.loc[matched_mask, "Frankenstein_Warhead"].astype(str)
    n_changed = int((before != after).sum())
    matched_pdbs = sorted(set(label_pdbs[matched_mask]))
    missing_pdbs = sorted(set(mapping) - set(label_pdbs))

    out.to_csv(output_path, index=False)

    print(f"[INFO] Protacs PDBs with warheads : {len(mapping):,}")
    print(f"[INFO] Labels rows matched        : {int(matched_mask.sum()):,}")
    print(f"[INFO] Unique matched PDBs        : {len(matched_pdbs):,}")
    print(f"[INFO] Frankenstein_Warhead edits : {n_changed:,}")
    print(f"[INFO] Protacs PDBs not in labels : {len(missing_pdbs):,}")
    if missing_pdbs:
        preview = ", ".join(missing_pdbs[:15])
        suffix = " ..." if len(missing_pdbs) > 15 else ""
        print(f"         {preview}{suffix}")
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
