#!/usr/bin/env python3
"""
Add an N_Terminal column to a training CSV by checking PDB files.

For each row, the PDB ID is taken from the Name column (text before the first
hyphen, e.g. 1BMQ-MN -> 1BMQ). N-terminal detection uses a tiered strategy:

  1. DBREF begResSeq (author numbering; handles missing N-term coordinates)
  2. First ATOM residue (fallback when DBREF is absent)
  3. SEQRES is used to validate chain/residue assignments when a Residue column
     is present

Usage:
    python adding_N_terminal.py training.csv --pdb-dir /path/to/pdbs
    python adding_N_terminal.py training.csv --pdb-dir /path/to/pdbs --output labeled.csv
    python adding_N_terminal.py training.csv --pdb-dir /path/to/pdbs --recursive
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

NAME_COLUMN_CANDIDATES = ("name",)
CHAIN_COLUMN_CANDIDATES = ("chain",)
RESNUM_COLUMN_CANDIDATES = ("resnum", "res_num", "residue_number", "residue_num")
RESIDUE_COLUMN_CANDIDATES = ("residue", "resname", "aa", "amino_acid")


@dataclass
class PdbNtermMetadata:
    """Per-chain N-terminal metadata parsed from a PDB file."""

    dbref_start: dict[str, int] = field(default_factory=dict)
    atom_start: dict[str, int] = field(default_factory=dict)
    seqres: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class NtermResult:
    n_terminal: int
    method: str
    n_term_resnum: Optional[int]
    residue_mismatch: bool = False


def find_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    explicit: Optional[str],
    required: bool = True,
) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            sys.exit(
                f"[ERROR] Column '{explicit}' not found in CSV.\n"
                f"Available: {list(df.columns)}"
            )
        return explicit

    colmap = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in colmap:
            return colmap[candidate]

    if required:
        sys.exit(
            f"[ERROR] Could not detect required column. "
            f"Tried {candidates}. Available: {list(df.columns)}"
        )
    return None


def extract_pdb_id(name: object) -> str:
    text = str(name).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("-", 1)[0].strip()


def normalize_chain(chain: object) -> str:
    text = str(chain).strip()
    if not text or text.lower() == "nan":
        return " "
    return text[0] if len(text) == 1 else text


def normalize_resnum(resnum: object) -> Optional[int]:
    if pd.isna(resnum):
        return None
    text = str(resnum).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_residue_name(residue: object) -> str:
    if pd.isna(residue):
        return ""
    return str(residue).strip().upper()[:3]


def find_pdb_file(pdb_id: str, pdb_dir: str, recursive: bool = False) -> Optional[str]:
    """Return the path to a PDB file matching pdb_id (case-insensitive)."""
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
                name, ext = os.path.splitext(fname)
                if ext.lower() not in {".pdb", ".ent"}:
                    continue
                if name.lower() == pdb_id_lower:
                    return os.path.join(root, fname)
        except FileNotFoundError:
            continue

    return None


def _parse_dbref_line(line: str, dbref_start: dict[str, int]) -> None:
    if len(line) < 22:
        return
    chain = line[12:13].strip() or line[12]
    if not chain:
        chain = " "
    beg_text = line[14:18].strip()
    if not beg_text:
        return
    try:
        beg = int(beg_text)
    except ValueError:
        return
    if chain not in dbref_start or beg < dbref_start[chain]:
        dbref_start[chain] = beg


def _parse_seqres_line(line: str, seqres: dict[str, list[str]]) -> None:
    if len(line) < 20:
        return
    chain = line[11:13].strip() or line[11]
    if not chain:
        chain = " "
    parts = line[13:].split()
    if len(parts) < 2:
        return
    residues = [token.upper() for token in parts[1:] if len(token) == 3 and token.isalpha()]
    if not residues:
        return
    seqres.setdefault(chain, []).extend(residues)


def parse_pdb_nterm_metadata(pdb_path: str) -> PdbNtermMetadata:
    """Parse DBREF, SEQRES, and first-ATOM records from a PDB file."""
    metadata = PdbNtermMetadata()
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            record = line[:6]
            if record in {"DBREF ", "SDBREF"}:
                _parse_dbref_line(line, metadata.dbref_start)
            elif record == "SEQRES":
                _parse_seqres_line(line, metadata.seqres)
            elif record == "ATOM  ":
                chain = line[21]
                if chain not in metadata.atom_start:
                    resnum = normalize_resnum(line[22:26].strip())
                    if resnum is not None:
                        metadata.atom_start[chain] = resnum
    return metadata


def get_nterm_for_chain(chain: str, metadata: PdbNtermMetadata) -> tuple[Optional[int], str]:
    if chain in metadata.dbref_start:
        return metadata.dbref_start[chain], "dbref"
    if chain in metadata.atom_start:
        return metadata.atom_start[chain], "atom"
    return None, "missing"


def classify_n_terminal(
    chain: object,
    resnum: object,
    residue: Optional[object],
    metadata: PdbNtermMetadata,
) -> NtermResult:
    chain_key = normalize_chain(chain)
    row_resnum = normalize_resnum(resnum)
    if row_resnum is None:
        return NtermResult(n_terminal=0, method="missing", n_term_resnum=None)

    n_term_resnum, method = get_nterm_for_chain(chain_key, metadata)
    if n_term_resnum is None:
        return NtermResult(n_terminal=0, method="missing", n_term_resnum=None)

    n_terminal = int(row_resnum == n_term_resnum)
    residue_mismatch = False

    if residue is not None and chain_key in metadata.seqres:
        seq_index = row_resnum - n_term_resnum
        seq = metadata.seqres[chain_key]
        if 0 <= seq_index < len(seq):
            expected = seq[seq_index].upper()
            observed = normalize_residue_name(residue)
            if observed and expected and observed != expected:
                residue_mismatch = True

    return NtermResult(
        n_terminal=n_terminal,
        method=method,
        n_term_resnum=n_term_resnum,
        residue_mismatch=residue_mismatch,
    )


def add_n_terminal_column(
    df: pd.DataFrame,
    pdb_dir: str,
    name_col: str,
    chain_col: str,
    resnum_col: str,
    residue_col: Optional[str],
    recursive: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    pdb_cache: dict[str, Optional[PdbNtermMetadata]] = {}
    missing_pdbs: list[str] = []
    residue_warnings: list[str] = []
    n_terminal: list[int] = []
    methods: list[str] = []
    n_term_resnums: list[object] = []

    for row_idx, row in df.iterrows():
        pdb_id = extract_pdb_id(row[name_col])
        if not pdb_id:
            n_terminal.append(0)
            methods.append("missing")
            n_term_resnums.append(pd.NA)
            continue

        if pdb_id not in pdb_cache:
            pdb_path = find_pdb_file(pdb_id, pdb_dir, recursive=recursive)
            if pdb_path is None:
                missing_pdbs.append(pdb_id)
                pdb_cache[pdb_id] = None
            else:
                pdb_cache[pdb_id] = parse_pdb_nterm_metadata(pdb_path)

        metadata = pdb_cache[pdb_id]
        if metadata is None:
            n_terminal.append(0)
            methods.append("missing")
            n_term_resnums.append(pd.NA)
            continue

        residue_value = row[residue_col] if residue_col else None
        result = classify_n_terminal(
            chain=row[chain_col],
            resnum=row[resnum_col],
            residue=residue_value,
            metadata=metadata,
        )
        n_terminal.append(result.n_terminal)
        methods.append(result.method)
        n_term_resnums.append(result.n_term_resnum if result.n_term_resnum is not None else pd.NA)

        if result.residue_mismatch:
            residue_warnings.append(
                f"row {row_idx}: {pdb_id} chain {row[chain_col]} "
                f"resnum {row[resnum_col]} residue {residue_value} "
                f"does not match SEQRES"
            )

    out = df.copy()
    out["N_Terminal"] = n_terminal
    out["N_Terminal_method"] = methods
    out["N_Terminal_resnum"] = n_term_resnums
    return out, sorted(set(missing_pdbs)), residue_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add N_Terminal (1/0) to a training CSV using DBREF numbering "
            "with SEQRES validation and first-ATOM fallback."
        )
    )
    parser.add_argument("training_csv", help="Input training CSV path")
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="Directory containing PDB files named like 1BMQ.pdb",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input_stem>_N_terminal.csv)",
    )
    parser.add_argument("--name-column", default=None, help="Name column (default: Name)")
    parser.add_argument("--chain-column", default=None, help="Chain column (default: Chain)")
    parser.add_argument("--resnum-column", default=None, help="ResNum column (default: ResNum)")
    parser.add_argument(
        "--residue-column",
        default=None,
        help="Residue column for SEQRES validation (default: auto-detect Residue)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PDB files in subdirectories of --pdb-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_csv = os.path.abspath(args.training_csv)
    pdb_dir = os.path.abspath(args.pdb_dir)

    if not os.path.isfile(training_csv):
        sys.exit(f"[ERROR] Training CSV not found: {training_csv}")
    if not os.path.isdir(pdb_dir):
        sys.exit(f"[ERROR] PDB directory not found: {pdb_dir}")

    try:
        df = pd.read_csv(training_csv)
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to read CSV '{training_csv}': {exc}")

    name_col = find_column(df, NAME_COLUMN_CANDIDATES, args.name_column)
    chain_col = find_column(df, CHAIN_COLUMN_CANDIDATES, args.chain_column)
    resnum_col = find_column(df, RESNUM_COLUMN_CANDIDATES, args.resnum_column)
    residue_col = find_column(
        df,
        RESIDUE_COLUMN_CANDIDATES,
        args.residue_column,
        required=False,
    )

    for col in ("N_Terminal", "N_Terminal_method", "N_Terminal_resnum"):
        if col in df.columns:
            print(f"[WARN] Replacing existing {col} column.")

    labeled_df, missing_pdbs, residue_warnings = add_n_terminal_column(
        df=df,
        pdb_dir=pdb_dir,
        name_col=name_col,
        chain_col=chain_col,
        resnum_col=resnum_col,
        residue_col=residue_col,
        recursive=args.recursive,
    )

    output_path = (
        os.path.abspath(args.output)
        if args.output
        else os.path.join(
            os.path.dirname(training_csv),
            f"{os.path.splitext(os.path.basename(training_csv))[0]}_N_terminal.csv",
        )
    )

    labeled_df.to_csv(output_path, index=False)

    n_terminal_count = int(labeled_df["N_Terminal"].sum())
    method_counts = labeled_df["N_Terminal_method"].value_counts().to_dict()

    print("\n[INFO] N_TERMINAL SUMMARY")
    print(f"  Input CSV         : {training_csv}")
    print(f"  PDB directory     : {pdb_dir}")
    print(f"  Rows processed    : {len(labeled_df)}")
    print(f"  N_Terminal = 1    : {n_terminal_count}")
    print(f"  N_Terminal = 0    : {len(labeled_df) - n_terminal_count}")
    print("  Methods used      :")
    for method, count in sorted(method_counts.items()):
        print(f"    - {method}: {count}")
    if residue_col:
        print(f"  SEQRES validation : enabled via column '{residue_col}'")
    else:
        print("  SEQRES validation : skipped (no Residue column found)")
    if missing_pdbs:
        print(f"  Missing PDB files : {len(missing_pdbs)}")
        for pdb_id in missing_pdbs[:20]:
            print(f"    - {pdb_id}")
        if len(missing_pdbs) > 20:
            print(f"    ... and {len(missing_pdbs) - 20} more")
    if residue_warnings:
        print(f"  Residue mismatches: {len(residue_warnings)}")
        for warning in residue_warnings[:20]:
            print(f"    - {warning}")
        if len(residue_warnings) > 20:
            print(f"    ... and {len(residue_warnings) - 20} more")
    print(f"\n[SUCCESS] Wrote {output_path}")


if __name__ == "__main__":
    main()
