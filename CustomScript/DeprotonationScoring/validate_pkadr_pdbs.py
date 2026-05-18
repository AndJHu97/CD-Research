#!/usr/bin/env python3
"""
validate_pkadr_pdbs.py

Check that each row in a PKAD-R CSV refers to a residue that exists in the corresponding PDB file.

Usage example:
    python validate_pkadr_pdbs.py path/to/pkadr.csv --pdb-dir /path/to/pdbs

Exits with code 0 when all rows validate; exits with code 2 when any missing.
"""
import argparse
import os
import re
import sys
import urllib.request
from typing import Tuple

import pandas as pd

# Map modified residue types to their parent types
MODIFIED_RESIDUES = {
    # Cysteine variants
    "CSD": "CYS",  # cysteinesulfinic acid
    "CSO": "CYS",  # S-hydroxycysteine
    "CME": "CYS",  # S,S-(2-hydroxyethyl)thiodisulfide
    "CSX": "CYS",  # oxidized cysteine
    "CSS": "CYS",  # S-sulfonylated cysteine
    "CYM": "CYS",  # deprotonated cysteine
    # Serine variants
    "SEP": "SER",  # phosphoserine
    # Threonine variants
    "TPO": "THR",  # phosphothreonine
    # Tyrosine variants
    "PTR": "TYR",  # O-phosphotyrosine
    "TYS": "TYR",  # sulfotyrosine
    # Histidine variants
    "HID": "HIS",  # protonated histidine
    "HIE": "HIS",  # neutral histidine
    "HIP": "HIS",  # double protonated histidine
    # Lysine variants
    "MLY": "LYS",  # N-dimethyl-lysine
    "M3L": "LYS",  # N-trimethyl-lysine
    # Aspartate variants
    "ASX": "ASP",  # protonated aspartate
    # Glutamate variants
    "GLX": "GLU",  # protonated glutamate
}


def find_pdb_path(pdb_id_or_path: str, pdb_dir: str) -> str:
    # If user provided a full path, use it directly.
    if os.path.isfile(pdb_id_or_path):
        return pdb_id_or_path

    # Strip optional chain suffix like '1EX3.A' -> '1EX3'
    pdb_id = str(pdb_id_or_path).lower().replace(".pdb", "").split(".")[0]

    # Check direct candidate in pdb_dir
    candidate = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if os.path.isfile(candidate):
        return candidate

    # Walk pdb_dir to find files that match pdb id in filename (case-insensitive)
    for root, dirs, files in os.walk(pdb_dir):
        for fn in files:
            name, ext = os.path.splitext(fn)
            if ext.lower() not in (".pdb", ".ent", ""):
                continue
            if name.lower() == pdb_id or name.lower().startswith(pdb_id):
                return os.path.join(root, fn)

    # Not found locally — try to download into pdb_dir using lowercase name
    os.makedirs(pdb_dir, exist_ok=True)
    dest = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        print(f"[pdb] Downloading {pdb_id} from RCSB to {dest}...")
        urllib.request.urlretrieve(url, dest)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[pdb] Saved to {dest}")
            return dest
    except Exception:
        pass

    raise FileNotFoundError(f"PDB file not found for '{pdb_id_or_path}' in {pdb_dir}")


def parse_pkadr_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    colmap = {c.lower(): c for c in df.columns}
    pdb_col = colmap.get("pdb", None)
    chain_col = colmap.get("chain", None)
    resid_col = colmap.get("resid in pdb", None) or colmap.get("resid", None) or colmap.get("resnum", None)
    resname_col = colmap.get("resname", None) or colmap.get("res name", None)
    pka_col = colmap.get("expt. pka", None) or colmap.get("expt pka", None) or colmap.get("pka", None)

    missing = [name for name, col in [
        ("PDB", pdb_col),
        ("Chain", chain_col),
        ("ResID", resid_col),
        ("ResName", resname_col),
        ("Expt. pKa", pka_col),
    ] if col is None]

    if missing:
        raise ValueError(f"Missing required columns in PKAD-R CSV: {', '.join(missing)}")

    return pdb_col, chain_col, resid_col, resname_col, pka_col


def parse_pdb_residue_line(line: str) -> Tuple[str, str, str]:
    # Returns (chain, resseq_str, resname)
    # PDB fixed-column parsing
    if len(line) < 26:
        return ("", "", "")
    resname = line[17:20].strip()
    chain = line[21].strip()
    resseq_str = line[22:26].strip()
    icode = line[26].strip() if len(line) > 26 else ""
    if icode:
        reskey = f"{resseq_str}{icode}"
    else:
        reskey = resseq_str
    return chain, reskey, resname


def collect_pdb_residues(pdb_path: str):
    residues = {}
    with open(pdb_path, "r") as fh:
        for ln in fh:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            chain, reskey, resname = parse_pdb_residue_line(ln)
            if not reskey:
                continue
            k = (chain or "", reskey)
            residues.setdefault(k, set()).add(resname.upper())
    return residues


def main():
    parser = argparse.ArgumentParser(description="Validate PKAD-R CSV rows against PDB files.")
    parser.add_argument("pkadr_csv", help="Path to PKAD-R CSV")
    parser.add_argument("--pdb-dir", default=".", help="Directory containing PDB files")
    parser.add_argument("--allow-resname-mismatch", action="store_true", help="Treat resname mismatches as warnings rather than missing")
    parser.add_argument("--missing-out", help="Write unique PDB paths (one per line) that lack the referenced residue")
    parser.add_argument("--out-csv", help="Write a filtered CSV excluding rows with missing residues")
    parser.add_argument("--exclude-pdb-missing", action="store_true", help="Also exclude rows where the PDB file could not be found")
    args = parser.parse_args()

    df_input = pd.read_csv(args.pkadr_csv)
    df = df_input.copy()
    # preserve original row indices so we can write a filtered CSV from the original frame
    df["_orig_index"] = df.index
    pdb_col, chain_col, resid_col, resname_col, _ = parse_pkadr_columns(df)

    df = df.rename(columns={
        pdb_col: "pdb",
        chain_col: "chain",
        resid_col: "resseq",
        resname_col: "resname",
    })

    df = df.dropna(subset=["pdb", "chain", "resseq", "resname"]) 

    missing_rows = []
    resname_mismatch = []

    pdb_cache = {}

    for idx, row in df.iterrows():
        pdb_id = str(row["pdb"]).strip()
        chain = str(row["chain"]).strip()
        resseq_raw = str(row["resseq"]).strip()
        resname = str(row["resname"]).strip().upper()

        try:
            pdb_path = find_pdb_path(pdb_id, args.pdb_dir)
        except FileNotFoundError:
            missing_rows.append((idx, None, pdb_id, chain, resseq_raw, resname, "PDB not found"))
            continue

        if pdb_path not in pdb_cache:
            pdb_cache[pdb_path] = collect_pdb_residues(pdb_path)

        residues = pdb_cache[pdb_path]

        # Attempt integer match first (common case)
        matched = False
        key_candidates = []
        try:
            # allow csv resseq to be int-like
            ival = int(re.sub(r"[^0-9]", "", resseq_raw))
            key_candidates.append((chain, str(ival)))
        except Exception:
            pass
        # also try raw string (handles insertion codes like 100A)
        key_candidates.append((chain, resseq_raw))

        for k in key_candidates:
            if k in residues:
                matched = True
                found_resnames = residues[k]
                # Check if CSV resname or any modified variant of it is in the PDB
                is_valid = False
                is_valid = resname in found_resnames
                if not is_valid:
                    resname_mismatch.append((idx, pdb_id, chain, resseq_raw, resname, sorted(found_resnames)))
                break

        if not matched:
            missing_rows.append((idx, pdb_path, pdb_id, chain, resseq_raw, resname, "Residue not found"))

    # Print results
    if missing_rows:
        print("\nERROR: Some CSV rows reference residues not present in PDB files:")
        for r in missing_rows:
            idx, pdb_path, pdb_id, chain, resseq_raw, resname, reason = r
            src = pdb_path if pdb_path else pdb_id
            print(f" Row {idx}: PDB={src} Chain={chain} Res={resname}{resseq_raw} -> {reason}")
        # print/write unique pdbs that lacked the residue (only include cases where PDB was available)
        missing_pdbs = sorted({p for (i, p, pid, c, rs, rn, rc) in missing_rows if p and rc == "Residue not found"})
        if missing_pdbs:
            print("\nPDB files missing referenced residues:")
            for p in missing_pdbs:
                print(" -", p)
            if args.missing_out:
                try:
                    with open(args.missing_out, "w") as outfh:
                        for p in missing_pdbs:
                            outfh.write(p + "\n")
                    print(f"Wrote missing PDB list to {args.missing_out}")
                except Exception as exc:
                    print(f"Failed to write missing-out file: {exc}")
    else:
        print("All CSV rows have corresponding residues in PDB files.")

    # Optionally write a filtered CSV excluding rows with missing residues.
    if args.out_csv:
        # Determine which original df_input indices to drop.
        # Determine which original df_input indices to drop.
        drop_idxs = set()

        # Remove rows with missing residues
        for (idx, pdb_path, pdb_id, chain, resseq_raw, resname, reason) in missing_rows:
            if reason == "Residue not found":
                drop_idxs.add(idx)
            elif reason == "PDB not found" and args.exclude_pdb_missing:
                drop_idxs.add(idx)

        # Also remove rows with residue-name mismatches
        for (idx, pdb_id, chain, resseq_raw, csv_resname, found) in resname_mismatch:
            drop_idxs.add(idx)

        if drop_idxs:
            out_df = df_input.drop(index=list(drop_idxs))
        else:
            out_df = df_input

        try:
            out_df.to_csv(args.out_csv, index=False)
            print(f"Wrote filtered CSV to {args.out_csv} "
                f"(removed {len(df_input) - len(out_df)} rows)")
        except Exception as exc:
            print(f"Failed to write out-csv: {exc}")
    else:
        print("All CSV rows have corresponding residues in PDB files.")

    if resname_mismatch:
        print("\nWARNING: Resname mismatches (CSV vs PDB):")
        for r in resname_mismatch:
            idx, pdb_id, chain, resseq_raw, csv_resname, found = r
            print(f" Row {idx}: PDB={pdb_id} Chain={chain} Res={resseq_raw} CSV={csv_resname} PDB_FOUND={found}")

    if missing_rows:
        sys.exit(2)


if __name__ == "__main__":
    main()
