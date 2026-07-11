"""
fix_warhead_smiles.py

Converts two protected/adduct warhead forms back to their active aldehyde:

  1. Bisulfite adduct  (Warhead == "Sulfonic acid")
       Pattern: C(O)S(=O)(=O)O  or  C(=O)S(=O)(=O)[O-]
       Replace terminal group with: C=O

  2. Hemiacetal        (Warhead == "Hemiacetal")
       Pattern: C(O)O  where the carbon also bears an additional O or ring O
       Uses RDKit to detect hemiacetal and cleave to aldehyde

For every modified row:
  - electrophile_smiles  : updated to free-aldehyde SMILES
  - Old Smile            : original SMILES
  - Warhead              : changed to "Aldehyde"
  - Frankenstein_Warhead : changed to "Aldehyde"
  - Warhead Classification Changed : "X"
  - Warhead Changed                : "X"

Modified rows are ALSO appended to rerun_incorrect.csv (never overwritten).

Usage:
    python fix_warhead_smiles.py \
        --input  batch_pdbs_bo_with_name.csv \
        --output batch_pdbs_bo_with_name.csv       # can overwrite in-place
        [--rerun rerun_incorrect.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# SMILES string-level conversions
# ---------------------------------------------------------------------------

# Bisulfite adduct patterns (both neutral and anionic form seen in data)
# Matches the terminal -C(O)S(=O)(=O)O  or  -C(=O)S(=O)(=O)[O-]
# (the =O variant appears in a few rows like 7JSU / 7LKV / 7LKW)
_BISULFITE_NEUTRAL  = re.compile(r'C\(O\)S\(=O\)\(=O\)O(?!\w)')   # C(O)S(=O)(=O)O
_BISULFITE_ANION    = re.compile(r'C\(=O\)S\(=O\)\(=O\)\[O-\]')   # C(=O)S(=O)(=O)[O-]  ← already =O so just strip sulfonyl
_BISULFITE_ANION2   = re.compile(r'C\(O\)S\(=O\)\(=O\)\[O-\]')
_BISULFITE_NEUTRAL2 = re.compile(r'C\(=O\)S\(=O\)\(=O\)O(?!\\w)')  # C(=O)S(=O)(=O)O — carbonyl-side bisulfite adduct    # C(O)S(=O)(=O)[O-]


def _convert_bisulfite(smi: str) -> str | None:
    """
    Replace bisulfite adduct terminal group with aldehyde C=O.
    Returns new SMILES or None if no pattern matched.
    """
    # C(O)S(=O)(=O)O  →  C=O
    if _BISULFITE_NEUTRAL.search(smi):
        return _BISULFITE_NEUTRAL.sub('C=O', smi)
    # C(O)S(=O)(=O)[O-]  →  C=O
    if _BISULFITE_ANION2.search(smi):
        return _BISULFITE_ANION2.sub('C=O', smi)
    # C(=O)S(=O)(=O)[O-]  →  C=O
    if _BISULFITE_ANION.search(smi):
        return _BISULFITE_ANION.sub('C=O', smi)
    # C(=O)S(=O)(=O)O  →  C=O  (carbonyl variant, neutral)
    if _BISULFITE_NEUTRAL2.search(smi):
        return _BISULFITE_NEUTRAL2.sub('C=O', smi)
    return None


# Hemiacetal patterns:
# General hemiacetal = sp3 carbon bearing -OH and -OR (or ring O)
# In SMILES this often appears as:
#   OC[...]O  (acyclic)  or  ring closures like C1(O)...O1
# We use a simple SMARTS-style string search; for robustness we also try RDKit.

def _convert_hemiacetal_smiles(smi: str) -> str | None:
    """
    Attempt to convert a hemiacetal SMILES to an aldehyde using RDKit.
    The hemiacetal carbon (C bonded to OH and OR) becomes C=O (aldehyde),
    and the OR oxygen is removed.
    Returns new canonical SMILES or None if conversion failed / RDKit unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors
    except ImportError:
        return _convert_hemiacetal_fallback(smi)

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # SMARTS for hemiacetal carbon: sp3 C with -OH and -O-C (ether oxygen)
    # [C;X4;H1,H0]([OH1])([OX2][#6])  — the carbon has an OH and an O-alkyl
    hemiacetal_smarts = Chem.MolFromSmarts('[C;X4]([OH1])([OX2][#6,#1])')
    if hemiacetal_smarts is None:
        return None

    matches = mol.GetSubstructMatches(hemiacetal_smarts)
    if not matches:
        return None

    # For the example given: OC[C@@H]1[C@@H](O)[C@H](O)[C@@H](F)[C@H](O1)O
    # The anomeric carbon (ring hemiacetal) is the C bonded to the ring O and the exo OH
    # We use an RW mol to edit it
    from rdkit.Chem import RWMol
    rw = RWMol(mol)

    # Take the first match — the hemiacetal carbon is index 0 in match tuple
    hemi_c_idx = matches[0][0]
    hemi_c = rw.GetAtomWithIdx(hemi_c_idx)

    # Find the -OH (will become =O) and the -OR (will be removed)
    oh_idx = None
    or_idx = None   # the ether oxygen atom index
    for neighbor in hemi_c.GetNeighbors():
        if neighbor.GetAtomicNum() == 8:
            # Check if it's an OH (has at least 1 H) or ether O (no H)
            if neighbor.GetTotalNumHs() >= 1 and oh_idx is None:
                oh_idx = neighbor.GetIdx()
            elif neighbor.GetTotalNumHs() == 0 and or_idx is None:
                or_idx = neighbor.GetIdx()

    if oh_idx is None or or_idx is None:
        return None

    # Change bond to OH from single → double (making it C=O aldehyde)
    rw.RemoveBond(hemi_c_idx, oh_idx)
    rw.AddBond(hemi_c_idx, oh_idx, Chem.BondType.DOUBLE)

    # Remove the OH hydrogen implicitly (RDKit handles valence)
    oh_atom = rw.GetAtomWithIdx(oh_idx)
    oh_atom.SetNumExplicitHs(0)
    oh_atom.SetNoImplicit(False)

    # Remove the ether oxygen and its bond (break the OR)
    or_atom = rw.GetAtomWithIdx(or_idx)
    # Get what the OR oxygen was bonded to (besides hemi_c)
    or_neighbors = [n.GetIdx() for n in or_atom.GetNeighbors() if n.GetIdx() != hemi_c_idx]
    rw.RemoveBond(hemi_c_idx, or_idx)

    # If the OR oxygen was a ring atom we need to be careful —
    # just remove the bond and let the ring open
    # Remove the now-dangling O atom if it has no other bonds
    try:
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception:
        return None


def _convert_hemiacetal_fallback(smi: str) -> str | None:
    """
    String-only fallback for hemiacetal → aldehyde.
    Handles the specific example: OC[C@@H]1[C@@H](O)[C@H](O)[C@@H](F)[C@H](O1)O
    The pattern is a pyranose-style ring hemiacetal ending in (O1)O.
    Replace terminal (O1)O with (O1)=O isn't right — instead we need to
    open the ring. This is chemistry-hard without RDKit, so we flag it.
    """
    return None


# ---------------------------------------------------------------------------
# Row-level processing
# ---------------------------------------------------------------------------

SMILES_COL   = "electrophile_smiles"
WARHEAD_COL  = "Warhead"
FRANK_COL    = "Frankenstein_Warhead"
OLD_SMILE_COL = "Old Smile"
WC_CHANGED_COL = "Warhead Classification Changed"
W_CHANGED_COL  = "Warhead Changed"

SULFONIC_ACID_LABELS = {"sulfonic acid", "sulfonic_acid"}
HEMIACETAL_LABELS    = {"hemiacetal"}


def _normalise(s: str) -> str:
    return s.strip().lower()


def process_row(row: dict) -> tuple[dict, bool]:
    """
    Inspect a row and apply warhead conversion if applicable.
    Returns (modified_row, was_changed).
    """
    warhead = _normalise(row.get(WARHEAD_COL, ""))
    smi = row.get(SMILES_COL, "").strip()
    changed = False
    new_smi = None

    if warhead in SULFONIC_ACID_LABELS:
        new_smi = _convert_bisulfite(smi)

    elif warhead in HEMIACETAL_LABELS:
        new_smi = _convert_hemiacetal_smiles(smi)

    if new_smi is not None and new_smi != smi:
        row = dict(row)  # don't mutate original
        row[OLD_SMILE_COL]    = smi
        row[SMILES_COL]       = new_smi
        row[WARHEAD_COL]      = "Aldehyde"
        row[FRANK_COL]        = "Aldehyde"
        row[WC_CHANGED_COL]   = "X"
        row[W_CHANGED_COL]    = "X"
        changed = True

    return row, changed


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_to_rerun(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Append rows to rerun CSV, writing header only if file doesn't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Convert bisulfite adduct / hemiacetal SMILES to free aldehyde.")
    parser.add_argument("--input",  required=True, help="Input labels CSV")
    parser.add_argument("--output", required=True, help="Output labels CSV (can be same as input)")
    parser.add_argument("--rerun",  default="rerun_incorrect.csv",
                        help="CSV to append changed rows to (default: rerun_incorrect.csv)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    rerun_path  = Path(args.rerun)

    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    fieldnames, rows = load_csv(input_path)

    # Ensure the new columns exist in fieldnames
    for col in [OLD_SMILE_COL, WC_CHANGED_COL, W_CHANGED_COL]:
        if col not in fieldnames:
            fieldnames.append(col)

    updated_rows = []
    changed_rows = []

    for row in rows:
        # Ensure new columns exist in every row dict
        for col in [OLD_SMILE_COL, WC_CHANGED_COL, W_CHANGED_COL]:
            row.setdefault(col, "")

        new_row, changed = process_row(row)
        updated_rows.append(new_row)
        if changed:
            changed_rows.append(new_row)

    write_csv(output_path, fieldnames, updated_rows)
    print(f"[INFO] Written {len(updated_rows)} rows to {output_path}")

    if changed_rows:
        append_to_rerun(rerun_path, fieldnames, changed_rows)
        print(f"[INFO] Appended {len(changed_rows)} changed rows to {rerun_path}")

        print(f"\n[SUMMARY] Rows modified:")
        for r in changed_rows:
            print(f"  {r.get('Name','?'):10}  {r.get(OLD_SMILE_COL,'')[:60]}  →  {r.get(SMILES_COL,'')[:40]}")
    else:
        print("[INFO] No rows required conversion.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())