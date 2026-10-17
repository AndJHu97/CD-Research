#!/usr/bin/env python3
"""
Extend a training CSV with ligand descriptors derived from electrophile SMILES.

SMILES are resolved from a labels CSV (one row per Name) using:
  1. electrophile_smiles (default column), when present
  2. LigID -> RCSB ideal ligand SDF (e.g. MNO from https://www.rcsb.org/ligand/MNO)
  3. Ligand suffix parsed from Name (text after the first hyphen) if LigID is absent

Each unique Name in the labels file covers all training rows sharing that Name.
Computed values are cached per Name.

Adds columns (when missing):
  - D_L_Ligand
  - LogP_Ligand
  - H_Bond_Donor_Ligand_Count
  - H_Bond_Acceptor_Ligand_Count
  - Formal_Charge_Ligand
  - TPSA_Ligand
  - Flatness_Ligand

Usage:
    python add_ligand_info.py training.csv labels_eval.csv
    python add_ligand_info.py training.csv labels_eval.csv --output training_with_ligand.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Descriptors3D, Lipinski
except ImportError:
    sys.exit(
        "[ERROR] RDKit is required. Install with: conda install -c conda-forge rdkit"
    )

# ── Defaults ─────────────────────────────────────────────────────────────────

N_CONFORMERS = 10
EMBED_SEED = 42

RCSB_SDF_URLS = (
    "https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf",
    "https://files.rcsb.org/ligands/download/{ligand_id}.sdf",
    "https://files.rcsb.org/ligands/view/{ligand_id}_ideal.sdf",
)

NEW_COLUMNS = (
    "D_L_Ligand",
    "LogP_Ligand",
    "H_Bond_Donor_Ligand_Count",
    "H_Bond_Acceptor_Ligand_Count",
    "Formal_Charge_Ligand",
    "TPSA_Ligand",
    "Flatness_Ligand",
)

SMILES_COLUMN_CANDIDATES = ("electrophile_smiles", "electrophile smiles", "smiles")
LIGID_COLUMN_CANDIDATES = ("ligid", "lig_id", "ligand_id")
NAME_COLUMN_CANDIDATES = ("name",)


@dataclass(frozen=True)
class LigandMetrics:
    d_l: float
    logp: float
    hbond_donor_count: int
    hbond_acceptor_count: int
    formal_charge: int
    tpsa: float
    flatness: float


# ── CSV helpers ──────────────────────────────────────────────────────────────

def find_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    explicit: Optional[str],
    required: bool = True,
) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            sys.exit(
                f"[ERROR] Column '{explicit}' not found.\n"
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


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def extract_ligand_id_from_name(name: str) -> str:
    if "-" not in name:
        return ""
    return name.split("-", 1)[1].strip()


# ── SMILES resolution ────────────────────────────────────────────────────────

def download_rcsb_ligand_sdf(ligand_id: str, cache_dir: str) -> Optional[str]:
    ligand_id = ligand_id.upper().strip()
    if not ligand_id:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{ligand_id}_ideal.sdf")

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 50:
        return out_path

    for template in RCSB_SDF_URLS:
        url = template.format(ligand_id=ligand_id)
        try:
            urllib.request.urlretrieve(url, out_path)
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 50:
                return out_path
        except Exception:
            continue
        if os.path.isfile(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
    return None


def smiles_from_sdf(sdf_path: str) -> Optional[str]:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in supplier:
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(mol)
    return None


def fetch_smiles_from_ligid(ligand_id: str, cache_dir: str) -> Optional[str]:
    sdf_path = download_rcsb_ligand_sdf(ligand_id, cache_dir)
    if sdf_path is None:
        return None
    return smiles_from_sdf(sdf_path)


def resolve_smiles_for_label_row(
    row: pd.Series,
    smiles_col: Optional[str],
    ligid_col: Optional[str],
    name_col: str,
    cache_dir: str,
) -> tuple[Optional[str], str]:
    if smiles_col:
        smiles = normalize_text(row[smiles_col])
        if smiles:
            return smiles, "electrophile_smiles"

    ligand_id = normalize_text(row[ligid_col]) if ligid_col else ""
    if not ligand_id:
        ligand_id = extract_ligand_id_from_name(normalize_text(row[name_col]))

    if not ligand_id:
        return None, "missing"

    smiles = fetch_smiles_from_ligid(ligand_id, cache_dir)
    if smiles:
        return smiles, f"rcsb:{ligand_id.upper()}"
    return None, f"rcsb_failed:{ligand_id.upper()}"


def build_name_smiles_lookup(
    labels_df: pd.DataFrame,
    smiles_col: Optional[str],
    ligid_col: Optional[str],
    name_col: str,
    cache_dir: str,
) -> tuple[dict[str, str], list[str]]:
    lookup: dict[str, str] = {}
    warnings: list[str] = []

    for _, row in labels_df.iterrows():
        name = normalize_text(row[name_col])
        if not name:
            continue
        if name in lookup:
            continue

        smiles, source = resolve_smiles_for_label_row(
            row, smiles_col, ligid_col, name_col, cache_dir
        )
        if smiles:
            lookup[name] = smiles
        else:
            warnings.append(f"{name}: could not resolve SMILES ({source})")

    return lookup, warnings


# ── Descriptor calculations ──────────────────────────────────────────────────

def max_pairwise_distance(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 0.0
    diffs = coords[:, None, :] - coords[None, :, :]
    return float(np.sqrt((diffs * diffs).sum(axis=2)).max())


def compute_2d_metrics(mol: Chem.Mol) -> dict[str, float | int]:
    return {
        "logp": float(Descriptors.MolLogP(mol)),
        "hbond_donor_count": int(Lipinski.NumHDonors(mol)),
        "hbond_acceptor_count": int(Lipinski.NumHAcceptors(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
    }


def heavy_atom_coords(mol_h: Chem.Mol, conf_id: int) -> np.ndarray:
    heavy_indices = [atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy_indices:
        return np.empty((0, 3), dtype=float)
    conf = mol_h.GetConformer(conf_id)
    return np.array([list(conf.GetAtomPosition(idx)) for idx in heavy_indices], dtype=float)


def compute_3d_metrics(
    mol: Chem.Mol,
    n_conformers: int = N_CONFORMERS,
    seed: int = EMBED_SEED,
) -> tuple[float, float]:
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed

    conf_ids = list(
        AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params)
    )
    if not conf_ids:
        status = AllChem.EmbedMolecule(mol_h, params)
        if status != 0:
            return float("nan"), float("nan")
        conf_ids = [0]

    dimensions: list[float] = []
    flatness_values: list[float] = []

    for conf_id in conf_ids:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)
        except Exception:
            pass

        coords = heavy_atom_coords(mol_h, conf_id)
        if coords.shape[0] >= 2:
            dimensions.append(max_pairwise_distance(coords))

        try:
            flatness_values.append(float(Descriptors3D.NPR2(mol_h, confId=conf_id)))
        except Exception:
            continue

    d_l = float(np.median(dimensions)) if dimensions else float("nan")
    flatness = float(np.median(flatness_values)) if flatness_values else float("nan")
    return d_l, flatness


def compute_ligand_metrics(
    smiles: str,
    n_conformers: int = N_CONFORMERS,
    seed: int = EMBED_SEED,
) -> Optional[LigandMetrics]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    metrics_2d = compute_2d_metrics(mol)
    d_l, flatness = compute_3d_metrics(mol, n_conformers=n_conformers, seed=seed)

    return LigandMetrics(
        d_l=d_l,
        logp=float(metrics_2d["logp"]),
        hbond_donor_count=int(metrics_2d["hbond_donor_count"]),
        hbond_acceptor_count=int(metrics_2d["hbond_acceptor_count"]),
        formal_charge=int(metrics_2d["formal_charge"]),
        tpsa=float(metrics_2d["tpsa"]),
        flatness=flatness,
    )


# ── Main pipeline ────────────────────────────────────────────────────────────

def add_ligand_info(
    training_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    smiles_col: Optional[str] = None,
    ligid_col: Optional[str] = None,
    training_name_col: Optional[str] = None,
    labels_name_col: Optional[str] = None,
    cache_dir: Optional[str] = None,
    n_conformers: int = N_CONFORMERS,
    embed_seed: int = EMBED_SEED,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    train_name_col = find_column(training_df, NAME_COLUMN_CANDIDATES, training_name_col)
    label_name_col = find_column(labels_df, NAME_COLUMN_CANDIDATES, labels_name_col)
    label_smiles_col = find_column(
        labels_df, SMILES_COLUMN_CANDIDATES, smiles_col, required=False
    )
    label_ligid_col = find_column(
        labels_df, LIGID_COLUMN_CANDIDATES, ligid_col, required=False
    )

    if label_smiles_col is None and label_ligid_col is None:
        sys.exit(
            "[ERROR] Labels CSV must contain electrophile_smiles and/or LigID "
            "(or provide --ligid-column)."
        )

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="ligand_sdf_cache_")

    name_smiles, resolve_warnings = build_name_smiles_lookup(
        labels_df,
        label_smiles_col,
        label_ligid_col,
        label_name_col,
        cache_dir,
    )

    metrics_cache: dict[str, LigandMetrics] = {}
    compute_errors: list[str] = []

    out = training_df.copy()
    for col in NEW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    for row_idx, row in out.iterrows():
        name = normalize_text(row[train_name_col])
        if not name:
            continue

        if name not in name_smiles:
            continue

        if name not in metrics_cache:
            metrics = compute_ligand_metrics(
                name_smiles[name],
                n_conformers=n_conformers,
                seed=embed_seed,
            )
            if metrics is None:
                compute_errors.append(f"{name}: invalid SMILES after resolution")
                continue
            metrics_cache[name] = metrics

        metrics = metrics_cache[name]
        out.at[row_idx, "D_L_Ligand"] = metrics.d_l
        out.at[row_idx, "LogP_Ligand"] = metrics.logp
        out.at[row_idx, "H_Bond_Donor_Ligand_Count"] = metrics.hbond_donor_count
        out.at[row_idx, "H_Bond_Acceptor_Ligand_Count"] = metrics.hbond_acceptor_count
        out.at[row_idx, "Formal_Charge_Ligand"] = metrics.formal_charge
        out.at[row_idx, "TPSA_Ligand"] = metrics.tpsa
        out.at[row_idx, "Flatness_Ligand"] = metrics.flatness

    return out, resolve_warnings, compute_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add ligand descriptor columns to a training CSV using SMILES "
            "resolved from a labels CSV."
        )
    )
    parser.add_argument("training_csv", help="Input training CSV path")
    parser.add_argument("labels_csv", help="Labels CSV with Name and SMILES/LigID")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: overwrite training CSV)",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Electrophile SMILES column in labels (default: electrophile_smiles)",
    )
    parser.add_argument(
        "--ligid-column",
        default=None,
        help="Ligand ID column in labels (default: LigID)",
    )
    parser.add_argument(
        "--training-name-column",
        default=None,
        help="Name column in training CSV (default: Name)",
    )
    parser.add_argument(
        "--labels-name-column",
        default=None,
        help="Name column in labels CSV (default: Name)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded RCSB ligand SDF files",
    )
    parser.add_argument(
        "--n-conformers",
        type=int,
        default=N_CONFORMERS,
        help=f"Number of 3D conformers for D_L / flatness (default: {N_CONFORMERS})",
    )
    parser.add_argument(
        "--embed-seed",
        type=int,
        default=EMBED_SEED,
        help=f"RDKit embedding random seed (default: {EMBED_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading training CSV: {args.training_csv}")
    training_df = pd.read_csv(args.training_csv)
    print(f"[INFO] {len(training_df)} training rows")

    print(f"[INFO] Loading labels CSV: {args.labels_csv}")
    labels_df = pd.read_csv(args.labels_csv)
    print(f"[INFO] {len(labels_df)} label rows")

    cache_dir = args.cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    out, resolve_warnings, compute_errors = add_ligand_info(
        training_df,
        labels_df,
        smiles_col=args.smiles_column,
        ligid_col=args.ligid_column,
        training_name_col=args.training_name_column,
        labels_name_col=args.labels_name_column,
        cache_dir=cache_dir,
        n_conformers=args.n_conformers,
        embed_seed=args.embed_seed,
    )

    output_path = args.output or args.training_csv
    out.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {len(out)} rows to {output_path}")

    unique_names = out[find_column(out, NAME_COLUMN_CANDIDATES, args.training_name_column)].map(
        normalize_text
    )
    filled = out.loc[unique_names != "", "D_L_Ligand"].notna().sum()
    print(f"[INFO] Rows with ligand metrics: {filled}/{len(out)}")

    if resolve_warnings:
        print(f"[WARN] {len(resolve_warnings)} SMILES resolution warning(s):", file=sys.stderr)
        for msg in resolve_warnings[:20]:
            print(f"  - {msg}", file=sys.stderr)
        if len(resolve_warnings) > 20:
            print(f"  ... and {len(resolve_warnings) - 20} more", file=sys.stderr)

    if compute_errors:
        print(f"[WARN] {len(compute_errors)} descriptor error(s):", file=sys.stderr)
        for msg in compute_errors[:20]:
            print(f"  - {msg}", file=sys.stderr)
        if len(compute_errors) > 20:
            print(f"  ... and {len(compute_errors) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()
