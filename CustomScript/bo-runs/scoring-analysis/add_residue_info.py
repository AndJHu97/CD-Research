#!/usr/bin/env python3
"""
Extend a training CSV with per-site structural environment features computed
from PDB coordinates.

Adds columns (when missing):
  - DCFS
  - Hydrophobic_Fraction_Residue
  - electrostatic_charge_residue
  - H_Bond_Donor_Residue_Count
  - H_Bond_Acceptor_Residue_Count

PDB files are resolved from the Name column (text before the first hyphen,
e.g. 1BMQ-MNO -> 1BMQ) inside --pdb-dir. Rows are processed in file order;
when the PDB id changes the previous structure is dropped and the next one
is loaded. Results for duplicate (Residue, Chain, ResNum) tuples are cached.

Usage:
    python add_residue_info.py training_eval_N_terminal.csv --pdb-dir /path/to/pdbs
    python add_residue_info.py training.csv --pdb-dir ./pdbs --output training_extended.csv
    python add_residue_info.py training.csv --pdb-dir ./pdbs --row-start 0 --row-end 500
    python add_residue_info.py training.csv --pdb-dir ./pdbs --row-start 500 --row-end 1000
    python add_residue_info.py training.csv --pdb-dir ./pdbs --download-missing --update-new

Batch mode (--row-start / --row-end): process a slice of rows (0-based, end
exclusive). When --output points to an existing CSV (including the input file),
prior values outside the slice are preserved so batches can be run sequentially.

--download-missing fetches PDBs from RCSB into --pdb-dir when not found locally.
--update-new skips rows that already have residue-info columns filled.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ── Defaults ─────────────────────────────────────────────────────────────────

MAX_DISTANCE = 8.0
STEP_SIZE = 0.4
N_RAYS = 20
DCFS_CONE_HALF_ANGLE_DEG = 30.0
ENV_CONE_HALF_ANGLE_DEG = 90.0
VDW_PROBE = 0.25

POSITIVE_RESIDUES = frozenset({"LYS", "ARG"})
NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})
HIS_VARIANTS = {"HIE", "HID", "HIP"}

REACTIVE_ATOMS = {
    "CYS": "SG",
    "SER": "OG",
    "THR": "OG1",
    "TYR": "OH",
    "LYS": "NZ",
    "HIS": "ND1",
}

BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT", "H", "HA", "HN"})

# Atom-level hydrophobic / polar classification (side-chain atoms only).
HYDROPHOBIC_ATOMS: dict[str, frozenset[str]] = {
    "ALA": frozenset({"CB"}),
    "VAL": frozenset({"CB", "CG1", "CG2"}),
    "LEU": frozenset({"CB", "CG", "CD1", "CD2"}),
    "ILE": frozenset({"CB", "CG1", "CG2", "CD1"}),
    "PRO": frozenset({"CB", "CG", "CD"}),
    "PHE": frozenset({"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "TRP": frozenset({"CB", "CG", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"}),
    "MET": frozenset({"CB", "CG", "SD", "CE"}),
    "TYR": frozenset({"CB", "CG", "CD1", "CD2", "CE1", "CE2"}),
    "THR": frozenset({"CG2"}),
}

POLAR_ATOMS: dict[str, frozenset[str]] = {
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "CYS": frozenset({"SG"}),
    "ASN": frozenset({"OD1", "ND2", "CG"}),
    "GLN": frozenset({"OE1", "NE2", "CD"}),
    "ASP": frozenset({"OD1", "OD2", "CG"}),
    "GLU": frozenset({"OE1", "OE2", "CD"}),
    "LYS": frozenset({"NZ", "CE"}),
    "ARG": frozenset({"NE", "NH1", "NH2", "CZ"}),
    "HIS": frozenset({"ND1", "NE2", "CG", "CD2", "CE1"}),
    "TYR": frozenset({"OH"}),
}

VDW_BY_ELEMENT = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}
DEFAULT_VDW = 1.70

NEW_COLUMNS = (
    "DCFS",
    "Hydrophobic_Fraction_Residue",
    "electrostatic_charge_residue",
    "H_Bond_Donor_Residue_Count",
    "H_Bond_Acceptor_Residue_Count",
)

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


@dataclass(frozen=True)
class Atom:
    name: str
    resname: str
    chain: str
    resseq: int
    xyz: np.ndarray
    vdw: float


@dataclass
class PdbStructure:
    atoms: list[Atom]
    coords: np.ndarray
    vdw_radii: np.ndarray


@dataclass(frozen=True)
class SiteKey:
    chain: str
    resseq: int
    resname: str


@dataclass(frozen=True)
class SiteMetrics:
    dcfs: float
    hydrophobic_fraction: float
    electrostatic_charge: float
    hbond_donor_count: int
    hbond_acceptor_count: int


# ── CSV / PDB helpers ────────────────────────────────────────────────────────

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


def normalize_resname(residue: object) -> str:
    if pd.isna(residue):
        return ""
    name = str(residue).strip().upper()[:3]
    if name in HIS_VARIANTS:
        return "HIS"
    return name


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


def download_pdb_from_rcsb(pdb_id: str, pdb_dir: str) -> Optional[str]:
    """Download a PDB from RCSB into pdb_dir. Returns path or None on failure."""
    pdb_id = str(pdb_id).strip().upper().replace(".pdb", "").split(".")[0]
    if not pdb_id:
        return None

    os.makedirs(pdb_dir, exist_ok=True)
    out_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    print(f"[INFO] Downloading {pdb_id} from RCSB...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, out_path)
    except (urllib.error.URLError, OSError) as exc:
        print(f"[WARN] Could not download {pdb_id}: {exc}", file=sys.stderr)
        if os.path.isfile(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        print(f"[WARN] Downloaded file for {pdb_id} is empty", file=sys.stderr)
        if os.path.isfile(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None

    print(f"[INFO] Saved {out_path}", file=sys.stderr)
    return out_path


def find_or_download_pdb_file(
    pdb_id: str,
    pdb_dir: str,
    recursive: bool = False,
    download_missing: bool = False,
) -> Optional[str]:
    pdb_path = find_pdb_file(pdb_id, pdb_dir, recursive=recursive)
    if pdb_path is not None or not download_missing:
        return pdb_path
    return download_pdb_from_rcsb(pdb_id, pdb_dir)


def row_has_residue_info(row: pd.Series) -> bool:
    """True when any residue-info column is already populated."""
    for col in NEW_COLUMNS:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def guess_element(atom_name: str) -> str:
    name = atom_name.strip().upper()
    if not name:
        return "C"
    if len(name) >= 2 and name[1].isalpha():
        return name[:2]
    return name[0]


def vdw_radius(atom_name: str) -> float:
    return VDW_BY_ELEMENT.get(guess_element(atom_name), DEFAULT_VDW)


def parse_pdb_structure(pdb_path: str) -> PdbStructure:
    atoms: list[Atom] = []
    with open(pdb_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                alt = line[16].strip()
                if alt not in ("", "A"):
                    continue
                atom = Atom(
                    name=line[12:16].strip().upper(),
                    resname=line[17:20].strip().upper(),
                    chain=line[21].strip() or " ",
                    resseq=int(line[22:26]),
                    xyz=np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=float,
                    ),
                    vdw=vdw_radius(line[12:16]),
                )
                if atom.resname in HIS_VARIANTS:
                    atom = Atom(
                        name=atom.name,
                        resname="HIS",
                        chain=atom.chain,
                        resseq=atom.resseq,
                        xyz=atom.xyz,
                        vdw=atom.vdw,
                    )
                atoms.append(atom)
            except (ValueError, IndexError):
                continue

    if not atoms:
        raise RuntimeError(f"No ATOM records found in {pdb_path}")

    coords = np.vstack([a.xyz for a in atoms])
    vdw_radii = np.array([a.vdw for a in atoms], dtype=float)
    return PdbStructure(atoms=atoms, coords=coords, vdw_radii=vdw_radii)


# ── Geometry ─────────────────────────────────────────────────────────────────

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(axis, helper))
    w = normalize(np.cross(axis, u))
    return u, w


def sample_cone_directions(axis: np.ndarray, n_rays: int, half_angle_deg: float) -> list[np.ndarray]:
    axis = normalize(axis)
    if n_rays <= 1:
        return [axis]

    half_angle = math.radians(half_angle_deg)
    u, w = orthonormal_basis(axis)
    directions = [axis]
    for i in range(1, n_rays):
        azimuth = 2.0 * math.pi * (i - 1) / max(n_rays - 1, 1)
        theta = half_angle * math.sqrt(i / max(n_rays - 1, 1))
        direction = (
            math.cos(theta) * axis
            + math.sin(theta) * (math.cos(azimuth) * u + math.sin(azimuth) * w)
        )
        directions.append(normalize(direction))
    return directions


def atom_in_cone(
    origin: np.ndarray,
    direction: np.ndarray,
    point: np.ndarray,
    max_distance: float,
    half_angle_rad: float,
) -> bool:
    vec = point - origin
    dist = float(np.linalg.norm(vec))
    if dist <= 1e-8 or dist > max_distance:
        return False
    cos_angle = float(np.dot(vec / dist, direction))
    return cos_angle >= math.cos(half_angle_rad)


def is_site_residue(
    atom: Atom,
    site_chain: str,
    site_resseq: int,
    site_resname: str,
) -> bool:
    return (
        atom.chain == site_chain
        and atom.resseq == site_resseq
        and atom.resname == site_resname
    )


def classify_atom(resname: str, atom_name: str) -> Optional[str]:
    resname = normalize_resname(resname)
    atom_name = atom_name.upper()
    if atom_name in HYDROPHOBIC_ATOMS.get(resname, frozenset()):
        return "hydrophobic"
    if atom_name in POLAR_ATOMS.get(resname, frozenset()):
        return "polar"
    return None


# ── Site-level calculations ──────────────────────────────────────────────────

def residue_atoms(
    structure: PdbStructure,
    chain: str,
    resseq: int,
    resname: str,
) -> list[Atom]:
    return [
        atom
        for atom in structure.atoms
        if atom.chain == chain and atom.resseq == resseq and atom.resname == resname
    ]


def find_atom(residue: list[Atom], atom_name: str) -> Optional[Atom]:
    target = atom_name.upper()
    for atom in residue:
        if atom.name == target:
            return atom
    return None


def side_chain_direction(residue: list[Atom]) -> np.ndarray:
    ca = find_atom(residue, "CA")
    cb = find_atom(residue, "CB")
    if ca is not None and cb is not None:
        return normalize(cb.xyz - ca.xyz)

    heavy = [
        atom.xyz
        for atom in residue
        if atom.name not in BACKBONE_ATOMS and not atom.name.startswith("H")
    ]
    if ca is not None and heavy:
        centroid = np.mean(np.vstack(heavy), axis=0)
        return normalize(centroid - ca.xyz)
    if ca is not None:
        return np.array([0.0, 0.0, 1.0])
    if heavy:
        centroid = np.mean(np.vstack(heavy), axis=0)
        return normalize(centroid - heavy[0])
    return np.array([0.0, 0.0, 1.0])


def reactive_atom(residue: list[Atom], resname: str) -> Optional[Atom]:
    preferred = REACTIVE_ATOMS.get(resname)
    if preferred:
        atom = find_atom(residue, preferred)
        if atom is not None:
            return atom
    for atom in residue:
        if atom.name not in BACKBONE_ATOMS:
            return atom
    return find_atom(residue, "CA")


def collision_mask(structure: PdbStructure, exclude_indices: set[int]) -> np.ndarray:
    mask = np.ones(len(structure.atoms), dtype=bool)
    for idx in exclude_indices:
        mask[idx] = False
    return mask


def ray_free_path_length(
    origin: np.ndarray,
    direction: np.ndarray,
    coords: np.ndarray,
    vdw_radii: np.ndarray,
    mask: np.ndarray,
    max_distance: float,
    step_size: float,
) -> float:
    direction = normalize(direction)
    collision_coords = coords[mask]
    collision_vdw = vdw_radii[mask]
    if collision_coords.size == 0:
        return max_distance

    traveled = 0.0
    while traveled <= max_distance:
        point = origin + direction * traveled
        deltas = collision_coords - point
        dist_sq = np.einsum("ij,ij->i", deltas, deltas)
        contact = dist_sq <= (collision_vdw + VDW_PROBE) ** 2
        if np.any(contact):
            return traveled
        traveled += step_size
    return max_distance


def compute_dcfs(
    origin: np.ndarray,
    direction: np.ndarray,
    coords: np.ndarray,
    vdw_radii: np.ndarray,
    mask: np.ndarray,
    n_rays: int,
    cone_half_angle_deg: float,
    max_distance: float,
    step_size: float,
) -> float:
    rays = sample_cone_directions(direction, n_rays, cone_half_angle_deg)
    lengths = [
        ray_free_path_length(
            origin, ray, coords, vdw_radii, mask, max_distance, step_size
        )
        for ray in rays
    ]
    return float(np.mean(lengths))


def compute_hydrophobic_fraction(
    origin: np.ndarray,
    direction: np.ndarray,
    structure: PdbStructure,
    site_chain: str,
    site_resseq: int,
    site_resname: str,
    max_distance: float,
    cone_half_angle_deg: float,
) -> float:
    direction = normalize(direction)
    half_angle = math.radians(cone_half_angle_deg)
    hydrophobic = 0
    polar = 0

    for atom in structure.atoms:
        if is_site_residue(atom, site_chain, site_resseq, site_resname):
            continue
        if not atom_in_cone(origin, direction, atom.xyz, max_distance, half_angle):
            continue
        label = classify_atom(atom.resname, atom.name)
        if label == "hydrophobic":
            hydrophobic += 1
        elif label == "polar":
            polar += 1

    total = hydrophobic + polar
    if total == 0:
        return float("nan")
    return hydrophobic / total


def compute_electrostatic_charge(
    origin: np.ndarray,
    direction: np.ndarray,
    structure: PdbStructure,
    site_chain: str,
    site_resseq: int,
    site_resname: str,
    max_distance: float,
    cone_half_angle_deg: float,
) -> float:
    direction = normalize(direction)
    half_angle = math.radians(cone_half_angle_deg)
    positive = 0
    negative = 0
    seen: set[tuple[str, str, int]] = set()

    for atom in structure.atoms:
        if is_site_residue(atom, site_chain, site_resseq, site_resname):
            continue
        if not atom_in_cone(origin, direction, atom.xyz, max_distance, half_angle):
            continue
        key = (atom.chain, atom.resname, atom.resseq)
        if key in seen:
            continue
        seen.add(key)
        if atom.resname in POSITIVE_RESIDUES:
            positive += 1
        elif atom.resname in NEGATIVE_RESIDUES:
            negative += 1

    return float(positive - negative)


def compute_hbond_counts(
    origin: np.ndarray,
    direction: np.ndarray,
    structure: PdbStructure,
    site_chain: str,
    site_resseq: int,
    site_resname: str,
    max_distance: float,
    cone_half_angle_deg: float,
) -> tuple[int, int]:
    """Crude H-bond proxy: count N atoms (donors) and O atoms (acceptors) in cone."""
    direction = normalize(direction)
    half_angle = math.radians(cone_half_angle_deg)
    donor_count = 0
    acceptor_count = 0

    for atom in structure.atoms:
        if is_site_residue(atom, site_chain, site_resseq, site_resname):
            continue
        if not atom_in_cone(origin, direction, atom.xyz, max_distance, half_angle):
            continue
        element = guess_element(atom.name)
        if element == "N":
            donor_count += 1
        elif element == "O":
            acceptor_count += 1

    return donor_count, acceptor_count


def compute_site_metrics(
    structure: PdbStructure,
    chain: str,
    resseq: int,
    resname: str,
    n_rays: int,
    dcfs_cone_half_angle_deg: float,
    env_cone_half_angle_deg: float,
    max_distance: float,
    step_size: float,
) -> SiteMetrics:
    residue = residue_atoms(structure, chain, resseq, resname)
    if not residue:
        raise RuntimeError(f"Residue not found: {chain}:{resname}:{resseq}")

    reactive = reactive_atom(residue, resname)
    if reactive is None:
        raise RuntimeError(f"No reactive atom for {chain}:{resname}:{resseq}")

    direction = side_chain_direction(residue)
    if float(np.linalg.norm(direction)) < 1e-8:
        ca = find_atom(residue, "CA")
        if ca is not None:
            direction = normalize(reactive.xyz - ca.xyz)
        if float(np.linalg.norm(direction)) < 1e-8:
            direction = np.array([0.0, 0.0, 1.0])

    exclude_indices = {
        idx
        for idx, atom in enumerate(structure.atoms)
        if atom.chain == chain and atom.resseq == resseq and atom.resname == resname
    }
    mask = collision_mask(structure, exclude_indices)

    dcfs = compute_dcfs(
        reactive.xyz,
        direction,
        structure.coords,
        structure.vdw_radii,
        mask,
        n_rays,
        dcfs_cone_half_angle_deg,
        max_distance,
        step_size,
    )
    hydrophobic_fraction = compute_hydrophobic_fraction(
        reactive.xyz,
        direction,
        structure,
        chain,
        resseq,
        resname,
        max_distance,
        env_cone_half_angle_deg,
    )
    electrostatic_charge = compute_electrostatic_charge(
        reactive.xyz,
        direction,
        structure,
        chain,
        resseq,
        resname,
        max_distance,
        env_cone_half_angle_deg,
    )
    hbond_donor_count, hbond_acceptor_count = compute_hbond_counts(
        reactive.xyz,
        direction,
        structure,
        chain,
        resseq,
        resname,
        max_distance,
        env_cone_half_angle_deg,
    )
    return SiteMetrics(
        dcfs=dcfs,
        hydrophobic_fraction=hydrophobic_fraction,
        electrostatic_charge=electrostatic_charge,
        hbond_donor_count=hbond_donor_count,
        hbond_acceptor_count=hbond_acceptor_count,
    )


# ── Main pipeline ────────────────────────────────────────────────────────────

def add_residue_info(
    df: pd.DataFrame,
    pdb_dir: str,
    recursive: bool = False,
    n_rays: int = N_RAYS,
    dcfs_cone_half_angle_deg: float = DCFS_CONE_HALF_ANGLE_DEG,
    env_cone_half_angle_deg: float = ENV_CONE_HALF_ANGLE_DEG,
    max_distance: float = MAX_DISTANCE,
    step_size: float = STEP_SIZE,
    name_col: Optional[str] = None,
    chain_col: Optional[str] = None,
    resnum_col: Optional[str] = None,
    residue_col: Optional[str] = None,
    row_start: int = 0,
    row_end: Optional[int] = None,
    download_missing: bool = False,
    update_new: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, int]]:
    name_col = find_column(df, ("name",), name_col)
    chain_col = find_column(df, ("chain",), chain_col)
    resnum_col = find_column(df, ("resnum", "res_num", "residue_number"), resnum_col)
    residue_col = find_column(df, ("residue", "resname", "aa"), residue_col)

    out = df.copy()
    for col in NEW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    current_pdb_id = ""
    current_structure: Optional[PdbStructure] = None
    site_cache: dict[SiteKey, SiteMetrics] = {}
    missing_pdbs: set[str] = set()
    site_errors: list[str] = []
    stats = {
        "rows_in_slice": 0,
        "rows_skipped_existing": 0,
        "rows_updated": 0,
        "pdbs_downloaded": 0,
    }
    downloaded_pdbs: set[str] = set()

    total = len(out)
    start = max(0, row_start)
    end = total if row_end is None else min(row_end, total)
    if start >= end:
        return out, [], [], stats

    batch_total = end - start
    stats["rows_in_slice"] = batch_total
    processed = 0
    for row_idx, row in out.iloc[start:end].iterrows():
        if update_new and row_has_residue_info(row):
            stats["rows_skipped_existing"] += 1
            continue

        processed += 1
        if processed % 500 == 0:
            print(
                f"[INFO] Processed {processed} new row(s) in slice "
                f"({stats['rows_skipped_existing']} skipped existing; "
                f"global {row_idx + 1}/{total})...",
                file=sys.stderr,
            )

        pdb_id = extract_pdb_id(row[name_col])
        chain = normalize_chain(row[chain_col])
        resseq = normalize_resnum(row[resnum_col])
        resname = normalize_resname(row[residue_col])

        if not pdb_id or resseq is None or not resname:
            continue

        site_key = SiteKey(chain=chain, resseq=resseq, resname=resname)
        if site_key in site_cache:
            metrics = site_cache[site_key]
        else:
            if pdb_id != current_pdb_id:
                had_local = find_pdb_file(pdb_id, pdb_dir, recursive=recursive) is not None
                pdb_path = find_or_download_pdb_file(
                    pdb_id,
                    pdb_dir,
                    recursive=recursive,
                    download_missing=download_missing,
                )
                if pdb_path is not None and download_missing and not had_local:
                    if pdb_id.upper() not in downloaded_pdbs:
                        downloaded_pdbs.add(pdb_id.upper())
                        stats["pdbs_downloaded"] += 1
                if pdb_path is None:
                    missing_pdbs.add(pdb_id)
                    current_pdb_id = pdb_id
                    current_structure = None
                    site_cache.clear()
                else:
                    try:
                        current_structure = parse_pdb_structure(pdb_path)
                        current_pdb_id = pdb_id
                        site_cache.clear()
                    except RuntimeError as exc:
                        site_errors.append(f"{pdb_id}: {exc}")
                        current_pdb_id = pdb_id
                        current_structure = None
                        site_cache.clear()

            if current_structure is None:
                continue

            try:
                metrics = compute_site_metrics(
                    current_structure,
                    chain,
                    resseq,
                    resname,
                    n_rays=n_rays,
                    dcfs_cone_half_angle_deg=dcfs_cone_half_angle_deg,
                    env_cone_half_angle_deg=env_cone_half_angle_deg,
                    max_distance=max_distance,
                    step_size=step_size,
                )
                site_cache[site_key] = metrics
            except RuntimeError as exc:
                site_errors.append(
                    f"row {row_idx} ({pdb_id} {chain}:{resname}:{resseq}): {exc}"
                )
                continue

        out.at[row_idx, "DCFS"] = metrics.dcfs
        out.at[row_idx, "Hydrophobic_Fraction_Residue"] = metrics.hydrophobic_fraction
        out.at[row_idx, "electrostatic_charge_residue"] = metrics.electrostatic_charge
        out.at[row_idx, "H_Bond_Donor_Residue_Count"] = metrics.hbond_donor_count
        out.at[row_idx, "H_Bond_Acceptor_Residue_Count"] = metrics.hbond_acceptor_count
        stats["rows_updated"] += 1

    return out, sorted(missing_pdbs), site_errors, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add DCFS, Hydrophobic_Fraction_Residue, electrostatic_charge_residue, "
            "H_Bond_Donor_Residue_Count, and H_Bond_Acceptor_Residue_Count "
            "columns to a training CSV."
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
        help=(
            "Output CSV path (default: overwrite input file). "
            "If the file already exists, rows outside --row-start/--row-end are kept."
        ),
    )
    parser.add_argument("--name-column", default=None, help="Name column (default: Name)")
    parser.add_argument("--chain-column", default=None, help="Chain column (default: Chain)")
    parser.add_argument("--resnum-column", default=None, help="ResNum column (default: ResNum)")
    parser.add_argument(
        "--residue-column",
        default=None,
        help="Residue column (default: auto-detect Residue)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --pdb-dir recursively for PDB files",
    )
    parser.add_argument("--n-rays", type=int, default=N_RAYS, help=f"Ray count (default: {N_RAYS})")
    parser.add_argument(
        "--dcfs-cone-half-angle",
        type=float,
        default=DCFS_CONE_HALF_ANGLE_DEG,
        help=f"DCFS ray-cone half-angle in degrees (default: {DCFS_CONE_HALF_ANGLE_DEG})",
    )
    parser.add_argument(
        "--env-cone-half-angle",
        type=float,
        default=ENV_CONE_HALF_ANGLE_DEG,
        help=(
            "Cone half-angle in degrees for hydrophobic, electrostatic, and "
            f"H-bond features (default: {ENV_CONE_HALF_ANGLE_DEG})"
        ),
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=MAX_DISTANCE,
        help=f"Maximum ray / cone distance in Angstroms (default: {MAX_DISTANCE})",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=STEP_SIZE,
        help=f"Ray marching step size in Angstroms (default: {STEP_SIZE})",
    )
    parser.add_argument(
        "--row-start",
        type=int,
        default=0,
        help="First row to process, 0-based inclusive (default: 0)",
    )
    parser.add_argument(
        "--row-end",
        type=int,
        default=None,
        help="Last row to process, 0-based exclusive (default: end of file)",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download PDBs from RCSB into --pdb-dir when not found locally",
    )
    parser.add_argument(
        "--update-new",
        action="store_true",
        help=(
            "Skip rows that already have residue-info columns filled "
            "(useful when re-running only missing sites)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.pdb_dir):
        sys.exit(f"[ERROR] PDB directory not found: {args.pdb_dir}")
    if args.row_start < 0:
        sys.exit("[ERROR] --row-start must be >= 0")
    if args.row_end is not None and args.row_end <= args.row_start:
        sys.exit("[ERROR] --row-end must be greater than --row-start")

    print(f"[INFO] Loading {args.training_csv}")
    df = pd.read_csv(args.training_csv)
    print(f"[INFO] {len(df)} rows, {len(df.columns)} columns")

    output_path = args.output or args.training_csv
    if os.path.isfile(output_path):
        print(f"[INFO] Loading existing output {output_path}")
        out = pd.read_csv(output_path)
        if len(out) != len(df):
            sys.exit(
                f"[ERROR] Output row count ({len(out)}) does not match input ({len(df)})"
            )
    else:
        out = df.copy()

    for col in NEW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    row_end = len(out) if args.row_end is None else min(args.row_end, len(out))
    print(f"[INFO] Processing rows {args.row_start}:{row_end} of {len(out)}")

    out, missing_pdbs, site_errors, stats = add_residue_info(
        out,
        pdb_dir=args.pdb_dir,
        recursive=args.recursive,
        n_rays=args.n_rays,
        dcfs_cone_half_angle_deg=args.dcfs_cone_half_angle,
        env_cone_half_angle_deg=args.env_cone_half_angle,
        max_distance=args.max_distance,
        step_size=args.step_size,
        name_col=args.name_column,
        chain_col=args.chain_column,
        resnum_col=args.resnum_column,
        residue_col=args.residue_column,
        row_start=args.row_start,
        row_end=args.row_end,
        download_missing=args.download_missing,
        update_new=args.update_new,
    )

    out.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {len(out)} rows to {output_path}")
    print(
        f"[INFO] Slice rows: {stats['rows_in_slice']:,}; "
        f"skipped existing: {stats['rows_skipped_existing']:,}; "
        f"updated: {stats['rows_updated']:,}"
    )
    if args.download_missing:
        print(f"[INFO] PDBs downloaded from RCSB: {stats['pdbs_downloaded']:,}")

    if missing_pdbs:
        print(
            f"[WARN] Missing PDB files for {len(missing_pdbs)} id(s): "
            + ", ".join(missing_pdbs[:20])
            + (" ..." if len(missing_pdbs) > 20 else ""),
            file=sys.stderr,
        )
    if site_errors:
        print(f"[WARN] {len(site_errors)} site-level error(s):", file=sys.stderr)
        for msg in site_errors[:20]:
            print(f"  - {msg}", file=sys.stderr)
        if len(site_errors) > 20:
            print(f"  ... and {len(site_errors) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()
