"""
Predict_Deprotonation.py

Run inference using a trained XGBoost deprotonation model.
Computes features for a specified residue in a PDB file and outputs
the predicted deprotonation probability P(deprot).

Features used (must match training):
    - ref_pka              : physiological pKa of the residue type
    - sasa                 : solvent accessible surface area (from CSV or freesasa)
    - electrostatic_potential : from APBS
    - arg_count, lys_count, asp_count, glu_count, his_count: charged residues near site
    - hbonds_weighted, hbonds_strict_flexible : from HBonds_Score.py

Usage:
    python Predict_Deprotonation.py deprot_xgb.pkl 4g5j.pdb A:CYS:797
    python Predict_Deprotonation.py deprot_xgb.pkl 4g5j.pdb A:CYS:797 --pdb-dir ./pdbs
    python Predict_Deprotonation.py deprot_xgb.pkl 4g5j.pdb A:HIS:42 --apbs-radius 12.0 --hbond-radius 6.0
    python Predict_Deprotonation.py deprot_xgb.pkl 4g5j.pdb A:HIS:42 --no-apbs
"""

import argparse
import json
import os
import re
import subprocess
import urllib.request
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHYSIOLOGIC_PH = 7.4

REFERENCE_PKA = {
    "ASP": 3.9,
    "GLU": 4.1,
    "HIS": 6.0,
    "CYS": 8.3,
    "LYS": 10.5,
    "TYR": 10.5,
    "ARG": 12.5,
    "SER": 13.0,
    "THR": 13.6,
}

# Target atoms per residue for distance-based neighborhood.
# Ordered from most side-chain distal to most proximal so incomplete residues
# can still contribute a reasonable center atom instead of failing outright.
TARGET_ATOMS = {
    "ALA": ["CA"],
    "ARG": ["CZ", "NH1", "NH2", "NE", "CD", "CG", "CB", "CA"],
    "ASN": ["CA"],
    "ASP": ["OD1", "OD2", "CG", "CB", "CA"],
    "CYS": ["SG", "CB", "CA"],
    "GLN": ["CA"],
    "GLU": ["OE1", "OE2", "CD", "CG", "CB", "CA"],
    "GLY": ["CA"],
    "HIS": ["ND1", "NE2", "CE1", "CD2", "CG", "CB", "CA"],
    "ILE": ["CA"],
    "LEU": ["CA"],
    "LYS": ["NZ", "CE", "CD", "CG", "CB", "CA"],
    "MET": ["CA"],
    "PHE": ["CA"],
    "PRO": ["CA"],
    "SER": ["OG", "CB", "CA"],
    "THR": ["OG1", "CG2", "CB", "CA"],
    "TRP": ["CA"],
    "TYR": ["OH", "CZ", "CE1", "CE2", "CD1", "CD2", "CG", "CB", "CA"],
    "VAL": ["CA"],
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APBS_SCRIPT = os.path.join(SCRIPT_DIR, "APBS_Deprotonation.py")
HBOND_SCRIPT = os.path.join(SCRIPT_DIR, "HBonds_Score.py")
PREDICTION_CACHE_FILE = os.path.join(SCRIPT_DIR, "deprotonation_prediction_cache.json")


def load_prediction_cache() -> Dict[str, dict]:
    if os.path.isfile(PREDICTION_CACHE_FILE):
        try:
            with open(PREDICTION_CACHE_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"[cache] Could not load prediction cache: {exc}")
    return {}


def save_prediction_cache(cache: Dict[str, dict]) -> None:
    try:
        with open(PREDICTION_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        print(f"[cache] Could not save prediction cache: {exc}")


def make_prediction_cache_key(pdb_path: str, chain: str, resname: str, resseq: int,
                              model_path: str, apbs_radius: float, hbond_radius: float,
                              ph: float, use_apbs: bool) -> str:
    pdb_path_abs = os.path.abspath(pdb_path)
    model_path_abs = os.path.abspath(model_path)
    pdb_mtime = os.path.getmtime(pdb_path_abs) if os.path.exists(pdb_path_abs) else 0.0
    model_mtime = os.path.getmtime(model_path_abs) if os.path.exists(model_path_abs) else 0.0
    return "|".join([
        pdb_path_abs,
        f"pdb_mtime={pdb_mtime}",
        str(chain).strip().upper(),
        str(resname).strip().upper(),
        str(int(resseq)),
        os.path.basename(model_path_abs),
        f"model_mtime={model_mtime}",
        f"apbs_radius={float(apbs_radius)}",
        f"hbond_radius={float(hbond_radius)}",
        f"ph={float(ph)}",
        f"use_apbs={int(bool(use_apbs))}",
    ])


# ---------------------------------------------------------------------------
# PDB utilities
# ---------------------------------------------------------------------------

def find_pdb_path(pdb_id_or_path: str, pdb_dir: str) -> str:
    if os.path.isfile(pdb_id_or_path):
        return pdb_id_or_path

    pdb_id = pdb_id_or_path.lower().replace(".pdb", "")
    candidate = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if os.path.isfile(candidate):
        return candidate

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[pdb] Downloading {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, candidate)
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            print(f"[pdb] Saved to {candidate}")
            return candidate
    except Exception as exc:
        raise FileNotFoundError(
            f"PDB file not found locally and download failed for '{pdb_id}'."
        ) from exc

    raise FileNotFoundError(f"PDB file not found for '{pdb_id_or_path}' in {pdb_dir}")


def extract_and_renumber_model1(pdb_path: str, out_path: str) -> None:
    """Extract MODEL 1 only and renumber atom serials from 1."""
    in_model1 = False
    serial = 1
    found_models = False

    with open(pdb_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if line.startswith("MODEL"):
                found_models = True
                model_num = line.split()[1] if len(line.split()) > 1 else ""
                in_model1 = (model_num == "1")
                continue
            if line.startswith("ENDMDL"):
                if in_model1:
                    break
                continue
            if not found_models or in_model1:
                if line.startswith(("ATOM", "HETATM")):
                    line = f"{line[:6]}{serial:5d}{line[11:]}"
                    serial += 1
                f_out.write(line)


# ---------------------------------------------------------------------------
# PDB atom helpers (copied from training script so this module is standalone)
# ---------------------------------------------------------------------------

def parse_pdb_atoms(pdb_path: str):
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atoms.append({
                    "record": line[0:6].strip(),
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "alt": line[16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resseq": int(line[22:26]),
                    "icode": line[26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def find_target_atom(atoms, chain: str, resname: str, resseq: int):
    target_atoms = TARGET_ATOMS.get(resname.upper(), ["CA"])

    # Prefer the most distal available atom in the residue-specific list.
    for preferred_name in target_atoms:
        preferred_name_u = preferred_name.upper()
        for atom in atoms:
            chain_match = (chain is None) or (atom["chain"] == chain)
            if (
                chain_match
                and atom["resname"].upper() == resname.upper()
                and atom["resseq"] == resseq
                and atom["name"].upper() == preferred_name_u
            ):
                return atom

    # Final fallback: use any atom from the residue instead of returning null.
    for atom in atoms:
        chain_match = (chain is None) or (atom["chain"] == chain)
        if chain_match and atom["resname"].upper() == resname.upper() and atom["resseq"] == resseq:
            return atom
    return None


def get_sphere_atoms(atoms, center_xyz, radius: float):
    cx, cy, cz = center_xyz
    result = []
    for atom in atoms:
        dx = atom["x"] - cx
        dy = atom["y"] - cy
        dz = atom["z"] - cz
        if (dx * dx + dy * dy + dz * dz) <= radius * radius:
            result.append(atom)
    return result


def compute_charged_residue_counts(pdb_path: str, chain: str, resname: str, resseq: int, radius: float) -> Dict[str, float]:
    atoms = parse_pdb_atoms(pdb_path)
    if not atoms:
        raise RuntimeError(f"No ATOM records found in {pdb_path}")
    atoms = [atom for atom in atoms if atom["alt"] in ("", "A")]
    target = find_target_atom(atoms, chain, resname, resseq)
    if target is None:
        raise RuntimeError(f"Could not find target atom for {chain}:{resname}:{resseq}")
    nuc_xyz = (target["x"], target["y"], target["z"])
    sphere_atoms = get_sphere_atoms(atoms, nuc_xyz, radius)
    sphere_residues = {(atom["chain"], atom["resname"], atom["resseq"]) for atom in sphere_atoms}
    counts = {"arg_count": 0, "lys_count": 0, "asp_count": 0, "glu_count": 0, "his_count": 0}
    for _, residue_name, _ in sphere_residues:
        residue_name_u = residue_name.upper()
        if residue_name_u in ("HIE", "HID", "HIP"):
            residue_name_u = "HIS"
        if residue_name_u == "ARG":
            counts["arg_count"] += 1
        elif residue_name_u == "LYS":
            counts["lys_count"] += 1
        elif residue_name_u == "ASP":
            counts["asp_count"] += 1
        elif residue_name_u == "GLU":
            counts["glu_count"] += 1
        elif residue_name_u == "HIS":
            counts["his_count"] += 1
    return counts


# ---------------------------------------------------------------------------
# SASA
# ---------------------------------------------------------------------------

def compute_sasa_freesasa(pdb_path: str, chain: str, resseq: int, resname: str) -> Optional[float]:
    try:
        import freesasa
    except ImportError:
        print("[sasa] freesasa not installed. Run: pip install freesasa")
        return np.nan

    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        rsa = result.residueAreas()  # per-residue breakdown including relative areas

        rows = []
        for chain_id, residues in rsa.items():
            for res_id, areas in residues.items():
                res_num_str = re.sub(r'[^0-9-]', '', res_id.strip())
                if not res_num_str:
                    continue
                r_resseq = int(res_num_str)
                r_resname = areas.residueType.strip()
                # relativeSideChain is 0-1, multiply by 100 to match Total-Side REL scale
                side_rel = areas.relativeSideChain
                side_rel = np.nan if side_rel is None else side_rel * 100.0
                rows.append((chain_id, r_resseq, r_resname, side_rel))

        sasa_df = pd.DataFrame(rows, columns=["chain", "resnum", "resname", "side_sasa"])

        pdb_base = os.path.splitext(pdb_path)[0]
        out_path = f"{pdb_base}_deprot_sasa.csv"
        sasa_df.to_csv(out_path, index=False)
        print(f"[sasa] Saved computed SASA to {out_path}")

        match = sasa_df[
            (sasa_df["chain"].astype(str) == str(chain)) &
            (sasa_df["resnum"].astype(int) == int(resseq)) &
            (sasa_df["resname"].str.upper() == resname.upper())
        ]
        if match.empty:
            print(f"[sasa] Residue {chain}:{resname}:{resseq} not found after freesasa computation")
            return np.nan
        return float(match.iloc[0]["side_sasa"])

    except Exception as exc:
        print(f"[sasa] freesasa computation failed for {chain}:{resname}:{resseq}: {exc}")
        return np.nan


def load_sasa(pdb_path: str, chain: str, resseq: int, resname: str) -> Optional[float]:
    pdb_dir = os.path.dirname(pdb_path)
    pdb_base = os.path.splitext(os.path.basename(pdb_path))[0]
    candidates = [
        os.path.join(pdb_dir, f"{pdb_base}_deprot_sasa.csv"),
        os.path.join(pdb_dir, f"{pdb_base}_sasa.csv"),
        os.path.join(pdb_dir, f"{pdb_base}_pdb_sasa.csv"),
        os.path.join(pdb_dir, "pdb_sasa.csv"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        chain_col = cols.get("chain")
        resnum_col = cols.get("resnum") or cols.get("resseq") or cols.get("resid")
        resname_col = cols.get("resname")
        sasa_col = (cols.get("side_sasa") or
                    cols.get("total-side rel") or
                    cols.get("total_side_rel"))
        if not all([chain_col, resnum_col, resname_col, sasa_col]):
            continue
        match = df[
            (df[chain_col].astype(str) == str(chain)) &
            (df[resnum_col].astype(int) == int(resseq)) &
            (df[resname_col].str.upper() == resname.upper())
        ]
        if not match.empty:
            return float(match.iloc[0][sasa_col])

    print(f"[sasa] No CSV found for {chain}:{resname}:{resseq}, computing via freesasa...")
    return compute_sasa_freesasa(pdb_path, chain, resseq, resname)


# ---------------------------------------------------------------------------
# Feature extraction (mirrors Deprotonation_Model.py)
# ---------------------------------------------------------------------------

def _extract_float(text: str, pattern: str) -> float:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return np.nan
    val = m.group(1).strip()
    return np.nan if val.lower() == "nan" else float(val)


def _extract_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"Pattern not found: {pattern}")
    return int(m.group(1))


def run_apbs(pdb_path: str, residue_spec: str, radius: float, ph: float) -> Dict[str, float]:
    cmd = [
        "python", APBS_SCRIPT,
        pdb_path, residue_spec,
        "--radius", str(radius),
        "--ph", str(ph),
        "--keep-files",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[APBS stdout]:\n{result.stdout[-3000:]}")
        print(f"[APBS stderr]:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"APBS failed: {result.stderr.strip()}")

    text = result.stdout
    return {
        "electrostatic_potential": _extract_float(text, r"Potential at .*?:\s*([\d.+-]+)\s*kT/e"),
        "arg_count": _extract_int(text, r"ARG:\s*(\d+)"),
        "lys_count": _extract_int(text, r"LYS:\s*(\d+)"),
        "asp_count": _extract_int(text, r"ASP:\s*(\d+)"),
        "glu_count": _extract_int(text, r"GLU:\s*(\d+)"),
        "his_count": _extract_int(text, r"HIS:\s*(\d+)"),
    }


def get_charged_residue_counts(pdb_path: str, chain: str, resseq: int, resname: str, radius: float) -> Dict[str, float]:
    base, ext = os.path.splitext(pdb_path)
    clean_pdb = pdb_path if base.endswith("_model1") else base + "_model1.pdb"
    if not os.path.exists(clean_pdb):
        extract_and_renumber_model1(pdb_path, clean_pdb)
        print(f"[preprocess] Extracted MODEL 1 → {clean_pdb}")
    return compute_charged_residue_counts(clean_pdb, chain, resname, resseq, radius)


def run_hbonds(pdb_path: str, residue_spec: str, radius: float) -> Dict[str, float]:
    cmd = [
        "python", HBOND_SCRIPT,
        pdb_path, residue_spec,
        "--radius", str(radius),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HBonds_Score failed: {result.stderr.strip()}")

    text = result.stdout
    return {
        "hbonds_weighted": _extract_float(text, r"Weighted score .*?:\s*([\d.+-]+)"),
        "hbonds_strict_flexible": _extract_float(text, r"Strict flexible score .*?:\s*([\d.+-]+)"),
    }


def build_features(pdb_path: str, chain: str, resname: str, resseq: int,
                   apbs_radius: float, hbond_radius: float, ph: float, use_apbs: bool = True) -> pd.DataFrame:
    residue_spec = f"{chain}:{resname}:{resseq}"

    # Extract MODEL 1 to handle multi-model NMR structures
    base, ext = os.path.splitext(pdb_path)
    clean_pdb = base + "_model1.pdb"
    if not os.path.exists(clean_pdb):
        extract_and_renumber_model1(pdb_path, clean_pdb)
        print(f"[preprocess] Extracted MODEL 1 → {clean_pdb}")

    ref_pka = REFERENCE_PKA.get(resname, np.nan)
    if np.isnan(ref_pka):
        print(f"[warn] No reference pKa for {resname} — ref_pka will be NaN")

    sasa = load_sasa(pdb_path, chain, resseq, resname)
    if use_apbs:
        apbs = run_apbs(clean_pdb, residue_spec, apbs_radius, ph)
    else:
        # Provide electrostatic_potential as NaN when APBS is skipped so
        # pipelines trained with APBS features don't error on missing columns.
        apbs = {
            "electrostatic_potential": np.nan,
            **get_charged_residue_counts(clean_pdb, chain, resseq, resname, apbs_radius),
        }
    hbonds = run_hbonds(clean_pdb, residue_spec, hbond_radius)

    features = {
        "ref_pka": ref_pka,
        "sasa": sasa,
        **apbs,
        **hbonds,
        "resname": resname,
    }

    # keep electrostatic_potential present (may be NaN) so models trained
    # with APBS still accept this feature set when APBS is disabled

    print("\n[features] Computed features:")
    for k, v in features.items():
        print(f"  {k:30s}: {v}")

    return pd.DataFrame([features])

def log1p_signed(x):
    """log1p for positive values, pass-through for negative."""
    return np.where(x < 0, x, np.log1p(x))


# ---------------------------------------------------------------------------
# Batch analyze from CSV
# ---------------------------------------------------------------------------

def run_analyze(args, pipeline):
    """
    Read a CSV of residues, run prediction for each row, append result
    columns, and write a sorted output CSV.
    """
    import warnings

    df = pd.read_csv(args.analyze)
    prediction_cache = load_prediction_cache()

    required = {"Residue", "Chain", "ResNum"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[analyze] CSV is missing required columns: {missing}")

    pdb_path = find_pdb_path(args.pdb, args.pdb_dir)
    print(f"[analyze] PDB: {pdb_path}")
    print(f"[analyze] Rows to process: {len(df)}\n")

    deprotonation_prob = []
    estimated_pka_list = []
    pka_shift_list     = []

    for idx, row in df.iterrows():
        chain   = str(row["Chain"]).strip().upper()
        resname = str(row["Residue"]).strip().upper()
        resseq  = int(re.sub(r'[^0-9-]', '', str(row["ResNum"]).strip()))
        spec    = f"{chain}:{resname}:{resseq}"
        cache_key = make_prediction_cache_key(
            pdb_path,
            chain,
            resname,
            resseq,
            args.model,
            args.apbs_radius,
            args.hbond_radius,
            args.ph,
            not args.no_apbs,
        )

        print(f"[analyze] ({idx + 1}/{len(df)}) {spec}")
        cached_row = prediction_cache.get(cache_key)
        if cached_row is not None:
            print(f"[cache] Reusing cached deprotonation output for {spec}")
            deprotonation_prob.append(cached_row.get("deprotonation_prob", np.nan))
            estimated_pka_list.append(cached_row.get("estimated_pKa", np.nan))
            pka_shift_list.append(cached_row.get("pKa_shift", np.nan))
            for column_name, value in cached_row.items():
                if column_name not in df.columns:
                    df.loc[idx, column_name] = value
                else:
                    df.at[idx, column_name] = value
            continue

        try:
            X    = build_features(pdb_path, chain, resname, resseq,
                                   args.apbs_radius, args.hbond_radius, args.ph, use_apbs=not args.no_apbs)
            prob = float(np.clip(pipeline.predict(X)[0], 0.0, 1.0))
            prob_clipped = np.clip(prob, 1e-6, 1 - 1e-6)
            ref_pka      = REFERENCE_PKA.get(resname, np.nan)
            est_pka      = args.ph - np.log10(prob_clipped / (1 - prob_clipped))
            shift        = est_pka - ref_pka if not np.isnan(ref_pka) else np.nan

            deprotonation_prob.append(prob)
            estimated_pka_list.append(round(est_pka, 4))
            pka_shift_list.append(round(shift, 4) if not np.isnan(shift) else np.nan)

            cached_row = row.to_dict()
            cached_row.update({
                "deprotonation_prob": prob,
                "estimated_pKa": round(est_pka, 4),
                "pKa_shift": round(shift, 4) if not np.isnan(shift) else np.nan,
            })
            prediction_cache[cache_key] = cached_row

        except Exception as exc:
            warnings.warn(f"[analyze] Failed for {spec}: {exc}")
            deprotonation_prob.append(np.nan)
            estimated_pka_list.append(np.nan)
            pka_shift_list.append(np.nan)

    # --- Attach prediction columns ---
    df["deprotonation_prob"]        = deprotonation_prob
    df["estimated_pKa"]             = estimated_pka_list
    df["pKa_shift"]                 = pka_shift_list

    # Absolute_Deprotonation_State: old 0.5 cutoff
    df["Absolute_Deprotonation_State"] = (df["deprotonation_prob"] >= 0.5).astype(int)

    # Deprotonation_State: new 0.2 cutoff based on the FPR analysis in Find_Optimal_Threshold.py
    df["Deprotonation_State"] = (df["deprotonation_prob"] >= 0.2).astype(int)

    # --- Helper: coerce binary columns safely ---
    def to_int(col):
        return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    bsd  = to_int("Binary_Score_Deprotonated")      # Binary_Score_Deprotonated
    bacc = to_int("Binary_Accessible")
    bod  = to_int("Binary_Orbital_Deprotonated")
    brd  = to_int("Binary_Reactivity_Deprotonated")
    ds   = df["Deprotonation_State"]

    # --- Phase 1 ---
    # Any row where Binary_Score_Deprotonated=1 AND Deprotonation_State=1 → score=1
    phase1_mask = (bsd == 1) & (ds == 1)
    all_bsd_zero = (bsd == 0).all()

    bsdwd = pd.Series(0, index=df.index)  # Binary_Score_Deprotonated_With_Deprot_Score

    if not all_bsd_zero:
        # Phase 1 applies
        bsdwd[phase1_mask] = 1
    else:
        # --- Phase 2: all Binary_Score_Deprotonated == 0 ---
        # Sort rows by Reactivity_Score_Deprotonated descending, check each individually
        rsd = pd.to_numeric(df["Reactivity_Score_Deprotonated"], errors="coerce").fillna(0)
        sorted_idx = rsd.sort_values(ascending=False).index

        for i in sorted_idx:
            if (bacc[i] == 1 and bod[i] == 1 and ds[i] == 1):
                bsdwd[i] = 1

    df["Binary_Score_Deprotonated_With_Deprot_Score"] = bsdwd

    # --- Weak_Bond ---
    df["Weak_Bond"] = ((bsdwd == 1) & (brd == 0)).astype(int)

    # --- Ranked_Success_Deprotonated ---
    # Priority: bsdwd=1 first (desc deprotonation_prob), then bsdwd=0 (desc deprotonation_prob)
    # --- Ranked_Success_Deprotonated ---
    df["_sort_bsdwd"] = bsdwd
    df["_sort_prob"]  = df["deprotonation_prob"]

    df_sorted = df.sort_values(
        ["_sort_bsdwd", "_sort_prob"],
        ascending=[False, False]
    )
    # Use rank with min method so ties get the same rank number
    df_sorted["Ranked_Success_Deprotonated"] = (
        df_sorted["_sort_prob"].fillna(-1.0)
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df = df_sorted.copy()

    # --- Ranked_Success_Reactivity ---
    rsd_col = pd.to_numeric(df["Reactivity_Score_Deprotonated"], errors="coerce").fillna(0)
    df["_sort_rsd"] = rsd_col
    df["_sort_brd"] = pd.to_numeric(df["Binary_Reactivity_Deprotonated"], errors="coerce").fillna(0)

    df_sorted2 = df.sort_values(
        ["_sort_bsdwd", "_sort_brd", "_sort_rsd"],
        ascending=[False, False, False]
    )
    # Rank within the full sorted order, ties get same rank
    df_sorted2["Ranked_Success_Reactivity"] = (
        df_sorted2["_sort_rsd"].fillna(-1.0)
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df = df_sorted2.copy()

    # --- Final sort ---
    # bsdwd=1 first desc deprotonation_prob, then bsdwd=0 desc deprotonation_prob
    df = df.sort_values(
        ["_sort_bsdwd", "_sort_prob"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Drop helper columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_sort")])

    save_prediction_cache(prediction_cache)

    # --- Output ---
    out_dir  = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.analyze))
    out_name = os.path.splitext(os.path.basename(args.analyze))[0] + "_deprot_predictions.csv"
    out_path = os.path.join(out_dir, out_name)

    df.to_csv(out_path, index=False)
    print(f"\n[analyze] Done. Results written to: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict deprotonation probability for a residue using a trained XGBoost model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Deprotonation_Model.py deprot_xgb.pkl 4g5j.pdb A:CYS:797
  python Deprotonation_Model.py deprot_xgb.pkl 4g5j.pdb A:HIS:42 --pdb-dir ./pdbs
  python Deprotonation_Model.py deprot_xgb.pkl 2rvq.pdb D:HIS:49 --apbs-radius 12.0
        """
    )
    parser.add_argument("model",        help="Path to trained pipeline (.pkl from joblib)")
    parser.add_argument("pdb",          help="PDB file path or 4-letter PDB ID")
    parser.add_argument("residue",      nargs="?", default=None,
                        help="Residue spec, e.g. A:CYS:797 (not needed in --analyze mode)")
    parser.add_argument("--pdb-dir",    default=".", help="Directory to search/download PDB files")
    parser.add_argument("--apbs-radius", type=float, default=12.0, help="APBS sphere radius (default: 12.0)")
    parser.add_argument("--hbond-radius", type=float, default=6.0,  help="H-bond search radius (default: 6.0)")
    parser.add_argument("--ph",         type=float, default=PHYSIOLOGIC_PH, help="pH for APBS (default: 7.4)")
    parser.add_argument("--no-apbs", action="store_true", help="Skip APBS electrostatics and use only charged-residue counts")
    parser.add_argument("--analyze",    default=None,
                        help="Path to input CSV for batch prediction mode")
    parser.add_argument("--out-dir",    default=None,
                        help="Directory for the output CSV (default: same as input CSV)")
    args = parser.parse_args()

    # Load model
    try:
        import joblib
    except ImportError:
        raise SystemExit("joblib is required. Install with: pip install joblib")

    if not os.path.isfile(args.model):
        raise SystemExit(f"Model file not found: {args.model}")

    print(f"[model] Loading pipeline from {args.model}...")
    pipeline = joblib.load(args.model)

    # ── Batch analyze mode ──────────────────────────────────────────────────
    if args.analyze:
        run_analyze(args, pipeline)
        return
    # ────────────────────────────────────────────────────────────────────────

    if args.residue is None:
        raise SystemExit("residue is required in single-residue mode (omit only with --analyze)")

    # Parse residue spec
    parts = args.residue.upper().replace("-", ":").split(":")
    if len(parts) != 3:
        raise SystemExit(f"Residue spec must be CHAIN:RESNAME:RESNUM, got: {args.residue}")
    chain, resname, resseq = parts[0], parts[1], int(parts[2])

    # Find PDB
    pdb_path = find_pdb_path(args.pdb, args.pdb_dir)
    print(f"[pdb] Using: {pdb_path}")
    print(f"[residue] {chain}:{resname}:{resseq}  |  ref pKa: {REFERENCE_PKA.get(resname, 'unknown')}")

    # Build features
    X = build_features(pdb_path, chain, resname, resseq,
                       args.apbs_radius, args.hbond_radius, args.ph, use_apbs=not args.no_apbs)

    # Predict
    prob = pipeline.predict(X)[0]
    prob = float(np.clip(prob, 0.0, 1.0))  # safety clip to [0, 1]

    prob_clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    estimated_pka = args.ph - np.log10(prob_clipped / (1 - prob_clipped))

    print(f"\n{'='*50}")
    print(f"  DEPROTONATION PREDICTION")
    print(f"{'='*50}")
    print(f"  Residue:          {chain}:{resname}:{resseq}")
    print(f"  Reference pKa:    {REFERENCE_PKA.get(resname, 'N/A')}")
    print(f"  pH:               {args.ph}")
    print(f"  P(deprotonation): {prob:.4f}")
    print(f"  Estimated pKa:    {estimated_pka:.2f}")
    print(f"  pKa shift:        {estimated_pka - REFERENCE_PKA.get(resname, estimated_pka):+.2f} from reference")
    if prob >= 0.5:
        verdict = "LIKELY DEPROTONATED"
    elif prob <= 0.5:
        verdict = "LIKELY PROTONATED"
    else:
        verdict = "MIXED / UNCERTAIN"
    print(f"  Verdict:          {verdict}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()