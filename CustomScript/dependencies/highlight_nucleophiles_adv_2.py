import os
import subprocess
import re
import importlib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA

# Approximate pKa thresholds for nucleophilic residues
PKA_THRESHOLDS = {
    "CYS": 8.5,
    "SER": 14.0,
    "THR": 14.0,
    "TYR": 11.0,
    "HIS": 7.0,
    "LYS": 10.5,
}

# Strict SASA cutoffs (Å²) for side-chain exposure based on Tien et al. 2013
SASA_CUTOFFS = {
    "CYS": 15.0,
    "SER": 15.0,
    "THR": 15.0,
    "TYR": 15.0,
    "HIS": 15.0,
    "LYS": 15.0,
}

WATER_RESIDUES = {"HOH", "WAT", "H2O", "DOD", "SOL"}

ION_HETATM_RESIDUES = {
    "NA", "NA1", "K", "K1", "CA", "CA2", "MG", "MG2", "ZN", "ZN2",
    "MN", "MN2", "FE", "FE2", "FE3", "CU", "CO", "NI", "CL", "BR", "IOD",
}

PRESERVE_HETATM_RESIDUES = ION_HETATM_RESIDUES | {
    "SO4", "PO4", "HEM", "HEME", "FAD", "FMN", "NAD",
    "NAP", "SAM", "SAH", "GDP", "GTP", "ADP", "ATP", "CMP", "UMP", "GMP",
    "AMP", "FUC", "MAN", "NAG", "BMA", "MSE",
}


def _cache_is_fresh(cache_path, source_path):
    """True when cache exists and is at least as new as the source file."""
    if not cache_path or not source_path or not os.path.exists(cache_path):
        return False
    try:
        return os.path.getmtime(cache_path) >= os.path.getmtime(source_path)
    except OSError:
        return False


def _pdb_record_name(line):
    return line[:6].replace("\t", " ").strip().upper()


def extract_ion_coords(pdb_path):
    """Return (x, y, z, resname) for crystallographic ion HETATM."""
    ions = []
    with open(pdb_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if _pdb_record_name(line) != "HETATM" or len(line) < 54:
                continue
            resname = line[17:20].strip().upper()
            if resname in WATER_RESIDUES or resname not in ION_HETATM_RESIDUES:
                continue
            ions.append(
                (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                    resname,
                )
            )
    return ions


def make_freesasa_structure(pdb_path, include_ions=False, ion_coords=None):
    """Build a FreeSASA Structure; ions are addAtom() spheres, not HETATM."""
    freesasa = importlib.import_module("freesasa")
    structure = freesasa.Structure(pdb_path)
    if not include_ions:
        return structure

    added = 0
    for i, (x, y, z, resname) in enumerate(ion_coords or (), start=1):
        try:
            # ALA CA is in the default classifier; residue ZN is not.
            structure.addAtom("CA", "ALA", str(9000 + i), "Z", x, y, z)
            added += 1
        except Exception as exc:
            print(f"[freesasa] WARNING: addAtom failed for {resname}: {exc}")
    if added == 0:
        print(
            "[freesasa] WARNING: --ions is on but no ion atoms were added. "
            "Check that Zn/Mg/Ca HETATM are present in the PDB."
        )
    else:
        print(f"[freesasa] Added {added} ion atom(s) via addAtom (ALA CA chain Z)")
    return structure


def clean_pdb_for_sasa(pdb_path, include_ions=False):
    """Write a cleaned PDB for SASA calculations.

    Keeps ATOM records and drops waters. Without include_ions, a small
    allowlist of ions/cofactors is written but FreeSASA still ignores HETATM
    by default. With include_ions, HETATM are dropped here and ions are added
    later with FreeSASA addAtom() using a known ALA CA probe.
    """
    suffix = "_sasa_cleaned_ions_v4.pdb" if include_ions else "_sasa_cleaned.pdb"
    cleaned_path = os.path.splitext(os.path.abspath(pdb_path))[0] + suffix

    if _cache_is_fresh(cleaned_path, pdb_path):
        return cleaned_path

    hetatm_keep = PRESERVE_HETATM_RESIDUES if not include_ions else set()

    with open(pdb_path, "r") as handle_in, open(cleaned_path, "w") as handle_out:
        for line in handle_in:
            record = _pdb_record_name(line)
            if record == "ATOM":
                handle_out.write(line)
                continue

            if record != "HETATM":
                continue

            resname = line[17:20].strip().upper()
            if resname in WATER_RESIDUES:
                continue
            if resname in hetatm_keep:
                handle_out.write(line)

    return cleaned_path

def estimate_electrophile_sasa(smiles):
    """Optional: calculate electrophile SASA for reference."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol_h = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    if success != 0:
        raise RuntimeError("3D coordinate embedding failed")

    AllChem.UFFOptimizeMolecule(mol_h)

    radii = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
             'F': 1.47, 'P': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}
    atom_radii = [radii.get(atom.GetSymbol(), 1.7) for atom in mol_h.GetAtoms()]

    sasa = rdFreeSASA.CalcSASA(mol_h, atom_radii)
    return sasa

def run_freesasa(pdb_path, include_ions=False):
    """Run FreeSASA via Python API and save RSA-style output to organized directory."""
    # Create sasa_output directory in the same location as the PDB file
    pdb_dir = os.path.dirname(os.path.abspath(pdb_path))
    sasa_output_dir = os.path.join(pdb_dir, "sasa_output")
    os.makedirs(sasa_output_dir, exist_ok=True)
    
    # Save .rsa file in the sasa_output directory
    pdb_basename = os.path.basename(pdb_path)
    rsa_stem = os.path.splitext(pdb_basename)[0]
    rsa_suffix = "_sasa_ions_v4.rsa" if include_ions else "_sasa.rsa"
    rsa_file = os.path.join(sasa_output_dir, rsa_stem + rsa_suffix)

    cleaned_pdb = clean_pdb_for_sasa(pdb_path, include_ions=include_ions)
    if _cache_is_fresh(rsa_file, pdb_path) and _cache_is_fresh(rsa_file, cleaned_pdb):
        return rsa_file

    ion_coords = extract_ion_coords(pdb_path) if include_ions else []
    try:
        freesasa = importlib.import_module("freesasa")
        structure = make_freesasa_structure(
            cleaned_pdb, include_ions=include_ions, ion_coords=ion_coords
        )
    except ImportError as exc:
        raise FileNotFoundError("freesasa Python package not installed") from exc
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()
    if include_ions:
        dummy_chains = [
            str(chain_id)
            for chain_id in residue_areas
            if str(chain_id).strip() == "Z"
        ]
        print(
            f"[freesasa] Dummy ion chain present in SASA: "
            f"{bool(dummy_chains)} (atoms requested={len(ion_coords)})"
        )

    with open(rsa_file, "w") as handle:
        for chain_id, residues in residue_areas.items():
            for res_id, areas in residues.items():
                res_num_match = re.search(r"-?\d+", str(res_id))
                res_num = res_num_match.group(0) if res_num_match else str(res_id).strip()
                res_name = str(getattr(areas, "residueType", "UNK")).strip().upper()
                side_abs = float(getattr(areas, "sideChain", 0.0) or 0.0)
                side_rel = getattr(areas, "relativeSideChain", 0.0)
                side_rel = 0.0 if side_rel is None else float(side_rel) * 100.0
                chain_str = str(chain_id).strip() if str(chain_id).strip() else "_"
                # Keep token positions compatible with parse_rsa_file.
                handle.write(
                    f"RES {res_name} {chain_str} {res_num} ABS {side_abs:.2f} REL {side_rel:.2f}\n"
                )

    return rsa_file

def run_propka(pdb_path):
    """Run PROPKA and move output to organized directory."""
    # Create pka_output directory in the same location as the PDB file
    pdb_dir = os.path.dirname(os.path.abspath(pdb_path))
    pka_output_dir = os.path.join(pdb_dir, "pka_output")
    os.makedirs(pka_output_dir, exist_ok=True)
    
    # PROPKA creates output in current working directory by default
    # We need to run it and then move the file to our organized directory
    pdb_basename = os.path.basename(pdb_path)
    pdb_base_no_ext = os.path.splitext(pdb_basename)[0]
    pka_file_dest = os.path.join(pka_output_dir, f"{pdb_base_no_ext}.pka")

    if os.path.exists(pka_file_dest):
        return pka_file_dest
    
    # Run PROPKA (it will create .pka file in current directory or PDB directory)
    subprocess.run(["propka3", pdb_path], check=True)
    
    # Look for the .pka file in multiple possible locations
    possible_pka_locations = [
        os.path.join(os.getcwd(), f"{pdb_base_no_ext}.pka"),  # Current directory
        os.path.join(pdb_dir, f"{pdb_base_no_ext}.pka"),      # PDB directory
        os.path.splitext(pdb_path)[0] + ".pka"                 # Next to PDB file
    ]
    
    pka_file_source = None
    for loc in possible_pka_locations:
        if os.path.exists(loc):
            pka_file_source = loc
            break
    
    if pka_file_source is None:
        raise FileNotFoundError(f"PROPKA output file not found. Checked: {possible_pka_locations}")
    
    # Move to organized directory
    if os.path.abspath(pka_file_source) != os.path.abspath(pka_file_dest):
        import shutil
        shutil.move(pka_file_source, pka_file_dest)
    
    return pka_file_dest

def parse_rsa_file(rsa_file):
    exposure = {}
    with open(rsa_file, 'r') as f:
        for line in f:
            if line.startswith('RES'):
                parts = line.split()
                if len(parts) < 8:
                    continue
                resname = parts[1]
                chain = parts[2]
                resnum = parts[3]
                try:
                    abs_side_sasa = float(parts[5]) if parts[5] != 'N/A' else -1.0
                    rel_side_sasa = float(parts[7]) if parts[7] != 'N/A' else -1.0
                except ValueError:
                    abs_side_sasa, rel_side_sasa = -1.0, -1.0
                exposure[(resname, chain, resnum)] = (abs_side_sasa, rel_side_sasa)
    return exposure

def parse_propka_file(path):
    pka_data = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isalpha() and parts[1].isdigit():
                try:
                    pka_data[(parts[0], parts[2], parts[1])] = float(parts[3])
                except ValueError:
                    continue
    return pka_data

def score_druggability(pKa, rel_side_sasa, resname):
    if rel_side_sasa == -1.0:
        return "n/a", "n/a", "n/a"

    sasa_cutoff = SASA_CUTOFFS.get(resname, 50.0)  # default cutoff 50 Å²
    is_accessible = rel_side_sasa >= sasa_cutoff
    is_deprotonated = resname in PKA_THRESHOLDS and pKa < PKA_THRESHOLDS[resname]

    if pKa == -1.0:
        return is_accessible, "n/a", "n/a"

    score = 1.0 if (is_accessible and is_deprotonated) else 0.5 if (is_accessible or is_deprotonated) else 0.0
    return is_accessible, is_deprotonated, score

def main(pdb_path, smiles=None):
    if smiles:
        electrophile_sasa = estimate_electrophile_sasa(smiles)
        print(f"\n🧪 Electrophile SASA (for reference): {electrophile_sasa:.2f} Å²")
    else:
        print("\n🧪 Electrophile SMILES not provided. Skipping electrophile SASA calculation.")

    print("📏 Using fixed SASA cutoffs per residue type (based on literature)\n")

    rsa_file = run_freesasa(pdb_path)
    pka_file = run_propka(pdb_path)

    exposure = parse_rsa_file(rsa_file)
    pka_data = parse_propka_file(pka_file)

    rows = []
    for (resname, chain, resnum), (abs_side_sasa, rel_side_sasa) in exposure.items():
        if resname not in PKA_THRESHOLDS:
            continue
        pKa = pka_data.get((resname, chain, resnum), -1.0)
        acc, dep, score = score_druggability(pKa, rel_side_sasa, resname)
        rows.append({
            "Residue": resname, "Chain": chain, "ResNum": resnum,
            "pKa": pKa, "Abs_Side_SASA": abs_side_sasa, "Rel_Side_SASA": rel_side_sasa,
            "Accessible": acc, "Deprotonated": dep, "Score": score
        })

    # Sort so that Accessible = True rows are at the top
    df = pd.DataFrame(rows)
    df.sort_values(by="Accessible", ascending=False, inplace=True)

    out_path = f"{os.path.splitext(pdb_path)[0]}_covalent_hotspots.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Analysis complete! Output written to: {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python highlight_nucleophiles_strict.py <pdb_file> [electrophile_smiles]")
    else:
        pdb_file = sys.argv[1]
        smiles = sys.argv[2] if len(sys.argv) > 2 else None
        main(pdb_file, smiles)
