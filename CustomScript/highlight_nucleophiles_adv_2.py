import os
import subprocess
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

def run_freesasa(pdb_path):
    rsa_file = os.path.splitext(pdb_path)[0] + "_sasa.rsa"
    subprocess.run(["freesasa", pdb_path, "--format=rsa", f"--output={rsa_file}"], check=True)
    return rsa_file

def run_propka(pdb_path):
    subprocess.run(["propka3", pdb_path], check=True)
    return os.path.splitext(pdb_path)[0] + ".pka"

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
                    total_sasa = float(parts[5]) if parts[5] != 'N/A' else -1.0
                    side_sasa = float(parts[7]) if parts[7] != 'N/A' else -1.0
                except ValueError:
                    total_sasa, side_sasa = -1.0, -1.0
                exposure[(resname, chain, resnum)] = (total_sasa, side_sasa)
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

def score_druggability(pKa, side_sasa, resname):
    if side_sasa == -1.0:
        return "n/a", "n/a", "n/a"

    sasa_cutoff = SASA_CUTOFFS.get(resname, 50.0)  # default cutoff 50 Å²
    is_accessible = side_sasa >= sasa_cutoff
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
    for (resname, chain, resnum), (total_sasa, side_sasa) in exposure.items():
        if resname not in PKA_THRESHOLDS:
            continue
        pKa = pka_data.get((resname, chain, resnum), -1.0)
        acc, dep, score = score_druggability(pKa, side_sasa, resname)
        rows.append({
            "Residue": resname, "Chain": chain, "ResNum": resnum,
            "pKa": pKa, "Side_SASA": side_sasa, "Total_SASA": total_sasa,
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
