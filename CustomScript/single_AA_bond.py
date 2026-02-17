#!/usr/bin/env python3
"""
intrinsic_reactivity_score.py

Dependencies:
  - RDKit
  - xTB (executable `xtb` on PATH)
  - numpy, scipy
"""

import os
import subprocess
import tempfile
import shutil
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import re

# ---------- USER INPUT ----------
#ELECTROPHILE_SMILES = "CC(C)CC"  # isopentane
#REACTIVE_SMARTS = "[C;H3]"       # any sp3 carbon

#ELECTROPHILE_SMILES = "CCO"  # isopentane
#REACTIVE_SMARTS = "[C;H3]"       # any sp3 carbon

#Afatinib and Cys
ELECTROPHILE_SMILES = "C=CC(=O)Nc1cc(F)cc(Cl)c1" 
REACTIVE_SMARTS = "[C]=[C]-C(=O)"           # example SMARTS to find reactive carbon (tweak per warhead)

#ZYA and Cruzain
#ELECTROPHILE_SMILES = "C[CH](NC(=O)[CH](Cc1ccc(O)cc1)NC(=O)OCc2ccccc2)C(=O)CF"
#REACTIVE_SMARTS = "C(=O)CF"  # reactive carbonyl carbon

#ZYA and Cruzain
#ELECTROPHILE_SMILES = "CC(C)C(NC(=O)c1ccc2ccccc2c1)C(=O)N1CCCC1C(=O)NC(CC(=O)NS(C)(=O)=O)C=O"
#REACTIVE_SMARTS = "[CX3](=O)[NX3][S]"  # reactive carbonyl carbon

#Papain CLIK148
#ELECTROPHILE_SMILES = "CN(C)C(=O)[C@H](Cc1ccccc1)NC(=O)C(=O)CC(=O)NCCc2ccccn2"
#REACTIVE_SMARTS = "NC(=O)C(=O)"  

#ELECTROPHILE_SMILES = "N#Cc3nc1c(ncn1C2CCCC2)c(n3)Nc5ccccc5OCCCn4ccnc4"  # or just "N#CC1=NC=NC=C1"
#REACTIVE_SMARTS = "[C]#N"  # reactive carbon
#NUCLEOPHILE_TYPE = "His"                  # choose from ["Cys","Ser","Lys","His"]
# --------------------------------

# default weights (heuristic)
# Delta E won't mean anything unless if I can do specifically for bond. Even if so, it ignores all the environment and entropy
WEIGHTS = {
    "deltaE": 0.0,
    "lumo":   0.1,
    "fukui":  0.4,
    "lg":     0.1,
    "homo_lumo_gap": 0.4
}
ALPHA = 5.0
BETA = 0.0

# surrogate nucleophiles (choose reactive chemical form)
SURROGATES = {
    "Cys": "CS",       # methyl thiol surrogate ("C-SH"); compute/deprotonate to S- if modelling thiolate
    "Ser": "CO",       # methyl alcohol surrogate ("C-OH") -> can model as neutral or alkoxide (O-)
    "Thr": "CCO",      # optional: threonine-like surrogate (beta-methyl alcohol)
    "Tyr": "c1ccc(O)cc1",  # phenol surrogate — can model as neutral phenol or phenoxide (O-) if deprotonated
    "Lys": "C[NH3+]",  # protonated methylammonium (–NH3+ at physiological pH)
    "His": "c1c[nH]cn1",   # neutral imidazole (protonated form)
    #c1cn[n-]c1 # deprotonated imidazolate
    #"Asp": "CC(=O)O",  # carboxylate surrogate (use deprotonated form COO- for nucleophilicity / general base behavior)
    #"Glu": "CCC(=O)O"   # longer chain carboxylate surrogate
}

XTBCMD = "xtb"  # ensure xtb in PATH

REF_DELTA_E_HARTREE = -0.016  # ~ -10 kcal/mol baseline for single-point correction

# ---------- helpers ----------
def rdkit_mol_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES: " + smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(m)
    return m

def write_xyz(mol, path):
    conf = mol.GetConformer()
    with open(path, 'w') as fh:
        fh.write(f"{mol.GetNumAtoms()}\n\n")
        for i, a in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            fh.write(f"{a.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

def run_xtb_xyz(xyz_path, charge=0, nproc=1, gbsa=None, opt=True):
    """
    Run xTB on the xyz file and parse:
      - Total energy (hartree)
      - HOMO / LUMO energies (eV)
      - Mulliken charges (list)
    Returns dict: { 'energy': float, 'homo': float, 'lumo': float, 'charges': [float], 'raw': str }
    """
    import subprocess, tempfile, shutil, os, re

    workdir = tempfile.mkdtemp(prefix="xtb_")
    shutil.copy(xyz_path, os.path.join(workdir, os.path.basename(xyz_path)))
    cmd = [XTBCMD, os.path.basename(xyz_path), "--gfn", "2", "--charge", str(charge)]
    if opt:
        cmd.append("--opt")
    if gbsa:
        cmd += ["--gbsa", gbsa]

    proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    energy = None
    homo = None
    lumo = None
    charges = []

    # --- Parse total energy ---
    for line in out.splitlines():
        if "TOTAL ENERGY" in line.upper():
            m = re.search(r"([-+]?\d*\.\d+|\d+)", line)
            if m:
                energy = float(m.group(1) if m.lastindex else m.group(0))
                break

    # --- Parse HOMO / LUMO ---
    # Try full orbital list first (preferred)
    for line in out.splitlines():
        line = line.strip()
        if "(HOMO)" in line:
            m = re.search(r"([-+]?\d*\.\d+)\s*\(HOMO\)", line)
            if m:
                homo = float(m.group(1)) # convert Eh -> eV if needed
        if "(LUMO)" in line:
            m = re.search(r"([-+]?\d*\.\d+)\s*\(LUMO\)", line)
            if m:
                lumo = float(m.group(1))

    # --- Parse Mulliken charges ---
    charge_file = os.path.join(workdir, "charges")
    if os.path.exists(charge_file):
        with open(charge_file) as fh:
            for line in fh:
                parts = line.split()
                try:
                    charges.append(float(parts[-1]))
                except:
                    continue
    else:
        for line in out.splitlines():
            if line.startswith("ATOM") and "CHARGE" in line:
                tokens = line.split()
                try:
                    charges.append(float(tokens[-1]))
                except:
                    continue

    # --- cleanup ---
    shutil.rmtree(workdir)

    # --- sanity checks ---
    if energy is None:
        raise RuntimeError("xTB failed to produce TOTAL ENERGY")
    if homo is None or lumo is None:
        print("Warning: HOMO/LUMO not found; check xTB output")
    if not charges:
        print("Warning: Charges not found; check xTB output")

    return {"energy": energy, "homo": homo, "lumo": lumo, "charges": charges, "raw": out}



def find_reactive_atom_index(rdkit_mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError("Invalid reactive SMARTS")
    matches = rdkit_mol.GetSubstructMatches(patt)
    if not matches:
        raise ValueError("No match for reactive SMARTS on electrophile")
    # choose first match's first atom. MAY NEED TO FIX
    print("match", matches)
    return matches[0][0]

def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

# ---------- main workflow ----------
def compute_score(electrophile_smiles, reactive_smarts, nucleophile_type):
    # build electrophile mol and write xyz
    e_mol = rdkit_mol_from_smiles(electrophile_smiles)
    #Find the reactive part of the electrophile
    reactive_idx = find_reactive_atom_index(e_mol, reactive_smarts)
    tmpdir = tempfile.mkdtemp(prefix="react_")
    e_xyz = os.path.join(tmpdir, "electrophile.xyz")
    write_xyz(e_mol, e_xyz)
    # run xTB on neutral electrophile
    e_neutral = run_xtb_xyz(e_xyz, charge=0, gbsa="water")
    # run xTB on anion (N+1 electrons) to compute Fukui f+ (for electrophile accept)
    e_anion = run_xtb_xyz(e_xyz, charge=-1, gbsa="water")
    # build surrogate nucleophile (simple RDKit molecule)
    sur_smiles = SURROGATES.get(nucleophile_type, None)
    if sur_smiles is None:
        raise ValueError("Unsupported nucleophile type: " + str(nucleophile_type))
    n_mol = rdkit_mol_from_smiles(sur_smiles)
    n_xyz = os.path.join(tmpdir, "nuc.xyz")
    write_xyz(n_mol, n_xyz)
    n_neutral = run_xtb_xyz(n_xyz, charge=0, gbsa="water")
    # Build very simple adduct: attach nucleophile to reactive atom by creating a bond in RDKit
    # combine molecules
    ad_mol = Chem.CombineMols(e_mol, n_mol)
    ad_mol = Chem.RWMol(ad_mol)

   # offset for nucleophile atoms
    offset = e_mol.GetNumAtoms()

    # last atom of nucleophile *before* combining
    nuc_last_idx = n_mol.GetNumAtoms() - 1

    # adjusted index in the combined molecule
    # Pick the heavy atom of nucleophile (not H)
    nuc_attach_idx = None
    for i, a in enumerate(n_mol.GetAtoms()):
        if a.GetAtomicNum() > 1:  # skip H
            nuc_attach_idx = offset + i
            break

    print("Nucleophile attach index (heavy atom):", nuc_attach_idx, n_mol.GetAtomWithIdx(nuc_attach_idx - offset).GetSymbol(), "")


    # --- remove one hydrogen from the reactive electrophile atom to satisfy valence ---
    react_atom = ad_mol.GetAtomWithIdx(reactive_idx)
    for nbr in react_atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:  # hydrogen
            print("Removing hydrogen at index:", nbr.GetIdx())
            ad_mol.RemoveAtom(nbr.GetIdx())
            break

    nuc_atom = ad_mol.GetAtomWithIdx(nuc_attach_idx)
    # Collect H neighbors first
    hydrogens_to_remove = [nbr.GetIdx() for nbr in nuc_atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
    print("Hydrogens to remove from nucleophile:", hydrogens_to_remove)
    # Remove them in **reverse order**
    for idx in sorted(hydrogens_to_remove, reverse=True):
        print("Removing hydrogen at index:", idx)
        ad_mol.RemoveAtom(idx)

    
    #for i, a in enumerate(ad_mol.GetAtoms()):
    #    print(f"Atom index {i}: symbol={a.GetSymbol()}, atomic_num={a.GetAtomicNum()}")
    
    print("Reactive index:", reactive_idx)
    print("Nucleophile attach index (with offset):", nuc_attach_idx, "(offset:", offset, "nuc_last_idx:", nuc_last_idx, ")")
    atom = ad_mol.GetAtomWithIdx(nuc_attach_idx)
    print(f"Atom at index {nuc_attach_idx}: symbol={atom.GetSymbol()}, atomic_num={atom.GetAtomicNum()}")
    print("Molecule atom count:", ad_mol.GetNumAtoms())

    if reactive_idx >= ad_mol.GetNumAtoms() or nuc_attach_idx >= ad_mol.GetNumAtoms():
        raise ValueError(
            f"Invalid bond indices: reactive={reactive_idx}, nuc={nuc_attach_idx}, "
            f"total_atoms={ad_mol.GetNumAtoms()}"
        )
    """
    # now add the new bond
    # --- after removing H from electrophile and nucleophile ---
    # 1. Add the new bond
    
    ad_mol.AddBond(int(reactive_idx), int(nuc_attach_idx), Chem.BondType.SINGLE)

    # 2. Update property cache & sanitize
    ad_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(ad_mol)

    # 3. Add explicit Hs again
    ad_mol = Chem.AddHs(ad_mol)

    # 4. **Embed 3D coordinates robustly**
    success = False
    for i in range(5):  # try up to 5 times for better chance
        if AllChem.EmbedMolecule(ad_mol, AllChem.ETKDGv3()) == 0:
            success = True
            break
        print(f"Embed attempt {i+1} failed, retrying")
    if not success:
        raise RuntimeError("RDKit embedding failed for adduct after bond addition")

    # 5. **Optimize geometry with MMFF, fallback to UFF if needed**
    try:
        res = AllChem.MMFFOptimizeMolecule(ad_mol)
        if res != 0:
            print("MMFF did not converge, falling back to UFF")
            AllChem.UFFOptimizeMolecule(ad_mol)
    except:
        print("MMFF failed completely, using UFF")
        AllChem.UFFOptimizeMolecule(ad_mol)

    # 6. Optional: scale slightly if atoms overlap
    conf = ad_mol.GetConformer()
    for i in range(ad_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, [pos.x*1.05, pos.y*1.05, pos.z*1.05])

    # 7. Debug print & write XYZ
    #for i, a in enumerate(ad_mol.GetAtoms()):
    #    print(f"Atom index {i}: symbol={a.GetSymbol()}, atomic_num={a.GetAtomicNum()}")
    write_xyz(ad_mol, os.path.join(tmpdir, "adduct.xyz"))
    print("Neighbors of nucleophile atom:")
    
    nuc_atom = ad_mol.GetAtomWithIdx(nuc_attach_idx)
    print("Nucleophile atom:", nuc_atom.GetSymbol(), "index:", nuc_attach_idx)
    for nbr in nuc_atom.GetNeighbors():
        print(f"Neighbor index: {nbr.GetIdx()}, symbol: {nbr.GetSymbol()}, atomic_num: {nbr.GetAtomicNum()}, explicit valence: {nbr.GetExplicitValence()}")
    print("Num hydrogens:", nuc_atom.GetTotalNumHs())
    print("Valence Nuc Atom:", {nuc_atom.GetExplicitValence()})
    
    ad_xyz = os.path.join(tmpdir, "adduct.xyz")

    write_xyz(ad_mol, ad_xyz)

    ad_res = run_xtb_xyz(ad_xyz, charge=0, gbsa="water")
    # compute descriptors
    # Use energies (units: hartree from xtb) convert to kcal/mol: 1 hartree = 627.509 kcal/mol
    HARTREE_TO_KCAL = 627.509
    with open("xtb_output.txt", "w") as f:
        f.write(e_neutral["raw"])
        f.write(n_neutral["raw"])
        f.write(ad_res["raw"])

    #print("Electrophile neutral xTB raw output:\n", ad_res["raw"])

    if None in (e_neutral["energy"], n_neutral["energy"], ad_res["energy"]):
        raise RuntimeError("xTB energy missing from one of the runs; inspect outputs")
    deltaE_hartree = ad_res["energy"] - (e_neutral["energy"] + n_neutral["energy"])
    """
    # Prepare data to save
    data_to_save = {
        #"adduct_energy_hartree": ad_res["energy"],
        "electrophile_energy_hartree": e_neutral["energy"],
        "nucleophile_energy_hartree": n_neutral["energy"],
        #"deltaE_hartree": deltaE_hartree
    }

    # Write to file (JSON format)
    import json
    with open("energies_output.json", "w") as f:
        json.dump(data_to_save, f, indent=2)


    #deltaE_kcal = deltaE_hartree * HARTREE_TO_KCAL
    # LUMO: use electrophile neutral lumo (xtb may give eV or hartree depending; assume eV parsed)
    lumo = e_neutral["lumo"]
    # Fukui f+ on reactive atom: q(N+1) - q(N)
    qN = None
    qNp1 = None
    if e_neutral["charges"] and e_anion["charges"]:
        try:
            qN = e_neutral["charges"][reactive_idx]
            qNp1 = e_anion["charges"][reactive_idx]
        except:
            # fallback: pick first or nearest length
            if len(e_neutral["charges"]) > reactive_idx:
                qN = e_neutral["charges"][reactive_idx]
            if len(e_anion["charges"]) > reactive_idx:
                qNp1 = e_anion["charges"][reactive_idx]
    if qN is None or qNp1 is None:
        # fallback small value
        fukui = 0.0
    else:
        fukui = qNp1 - qN
    # partial charge at reactive atom (neutral)
    partial_charge = qN if qN is not None else 0.0
    # leaving group score heuristic: if electrophile contains halide -> good; else moderate
    lg_score = 0.0
    smi = electrophile_smiles.lower()

    min_gap = 1.0  # eV, very reactive
    max_gap = 6.0  # eV, less reactive

    # Compute HOMO-LUMO gap
    if n_neutral["homo"] is not None and e_neutral["lumo"] is not None:
        homo_lumo_gap = e_neutral["lumo"] - n_neutral["homo"]  # eV
    else:
        homo_lumo_gap = 12.0  # fallback large gap

    # Normalize to 0-1
    if homo_lumo_gap <= min_gap:
        HL_n = 1.0
    elif homo_lumo_gap >= max_gap:
        HL_n = 0.0
    else:
        HL_n = (max_gap - homo_lumo_gap) / (max_gap - min_gap)


    lg_score = get_lg_score(smi)
    # Compose raw score components (we will normalize with simple transforms; better: calibrate with dataset)
    # We want: E = -deltaE_kcal (exergonic -> positive), L = -lumo (lower LUMO -> positive), F = fukui, G = lg_score
    #E = -deltaE_kcal
    L = - (lumo if lumo is not None else 0.0)
    F = fukui
    G = lg_score
    # Simple min-max normalizations with heuristic ranges (replace with dataset-based min/max for production)
    # Heuristic ranges:
    # deltaE_kcal: [-100, 10] -> after E=-delta: [-10,100] ; set min= -100-> maps to 0
    def norm(x, xmin, xmax):
        if xmax == xmin: return 0.5
        return max(0.0, min(1.0, (x - xmin) / (xmax - xmin)))
    #E_n = norm(E, -10.0, 80.0)      # heuristic
    L_n = norm(L, -10.0, 10.0)
    F_n = norm(F, -0.5, 0.5)
    G_n = G  # already 0..1
    # weighted sum
    S_raw = WEIGHTS["lumo"]*L_n + WEIGHTS["fukui"]*F_n + WEIGHTS["lg"]*G_n + WEIGHTS["homo_lumo_gap"]*HL_n
    #Sigmoid is useless because why would I need this if my score is already between 0 and 1
    #score = sigmoid(ALPHA * S_raw + BETA)
    # cleanup temp
    shutil.rmtree(tmpdir)
    # return dict
    return {
        #"deltaE_kcal": deltaE_kcal, "E_n": E_n, 
        "L_n": L_n, "F_n": F_n, "G": G, "HL_n": HL_n,
        "score_raw": S_raw,
        "partial_charge": partial_charge, "lumo": lumo, "fukui": fukui,
        "homo_lumo_gap": homo_lumo_gap
    }

def get_lg_score(smi):
    """
    Returns a leaving group score between 0 and 1.
    Rough approximation based on common leaving groups.
    """
    # Simple lookup of common leaving groups
    lg_dict = {
        "I": 1.0,       # iodide – excellent
        "Br": 0.9,      # bromide – very good
        "Cl": 0.7,      # chloride – moderate
        "OTs": 1.0,     # tosylate – excellent
        "OMs": 0.9,     # mesylate – very good
        "OH": 0.2,      # hydroxide – poor
        "F": 0.1,       # fluoride – very poor
        "H": 0.0,       # hydrogen – not leaving
    }

    # Default score if no match found
    score = 0.5

    # Check for exact leaving group matches in SMILES
    for lg, lg_score in lg_dict.items():
        if lg in smi:
            score = lg_score
            break

    return score


if __name__ == "__main__":
    results = {}
    for nuc_type in SURROGATES.keys():
        print(f"Computing score for nucleophile: {nuc_type}")
        try:
            out = compute_score(ELECTROPHILE_SMILES, REACTIVE_SMARTS, nuc_type)
            results[nuc_type] = out
        except Exception as e:
            print(f"Error for {nuc_type}: {e}")
            results[nuc_type] = {"error": str(e)}

    # Print nicely
    import json
    print(json.dumps(results, indent=2))

    # Optional: save to JSON file
    with open("all_nucleophile_scores.json", "w") as f:
        json.dump(results, f, indent=2)

