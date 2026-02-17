#!/usr/bin/env python3
"""
Frankenstein.py - Comprehensive Covalent Binding Detector

Combines nucleophile identification (from highlight_nucleophiles_adv_2.py) with 
electrophile-nucleophile reactivity scoring (from single_AA_bond.py).

This script:
1. Identifies accessible nucleophilic residues in a protein structure (via SASA and pKa)
2. Takes an electrophile SMILES string as input
3. Auto-detects reactive electrophilic centers/warheads in the electrophile
4. Scores each accessible nucleophile against the detected reactive centers
5. Outputs ranked matches for covalent binding hotspots

Dependencies:
  - RDKit
  - xTB (executable `xtb` on PATH)
  - numpy, scipy, pandas
  - freesasa, propka3

Author: Combined from highlight_nucleophiles_adv_2.py and single_AA_bond.py
"""

import os
import sys
import json
import pandas as pd
from rdkit import Chem

# Add paths to import existing scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
existing_structures_path = os.path.join(project_root, 'Existing_Structures')
sys.path.insert(0, script_dir)
sys.path.insert(0, existing_structures_path)

# Import from existing scripts
try:
    # Import from highlight_nucleophiles_adv_2.py
    import highlight_nucleophiles_adv_2 as hn_adv
    
    # Import from single_AA_bond.py
    import single_AA_bond as saa
    
    # Import from covalent_orbital_requirements.py
    import covalent_orbital_requirements as cor
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Make sure highlight_nucleophiles_adv_2.py, single_AA_bond.py, and covalent_orbital_requirements.py are accessible.")
    sys.exit(1)

# ============================================================================
# CONFIGURATION - Import from existing scripts
# ============================================================================

# Use configurations from the original scripts
PKA_THRESHOLDS = hn_adv.PKA_THRESHOLDS
SASA_CUTOFFS = hn_adv.SASA_CUTOFFS
SURROGATES = saa.SURROGATES
WEIGHTS = saa.WEIGHTS
XTBCMD = saa.XTBCMD

# Deprotonated (anionic) forms for nucleophiles - more reactive
SURROGATES_DEPROTONATED = {
    "Cys": "C[S-]",              # thiolate anion
    "Ser": "C[O-]",              # methoxide anion
    "Thr": "CC[O-]",             # ethoxide anion
    "Tyr": "c1ccc([O-])cc1",     # phenoxide anion
    "Lys": "NCC",                # neutral amine (deprotonated from NH3+)
    "His": "c1cn[n-]c1"          # imidazolate (already anionic)
}

# Binary scoring configuration
REACTIVITY_CUTOFF = 0.5  # Minimum reactivity score for covalent binding consideration

# HSAB (Hard-Soft Acid-Base) classification for amino acid targeting
# Note: Borderline electrophiles can interact with all nucleophile types (less selective)
HSAB_TO_AA = {
    "soft": ["CYS", "SEC"],                    # thiols, selenols - prefer soft electrophiles
    "borderline": ["CYS", "SEC", "HIS", "TYR", "LYS", "SER", "THR", "ASP", "GLU"],  # can react with all
    "hard": ["LYS", "SER", "THR", "ASP", "GLU"]  # amines, alcohols, carboxylates - prefer hard electrophiles
}

# Nucleophile SMILES mapping (for orbital compatibility check)
NUCLEOPHILE_SMILES_MAP = {
    "CYS": "CS",              # Cys surrogate
    "SER": "CO",              # Ser surrogate
    "THR": "CCO",             # Thr surrogate
    "TYR": "c1ccc(O)cc1",     # Tyr surrogate
    "LYS": "NCC",             # Lys surrogate
    "HIS": "c1cn[n-]c1"       # His surrogate
}

# Nucleophile reactive atom indices
NUCLEOPHILE_ATOM_INDEX = {
    "CS": 1,              # Cys: sulfur
    "CO": 1,              # Ser: oxygen
    "CCO": 2,             # Thr: oxygen
    "c1ccc(O)cc1": 4,     # Tyr: oxygen
    "NCC": 0,             # Lys: nitrogen
    "c1cn[n-]c1": 4,      # His: carbon in ring
    # Deprotonated forms
    "C[S-]": 1,           # Cys thiolate: sulfur
    "C[O-]": 1,           # Ser methoxide: oxygen
    "CC[O-]": 2,          # Thr ethoxide: oxygen
    "c1ccc([O-])cc1": 4   # Tyr phenoxide: oxygen
}

# ============================================================================
# ELECTROPHILE REACTIVE CENTER DETECTION
# ============================================================================

# TODO: Expand this list with more electrophilic warheads
# Each entry: (name, SMARTS pattern, reactive_atom_index, hsab_classification)
# reactive_atom_index: which atom in the SMARTS match is the reactive center (0-indexed)
# hsab_classification: "soft", "borderline", or "hard" for HSAB-based amino acid targeting
ELECTROPHILE_WARHEADS = [
    # Michael acceptors
    ("Alpha-beta unsaturated carbonyl (Michael acceptor)", "[CX3](=O)[CX3]=[CX3]", 3, "soft"),  # β-carbon (index 3 in SMARTS match = C(=O)-C=C pattern where β is last)
    ("Acrylamide warhead", "C=CC(=O)N", 0, "soft"),  # β-carbon of C=CC
    ("Vinyl sulfone", "C=CS(=O)(=O)", 0, "soft"),    # β-carbon (double bond carbon)

    # Carbonyl-based electrophiles
    ("Aldehyde", "[CX3H1](=O)[#6]", 0, "hard"),      # carbonyl carbon
    ("Ketone (reactive)", "[CX3](=O)([#6])[#6]", 0, "soft"),  # carbonyl carbon
    ("Activated ketone", "C(=O)C(=O)", 0, "soft"),   # carbonyl carbon (1,2-dicarbonyl)

    # Esters (acyl transfer agents)
    ("Phenyl ester", "[CX3](=O)Oc1ccccc1", 0, "hard"),  # aspirin-like, carbonyl carbon
    ("Activated ester", "[CX3](=O)O[#6]", 0, "hard"),   # general ester, carbonyl carbon

    # Halogenated carbonyls
    ("Alpha-halo carbonyl", "[CX4][F,Cl,Br,I]C(=O)", 0, "soft"),  # carbon attached to halogen
    ("Fluoromethyl ketone", "C(=O)CF", 1, "soft"),                # carbonyl carbon

    # Epoxides and aziridines
    ("Epoxide", "C1OC1", 0, "hard"),     # one of the ring carbons (usually less hindered)
    ("Aziridine", "C1NC1", 0, "hard"),   # carbon in the three-membered ring

    # Nitriles
    ("Nitrile (electrophilic)", "[C]#N", 0, "soft"),  # nitrile carbon

    # Aminium ions (positively charged nitrogen)
    ("Aminium ion", "[NX4+]", 0, "hard"),  # quaternary ammonium nitrogen
    ("Protonated amine", "[NX3+]", 0, "hard"),  # protonated tertiary amine

    # Leaving group displacement
    ("Alkyl halide (good LG)", "[CX4][Br,I]", 0, "soft"),  # carbon attached to halogen
    ("Alkyl chloride", "[CX4]Cl", 0, "borderline"),             # carbon attached to Cl

    # Sulfonyl-based
    ("Sulfonyl fluoride", "S(=O)(=O)F", 0, "borderline"),      # sulfur
    ("Sulfonamide (activated)", "[NX3][S](=O)(=O)", 1, "borderline"),  # sulfur as reactive center

    # Disulfides
    ("Disulfide", "[SX2][SX2]", 0, "soft"),              # sulfur

    # Lactones/lactams (strained rings)
    ("Beta-lactone", "C1OC(=O)C1", 2, "hard"),  # carbonyl carbon in ring
    ("Beta-lactam", "C1NC(=O)C1", 2, "hard"),   # carbonyl carbon in ring

    # Isocyanates/isothiocyanates
    ("Isocyanate", "N=C=O", 1, "hard"),         # carbon between N and O
    ("Isothiocyanate", "N=C=S", 1, "soft"),     # carbon between N and S

    # Boronic acids (reversible covalent)
    ("Boronic acid", "[BX3](O)O", 0, "hard"),   # boron
]


def detect_electrophile_warheads(mol):
    """
    Detects reactive electrophilic centers in a molecule using SMARTS patterns.
    
    Returns:
        List of tuples: [(warhead_name, warhead_smarts, reactive_atom_idx, hsab_class), ...]
        
    NOTE: reactive_atom_idx uses the index specified in ELECTROPHILE_WARHEADS
          to select the correct atom from the SMARTS match.
    """
    if mol is None:
        return []
    
    detected = []
    for name, smarts, reactive_atom_index, hsab_class in ELECTROPHILE_WARHEADS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            print(f"Warning: Invalid SMARTS pattern for {name}: {smarts}")
            continue
        # Find match of warhead
        #Getsubtract returns an array and each index is the atom index in the warhead and the value is the absolute atom index in the electrophile
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            # Use the specified atom index from the SMARTS match as reactive center
            if reactive_atom_index < len(match):
                #Return the absolute atom index
                reactive_idx = match[reactive_atom_index]
                print(f"Detected warhead: {name} at atom index {reactive_idx} with match {match}")
                detected.append((name, smarts, reactive_idx, hsab_class))
            else:
                print(f"Warning: reactive_atom_index {reactive_atom_index} out of range for {name}")
    
    return detected

def get_hsab_target_residues(warhead_hsab):
    """
    Get the list of target amino acid types based on HSAB classification.
    
    Args:
        warhead_hsab: HSAB classification ("soft", "borderline", or "hard")
        
    Returns:
        List of amino acid types (uppercase 3-letter codes)
    """
    return HSAB_TO_AA.get(warhead_hsab, [])

def check_hsab_match(residue_type, warhead_hsab):
    """
    Check if a nucleophile residue type matches the HSAB-predicted targets.
    
    Args:
        residue_type: 3-letter amino acid code (e.g., "CYS")
        warhead_hsab: HSAB classification ("soft", "borderline", or "hard")
        
    Returns:
        Boolean: True if residue matches HSAB prediction, False otherwise
    """
    target_residues = get_hsab_target_residues(warhead_hsab)
    return residue_type.upper() in target_residues

# ============================================================================
# PROTEIN NUCLEOPHILE IDENTIFICATION (use functions from highlight_nucleophiles_adv_2.py)
# ============================================================================

# Use functions directly from imported module
estimate_electrophile_sasa = hn_adv.estimate_electrophile_sasa
run_freesasa = hn_adv.run_freesasa
run_propka = hn_adv.run_propka
parse_rsa_file = hn_adv.parse_rsa_file
parse_propka_file = hn_adv.parse_propka_file
score_nucleophile_accessibility = hn_adv.score_druggability  # Note: renamed in original

def identify_accessible_nucleophiles(pdb_path):
    """
    Identify accessible nucleophilic residues in protein structure.
    Uses functions from highlight_nucleophiles_adv_2.py.
    
    Returns:
        DataFrame with columns: Residue, Chain, ResNum, pKa, Side_SASA, Total_SASA,
                                Accessible, Deprotonated, Score
    """
    print("🔬 Running FreeSASA and PROPKA...")
    
    # Convert pdb_path to absolute path to ensure consistent file resolution
    pdb_path = os.path.abspath(pdb_path)
    pdb_dir = os.path.dirname(pdb_path)
    pdb_base = os.path.splitext(os.path.basename(pdb_path))[0]
    
    try:
        # Run FreeSASA - it creates output in the same directory as input
        rsa_file = run_freesasa(pdb_path)
        # Check if it exists, if not look in current directory
        if not os.path.exists(rsa_file):
            rsa_file_alt = os.path.join(os.getcwd(), f"{pdb_base}_sasa.rsa")
            if os.path.exists(rsa_file_alt):
                rsa_file = rsa_file_alt
        if not os.path.exists(rsa_file):
            raise FileNotFoundError(f"FreeSASA output file not found: {rsa_file}")
    except FileNotFoundError as e:
        if "freesasa" in str(e).lower() or "No such file" in str(e):
            print("\n❌ ERROR: 'freesasa' command not found!")
            print("Please install FreeSASA:")
            print("  - Windows: Download from https://freesasa.github.io/")
            print("  - Linux/Mac: conda install -c salilab freesasa")
            print("  - Or: pip install freesasa")
            print("\nMake sure it's in your PATH after installation.")
        raise
    
    try:
        # Run PROPKA - it creates output in the CURRENT directory by default
        pka_file = run_propka(pdb_path)
        # PROPKA creates the file in current directory, not with the input
        pka_file_basename = os.path.basename(pka_file)
        pka_file_in_cwd = os.path.join(os.getcwd(), pka_file_basename)
        
        # Check multiple possible locations for the .pka file
        if os.path.exists(pka_file):
            # Already at expected location
            pass
        elif os.path.exists(pka_file_in_cwd):
            # Found in current working directory
            pka_file = pka_file_in_cwd
        else:
            raise FileNotFoundError(f"PROPKA output file not found. Checked: {pka_file} and {pka_file_in_cwd}")
    except FileNotFoundError as e:
        if "propka" in str(e).lower() and "not found" in str(e).lower() and "output" not in str(e).lower():
            print("\n❌ ERROR: 'propka3' command not found!")
            print("Please install PROPKA:")
            print("  - pip install propka")
            print("\nMake sure it's in your PATH after installation.")
        raise

    exposure = parse_rsa_file(rsa_file)
    pka_data = parse_propka_file(pka_file)

    rows = []
    for (resname, chain, resnum), (total_sasa, side_sasa) in exposure.items():
        if resname not in PKA_THRESHOLDS:
            continue
        pKa = pka_data.get((resname, chain, resnum), -1.0)
        acc, dep, score = hn_adv.score_druggability(pKa, side_sasa, resname)
        rows.append({
            "Residue": resname,
            "Chain": chain,
            "ResNum": resnum,
            "pKa": pKa,
            "Side_SASA": side_sasa,
            "Total_SASA": total_sasa,
            "Accessible": acc,
            "Deprotonated": dep,
            "Accessibility_Score": 1.0 if acc else 0.0  # Use only accessibility (ignore deprotonation)
        })

    df = pd.DataFrame(rows)
    # Return all nucleophiles (both accessible and inaccessible)
    return df

# ============================================================================
# QUANTUM REACTIVITY CALCULATIONS (use functions from single_AA_bond.py)
# ============================================================================

# Use functions directly from imported module
rdkit_mol_from_smiles = saa.rdkit_mol_from_smiles
write_xyz = saa.write_xyz
run_xtb_xyz = saa.run_xtb_xyz
get_lg_score = saa.get_lg_score

def compute_reactivity_score(electrophile_smiles, reactive_idx, nucleophile_type, protonation_state="protonated"):
    """
    Compute intrinsic reactivity score between electrophile and nucleophile.
    Uses core calculation functions from single_AA_bond.py.
    
    Args:
        electrophile_smiles: SMILES of electrophile
        reactive_idx: Index of reactive atom in electrophile
        nucleophile_type: Type of amino acid (e.g., "CYS", "SER")
        protonation_state: "protonated" or "deprotonated" (default: "protonated")
    
    NOTE: This is a simplified version that computes individual descriptors.
    The full compute_score function from single_AA_bond.py includes adduct formation,
    which is currently problematic. We extract the working parts here.
    
    TODO: Consider using saa.compute_score() directly if adduct issues are resolved.
    
    Returns:
        dict with scoring components
    """
    import tempfile
    import shutil
    
    tmpdir = tempfile.mkdtemp(prefix="react_")
    
    try:
        # Build electrophile
        e_mol = rdkit_mol_from_smiles(electrophile_smiles)
        e_xyz = os.path.join(tmpdir, "electrophile.xyz")
        write_xyz(e_mol, e_xyz)
        
        # Run xTB on neutral electrophile
        e_neutral = run_xtb_xyz(e_xyz, charge=0, gbsa="water")
        
        # Run xTB on anion for Fukui function
        e_anion = run_xtb_xyz(e_xyz, charge=-1, gbsa="water")
        
        # Build nucleophile surrogate based on protonation state
        if protonation_state == "deprotonated":
            sur_smiles = SURROGATES_DEPROTONATED.get(nucleophile_type, None)
        else:
            sur_smiles = SURROGATES.get(nucleophile_type, None)
            
        if sur_smiles is None:
            raise ValueError(f"Unsupported nucleophile type: {nucleophile_type} ({protonation_state})")
        
        n_mol = rdkit_mol_from_smiles(sur_smiles)
        n_xyz = os.path.join(tmpdir, "nuc.xyz")
        write_xyz(n_mol, n_xyz)
        
        # Determine charge for nucleophile based on protonation state
        # Deprotonated forms of acids (Cys, Ser, Thr, Tyr) have -1 charge
        # His already has -1 in both forms, Lys is neutral in both
        if protonation_state == "deprotonated" and nucleophile_type in ["Cys", "Ser", "Thr", "Tyr", "His"]:
            nuc_charge = -1
        else:
            nuc_charge = 0
            
        n_neutral = run_xtb_xyz(n_xyz, charge=nuc_charge, gbsa="water")
        
        # TODO: Implement robust adduct formation here
        # Currently disabled due to geometry/valence issues in original script
        # Would compute: deltaE = E(adduct) - E(electrophile) - E(nucleophile)
        
        # Compute descriptors
        lumo = e_neutral["lumo"]
        
        # Fukui f+ on reactive atom: q(N+1) - q(N)
        fukui = 0.0
        partial_charge = 0.0
        if e_neutral["charges"] and e_anion["charges"]:
            if len(e_neutral["charges"]) > reactive_idx and len(e_anion["charges"]) > reactive_idx:
                qN = e_neutral["charges"][reactive_idx]
                qNp1 = e_anion["charges"][reactive_idx]
                fukui = qNp1 - qN
                partial_charge = qN
        
        # Leaving group score
        lg_score = get_lg_score(electrophile_smiles.lower())
        
        # HOMO-LUMO gap
        min_gap = 1.0  # eV, very reactive
        max_gap = 6.0  # eV, less reactive
        
        if n_neutral["homo"] is not None and e_neutral["lumo"] is not None:
            homo_lumo_gap = e_neutral["lumo"] - n_neutral["homo"]
        else:
            homo_lumo_gap = 12.0  # fallback
        
        # Normalize HOMO-LUMO gap to 0-1 (lower gap = higher score)
        if homo_lumo_gap <= min_gap:
            HL_n = 1.0
        elif homo_lumo_gap >= max_gap:
            HL_n = 0.0
        else:
            HL_n = (max_gap - homo_lumo_gap) / (max_gap - min_gap)
        
        # Normalize other components
        def norm(x, xmin, xmax):
            if xmax == xmin:
                return 0.5
            return max(0.0, min(1.0, (x - xmin) / (xmax - xmin)))
        
        L_n = norm(-lumo, -10.0, 10.0)  # lower LUMO = higher score
        F_n = norm(fukui, -0.5, 0.5)
        G_n = lg_score  # already 0-1
        
        # Weighted composite score (using weights from single_AA_bond.py)
        S_raw = (WEIGHTS["lumo"] * L_n + 
                 WEIGHTS["fukui"] * F_n + 
                 WEIGHTS["lg"] * G_n + 
                 WEIGHTS["homo_lumo_gap"] * HL_n)
        
        return {
            "reactivity_score": S_raw,
            "lumo": lumo,
            "lumo_normalized": L_n,
            "fukui": fukui,
            "fukui_normalized": F_n,
            "leaving_group_score": G_n,
            "homo_lumo_gap": homo_lumo_gap,
            "homo_lumo_gap_normalized": HL_n,
            "partial_charge": partial_charge,
        }
    
    finally:
        shutil.rmtree(tmpdir)

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(pdb_path, electrophile_smiles, output_prefix="Frankenstein", top_n_types=3):
    """
    Main workflow: 
    1. Determine top N most reactive amino acid TYPES for the electrophile (quantum calculations)
    2. Identify accessible nucleophiles in protein (all types)
    3. Filter to only keep accessible nucleophiles matching the top N reactive types
    4. Combine reactivity scores (from step 1) with accessibility scores (from step 2)
    5. Output ranked results
    
    Args:
        pdb_path: Path to PDB file
        electrophile_smiles: SMILES string of electrophile
        output_prefix: Prefix for output files
        top_n_types: Number of top reactive amino acid types to consider (default: 3)
    """
    # Create output directory
    output_dir = f"{output_prefix}_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🧟 FRANKENSTEIN - Covalent Binding Detector 🧟")
    print("=" * 80)
    print(f"📁 Output directory: {output_dir}/")
    
    # Step 1: Determine top reactive amino acid TYPES for this electrophile
    print("\n⚡ STEP 1: Determining most reactive amino acid types for electrophile...")
    print(f"   Electrophile: {electrophile_smiles}")
    
    # Detect warheads
    e_mol = Chem.MolFromSmiles(electrophile_smiles)
    if e_mol is None:
        print("❌ Error: Invalid electrophile SMILES")
        return
    
    warheads = detect_electrophile_warheads(e_mol)
    if len(warheads) == 0:
        print("⚠️  No reactive warheads detected in electrophile.")
        print("TODO: The warhead detection may need expansion for this molecule.")
        return
    
    print(f"   Detected {len(warheads)} warhead(s):")
    for i, (name, smarts, idx, hsab) in enumerate(warheads, 1):
        hsab_targets = get_hsab_target_residues(hsab)
        print(f"   {i}. {name} (reactive atom index: {idx})")
        print(f"      HSAB: {hsab} → Target residues: {', '.join(hsab_targets)}")
    
    # Process ALL warheads (not just the top one)
    print(f"\n   Processing all {len(warheads)} warhead(s)...")
    
    all_warhead_results = {}  # Store results for each warhead
    
    for warhead_idx, (warhead_name, warhead_smarts, reactive_idx, hsab_class) in enumerate(warheads, 1):
        print(f"\n   [{warhead_idx}/{len(warheads)}] Warhead: {warhead_name}")
        print(f"   Reactive atom index: {reactive_idx} (atom: {e_mol.GetAtomWithIdx(reactive_idx).GetSymbol()})")
        print(f"   HSAB classification: {hsab_class}")
        print(f"   Computing intrinsic reactivity for each amino acid type...")
        
        type_scores_protonated = {}
        type_scores_deprotonated = {}
        
        for aa_type in SURROGATES.keys():
            print(f"      - Testing {aa_type}...")
            
            # Test protonated form
            print(f"        Protonated: ", end="")
            try:
                score_data = compute_reactivity_score(
                    electrophile_smiles,
                    reactive_idx,
                    aa_type,
                    protonation_state="protonated"
                )
                type_scores_protonated[aa_type] = score_data
                print(f"Reactivity: {score_data['reactivity_score']:.3f}")
            except Exception as e:
                print(f"Error: {e}")
                type_scores_protonated[aa_type] = {"reactivity_score": 0.0}
            
            # Test deprotonated form
            print(f"        Deprotonated: ", end="")
            try:
                score_data = compute_reactivity_score(
                    electrophile_smiles,
                    reactive_idx,
                    aa_type,
                    protonation_state="deprotonated"
                )
                type_scores_deprotonated[aa_type] = score_data
                print(f"Reactivity: {score_data['reactivity_score']:.3f}")
            except Exception as e:
                print(f"Error: {e}")
                type_scores_deprotonated[aa_type] = {"reactivity_score": 0.0}
        
        # Get top N amino acid types for this warhead (separate for protonated and deprotonated)
        sorted_types_protonated = sorted(type_scores_protonated.items(), key=lambda x: x[1]["reactivity_score"], reverse=True)
        sorted_types_deprotonated = sorted(type_scores_deprotonated.items(), key=lambda x: x[1]["reactivity_score"], reverse=True)
        
        top_types_protonated = [aa_type.strip().upper() for aa_type, score_dict in sorted_types_protonated[:top_n_types]]
        top_types_deprotonated = [aa_type.strip().upper() for aa_type, score_dict in sorted_types_deprotonated[:top_n_types]]
        
        # Create uppercase version of type_scores for matching with uppercase residue names
        type_scores_protonated_upper = {aa_type.upper(): score_dict for aa_type, score_dict in type_scores_protonated.items()}
        type_scores_deprotonated_upper = {aa_type.upper(): score_dict for aa_type, score_dict in type_scores_deprotonated.items()}
        
        print(f"      ✅ Top {top_n_types} reactive types for this warhead:")
        print(f"         Protonated:")
        for i, (aa_type, score_dict) in enumerate(sorted_types_protonated[:top_n_types], 1):
            print(f"           {i}. {aa_type} (Score: {score_dict['reactivity_score']:.3f})")
        print(f"         Deprotonated:")
        for i, (aa_type, score_dict) in enumerate(sorted_types_deprotonated[:top_n_types], 1):
            print(f"           {i}. {aa_type} (Score: {score_dict['reactivity_score']:.3f})")
        
        # Store results for this warhead
        all_warhead_results[warhead_name] = {
            "warhead_smarts": warhead_smarts,
            "reactive_idx": reactive_idx,
            "hsab_class": hsab_class,
            "hsab_targets": get_hsab_target_residues(hsab_class),
            "type_scores_protonated": type_scores_protonated_upper,
            "type_scores_deprotonated": type_scores_deprotonated_upper,
            "top_types_protonated": top_types_protonated,
            "top_types_deprotonated": top_types_deprotonated
        }
    
    # Save all warhead reactivity scores
    all_scores_json = os.path.join(output_dir, "all_warheads_reactivity.json")
    with open(all_scores_json, 'w') as f:
        json.dump(all_warhead_results, f, indent=2)
    print(f"\n   💾 Saved all warhead reactivity scores to: {all_scores_json}")
    
    # Step 2: Identify all nucleophiles in protein (accessible and inaccessible)
    print("\n📍 STEP 2: Identifying nucleophiles in protein...")
    all_nucleophiles = identify_accessible_nucleophiles(pdb_path)
    accessible_count = all_nucleophiles[all_nucleophiles["Accessible"] == True].shape[0]
    print(f"   Found {len(all_nucleophiles)} total nucleophiles ({accessible_count} accessible, {len(all_nucleophiles) - accessible_count} inaccessible)")
    
    if len(all_nucleophiles) == 0:
        print("⚠️  No nucleophiles found. Exiting.")
        return
    
    # Save all nucleophiles for reference
    all_nucleophiles_csv = os.path.join(output_dir, "all_accessibility_nucleophiles.csv")
    all_nucleophiles.to_csv(all_nucleophiles_csv, index=False)
    print(f"   💾 Saved all nucleophiles to: {all_nucleophiles_csv}")
    
    # Filter to only accessible nucleophiles for further processing
    all_accessible = all_nucleophiles[all_nucleophiles["Accessible"] == True].copy()
    
    if len(all_accessible) == 0:
        print("⚠️  No accessible nucleophiles found. Exiting.")
        return
    
    # Strip whitespace and convert to uppercase for case-insensitive matching
    all_accessible["Residue"] = all_accessible["Residue"].str.strip().str.upper()
    
    # Step 3-5: Process results for each warhead
    print(f"\n🔬 STEP 3-5: Processing results for each warhead...")
    
    all_results = []  # Collect all results across warheads
    
    for warhead_name, warhead_data in all_warhead_results.items():
        print(f"\n   Processing warhead: {warhead_name}")
        
        top_types_protonated = warhead_data["top_types_protonated"]
        top_types_deprotonated = warhead_data["top_types_deprotonated"]
        type_scores_protonated = warhead_data["type_scores_protonated"]
        type_scores_deprotonated = warhead_data["type_scores_deprotonated"]
        hsab_class = warhead_data["hsab_class"]
        hsab_targets = warhead_data["hsab_targets"]
        
        # Get union of top types from both protonation states for initial filtering
        all_top_types = list(set(top_types_protonated + top_types_deprotonated))
        filtered_nucleophiles = all_accessible[all_accessible["Residue"].isin(all_top_types)].copy()
        
        if len(filtered_nucleophiles) == 0:
            print(f"      ⚠️  No accessible nucleophiles match top {top_n_types} reactive types for this warhead")
            continue
        
        print(f"      Found {len(filtered_nucleophiles)} matching nucleophiles (union of protonated and deprotonated top types)")
        
        # Convert scores to numeric, handling 'n/a' and other non-numeric values
        filtered_nucleophiles["Accessibility_Score"] = pd.to_numeric(
            filtered_nucleophiles["Accessibility_Score"], errors='coerce'
        )
        
        # Add warhead name and reactivity scores (both protonated and deprotonated)
        filtered_nucleophiles["Warhead"] = warhead_name
        
        # Extract reactivity scores for both protonation states
        filtered_nucleophiles["Reactivity_Score_Protonated"] = filtered_nucleophiles["Residue"].map(
            lambda res: type_scores_protonated.get(res, {}).get("reactivity_score", 0.0) if isinstance(type_scores_protonated.get(res), dict) else 0.0
        )
        filtered_nucleophiles["Reactivity_Score_Deprotonated"] = filtered_nucleophiles["Residue"].map(
            lambda res: type_scores_deprotonated.get(res, {}).get("reactivity_score", 0.0) if isinstance(type_scores_deprotonated.get(res), dict) else 0.0
        )
        
        # Use deprotonated score as primary reactivity score
        filtered_nucleophiles["Reactivity_Score"] = filtered_nucleophiles["Reactivity_Score_Deprotonated"]
        
        # Remove rows with NaN scores
        filtered_nucleophiles = filtered_nucleophiles.dropna(subset=["Accessibility_Score", "Reactivity_Score_Protonated", "Reactivity_Score_Deprotonated"])
        
        if len(filtered_nucleophiles) == 0:
            print(f"      ⚠️  No valid scores after filtering NaN values")
            continue
        
        # Calculate combined score (accessibility × reactivity)
        filtered_nucleophiles["Combined_Score"] = (
            filtered_nucleophiles["Accessibility_Score"] * 
            filtered_nucleophiles["Reactivity_Score"]
        )
        
        # Add binary filter columns
        print(f"      Computing binary filters...")
        
        # Binary filter 1: Accessible (already True from filtering, but make explicit)
        filtered_nucleophiles["Binary_Accessible"] = filtered_nucleophiles["Accessible"].astype(int)
        
        # Binary filter 2: Top N reactive type (will be applied separately for each protonation state)
        # Mark as 1 if in either top list (will be refined per output file)
        filtered_nucleophiles["Binary_Reactivity_Protonated"] = (
            filtered_nucleophiles["Residue"].isin(top_types_protonated)
        ).astype(int)
        filtered_nucleophiles["Binary_Reactivity_Deprotonated"] = (
            filtered_nucleophiles["Residue"].isin(top_types_deprotonated)
        ).astype(int)
        
        # Binary filter 3: Orbital compatibility check
        reactive_idx = warhead_data["reactive_idx"]
        orbital_compatible = []
        
        for idx, row in filtered_nucleophiles.iterrows():
            residue_type = row["Residue"]
            nuc_smiles = NUCLEOPHILE_SMILES_MAP.get(residue_type)
            
            if nuc_smiles:
                nuc_atom_idx = NUCLEOPHILE_ATOM_INDEX.get(nuc_smiles, 0)
                try:
                    result = cor.paper_check_interaction(
                        nuc_smiles, nuc_atom_idx, 
                        electrophile_smiles, reactive_idx
                    )
                    orbital_compatible.append(1 if result["covalent_bond_possible"] else 0)
                except Exception as e:
                    print(f"         Warning: Orbital check failed for {residue_type}: {e}")
                    orbital_compatible.append(0)
            else:
                orbital_compatible.append(0)
        
        filtered_nucleophiles["Binary_Orbital"] = orbital_compatible
        
        # Binary filter 4: HSAB match check
        hsab_matches = []
        for idx, row in filtered_nucleophiles.iterrows():
            residue_type = row["Residue"]
            is_match = check_hsab_match(residue_type, hsab_class)
            hsab_matches.append(1 if is_match else 0)
        
        filtered_nucleophiles["Binary_HSAB"] = hsab_matches
        
        # Calculate final binary scores (separate for each protonation state)
        filtered_nucleophiles["Binary_Score_Protonated"] = (
            (filtered_nucleophiles["Binary_Accessible"] == 1) &
            (filtered_nucleophiles["Binary_Reactivity_Protonated"] == 1) &
            (filtered_nucleophiles["Binary_Orbital"] == 1) &
            (filtered_nucleophiles["Binary_HSAB"] == 1)
        ).astype(int)
        filtered_nucleophiles["Binary_Score_Deprotonated"] = (
            (filtered_nucleophiles["Binary_Accessible"] == 1) &
            (filtered_nucleophiles["Binary_Reactivity_Deprotonated"] == 1) &
            (filtered_nucleophiles["Binary_Orbital"] == 1) &
            (filtered_nucleophiles["Binary_HSAB"] == 1)
        ).astype(int)
        
        print(f"      Computed scores for {len(filtered_nucleophiles)} valid nucleophiles")
        print(f"      Binary filters: Accessible={filtered_nucleophiles['Binary_Accessible'].sum()}, "
              f"Orbital={filtered_nucleophiles['Binary_Orbital'].sum()}, "
              f"HSAB={filtered_nucleophiles['Binary_HSAB'].sum()}")
        print(f"      Binary score (Protonated): {filtered_nucleophiles['Binary_Score_Protonated'].sum()} pass all filters")
        print(f"      Binary score (Deprotonated): {filtered_nucleophiles['Binary_Score_Deprotonated'].sum()} pass all filters")
        
        # Create separate dataframes for protonated and deprotonated
        warhead_safe_name = warhead_name.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Protonated version - filter to only top N protonated types
        df_protonated = filtered_nucleophiles[filtered_nucleophiles["Residue"].isin(top_types_protonated)].copy()
        df_protonated["Reactivity_Score"] = df_protonated["Reactivity_Score_Protonated"]
        df_protonated["Binary_Reactivity"] = df_protonated["Binary_Reactivity_Protonated"]
        df_protonated["Binary_Score"] = df_protonated["Binary_Score_Protonated"]
        df_protonated["Combined_Score"] = (
            df_protonated["Accessibility_Score"] * 
            df_protonated["Reactivity_Score_Protonated"]
        )
        df_protonated_sorted = df_protonated.sort_values(
            by=["Binary_Score", "Combined_Score"], 
            ascending=[False, False]
        )
        protonated_csv = os.path.join(output_dir, f"protonated_ranked_targets_{warhead_safe_name}.csv")
        df_protonated_sorted.to_csv(protonated_csv, index=False)
        print(f"      💾 Saved protonated (top {top_n_types} protonated types) to: {protonated_csv}")
        
        # Deprotonated version - filter to only top N deprotonated types
        df_deprotonated = filtered_nucleophiles[filtered_nucleophiles["Residue"].isin(top_types_deprotonated)].copy()
        df_deprotonated["Reactivity_Score"] = df_deprotonated["Reactivity_Score_Deprotonated"]
        df_deprotonated["Binary_Reactivity"] = df_deprotonated["Binary_Reactivity_Deprotonated"]
        df_deprotonated["Binary_Score"] = df_deprotonated["Binary_Score_Deprotonated"]
        df_deprotonated["Combined_Score"] = (
            df_deprotonated["Accessibility_Score"] * 
            df_deprotonated["Reactivity_Score_Deprotonated"]
        )
        df_deprotonated_sorted = df_deprotonated.sort_values(
            by=["Binary_Score", "Combined_Score"], 
            ascending=[False, False]
        )
        deprotonated_csv = os.path.join(output_dir, f"deprotonated_ranked_targets_{warhead_safe_name}.csv")
        df_deprotonated_sorted.to_csv(deprotonated_csv, index=False)
        print(f"      💾 Saved deprotonated (top {top_n_types} deprotonated types) to: {deprotonated_csv}")
        
        # Add to combined results
        all_results.append({
            "protonated": df_protonated,
            "deprotonated": df_deprotonated
        })
    
    # Combine all warhead results
    if len(all_results) == 0:
        print("\n⚠️  No results generated for any warhead. Exiting.")
        return
    
    # Separate protonated and deprotonated results
    all_protonated = [r["protonated"] for r in all_results]
    all_deprotonated = [r["deprotonated"] for r in all_results]
    
    combined_protonated = pd.concat(all_protonated, ignore_index=True)
    combined_protonated.sort_values(
        by=["Binary_Score", "Combined_Score"], 
        ascending=[False, False], 
        inplace=True
    )
    
    combined_deprotonated = pd.concat(all_deprotonated, ignore_index=True)
    combined_deprotonated.sort_values(
        by=["Binary_Score", "Combined_Score"], 
        ascending=[False, False], 
        inplace=True
    )
    
    # Save combined results for both states
    protonated_combined_csv = os.path.join(output_dir, "protonated_ranked_covalent_targets_all_warheads.csv")
    combined_protonated.to_csv(protonated_combined_csv, index=False)
    print(f"\n   💾 Saved combined protonated results to: {protonated_combined_csv}")
    
    deprotonated_combined_csv = os.path.join(output_dir, "deprotonated_ranked_covalent_targets_all_warheads.csv")
    combined_deprotonated.to_csv(deprotonated_combined_csv, index=False)
    print(f"   💾 Saved combined deprotonated results to: {deprotonated_combined_csv}")
    
    # Display top hits across all warheads (using deprotonated as primary)
    print("\n🏆 TOP 10 COVALENT BINDING CANDIDATES - DEPROTONATED (across all warheads):")
    print("=" * 90)
    print(f"{'Residue':<12} | {'Warhead':<38} | {'Score':<8} | {'Binary':>6}")
    print("-" * 90)
    top_10_deprot = combined_deprotonated.head(10)
    for idx, row in top_10_deprot.iterrows():
        nuc_id = f"{row['Residue']}{row['ResNum']}:{row['Chain']}"
        warhead_short = row['Warhead'][:35] + "..." if len(row['Warhead']) > 38 else row['Warhead']
        binary_status = "✓" if row.get('Binary_Score', 0) == 1 else "✗"
        print(f"{nuc_id:12} | {warhead_short:38} | {row['Combined_Score']:8.3f} | {binary_status:>6}")
    
    print("\n🏆 TOP 10 COVALENT BINDING CANDIDATES - PROTONATED (across all warheads):")
    print("=" * 90)
    print(f"{'Residue':<12} | {'Warhead':<38} | {'Score':<8} | {'Binary':>6}")
    print("-" * 90)
    top_10_prot = combined_protonated.head(10)
    for idx, row in top_10_prot.iterrows():
        nuc_id = f"{row['Residue']}{row['ResNum']}:{row['Chain']}"
        warhead_short = row['Warhead'][:35] + "..." if len(row['Warhead']) > 38 else row['Warhead']
        binary_status = "✓" if row.get('Binary_Score', 0) == 1 else "✗"
        print(f"{nuc_id:12} | {warhead_short:38} | {row['Combined_Score']:8.3f} | {binary_status:>6}")
    
    # Display summary statistics
    total_candidates_deprot = len(combined_deprotonated)
    binary_pass_deprot = combined_deprotonated.get('Binary_Score', pd.Series([0])).sum()
    total_candidates_prot = len(combined_protonated)
    binary_pass_prot = combined_protonated.get('Binary_Score', pd.Series([0])).sum()
    print("\n" + "=" * 90)
    print(f"📊 Summary (Deprotonated): {binary_pass_deprot}/{total_candidates_deprot} candidates pass all binary filters")
    print(f"📊 Summary (Protonated): {binary_pass_prot}/{total_candidates_prot} candidates pass all binary filters")
    print(f"   Filters: Accessible=1, Top {top_n_types} Reactive Type=True, Orbital Compatible=True, HSAB Match=True")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"📂 All output files saved to: {output_dir}/")
    print(f"   📄 all_warheads_reactivity.json - Reactivity scores for all warheads")
    print(f"   📄 all_accessibility_nucleophiles.csv - All nucleophiles (accessible & inaccessible)")
    print(f"   📄 protonated_ranked_targets_<warhead>.csv - Protonated results per warhead")
    print(f"   📄 deprotonated_ranked_targets_<warhead>.csv - Deprotonated results per warhead")
    print(f"   📄 protonated_ranked_covalent_targets_all_warheads.csv - Combined protonated results")
    print(f"   📄 deprotonated_ranked_covalent_targets_all_warheads.csv - Combined deprotonated results")
    print("=" * 80)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Frankenstein.py <pdb_file> <electrophile_smiles> [output_prefix] [top_n_types]")
        print()
        print("Arguments:")
        print("  pdb_file           : Path to protein PDB file")
        print("  electrophile_smiles: SMILES string of electrophile")
        print("  output_prefix      : Prefix for output files (default: Frankenstein)")
        print("  top_n_types        : Number of top reactive AA types to consider (default: 3)")
        print()
        print("Example:")
        print('  python Frankenstein.py protein.pdb "C=CC(=O)Nc1cc(F)cc(Cl)c1" afatinib 3')
        print()
        print("Workflow:(quantum calculations once per type)")
        print("  2. Identifies accessible nucleophiles in protein (SASA/pKa analysis)")
        print("  3. Filters to only accessible residues matching top N reactive types")
        print("  4. Combines reactivity × accessibility scores")
        print("  5. Outputs ranked covalent binding predictionng top N reactive types")
        print("  4. Scores filtered nucleophiles and outputs results")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    electrophile_smiles = sys.argv[2]
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else "Frankenstein"
    top_n_types = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    
    if not os.path.exists(pdb_file):
        print(f"❌ Error: PDB file not found: {pdb_file}")
        sys.exit(1)
    
    main(pdb_file, electrophile_smiles, output_prefix, top_n_types)
