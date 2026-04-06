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
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    "His": "c1c[nH]cn1"          # neutral imidazole — nucleophilic form is neutral, not imidazolate anion
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

# Helper function to convert residue type to surrogate key format
def get_surrogate_key(residue_type_uppercase):
    """Convert 3-letter uppercase code (CYS) to surrogate key format (Cys)"""
    mapping = {
        "CYS": "Cys",
        "SER": "Ser", 
        "THR": "Thr",
        "TYR": "Tyr",
        "LYS": "Lys",
        "HIS": "His"
    }
    return mapping.get(residue_type_uppercase)

# Nucleophile reactive atom indices
NUCLEOPHILE_ATOM_INDEX = {
    # Protonated/neutral forms
    "CS": 1,              # Cys: sulfur
    "CO": 1,              # Ser: oxygen
    "CCO": 2,             # Thr: oxygen
    "c1ccc(O)cc1": 4,     # Tyr: oxygen
    "C[NH3+]": 1,         # Lys protonated: nitrogen (non-nucleophilic, no lone pair at N+)
    "NCC": 0,             # Lys deprotonated: nitrogen
    "c1c[nH]cn1": 4,      # His: free (pyridine-type) N is the nucleophile, not the N-H (index 2)
    "c1cn[n-]c1": 3,      # His imidazolate: anionic nitrogen (kept for reference) 
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
    ("Vinyl sulfone", "C=C[SX4](=O)(=O)[#6]", 0, "soft"),        # carbon on both sides
    ("Vinyl sulfonate ester", "C=C[SX4](=O)(=O)O[#6]", 0, "soft"),  # sulfur bonded to alkoxy leaving group
    ("Vinylsulfonamide", "C=C[SX4](=O)(=O)[NX3]", 0, "soft"),    # nitrogen on sulfonyl

    # Carbonyl-based electrophiles
    ("Aldehyde", "[CX3H1](=O)[#6]", 0, "hard"),      # carbonyl carbon
    ("Ketone (reactive)", "[CX3](=O)([#6])[#6]", 0, "soft"),  # carbonyl carbon
    ("Activated ketone", "[#6X3](=O)[#6X3](=O)", 0, "soft"),   # carbonyl carbon (1,2-dicarbonyl, includes aromatic quinone-like motifs)

    # Esters (acyl transfer agents)
    ("Phenyl ester", "[CX3](=O)Oc1ccccc1", 0, "hard"),  # aspirin-like, carbonyl carbon
    ("Activated ester", "[CX3;!$(C(=O)N)](=O)O[#6]", 0, "hard"),   # exclude amide-like carbamates
    ("Carbamate (amide-like)", "[NX3][CX3](=O)O[#6]", 1, "hard"),  # carbonyl carbon is less electrophilic due amide resonance

    # Halogenated carbonyls
    ("Alpha-halo carbonyl", "[CX4][F,Cl,Br,I]C(=O)", 0, "soft"),  # carbon attached to halogen
    ("Fluoromethyl ketone", "C(=O)CF", 1, "soft"),                # carbonyl carbon

    # Epoxides and aziridines
    ("Epoxide", "C1OC1", 0, "hard"),     # one of the ring carbons (usually less hindered)
    ("Alpha-beta epoxyketone (epoxide)", "C(=O)C1OC1", 4, "borderline"),  # proteasome inhibitor - beta carbon in epoxide ring (more reactive)
    ("Alpha-beta epoxyketone (carbonyl)", "C(=O)C1OC1", 0, "hard"),  # proteasome inhibitor - ketone carbon
    ("Aziridine", "C1NC1", 0, "hard"),   # carbon in the three-membered ring

    # Nitriles
    ("Nitrile (electrophilic)", "[C]#N", 0, "soft"),  # nitrile carbon
    ("Aromatic nitrile (cathepsin-like)", "[c][CX2]#[NX1]", 0, "soft"),  # aryl nitrile specifically

    # Aminium ions (positively charged nitrogen)
    ("Aminium ion", "[NX4+;H0]", 0, "hard"),  # quaternary ammonium nitrogen
    ("Protonated amine", "[NX4+;H1,H2,H3]", 0, "hard"),  # protonated amine (exclude nitro-like N+)

    # Leaving group displacement
    ("Alkyl halide (good LG)", "[CX4][Br,I]", 0, "soft"),  # carbon attached to halogen
    ("Alkyl chloride", "[CX4]Cl", 0, "borderline"),             # carbon attached to Cl
    ("Nitro-activated aryl halide (SNAr)", "[c]1[c]([N+](=O)[O-])[c][c][c]([F,Cl,Br,I])[c]1", 7, "soft"),  # aryl carbon bearing halide is the SNAr attack site

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

    # Imine-based (reversible covalent, mainly Lys targeting). COULD REMOVE?? Not strong? Dont' include for now because are usually reversible
    #("Imine (Schiff base former)", "[CX3]=[NX2]", 0, "hard"),           # imine carbon
    #("Cyclic imine", "[CX3R]=[NX2R]", 0, "hard"),                       # ring-constrained imine
    #("Aldehyde-derived imine", "[CX3H1]=[NX2]", 0, "hard"),             # more reactive imine
    ("Cyanamide", "N[C]#N", 1, "soft"),                                  # nitrile carbon (Cys targeted). This is the only one strong enough to consider
    #("Alpha-beta unsaturated imine", "[CX3]=[CX3][CX3]=[NX2]", 0, "soft"), # vinylogous imine
    ("Aliphatic imine (Schiff base former)", "[CX3H1]=[NX2H1]", 0, "hard"),  # Lys targeting
    ("Vinyl imine", "[CX3]=[CX3][CX3]=[NX2]", 0, "borderline"),  # extended system
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


# ============================================================================
# PDB DOWNLOAD FUNCTIONS
# ============================================================================

def download_pdb_file(pdb_id, output_dir="."):
    """
    Download a PDB file from RCSB PDB database.
    
    Args:
        pdb_id: PDB ID (e.g., '4g5j' or '4g5j.pdb')
        output_dir: Directory to save the PDB file
    
    Returns:
        Path to downloaded PDB file, or None if download failed
    """
    # Clean PDB ID (remove .pdb extension if present, convert to lowercase)
    pdb_id_clean = pdb_id.lower().replace('.pdb', '')
    
    # Construct URL
    url = f"https://files.rcsb.org/download/{pdb_id_clean}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id_clean}.pdb")
    
    print(f"   📥 Downloading PDB {pdb_id_clean} from RCSB...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"   ✅ Successfully downloaded to: {output_path}")
        return output_path
    except Exception as e:
        print(f"   ❌ Failed to download PDB {pdb_id_clean}: {e}")
        return None


def download_ligand_sdf(ligand_id, output_dir="."):
    """
    Download a ligand SDF file from RCSB PDB ligand database.
    
    Args:
        ligand_id: Ligand ID (e.g., 'AFA', 'ATP')
        output_dir: Directory to save the SDF file
    
    Returns:
        Path to downloaded SDF file, or None if download failed
    """
    # Clean ligand ID (uppercase, strip whitespace)
    ligand_id_clean = ligand_id.upper().strip()
    
    # Try two possible URLs for ligand SDF files
    urls = [
        f"https://files.rcsb.org/ligands/view/{ligand_id_clean}_ideal.sdf",
        f"https://files.rcsb.org/ligands/download/{ligand_id_clean}.sdf"
    ]
    
    output_path = os.path.join(output_dir, f"{ligand_id_clean}.sdf")
    
    print(f"   📥 Downloading ligand SDF {ligand_id_clean} from RCSB...")
    
    for url in urls:
        try:
            urllib.request.urlretrieve(url, output_path)
            # Verify file was downloaded and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                print(f"   ✅ Successfully downloaded ligand SDF to: {output_path}")
                return output_path
            else:
                # File too small or doesn't exist, try next URL
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
        except Exception as e:
            # Try next URL
            continue
    
    print(f"   ❌ Failed to download ligand SDF {ligand_id_clean}")
    return None


def extract_ligand_from_pdb(pdb_path, ligand_id):
    """
    Extract a ligand and convert to SMILES. Prefers SDF files over PDB.
    
    Will look for:
    1. SDF file with same base name as PDB (e.g., 4g5j.sdf for 4g5j.pdb)
    2. SDF file named <pdb_base>_<ligand_id>.sdf (e.g., 4g5j_AFA.sdf)
    3. Extract from PDB HETATM records as fallback
    
    Args:
        pdb_path: Path to PDB file
        ligand_id: 3-letter ligand code (e.g., 'MNO', 'AFA')
    
    Returns:
        SMILES string, or None if extraction failed
    """
    print(f"   🔍 Extracting ligand {ligand_id} from structure files...")
    
    # Get base paths for looking for SDF files
    pdb_dir = os.path.dirname(pdb_path)
    pdb_base = os.path.splitext(os.path.basename(pdb_path))[0]
    
    # Try to find SDF file (much better for small molecules)
    sdf_candidates = [
        os.path.join(pdb_dir, f"{pdb_base}.sdf"),
        os.path.join(pdb_dir, f"{pdb_base}_{ligand_id}.sdf"),
        os.path.join(pdb_dir, f"{pdb_base}_{ligand_id.upper()}.sdf"),
        os.path.join(pdb_dir, f"{pdb_base}_{ligand_id.lower()}.sdf"),
        os.path.join(pdb_dir, f"{ligand_id}.sdf"),
        os.path.join(pdb_dir, f"{ligand_id.upper()}.sdf"),
        os.path.join(pdb_dir, f"{ligand_id.lower()}.sdf")
    ]
    
    # Check for SDF file
    for sdf_path in sdf_candidates:
        if os.path.exists(sdf_path):
            print(f"   ✓ Found SDF file: {sdf_path}")
            return extract_ligand_from_sdf(sdf_path, ligand_id)
    
    # No SDF found locally, try downloading from RCSB
    print(f"   ℹ️  No local SDF file found, attempting to download from RCSB...")
    downloaded_sdf = download_ligand_sdf(ligand_id, pdb_dir)
    if downloaded_sdf:
        return extract_ligand_from_sdf(downloaded_sdf, ligand_id)
    
    # No SDF available, extract from PDB file
    print(f"   ℹ️  No SDF available, extracting from PDB (less reliable for small molecules)")
    
    try:
        # Try using RDKit first
        from rdkit import Chem
        from rdkit.Chem import MolFromPDBFile, MolToSmiles
        
        # Read PDB file and look for the ligand
        with open(pdb_path, 'r') as f:
            pdb_lines = f.readlines()
        
        # Extract ligand lines (HETATM records matching the ligand_id)
        ligand_lines = []
        for line in pdb_lines:
            if line.startswith('HETATM'):
                # PDB format: residue name is at columns 17-20 (0-indexed: 17-19)
                res_name = line[17:20].strip()
                if res_name.upper() == ligand_id.upper():
                    ligand_lines.append(line)
        
        if not ligand_lines:
            print(f"   ⚠️  No ligand with ID '{ligand_id}' found in PDB file")
            return None
        
        # Write temporary PDB file with just the ligand
        import tempfile
        temp_ligand_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        temp_ligand_pdb.write(''.join(ligand_lines))
        temp_ligand_pdb.write('END\n')
        temp_ligand_pdb.close()
        
        # Try to read with RDKit
        mol = MolFromPDBFile(temp_ligand_pdb.name, removeHs=False)
        
        # Clean up temp file
        os.unlink(temp_ligand_pdb.name)
        
        if mol is None:
            print(f"   ⚠️  Failed to parse ligand with RDKit, trying obabel...")
            # Try with obabel as fallback
            return extract_ligand_with_obabel(pdb_path, ligand_id)
        
        # Remove explicit hydrogens to match warhead atom indexing
        mol = Chem.RemoveHs(mol)
        smiles = MolToSmiles(mol)
        print(f"   ✅ Extracted ligand SMILES from PDB: {smiles}")
        return smiles
        
    except Exception as e:
        print(f"   ⚠️  RDKit extraction failed: {e}")
        print(f"   Trying obabel...")
        return extract_ligand_with_obabel(pdb_path, ligand_id)


def extract_ligand_from_sdf(sdf_path, ligand_id):
    """
    Extract ligand from SDF file and convert to SMILES.
    SDF files are much better for small molecules than PDB files.
    
    Args:
        sdf_path: Path to SDF file
        ligand_id: Ligand ID (for logging)
    
    Returns:
        SMILES string, or None if extraction failed
    """
    try:
        from rdkit import Chem
        
        # Read SDF file (may contain multiple molecules)
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        
        # Try to get the first valid molecule
        for mol in supplier:
            if mol is not None:
                # Remove explicit hydrogens to match warhead atom indexing
                mol = Chem.RemoveHs(mol)
                smiles = Chem.MolToSmiles(mol)
                print(f"   ✅ Extracted ligand SMILES from SDF: {smiles}")
                return smiles
        
        print(f"   ⚠️  No valid molecules found in SDF file")
        return None
        
    except Exception as e:
        print(f"   ⚠️  Failed to read SDF file: {e}")
        
        # Try obabel as fallback
        try:
            import subprocess
            result = subprocess.run(
                ['obabel', sdf_path, '-osmi', '-d'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                smiles = result.stdout.strip().split()[0]
                print(f"   ✅ Extracted ligand SMILES from SDF (obabel): {smiles}")
                return smiles
            else:
                print(f"   ❌ obabel failed: {result.stderr}")
                return None
        except Exception as e2:
            print(f"   ❌ obabel fallback failed: {e2}")
            return None


def extract_ligand_with_obabel(pdb_path, ligand_id):
    """
    Extract ligand using obabel (Open Babel) as fallback.
    
    Args:
        pdb_path: Path to PDB file
        ligand_id: 3-letter ligand code
    
    Returns:
        SMILES string, or None if extraction failed
    """
    try:
        import subprocess
        import tempfile
        
        # Extract ligand lines to temporary file
        with open(pdb_path, 'r') as f:
            pdb_lines = f.readlines()
        
        ligand_lines = []
        for line in pdb_lines:
            if line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name.upper() == ligand_id.upper():
                    ligand_lines.append(line)
        
        if not ligand_lines:
            return None
        
        # Write temporary PDB file with just the ligand
        temp_ligand_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        temp_ligand_pdb.write(''.join(ligand_lines))
        temp_ligand_pdb.write('END\n')
        temp_ligand_pdb.close()
        
        # Use obabel to convert to SMILES (with -d to remove hydrogens)
        result = subprocess.run(
            ['obabel', temp_ligand_pdb.name, '-osmi', '-d'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Clean up temp file
        os.unlink(temp_ligand_pdb.name)
        
        if result.returncode == 0:
            smiles = result.stdout.strip().split()[0]  # First column is SMILES
            print(f"   ✅ Extracted ligand SMILES from PDB (obabel): {smiles}")
            return smiles
        else:
            print(f"   ❌ obabel failed: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print(f"   ❌ obabel not found. Please install Open Babel.")
        return None
    except Exception as e:
        print(f"   ❌ obabel extraction failed: {e}")
        return None


# ============================================================================
# NUCLEOPHILE IDENTIFICATION FUNCTIONS
# ============================================================================

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
        # Run FreeSASA - now saves to organized sasa_output/ directory
        rsa_file = run_freesasa(pdb_path)
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
        # Run PROPKA - now saves to organized pka_output/ directory
        pka_file = run_propka(pdb_path)
        if not os.path.exists(pka_file):
            raise FileNotFoundError(f"PROPKA output file not found: {pka_file}")
    except FileNotFoundError as e:
        if "propka" in str(e).lower() and "not found" in str(e).lower() and "output" not in str(e).lower():
            print("\n❌ ERROR: 'propka3' command not found!")
            print("Please install PROPKA:")
            print("  - pip install propka")
            print("\nMake sure it's in your PATH after installation.")
        raise

    exposure = parse_rsa_file(rsa_file)
    pka_data = parse_propka_file(pka_file)
    
    # Count ALL residues in PDB (before any filtering)
    total_residues_in_pdb = len(exposure)

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
    # Return all nucleophiles (both accessible and inaccessible) AND total residue count
    return df, total_residues_in_pdb

# ============================================================================
# QUANTUM REACTIVITY CALCULATIONS (use functions from single_AA_bond.py)
# ============================================================================

# Use functions directly from imported module
rdkit_mol_from_smiles = saa.rdkit_mol_from_smiles
write_xyz = saa.write_xyz
run_xtb_xyz = saa.run_xtb_xyz
get_lg_score = saa.get_lg_score

# Cache files for quantum calculations (to avoid redundant xTB calculations)
CACHE_FILE = os.path.join(script_dir, "reactivity_cache.json")
NUCLEOPHILE_CACHE_FILE = os.path.join(script_dir, "nucleophile_cache.json")

def load_reactivity_cache():
    """Load cached reactivity scores from file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
            return {}
    return {}

def save_reactivity_cache(cache):
    """Save reactivity scores cache to file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

def load_nucleophile_cache():
    """Load cached nucleophile HOMO energies from file."""
    if os.path.exists(NUCLEOPHILE_CACHE_FILE):
        try:
            with open(NUCLEOPHILE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load nucleophile cache file: {e}")
            return {}
    return {}

def save_nucleophile_cache(cache):
    """Save nucleophile HOMO energies cache to file."""
    try:
        with open(NUCLEOPHILE_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save nucleophile cache file: {e}")

def get_nucleophile_cache_key(nucleophile_type, protonation_state):
    """Generate a unique cache key for a nucleophile (type + protonation state)."""
    return f"{nucleophile_type}|{protonation_state}"

def get_cache_key(electrophile_smiles, reactive_idx, nucleophile_type, protonation_state):
    """Generate a unique cache key for a warhead-nucleophile combination."""
    # Canonicalize SMILES to ensure consistency
    try:
        mol = Chem.MolFromSmiles(electrophile_smiles)
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    except:
        canonical_smiles = electrophile_smiles
    
    return f"{canonical_smiles}|{reactive_idx}|{nucleophile_type}|{protonation_state}"

def compute_reactivity_score(electrophile_smiles, reactive_idx, nucleophile_type, protonation_state="protonated", cache=None, nucleophile_cache=None, use_cache=True):
    """
    Compute intrinsic reactivity score between electrophile and nucleophile.
    Uses core calculation functions from single_AA_bond.py.
    Implements caching to avoid redundant xTB calculations.
    
    Args:
        electrophile_smiles: SMILES of electrophile
        reactive_idx: Index of reactive atom in electrophile
        nucleophile_type: Type of amino acid (e.g., "CYS", "SER")
        protonation_state: "protonated" or "deprotonated" (default: "protonated")
        cache: Optional cache dictionary for full reactivity results (if None, caching is disabled)
        nucleophile_cache: Optional cache dictionary for nucleophile HOMO energies (if None, caching is disabled)
        use_cache: If True, read from cache; if False, force recalculation but still save to cache (default: True)
    
    NOTE: This is a simplified version that computes individual descriptors.
    The full compute_score function from single_AA_bond.py includes adduct formation,
    which is currently problematic. We extract the working parts here.
    
    TODO: Consider using saa.compute_score() directly if adduct issues are resolved.
    
    Returns:
        dict with scoring components
    """
    # Check cache first (only if use_cache=True)
    if cache is not None and use_cache:
        cache_key = get_cache_key(electrophile_smiles, reactive_idx, nucleophile_type, protonation_state)
        if cache_key in cache:
            return cache[cache_key]
    
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
        
        # Check nucleophile cache for HOMO energy (only if use_cache=True)
        nuc_cache_key = get_nucleophile_cache_key(nucleophile_type, protonation_state)
        n_homo = None
        
        if nucleophile_cache is not None and use_cache and nuc_cache_key in nucleophile_cache:
            # Use cached nucleophile HOMO
            n_homo = nucleophile_cache[nuc_cache_key].get("homo")
        
        if n_homo is None:
            # Need to compute nucleophile properties
            n_mol = rdkit_mol_from_smiles(sur_smiles)
            n_xyz = os.path.join(tmpdir, "nuc.xyz")
            write_xyz(n_mol, n_xyz)
            
            # Determine charge for nucleophile based on protonation state
            # Deprotonated forms of acids (Cys, Ser, Thr, Tyr) have -1 charge
            # Lys neutral in both forms; His nucleophilic form is neutral imidazole (not imidazolate)
            if protonation_state == "deprotonated" and nucleophile_type in ["Cys", "Ser", "Thr", "Tyr"]:
                nuc_charge = -1
            else:
                nuc_charge = 0
                
            n_neutral = run_xtb_xyz(n_xyz, charge=nuc_charge, gbsa="water")
            n_homo = n_neutral["homo"]
            
            # Cache the nucleophile HOMO energy
            if nucleophile_cache is not None:
                nucleophile_cache[nuc_cache_key] = {
                    "homo": n_homo,
                    "smiles": sur_smiles,
                    "charge": nuc_charge
                }
        
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
        
        if n_homo is not None and e_neutral["lumo"] is not None:
            homo_lumo_gap = e_neutral["lumo"] - n_homo
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
        
        result = {
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
        
        # Save to cache if provided
        if cache is not None:
            cache_key = get_cache_key(electrophile_smiles, reactive_idx, nucleophile_type, protonation_state)
            cache[cache_key] = result
        
        return result
    
    finally:
        shutil.rmtree(tmpdir)

# ============================================================================
# PARALLEL COMPUTATION WORKER
# ============================================================================

def _compute_aa_reactivity_worker(args):
    """
    Worker function for parallel amino acid reactivity computation.
    Must be at module level for pickling.
    
    Args:
        args: Tuple of (electrophile_smiles, reactive_idx, aa_type, protonation_state,
                       reactivity_cache_dict, nucleophile_cache_dict, use_cache)
    
    Returns:
        Tuple of (aa_type, protonation_state, score_data, cache_was_hit, nuc_cache_was_hit,
                 reactivity_cache_updates, nucleophile_cache_updates)
    """
    electrophile_smiles, reactive_idx, aa_type, protonation_state, reactivity_cache_dict, nucleophile_cache_dict, use_cache = args
    
    # Create temporary cache objects for this worker
    reactivity_cache = reactivity_cache_dict.copy() if reactivity_cache_dict else {}
    nucleophile_cache = nucleophile_cache_dict.copy() if nucleophile_cache_dict else {}
    
    # Check if this was a cache hit (only meaningful if use_cache=True)
    cache_key = get_cache_key(electrophile_smiles, reactive_idx, aa_type, protonation_state)
    nuc_cache_key = get_nucleophile_cache_key(aa_type, protonation_state)
    cache_was_hit = use_cache and (cache_key in reactivity_cache)
    nuc_cache_was_hit = use_cache and (nuc_cache_key in nucleophile_cache)
    
    # Compute the reactivity score
    try:
        score_data = compute_reactivity_score(
            electrophile_smiles,
            reactive_idx,
            aa_type,
            protonation_state=protonation_state,
            cache=reactivity_cache,
            nucleophile_cache=nucleophile_cache,
            use_cache=use_cache
        )
    except Exception as e:
        score_data = {"reactivity_score": 0.0, "error": str(e)}
    
    # Collect cache updates (only new entries)
    reactivity_cache_updates = {}
    if cache_key in reactivity_cache and cache_key not in reactivity_cache_dict:
        reactivity_cache_updates[cache_key] = reactivity_cache[cache_key]
    
    nucleophile_cache_updates = {}
    if nuc_cache_key in nucleophile_cache and nuc_cache_key not in nucleophile_cache_dict:
        nucleophile_cache_updates[nuc_cache_key] = nucleophile_cache[nuc_cache_key]
    
    return (aa_type, protonation_state, score_data, cache_was_hit, nuc_cache_was_hit,
            reactivity_cache_updates, nucleophile_cache_updates)

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(pdb_path, electrophile_smiles, output_prefix="Frankenstein", top_n_types=3, n_workers=1, use_cache=True):
    """
    Main workflow: 
    1. Determine top N most reactive amino acid TYPES for the electrophile (quantum calculations)
    2. Identify accessible nucleophiles in protein (all types)
    3. Filter to only keep accessible nucleophiles matching the top N reactive types (Accessibility is essentially the first filter though since calculated first but just implemented here)
    4. Combine reactivity scores (from step 1) with accessibility scores (from step 2)
    5. Output ranked results
    
    Args:
        pdb_path: Path to PDB file
        electrophile_smiles: SMILES string of electrophile
        output_prefix: Prefix for output files
        top_n_types: Number of top reactive amino acid types to consider (default: 3)
        n_workers: Number of parallel workers for reactivity calculations (default: 1 = sequential)
        use_cache: If True, use cached results; if False, force recalculation but still save to cache (default: True)
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
    
    # Load reactivity cache to avoid redundant calculations
    print(f"   Loading reactivity cache...")
    reactivity_cache = load_reactivity_cache()
    print(f"   Loading nucleophile cache...")
    nucleophile_cache = load_nucleophile_cache()
    cache_hits = 0
    cache_misses = 0
    nuc_cache_hits = 0
    nuc_cache_misses = 0
    
    # Determine execution mode
    use_parallel = n_workers > 1
    if use_parallel:
        print(f"   ⚡ Using parallel execution with {n_workers} workers")
    
    if not use_cache:
        print(f"   🔄 Cache reading DISABLED - forcing recalculation (cache will still be updated)")
    else:
        print(f"   Using sequential execution (use --workers N for parallel)")
    
    all_warhead_results = {}  # Store results for each warhead
    
    for warhead_idx, (warhead_name, warhead_smarts, reactive_idx, hsab_class) in enumerate(warheads, 1):
        print(f"\n   [{warhead_idx}/{len(warheads)}] Warhead: {warhead_name}")
        print(f"   Reactive atom index: {reactive_idx} (atom: {e_mol.GetAtomWithIdx(reactive_idx).GetSymbol()})")
        print(f"   HSAB classification: {hsab_class}")
        print(f"   Computing intrinsic reactivity for each amino acid type...")
        
        type_scores_protonated = {}
        type_scores_deprotonated = {}
        
        if use_parallel:
            # Parallel execution using ProcessPoolExecutor
            tasks = []
            aa_types_list = list(SURROGATES.keys())
            
            # Prepare all tasks (protonated + deprotonated for each AA type)
            for aa_type in aa_types_list:
                tasks.append((electrophile_smiles, reactive_idx, aa_type, "protonated",
                             reactivity_cache, nucleophile_cache, use_cache))
                tasks.append((electrophile_smiles, reactive_idx, aa_type, "deprotonated",
                             reactivity_cache, nucleophile_cache, use_cache))
            
            print(f"      Submitting {len(tasks)} calculations to {n_workers} workers...")
            
            # Execute in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_compute_aa_reactivity_worker, task): task for task in tasks}
                
                for future in as_completed(futures):
                    task = futures[future]
                    aa_type, protonation_state = task[2], task[3]
                    
                    try:
                        aa_type, prot_state, score_data, cache_hit, nuc_cache_hit, \
                            react_cache_updates, nuc_cache_updates = future.result()
                        
                        # Update statistics
                        if cache_hit:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                        if nuc_cache_hit:
                            nuc_cache_hits += 1
                        else:
                            nuc_cache_misses += 1
                        
                        # Merge cache updates
                        reactivity_cache.update(react_cache_updates)
                        nucleophile_cache.update(nuc_cache_updates)
                        
                        # Store results
                        if prot_state == "protonated":
                            type_scores_protonated[aa_type] = score_data
                        else:
                            type_scores_deprotonated[aa_type] = score_data
                        
                        print(f"      ✓ {aa_type} ({prot_state}): {score_data.get('reactivity_score', 0.0):.3f}" +
                              (" [Cached]" if cache_hit else ""))
                    
                    except Exception as e:
                        print(f"      ✗ {aa_type} ({protonation_state}): Error - {e}")
                        if protonation_state == "protonated":
                            type_scores_protonated[aa_type] = {"reactivity_score": 0.0}
                        else:
                            type_scores_deprotonated[aa_type] = {"reactivity_score": 0.0}
        
        else:
            # Sequential execution (original code)
            for aa_type in SURROGATES.keys():
                print(f"      - Testing {aa_type}...")
                
                # Test protonated form
                print(f"        Protonated: ", end="")
                cache_key_prot = get_cache_key(electrophile_smiles, reactive_idx, aa_type, "protonated")
                nuc_cache_key_prot = get_nucleophile_cache_key(aa_type, "protonated")
                if use_cache and cache_key_prot in reactivity_cache:
                    print(f"[Cached] ", end="")
                    cache_hits += 1
                else:
                    cache_misses += 1
                if use_cache and nuc_cache_key_prot in nucleophile_cache:
                    nuc_cache_hits += 1
                else:
                    nuc_cache_misses += 1
                try:
                    score_data = compute_reactivity_score(
                        electrophile_smiles,
                        reactive_idx,
                        aa_type,
                        protonation_state="protonated",
                        cache=reactivity_cache,
                        nucleophile_cache=nucleophile_cache,
                        use_cache=use_cache
                    )
                    type_scores_protonated[aa_type] = score_data
                    print(f"Reactivity: {score_data['reactivity_score']:.3f}")
                except Exception as e:
                    print(f"Error: {e}")
                    type_scores_protonated[aa_type] = {"reactivity_score": 0.0}
                
                # Test deprotonated form
                print(f"        Deprotonated: ", end="")
                cache_key_deprot = get_cache_key(electrophile_smiles, reactive_idx, aa_type, "deprotonated")
                nuc_cache_key_deprot = get_nucleophile_cache_key(aa_type, "deprotonated")
                if use_cache and cache_key_deprot in reactivity_cache:
                    print(f"[Cached] ", end="")
                    cache_hits += 1
                else:
                    cache_misses += 1
                if use_cache and nuc_cache_key_deprot in nucleophile_cache:
                    nuc_cache_hits += 1
                else:
                    nuc_cache_misses += 1
                try:
                    score_data = compute_reactivity_score(
                        electrophile_smiles,
                        reactive_idx,
                        aa_type,
                        protonation_state="deprotonated",
                        cache=reactivity_cache,
                        nucleophile_cache=nucleophile_cache,
                        use_cache=use_cache
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
        
        # Compute orbital compatibility for all amino acid types (compute once, reuse later)
        print(f"      Computing orbital compatibility for all amino acid types...")
        orbital_compatibility_protonated = {}
        orbital_compatibility_deprotonated = {}
        
        for aa_type in SURROGATES.keys():
            surrogate_key = get_surrogate_key(aa_type.upper())
            
            # Check protonated form
            nuc_smiles_prot = SURROGATES.get(surrogate_key)
            if nuc_smiles_prot:
                nuc_atom_idx_prot = NUCLEOPHILE_ATOM_INDEX.get(nuc_smiles_prot, 0)
                try:
                    result_prot = cor.paper_check_interaction(
                        nuc_smiles_prot, nuc_atom_idx_prot,
                        electrophile_smiles, reactive_idx
                    )
                    orbital_compatibility_protonated[aa_type.upper()] = {
                        "compatible": result_prot["covalent_bond_possible"],
                        "orbital_score": result_prot.get("orbital_score", 0.0),
                        "nucleophile_info": result_prot.get("nucleophile", {}),
                        "electrophile_info": result_prot.get("electrophile", {})
                    }
                except Exception as e:
                    orbital_compatibility_protonated[aa_type.upper()] = {
                        "compatible": False,
                        "orbital_score": 0.0,
                        "error": str(e)
                    }
            else:
                orbital_compatibility_protonated[aa_type.upper()] = {
                    "compatible": False,
                    "orbital_score": 0.0,
                    "error": "No SMILES surrogate found"
                }
            
            # Check deprotonated form
            nuc_smiles_deprot = SURROGATES_DEPROTONATED.get(surrogate_key)
            if nuc_smiles_deprot:
                nuc_atom_idx_deprot = NUCLEOPHILE_ATOM_INDEX.get(nuc_smiles_deprot, 0)
                try:
                    result_deprot = cor.paper_check_interaction(
                        nuc_smiles_deprot, nuc_atom_idx_deprot,
                        electrophile_smiles, reactive_idx
                    )
                    orbital_compatibility_deprotonated[aa_type.upper()] = {
                        "compatible": result_deprot["covalent_bond_possible"],
                        "orbital_score": result_deprot.get("orbital_score", 0.0),
                        "nucleophile_info": result_deprot.get("nucleophile", {}),
                        "electrophile_info": result_deprot.get("electrophile", {})
                    }
                except Exception as e:
                    orbital_compatibility_deprotonated[aa_type.upper()] = {
                        "compatible": False,
                        "orbital_score": 0.0,
                        "error": str(e)
                    }
            else:
                orbital_compatibility_deprotonated[aa_type.upper()] = {
                    "compatible": False,
                    "orbital_score": 0.0,
                    "error": "No SMILES surrogate found"
                }
        
        # Compute HSAB matches for all amino acid types (compute once, reuse later)
        print(f"      Computing HSAB matches for all amino acid types...")
        hsab_matches_all_types = {}
        hsab_class_targets = get_hsab_target_residues(hsab_class)
        
        for aa_type in SURROGATES.keys():
            aa_type_upper = aa_type.upper()
            is_match = check_hsab_match(aa_type_upper, hsab_class)
            hsab_matches_all_types[aa_type_upper] = {
                "hsab_match": is_match,
                "warhead_hsab_class": hsab_class,
                "target_residues": hsab_class_targets
            }
        
        # Store results for this warhead
        all_warhead_results[warhead_name] = {
            "warhead_smarts": warhead_smarts,
            "reactive_idx": reactive_idx,
            "hsab_class": hsab_class,
            "hsab_targets": get_hsab_target_residues(hsab_class),
            "type_scores_protonated": type_scores_protonated_upper,
            "type_scores_deprotonated": type_scores_deprotonated_upper,
            "top_types_protonated": top_types_protonated,
            "top_types_deprotonated": top_types_deprotonated,
            "orbital_compatibility_protonated": orbital_compatibility_protonated,
            "orbital_compatibility_deprotonated": orbital_compatibility_deprotonated,
            "hsab_matches": hsab_matches_all_types
        }
    
    # Save updated caches
    print(f"\n   Saving reactivity cache...")
    print(f"   Reactivity cache statistics: {cache_hits} hits, {cache_misses} misses (computed)")
    save_reactivity_cache(reactivity_cache)
    
    print(f"   Saving nucleophile cache...")
    print(f"   Nucleophile cache statistics: {nuc_cache_hits} hits, {nuc_cache_misses} misses (computed)")
    print(f"   ⚡ Efficiency gain: Nucleophile calculations reduced by {nuc_cache_hits}/{nuc_cache_hits + nuc_cache_misses} = {100 * nuc_cache_hits / (nuc_cache_hits + nuc_cache_misses) if (nuc_cache_hits + nuc_cache_misses) > 0 else 0:.1f}%")
    save_nucleophile_cache(nucleophile_cache)
    
    # Save all warhead reactivity scores
    all_scores_json = os.path.join(output_dir, "all_warheads_reactivity.json")
    with open(all_scores_json, 'w') as f:
        json.dump(all_warhead_results, f, indent=2)
    print(f"\n   💾 Saved all warhead reactivity scores to: {all_scores_json}")
    
    # Extract and save orbital compatibility data
    all_orbital_data = {}
    for warhead_name, warhead_data in all_warhead_results.items():
        all_orbital_data[warhead_name] = {
            "warhead_smarts": warhead_data["warhead_smarts"],
            "reactive_idx": warhead_data["reactive_idx"],
            "protonated": warhead_data["orbital_compatibility_protonated"],
            "deprotonated": warhead_data["orbital_compatibility_deprotonated"]
        }
    
    orbital_json = os.path.join(output_dir, "all_warheads_orbital_compatibility.json")
    with open(orbital_json, 'w') as f:
        json.dump(all_orbital_data, f, indent=2)
    print(f"   💾 Saved all warhead orbital compatibility data to: {orbital_json}")
    
    # Extract and save HSAB match data
    all_hsab_data = {}
    for warhead_name, warhead_data in all_warhead_results.items():
        all_hsab_data[warhead_name] = {
            "warhead_smarts": warhead_data["warhead_smarts"],
            "reactive_idx": warhead_data["reactive_idx"],
            "hsab_class": warhead_data["hsab_class"],
            "hsab_targets": warhead_data["hsab_targets"],
            "matches": warhead_data["hsab_matches"]
        }
    
    hsab_json = os.path.join(output_dir, "all_warheads_hsab_matches.json")
    with open(hsab_json, 'w') as f:
        json.dump(all_hsab_data, f, indent=2)
    print(f"   💾 Saved all warhead HSAB match data to: {hsab_json}")
    
    # Step 2: Identify all nucleophiles in protein (accessible and inaccessible)
    print("\n📍 STEP 2: Identifying nucleophiles in protein...")
    all_nucleophiles, total_residues_in_pdb = identify_accessible_nucleophiles(pdb_path)
    accessible_count = all_nucleophiles[all_nucleophiles["Accessible"] == True].shape[0]
    print(f"   Found {total_residues_in_pdb} total residues in PDB")
    print(f"   Found {len(all_nucleophiles)} nucleophilic residues ({accessible_count} accessible, {len(all_nucleophiles) - accessible_count} inaccessible)")
    
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
    filter_statistics = {}  # Track filtering statistics for each warhead
    
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
        
        # Binary filter 3: Orbital compatibility check (lookup from pre-computed values)
        # Reuse the comprehensive computation done earlier to avoid redundant calculations
        print(f"      Looking up orbital compatibility from pre-computed values...")
        reactive_idx = warhead_data["reactive_idx"]
        orbital_compatible_protonated = []
        orbital_compatible_deprotonated = []
        
        orbital_data_prot = warhead_data["orbital_compatibility_protonated"]
        orbital_data_deprot = warhead_data["orbital_compatibility_deprotonated"]
        
        for idx, row in filtered_nucleophiles.iterrows():
            residue_type = row["Residue"]
            
            # Lookup protonated form from pre-computed values
            if residue_type in orbital_data_prot:
                orbital_compatible_protonated.append(1 if orbital_data_prot[residue_type]["compatible"] else 0)
            else:
                print(f"         Warning: No pre-computed orbital data (protonated) for {residue_type}")
                orbital_compatible_protonated.append(0)
            
            # Lookup deprotonated form from pre-computed values
            if residue_type in orbital_data_deprot:
                orbital_compatible_deprotonated.append(1 if orbital_data_deprot[residue_type]["compatible"] else 0)
            else:
                print(f"         Warning: No pre-computed orbital data (deprotonated) for {residue_type}")
                orbital_compatible_deprotonated.append(0)
        
        filtered_nucleophiles["Binary_Orbital_Protonated"] = orbital_compatible_protonated
        filtered_nucleophiles["Binary_Orbital_Deprotonated"] = orbital_compatible_deprotonated
        
        # Binary filter 4: HSAB match check (lookup from pre-computed values)
        # Reuse the comprehensive computation done earlier to avoid redundant calculations
        print(f"      Looking up HSAB matches from pre-computed values...")
        hsab_matches = []
        hsab_data = warhead_data["hsab_matches"]
        
        for idx, row in filtered_nucleophiles.iterrows():
            residue_type = row["Residue"]
            if residue_type in hsab_data:
                hsab_matches.append(1 if hsab_data[residue_type]["hsab_match"] else 0)
            else:
                print(f"         Warning: No pre-computed HSAB data for {residue_type}")
                hsab_matches.append(0)
        
        filtered_nucleophiles["Binary_HSAB"] = hsab_matches
        
        # Calculate final binary scores WITHOUT HSAB (separate for each protonation state)
        filtered_nucleophiles["Binary_Score_Protonated"] = (
            (filtered_nucleophiles["Binary_Accessible"] == 1) &
            (filtered_nucleophiles["Binary_Reactivity_Protonated"] == 1) &
            (filtered_nucleophiles["Binary_Orbital_Protonated"] == 1)
        ).astype(int)
        filtered_nucleophiles["Binary_Score_Deprotonated"] = (
            (filtered_nucleophiles["Binary_Accessible"] == 1) &
            (filtered_nucleophiles["Binary_Reactivity_Deprotonated"] == 1) &
            (filtered_nucleophiles["Binary_Orbital_Deprotonated"] == 1)
        ).astype(int)
        
        # Calculate final binary scores WITH HSAB (separate for each protonation state)
        filtered_nucleophiles["Binary_Score_With_HSAB_Protonated"] = (
            (filtered_nucleophiles["Binary_Accessible"] == 1) &
            (filtered_nucleophiles["Binary_Reactivity_Protonated"] == 1) &
            (filtered_nucleophiles["Binary_Orbital_Protonated"] == 1) &
            (filtered_nucleophiles["Binary_HSAB"] == 1)
        ).astype(int)
        filtered_nucleophiles["Binary_Score_With_HSAB_Deprotonated"] = (
            (filtered_nucleophiles["Binary_Accessible"] == 1) &
            (filtered_nucleophiles["Binary_Reactivity_Deprotonated"] == 1) &
            (filtered_nucleophiles["Binary_Orbital_Deprotonated"] == 1) &
            (filtered_nucleophiles["Binary_HSAB"] == 1)
        ).astype(int)
        
        print(f"      Computed scores for {len(filtered_nucleophiles)} valid nucleophiles")
        print(f"      Binary filters: Accessible={filtered_nucleophiles['Binary_Accessible'].sum()}, "
              f"Orbital_Prot={filtered_nucleophiles['Binary_Orbital_Protonated'].sum()}, "
              f"Orbital_Deprot={filtered_nucleophiles['Binary_Orbital_Deprotonated'].sum()}, "
              f"HSAB={filtered_nucleophiles['Binary_HSAB'].sum()}")
        print(f"      Binary score (Protonated, no HSAB): {filtered_nucleophiles['Binary_Score_Protonated'].sum()}")
        print(f"      Binary score (Deprotonated, no HSAB): {filtered_nucleophiles['Binary_Score_Deprotonated'].sum()}")
        print(f"      Binary score with HSAB (Protonated): {filtered_nucleophiles['Binary_Score_With_HSAB_Protonated'].sum()}")
        print(f"      Binary score with HSAB (Deprotonated): {filtered_nucleophiles['Binary_Score_With_HSAB_Deprotonated'].sum()}")
        
        # Track statistics for this warhead
        total_nucleophiles = len(all_nucleophiles)
        accessible_nucleophiles = len(all_accessible)
        
        # Protonated statistics
        after_reactivity_prot = len(filtered_nucleophiles[filtered_nucleophiles["Residue"].isin(top_types_protonated)])
        after_orbital_prot = (
            (filtered_nucleophiles["Residue"].isin(top_types_protonated)) &
            (filtered_nucleophiles["Binary_Orbital_Protonated"] == 1)
        ).sum()
        after_hsab_prot = (
            (filtered_nucleophiles["Residue"].isin(top_types_protonated)) &
            (filtered_nucleophiles["Binary_Orbital_Protonated"] == 1) &
            (filtered_nucleophiles["Binary_HSAB"] == 1)
        ).sum()
        final_pass_prot = filtered_nucleophiles["Binary_Score_Protonated"].sum()
        
        # Deprotonated statistics
        after_reactivity_deprot = len(filtered_nucleophiles[filtered_nucleophiles["Residue"].isin(top_types_deprotonated)])
        after_orbital_deprot = (
            (filtered_nucleophiles["Residue"].isin(top_types_deprotonated)) &
            (filtered_nucleophiles["Binary_Orbital_Deprotonated"] == 1)
        ).sum()
        after_hsab_deprot = (
            (filtered_nucleophiles["Residue"].isin(top_types_deprotonated)) &
            (filtered_nucleophiles["Binary_Orbital_Deprotonated"] == 1) &
            (filtered_nucleophiles["Binary_HSAB"] == 1)
        ).sum()
        final_pass_deprot = filtered_nucleophiles["Binary_Score_Deprotonated"].sum()
        
        # Store statistics
        filter_statistics[warhead_name] = {
            "total_residues_in_pdb": total_residues_in_pdb,
            "total_nucleophiles": total_nucleophiles,
            "accessible_nucleophiles": accessible_nucleophiles,
            "protonated": {
                "after_reactivity": after_reactivity_prot,
                "after_orbital": after_orbital_prot,
                "after_hsab": after_hsab_prot,
                "final": final_pass_prot
            },
            "deprotonated": {
                "after_reactivity": after_reactivity_deprot,
                "after_orbital": after_orbital_deprot,
                "after_hsab": after_hsab_deprot,
                "final": final_pass_deprot
            }
        }
        
        # Create separate dataframes for protonated and deprotonated
        warhead_safe_name = warhead_name.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Protonated version - filter to only top N protonated types
        df_protonated = filtered_nucleophiles[filtered_nucleophiles["Residue"].isin(top_types_protonated)].copy()
        df_protonated["Reactivity_Score"] = df_protonated["Reactivity_Score_Protonated"]
        df_protonated["Binary_Reactivity"] = df_protonated["Binary_Reactivity_Protonated"]
        df_protonated["Binary_Orbital"] = df_protonated["Binary_Orbital_Protonated"]
        df_protonated["Binary_Score"] = df_protonated["Binary_Score_Protonated"]
        df_protonated["Binary_Score_With_HSAB"] = df_protonated["Binary_Score_With_HSAB_Protonated"]
        df_protonated["Combined_Score"] = (
            df_protonated["Accessibility_Score"] * 
            df_protonated["Reactivity_Score_Protonated"]
        )
        df_protonated_sorted = df_protonated.sort_values(
            by=["Binary_Score", "Binary_Score_With_HSAB", "Combined_Score"], 
            ascending=[False, False, False]
        )
        protonated_csv = os.path.join(output_dir, f"protonated_ranked_targets_{warhead_safe_name}.csv")
        df_protonated_sorted.to_csv(protonated_csv, index=False)
        print(f"      💾 Saved protonated (top {top_n_types} protonated types) to: {protonated_csv}")
        
        # Deprotonated version - filter to only top N deprotonated types
        df_deprotonated = filtered_nucleophiles[filtered_nucleophiles["Residue"].isin(top_types_deprotonated)].copy()
        df_deprotonated["Reactivity_Score"] = df_deprotonated["Reactivity_Score_Deprotonated"]
        df_deprotonated["Binary_Reactivity"] = df_deprotonated["Binary_Reactivity_Deprotonated"]
        df_deprotonated["Binary_Orbital"] = df_deprotonated["Binary_Orbital_Deprotonated"]
        df_deprotonated["Binary_Score"] = df_deprotonated["Binary_Score_Deprotonated"]
        df_deprotonated["Binary_Score_With_HSAB"] = df_deprotonated["Binary_Score_With_HSAB_Deprotonated"]
        df_deprotonated["Combined_Score"] = (
            df_deprotonated["Accessibility_Score"] * 
            df_deprotonated["Reactivity_Score_Deprotonated"]
        )
        df_deprotonated_sorted = df_deprotonated.sort_values(
            by=["Binary_Score", "Binary_Score_With_HSAB", "Combined_Score"], 
            ascending=[False, False, False]
        )
        deprotonated_csv = os.path.join(output_dir, f"deprotonated_ranked_targets_{warhead_safe_name}.csv")
        df_deprotonated_sorted.to_csv(deprotonated_csv, index=False)
        print(f"      💾 Saved deprotonated (top {top_n_types} deprotonated types) to: {deprotonated_csv}")
        
        # Add to combined results (include warhead name for per-warhead checking in test mode)
        all_results.append({
            "warhead_name": warhead_name,
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
        by=["Binary_Score", "Binary_Score_With_HSAB", "Combined_Score"], 
        ascending=[False, False, False], 
        inplace=True
    )
    
    combined_deprotonated = pd.concat(all_deprotonated, ignore_index=True)
    combined_deprotonated.sort_values(
        by=["Binary_Score", "Binary_Score_With_HSAB", "Combined_Score"], 
        ascending=[False, False, False], 
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
    binary_pass_with_hsab_deprot = combined_deprotonated.get('Binary_Score_With_HSAB', pd.Series([0])).sum()
    total_candidates_prot = len(combined_protonated)
    binary_pass_prot = combined_protonated.get('Binary_Score', pd.Series([0])).sum()
    binary_pass_with_hsab_prot = combined_protonated.get('Binary_Score_With_HSAB', pd.Series([0])).sum()
    print("\n" + "=" * 90)
    print(f"📊 Summary (Deprotonated): {binary_pass_deprot}/{total_candidates_deprot} pass filters (no HSAB), {binary_pass_with_hsab_deprot}/{total_candidates_deprot} pass all filters (with HSAB)")
    print(f"📊 Summary (Protonated): {binary_pass_prot}/{total_candidates_prot} pass filters (no HSAB), {binary_pass_with_hsab_prot}/{total_candidates_prot} pass all filters (with HSAB)")
    print(f"   Filters (no HSAB): Accessible=1, Top {top_n_types} Reactive Type=True, Orbital Compatible=True")
    print(f"   Filters (with HSAB): Above + HSAB Match=True")
    
    # Generate filtering statistics report
    print("\n📈 Generating filtering statistics report...")
    generate_filtering_statistics_report(filter_statistics, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"📂 All output files saved to: {output_dir}/")
    print(f"   📄 all_warheads_reactivity.json - Reactivity scores for all warheads")
    print(f"   📄 all_warheads_orbital_compatibility.json - Orbital compatibility for all warheads & AA types")
    print(f"   📄 all_warheads_hsab_matches.json - HSAB classification matches for all warheads")
    print(f"   📄 all_accessibility_nucleophiles.csv - All nucleophiles (accessible & inaccessible)")
    print(f"   📄 protonated_ranked_targets_<warhead>.csv - Protonated results per warhead")
    print(f"   📄 deprotonated_ranked_targets_<warhead>.csv - Deprotonated results per warhead")
    print(f"   📄 protonated_ranked_covalent_targets_all_warheads.csv - Combined protonated results")
    print(f"   📄 deprotonated_ranked_covalent_targets_all_warheads.csv - Combined deprotonated results")
    print(f"   📄 filtering_statistics.txt - Detailed filtering statistics report")
    print("=" * 80)
    
    # Return statistics and results for batch processing
    warhead_analysis = {}
    for warhead_name, warhead_data in all_warhead_results.items():
        warhead_analysis[warhead_name] = {
            "top_types_protonated": warhead_data.get("top_types_protonated", []),
            "top_types_deprotonated": warhead_data.get("top_types_deprotonated", []),
            "orbital_compatibility_protonated": warhead_data.get("orbital_compatibility_protonated", {}),
            "orbital_compatibility_deprotonated": warhead_data.get("orbital_compatibility_deprotonated", {}),
        }

    nucleophile_site_lookup = {}
    for _, row in all_nucleophiles.iterrows():
        lookup_key = (
            str(row.get("Residue", "")).strip().upper(),
            str(row.get("ResNum", "")).strip(),
            str(row.get("Chain", "")).strip(),
        )
        nucleophile_site_lookup[lookup_key] = {
            "accessible": bool(row.get("Accessible", False))
        }

    return {
        "filter_statistics": filter_statistics,
        "output_prefix": output_prefix,
        "electrophile_smiles": electrophile_smiles,
        "combined_protonated": combined_protonated if len(all_results) > 0 else None,
        "combined_deprotonated": combined_deprotonated if len(all_results) > 0 else None,
        "per_warhead_results": all_results,  # Include per-warhead DataFrames for test mode
        "warhead_analysis": warhead_analysis,
        "nucleophile_site_lookup": nucleophile_site_lookup,
    }


def generate_filtering_statistics_report(filter_statistics, output_dir):
    """
    Generate a detailed filtering statistics report showing the reduction
    in search space at each filtering step.
    
    Args:
        filter_statistics: Dictionary containing statistics for each warhead
        output_dir: Output directory path
    """
    report_path = os.path.join(output_dir, "filtering_statistics.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("FILTERING STATISTICS REPORT\n")
        f.write("Search Space Reduction Analysis\n")
        f.write("=" * 100 + "\n\n")
        
        for warhead_name, stats in filter_statistics.items():
            f.write("=" * 100 + "\n")
            f.write(f"WARHEAD: {warhead_name}\n")
            f.write("=" * 100 + "\n\n")
            
            total_residues = stats["total_residues_in_pdb"]
            total_nucleophiles = stats["total_nucleophiles"]
            accessible = stats["accessible_nucleophiles"]
            
            # Helper function to calculate percentages
            def calc_percentages(current, previous, total_baseline):
                if previous == 0:
                    relative_pct = 0.0
                else:
                    relative_pct = ((previous - current) / previous) * 100
                
                if total_baseline == 0:
                    absolute_pct = 0.0
                else:
                    absolute_pct = ((total_baseline - current) / total_baseline) * 100
                
                return relative_pct, absolute_pct
            
            # Process protonated
            f.write("-" * 100 + "\n")
            f.write("PROTONATED FORM\n")
            f.write("-" * 100 + "\n\n")
            
            prot = stats["protonated"]
            
            # Initial State: Total residues in PDB (all amino acids)
            f.write(f"Initial State: All Residues in PDB\n")
            f.write(f"  Count: {total_residues:,}\n")
            f.write(f"  Description: Starting point - all amino acid residues in structure\n\n")
            
            # Step 0: FILTER to nucleophilic residues only (Cys, Ser, Thr, Tyr, Lys, His)
            rel_pct, abs_pct = calc_percentages(total_nucleophiles, total_residues, total_residues)
            f.write(f"Step 0: Filter to Nucleophilic Residues Only\n")
            f.write(f"  Residue Types: Cys, Ser, Thr, Tyr, Lys, His\n")
            f.write(f"  Count: {total_nucleophiles:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {total_residues:,} to {total_nucleophiles:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 1: After accessibility filter
            rel_pct, abs_pct = calc_percentages(accessible, total_nucleophiles, total_residues)
            f.write(f"Step 1: Filter to Accessible Residues Only (Solvent Exposed)\n")
            f.write(f"  Count: {accessible:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {total_nucleophiles:,} to {accessible:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 2: After reactivity filter (top N types)
            after_react = prot["after_reactivity"]
            rel_pct, abs_pct = calc_percentages(after_react, accessible, total_residues)
            f.write(f"Step 2: Filter to Top N Reactive Types\n")
            f.write(f"  Count: {after_react:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {accessible:,} to {after_react:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 3: After orbital compatibility filter
            after_orb = prot["after_orbital"]
            rel_pct, abs_pct = calc_percentages(after_orb, after_react, total_residues)
            f.write(f"Step 3: Filter by Orbital Compatibility\n")
            f.write(f"  Count: {after_orb:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {after_react:,} to {after_orb:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 4: After HSAB match filter
            after_hsab = prot["after_hsab"]
            rel_pct, abs_pct = calc_percentages(after_hsab, after_orb, total_residues)
            f.write(f"Step 4: Filter by HSAB Match\n")
            f.write(f"  Count: {after_hsab:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {after_orb:,} to {after_hsab:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Final: All filters pass
            final = prot["final"]
            f.write(f"Final: All Filters Pass (Accessible + Reactive + Orbital)\n")
            f.write(f"  Count: {final:,}\n")
            f.write(f"  Total Reduction: {((total_residues - final) / total_residues * 100):.1f}% (from {total_residues:,} to {final:,})\n\n")
            
            # Final with HSAB: All filters pass including HSAB
            f.write(f"Final: All Filters Pass with HSAB (Accessible + Reactive + Orbital + HSAB)\n")
            f.write(f"  Count: {after_hsab:,}\n")
            f.write(f"  Total Reduction: {((total_residues - after_hsab) / total_residues * 100):.1f}% (from {total_residues:,} to {after_hsab:,})\n\n")
            
            # Process deprotonated
            f.write("-" * 100 + "\n")
            f.write("DEPROTONATED FORM\n")
            f.write("-" * 100 + "\n\n")
            
            deprot = stats["deprotonated"]
            
            # Initial State: Total residues in PDB (all amino acids)
            f.write(f"Initial State: All Residues in PDB\n")
            f.write(f"  Count: {total_residues:,}\n")
            f.write(f"  Description: Starting point - all amino acid residues in structure\n\n")
            
            # Step 0: FILTER to nucleophilic residues only
            rel_pct, abs_pct = calc_percentages(total_nucleophiles, total_residues, total_residues)
            f.write(f"Step 0: Filter to Nucleophilic Residues Only\n")
            f.write(f"  Residue Types: Cys, Ser, Thr, Tyr, Lys, His\n")
            f.write(f"  Count: {total_nucleophiles:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {total_residues:,} to {total_nucleophiles:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 1: After accessibility filter
            rel_pct, abs_pct = calc_percentages(accessible, total_nucleophiles, total_residues)
            f.write(f"Step 1: Filter to Accessible Residues Only (Solvent Exposed)\n")
            f.write(f"  Count: {accessible:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {total_nucleophiles:,} to {accessible:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 2: After reactivity filter (top N types)
            after_react = deprot["after_reactivity"]
            rel_pct, abs_pct = calc_percentages(after_react, accessible, total_residues)
            f.write(f"Step 2: Filter to Top N Reactive Types\n")
            f.write(f"  Count: {after_react:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {accessible:,} to {after_react:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 3: After orbital compatibility filter
            after_orb = deprot["after_orbital"]
            rel_pct, abs_pct = calc_percentages(after_orb, after_react, total_residues)
            f.write(f"Step 3: Filter by Orbital Compatibility\n")
            f.write(f"  Count: {after_orb:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {after_react:,} to {after_orb:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Step 4: After HSAB match filter
            after_hsab = deprot["after_hsab"]
            rel_pct, abs_pct = calc_percentages(after_hsab, after_orb, total_residues)
            f.write(f"Step 4: Filter by HSAB Match\n")
            f.write(f"  Count: {after_hsab:,}\n")
            f.write(f"  Relative Reduction: {rel_pct:.1f}% (from {after_orb:,} to {after_hsab:,})\n")
            f.write(f"  Absolute Reduction: {abs_pct:.1f}% (from original {total_residues:,})\n\n")
            
            # Final: All filters pass
            final = deprot["final"]
            f.write(f"Final: All Filters Pass (Accessible + Reactive + Orbital)\n")
            f.write(f"  Count: {final:,}\n")
            f.write(f"  Total Reduction: {((total_residues - final) / total_residues * 100):.1f}% (from {total_residues:,} to {final:,})\n\n")
            
            # Final with HSAB: All filters pass including HSAB
            f.write(f"Final: All Filters Pass with HSAB (Accessible + Reactive + Orbital + HSAB)\n")
            f.write(f"  Count: {after_hsab:,}\n")
            f.write(f"  Total Reduction: {((total_residues - after_hsab) / total_residues * 100):.1f}% (from {total_residues:,} to {after_hsab:,})\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"   💾 Saved filtering statistics report to: {report_path}")


def generate_batch_statistics_csv(batch_results, output_file="batch_filtering_statistics.csv", test_mode=False):
    """
    Generate a CSV file with filtering statistics for all batch processed entries.
    
    Each row represents one warhead (protonated or deprotonated state).
    If no warhead found, outputs one row with "No warhead found".
    
    Args:
        batch_results: List of dicts with keys: name, pdb_file, electrophile_smiles, filter_stats, 
                       test_residue (optional), test_resnum (optional), test_chain (optional),
                       combined_protonated (optional), combined_deprotonated (optional),
                       per_warhead_results (optional)
        output_file: Path to output CSV file
        test_mode: If True, include "found_site" and "found_site_with_HSAB" columns for testing known binding sites
    """
    print(f"\n📊 Generating batch filtering statistics CSV...")
    
    rows = []
    
    for result in batch_results:
        name = result["name"]
        pdb_file = result["pdb_file"]
        electrophile_smiles = result["electrophile_smiles"]
        filter_stats = result["filter_stats"]
        
        # Test mode information
        test_residue = result.get("test_residue")
        test_resnum = result.get("test_resnum")
        test_chain = result.get("test_chain")
        combined_protonated = result.get("combined_protonated")
        combined_deprotonated = result.get("combined_deprotonated")
        per_warhead_results = result.get("per_warhead_results", [])
        warhead_analysis = result.get("warhead_analysis", {})
        nucleophile_site_lookup = result.get("nucleophile_site_lookup", {})
        
        # Helper function to check if test site is in final results
        def check_site_found(df, residue, resnum, chain, binary_col="Binary_Score"):
            """Check if a specific residue is in the dataframe with binary column == 1
            Returns: True if found, False if not found, 'no matches' if no residues pass filter
            """
            if df is None:
                print(f"  DEBUG: DataFrame is None for {binary_col}")
                return "no matches"
            if df.empty:
                print(f"  DEBUG: DataFrame is empty for {binary_col}")
                return "no matches"
            # Filter for residues that pass all filters (binary column == 1)
            passing = df[df[binary_col] == 1]
            if len(passing) == 0:
                print(f"  DEBUG: No residues with {binary_col}=1 (total rows: {len(df)})")
                return "no matches"
            # Clean inputs for robust string matching
            residue_clean = str(residue).strip().upper()
            resnum_str = str(resnum).strip()
            chain_clean = str(chain).strip()
            
            # Check if the test site is in the passing results
            # Note: ResNum is stored as string in CSV, so compare as strings
            match = passing[
                (passing["Residue"].astype(str).str.strip().str.upper() == residue_clean) &
                (passing["ResNum"].astype(str).str.strip() == resnum_str) &
                (passing["Chain"].astype(str).str.strip() == chain_clean)
            ]
            
            # Debug output
            if len(match) > 0:
                print(f"  ✓ Found {residue_clean} {resnum_str} {chain_clean} in {binary_col}")
                print(f"    Passing warheads: {match['Warhead'].unique().tolist()}")
            else:
                print(f"  ✗ Did not find {residue_clean} {resnum_str} {chain_clean} in {binary_col} (checked {len(passing)} passing residues)")
            
            return len(match) > 0

        def evaluate_site_steps(warhead_name, is_protonated):
            """Evaluate site-level filter progression for one warhead/state."""
            if not (test_mode and test_residue and test_resnum is not None and test_chain):
                return {
                    "found_site": False,
                    "step0_nucleophilic_pass": "",
                    "step1_accessible_pass": "",
                    "step2_reactivity_pass": "",
                    "step3_orbital_pass": "",
                    "failed_step": "",
                    "succeeded_steps": "",
                }

            residue_clean = str(test_residue).strip().upper()
            resnum_clean = str(test_resnum).strip()
            chain_clean = str(test_chain).strip()
            site_key = (residue_clean, resnum_clean, chain_clean)

            step0_nucleophilic_pass = False
            step1_accessible_pass = False
            step2_reactivity_pass = False
            step3_orbital_pass = False

            site_info = nucleophile_site_lookup.get(site_key)
            if site_info is not None:
                step0_nucleophilic_pass = True
                if bool(site_info.get("accessible", False)):
                    step1_accessible_pass = True

            wh_analysis = warhead_analysis.get(warhead_name, {})
            if is_protonated:
                top_types = [x.strip().upper() for x in wh_analysis.get("top_types_protonated", [])]
                orbital_map = wh_analysis.get("orbital_compatibility_protonated", {})
            else:
                top_types = [x.strip().upper() for x in wh_analysis.get("top_types_deprotonated", [])]
                orbital_map = wh_analysis.get("orbital_compatibility_deprotonated", {})

            if step1_accessible_pass and residue_clean in top_types:
                step2_reactivity_pass = True

            if step2_reactivity_pass:
                orbital_entry = orbital_map.get(residue_clean, {})
                step3_orbital_pass = bool(orbital_entry.get("compatible", False))

            succeeded = []
            if step0_nucleophilic_pass:
                succeeded.append("step0_nucleophilic")
            if step1_accessible_pass:
                succeeded.append("step1_accessible")
            if step2_reactivity_pass:
                succeeded.append("step2_reactivity")
            if step3_orbital_pass:
                succeeded.append("step3_orbital")

            found_site = step0_nucleophilic_pass and step1_accessible_pass and step2_reactivity_pass and step3_orbital_pass
            if found_site:
                failed_step = ""
            elif not step0_nucleophilic_pass:
                failed_step = "step0_nucleophilic"
            elif not step1_accessible_pass:
                failed_step = "step1_accessible"
            elif not step2_reactivity_pass:
                failed_step = "step2_reactivity"
            else:
                failed_step = "step3_orbital"

            return {
                "found_site": found_site,
                "step0_nucleophilic_pass": step0_nucleophilic_pass,
                "step1_accessible_pass": step1_accessible_pass,
                "step2_reactivity_pass": step2_reactivity_pass,
                "step3_orbital_pass": step3_orbital_pass,
                "failed_step": failed_step,
                "succeeded_steps": ";".join(succeeded),
            }
        
        # Check if any warheads were found
        if not filter_stats or len(filter_stats) == 0:
            # No warhead found - output single row
            row_data = {
                "name": name,
                "pdb_file": pdb_file,
                "electrophile_smiles": electrophile_smiles,
                "warhead_type": "No warhead found",
                "is_protonated": "",
                "total_residues": "",
                "step0_starting_count": "",
                "step0_filtered_count": "",
                "step0_relative_reduction_pct": "",
                "step0_absolute_reduction_pct": "",
                "step1_starting_count": "",
                "step1_filtered_count": "",
                "step1_relative_reduction_pct": "",
                "step1_absolute_reduction_pct": "",
                "step2_starting_count": "",
                "step2_filtered_count": "",
                "step2_relative_reduction_pct": "",
                "step2_absolute_reduction_pct": "",
                "step3_starting_count": "",
                "step3_filtered_count": "",
                "step3_relative_reduction_pct": "",
                "step3_absolute_reduction_pct": "",
                "step4_starting_count": "",
                "step4_filtered_count": "",
                "step4_relative_reduction_pct": "",
                "step4_absolute_reduction_pct": ""
            }
            if test_mode:
                row_data["found_site"] = ""
                row_data["found_site_with_HSAB"] = ""
                row_data["step0_nucleophilic_pass"] = ""
                row_data["step1_accessible_pass"] = ""
                row_data["step2_reactivity_pass"] = ""
                row_data["step3_orbital_pass"] = ""
                row_data["failed_step"] = ""
                row_data["succeeded_steps"] = ""
            rows.append(row_data)
            continue
        
        # Create a lookup for per-warhead DataFrames (for test mode)
        warhead_dfs = {}
        if test_mode:
            print(f"  DEBUG: Building warhead lookup for {name}...")
            print(f"  DEBUG: per_warhead_results has {len(per_warhead_results)} entries")
        for wh_result in per_warhead_results:
            wh_name = wh_result.get("warhead_name")
            if wh_name:
                warhead_dfs[wh_name] = {
                    "protonated": wh_result.get("protonated"),
                    "deprotonated": wh_result.get("deprotonated")
                }
                if test_mode:
                    prot_df = wh_result.get("protonated")
                    deprot_df = wh_result.get("deprotonated")
                    print(f"  DEBUG: Added warhead '{wh_name}' - protonated: {len(prot_df) if prot_df is not None else 0} rows, deprotonated: {len(deprot_df) if deprot_df is not None else 0} rows")
        
        if test_mode:
            print(f"  DEBUG: warhead_dfs keys: {list(warhead_dfs.keys())}")
            print(f"  DEBUG: filter_stats keys: {list(filter_stats.keys())}")
        
        # Process each warhead
        for warhead_name, stats in filter_stats.items():
            total_residues = stats["total_residues_in_pdb"]
            total_nucleophiles = stats["total_nucleophiles"]
            accessible = stats["accessible_nucleophiles"]
            
            # Helper function to calculate reduction percentages
            def calc_reduction(filtered, starting, absolute_baseline):
                rel_pct = ((starting - filtered) / starting * 100) if starting > 0 else 0.0
                abs_pct = ((absolute_baseline - filtered) / absolute_baseline * 100) if absolute_baseline > 0 else 0.0
                return rel_pct, abs_pct
            
            # Process protonated
            prot = stats["protonated"]
            after_react_prot = prot["after_reactivity"]
            after_orb_prot = prot["after_orbital"]
            after_hsab_prot = prot["after_hsab"]
            final_prot = prot["final"]
            
            # Step 0: Nucleophilic filter
            step0_rel, step0_abs = calc_reduction(total_nucleophiles, total_residues, total_residues)
            # Step 1: Accessibility filter
            step1_rel, step1_abs = calc_reduction(accessible, total_nucleophiles, total_residues)
            # Step 2: Reactivity filter
            step2_rel, step2_abs = calc_reduction(after_react_prot, accessible, total_residues)
            # Step 3: Orbital filter
            step3_rel, step3_abs = calc_reduction(after_orb_prot, after_react_prot, total_residues)
            # Step 4: HSAB filter
            step4_rel, step4_abs = calc_reduction(after_hsab_prot, after_orb_prot, total_residues)
            
            # Check if test site is found (for test mode) - check THIS WARHEAD ONLY
            site_found_prot = False
            site_found_with_hsab_prot = False
            site_eval_prot = {
                "found_site": False,
                "step0_nucleophilic_pass": "",
                "step1_accessible_pass": "",
                "step2_reactivity_pass": "",
                "step3_orbital_pass": "",
                "failed_step": "",
                "succeeded_steps": "",
            }
            if test_mode and test_residue and test_resnum is not None and test_chain:
                # Get this specific warhead's DataFrame
                print(f"  DEBUG: Looking up warhead '{warhead_name}' in warhead_dfs...")
                warhead_df_prot = warhead_dfs.get(warhead_name, {}).get("protonated")
                if warhead_df_prot is not None:
                    print(f"  DEBUG: Found DataFrame with {len(warhead_df_prot)} rows for protonated {warhead_name}")
                else:
                    print(f"  DEBUG: No DataFrame found for protonated {warhead_name}")
                site_found_prot = check_site_found(warhead_df_prot, test_residue, test_resnum, test_chain, "Binary_Score")
                site_found_with_hsab_prot = check_site_found(warhead_df_prot, test_residue, test_resnum, test_chain, "Binary_Score_With_HSAB")
                site_eval_prot = evaluate_site_steps(warhead_name, is_protonated=True)
                site_found_prot = site_eval_prot["found_site"]
            
            row_data_prot = {
                "name": name,
                "pdb_file": pdb_file,
                "electrophile_smiles": electrophile_smiles,
                "warhead_type": warhead_name,
                "is_protonated": True,
                "total_residues": total_residues,
                "step0_starting_count": total_residues,
                "step0_filtered_count": total_nucleophiles,
                "step0_relative_reduction_pct": f"{step0_rel:.2f}",
                "step0_absolute_reduction_pct": f"{step0_abs:.2f}",
                "step1_starting_count": total_nucleophiles,
                "step1_filtered_count": accessible,
                "step1_relative_reduction_pct": f"{step1_rel:.2f}",
                "step1_absolute_reduction_pct": f"{step1_abs:.2f}",
                "step2_starting_count": accessible,
                "step2_filtered_count": after_react_prot,
                "step2_relative_reduction_pct": f"{step2_rel:.2f}",
                "step2_absolute_reduction_pct": f"{step2_abs:.2f}",
                "step3_starting_count": after_react_prot,
                "step3_filtered_count": after_orb_prot,
                "step3_relative_reduction_pct": f"{step3_rel:.2f}",
                "step3_absolute_reduction_pct": f"{step3_abs:.2f}",
                "step4_starting_count": after_orb_prot,
                "step4_filtered_count": after_hsab_prot,
                "step4_relative_reduction_pct": f"{step4_rel:.2f}",
                "step4_absolute_reduction_pct": f"{step4_abs:.2f}"
            }
            if test_mode:
                row_data_prot["found_site"] = site_found_prot
                row_data_prot["found_site_with_HSAB"] = site_found_with_hsab_prot
                row_data_prot["step0_nucleophilic_pass"] = site_eval_prot["step0_nucleophilic_pass"]
                row_data_prot["step1_accessible_pass"] = site_eval_prot["step1_accessible_pass"]
                row_data_prot["step2_reactivity_pass"] = site_eval_prot["step2_reactivity_pass"]
                row_data_prot["step3_orbital_pass"] = site_eval_prot["step3_orbital_pass"]
                row_data_prot["failed_step"] = site_eval_prot["failed_step"]
                row_data_prot["succeeded_steps"] = site_eval_prot["succeeded_steps"]
            rows.append(row_data_prot)
            
            # Process deprotonated
            deprot = stats["deprotonated"]
            after_react_deprot = deprot["after_reactivity"]
            after_orb_deprot = deprot["after_orbital"]
            after_hsab_deprot = deprot["after_hsab"]
            final_deprot = deprot["final"]
            
            # Step 0: Nucleophilic filter (same as protonated)
            # Step 1: Accessibility filter (same as protonated)
            # Step 2: Reactivity filter
            step2_rel, step2_abs = calc_reduction(after_react_deprot, accessible, total_residues)
            # Step 3: Orbital filter
            step3_rel, step3_abs = calc_reduction(after_orb_deprot, after_react_deprot, total_residues)
            # Step 4: HSAB filter
            step4_rel, step4_abs = calc_reduction(after_hsab_deprot, after_orb_deprot, total_residues)
            
            # Check if test site is found (for test mode) - check THIS WARHEAD ONLY
            site_found_deprot = False
            site_found_with_hsab_deprot = False
            site_eval_deprot = {
                "found_site": False,
                "step0_nucleophilic_pass": "",
                "step1_accessible_pass": "",
                "step2_reactivity_pass": "",
                "step3_orbital_pass": "",
                "failed_step": "",
                "succeeded_steps": "",
            }
            if test_mode and test_residue and test_resnum is not None and test_chain:
                # Get this specific warhead's DataFrame
                print(f"  DEBUG: Looking up warhead '{warhead_name}' in warhead_dfs...")
                warhead_df_deprot = warhead_dfs.get(warhead_name, {}).get("deprotonated")
                if warhead_df_deprot is not None:
                    print(f"  DEBUG: Found DataFrame with {len(warhead_df_deprot)} rows for deprotonated {warhead_name}")
                else:
                    print(f"  DEBUG: No DataFrame found for deprotonated {warhead_name}")
                site_found_deprot = check_site_found(warhead_df_deprot, test_residue, test_resnum, test_chain, "Binary_Score")
                site_found_with_hsab_deprot = check_site_found(warhead_df_deprot, test_residue, test_resnum, test_chain, "Binary_Score_With_HSAB")
                site_eval_deprot = evaluate_site_steps(warhead_name, is_protonated=False)
                site_found_deprot = site_eval_deprot["found_site"]
            
            row_data_deprot = {
                "name": name,
                "pdb_file": pdb_file,
                "electrophile_smiles": electrophile_smiles,
                "warhead_type": warhead_name,
                "is_protonated": False,
                "total_residues": total_residues,
                "step0_starting_count": total_residues,
                "step0_filtered_count": total_nucleophiles,
                "step0_relative_reduction_pct": f"{step0_rel:.2f}",
                "step0_absolute_reduction_pct": f"{step0_abs:.2f}",
                "step1_starting_count": total_nucleophiles,
                "step1_filtered_count": accessible,
                "step1_relative_reduction_pct": f"{step1_rel:.2f}",
                "step1_absolute_reduction_pct": f"{step1_abs:.2f}",
                "step2_starting_count": accessible,
                "step2_filtered_count": after_react_deprot,
                "step2_relative_reduction_pct": f"{step2_rel:.2f}",
                "step2_absolute_reduction_pct": f"{step2_abs:.2f}",
                "step3_starting_count": after_react_deprot,
                "step3_filtered_count": after_orb_deprot,
                "step3_relative_reduction_pct": f"{step3_rel:.2f}",
                "step3_absolute_reduction_pct": f"{step3_abs:.2f}",
                "step4_starting_count": after_orb_deprot,
                "step4_filtered_count": after_hsab_deprot,
                "step4_relative_reduction_pct": f"{step4_rel:.2f}",
                "step4_absolute_reduction_pct": f"{step4_abs:.2f}"
            }
            if test_mode:
                row_data_deprot["found_site"] = site_found_deprot
                row_data_deprot["found_site_with_HSAB"] = site_found_with_hsab_deprot
                row_data_deprot["step0_nucleophilic_pass"] = site_eval_deprot["step0_nucleophilic_pass"]
                row_data_deprot["step1_accessible_pass"] = site_eval_deprot["step1_accessible_pass"]
                row_data_deprot["step2_reactivity_pass"] = site_eval_deprot["step2_reactivity_pass"]
                row_data_deprot["step3_orbital_pass"] = site_eval_deprot["step3_orbital_pass"]
                row_data_deprot["failed_step"] = site_eval_deprot["failed_step"]
                row_data_deprot["succeeded_steps"] = site_eval_deprot["succeeded_steps"]
            rows.append(row_data_deprot)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"   💾 Saved batch filtering statistics to: {output_file}")
    print(f"   📊 Total rows: {len(df)}")


def find_pdb_file(pdb_filename, pdb_dir=None):
    """
    Find a PDB file with flexible matching (case-insensitive, auto-add .pdb extension).
    
    Args:
        pdb_filename: PDB filename from CSV (e.g., "1BMQ", "1bmq.pdb", "4g5j")
        pdb_dir: Optional directory to search in
    
    Returns:
        Full path to PDB file if found, None otherwise
    """
    # Try different variations of the filename
    variations = []
    
    # Original filename
    variations.append(pdb_filename)
    
    # Add .pdb extension if not present
    if not pdb_filename.lower().endswith('.pdb'):
        variations.append(pdb_filename + '.pdb')
    
    # Try lowercase versions
    variations.append(pdb_filename.lower())
    if not pdb_filename.lower().endswith('.pdb'):
        variations.append(pdb_filename.lower() + '.pdb')
    
    # Try uppercase versions
    variations.append(pdb_filename.upper())
    if not pdb_filename.upper().endswith('.PDB'):
        variations.append(pdb_filename.upper() + '.PDB')
    
    # Check each variation
    for variant in variations:
        # Try direct path first
        if os.path.exists(variant):
            return os.path.abspath(variant)
        
        # Try with pdb_dir if provided
        if pdb_dir is not None:
            full_path = os.path.join(pdb_dir, variant)
            if os.path.exists(full_path):
                return os.path.abspath(full_path)
    
    # If not found with variations, try case-insensitive directory search
    if pdb_dir is not None and os.path.exists(pdb_dir):
        # Get the base filename without path
        base_filename = os.path.basename(pdb_filename)
        # Add .pdb if not present
        if not base_filename.lower().endswith('.pdb'):
            base_filename_with_ext = base_filename + '.pdb'
        else:
            base_filename_with_ext = base_filename
        
        # List all files in directory and compare case-insensitively
        try:
            for file in os.listdir(pdb_dir):
                if file.lower() == base_filename.lower() or file.lower() == base_filename_with_ext.lower():
                    return os.path.abspath(os.path.join(pdb_dir, file))
        except Exception:
            pass
    
    return None


def get_csv_value(row, *column_names, default=None):
    """Return the first present non-NaN CSV value from a row using alias column names."""
    for column_name in column_names:
        if column_name in row.index:
            value = row[column_name]
            if pd.notna(value):
                return value
    return default


def batch_process(csv_file, pdb_dir=None, pdb_download_dir=None, test_mode=False, n_workers=1, use_cache=True):
    """
    Process multiple protein-electrophile pairs from a CSV file.
    
    CSV format (with header):
        Required: name, protein pdb, top n types
        Optional: electrophile smiles, LigID (ligand ID from PDB)
        Test mode: Residue, ResNum, Chain (known binding site to test)
    
    Example CSV:
        name,protein pdb,electrophile smiles,top n types
        afatinib,protein1.pdb,C=CC(=O)Nc1cc(F)cc(Cl)c1,3
        
    Example CSV with ligand extraction:
        name,protein pdb,electrophile smiles,LigID,top n types
        test1,4g5j.pdb,,AFA,3
        
    Example CSV for testing:
        name,protein pdb,electrophile smiles,top n types,Residue,ResNum,Chain
        test1,4g5j.pdb,SMILES_HERE,3,Cys,285,A
    
    Args:
        csv_file: Path to CSV file with batch processing parameters
        pdb_dir: Directory containing PDB files (prepended to relative paths in CSV)
        pdb_download_dir: Directory to download PDB files if not found locally (optional)
        test_mode: If True, expects Residue, ResNum, Chain columns for testing known sites
        n_workers: Number of parallel workers for calculations (default: 1)
        use_cache: If True, use cached results; if False, force recalculation but still save to cache (default: True)
    """
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
        sys.exit(1)
    
    print("=" * 80)
    if test_mode:
        print("🧪 BATCH TESTING MODE")
    else:
        print("📋 BATCH PROCESSING MODE")
    print("=" * 80)
    print(f"Reading from: {csv_file}\n")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_column_groups = [
        ['name'],
        ['protein pdb'],
        ['top n types'],
    ]
    missing_columns = [" / ".join(group) for group in required_column_groups if not any(col in df.columns for col in group)]
    if missing_columns:
        print(f"❌ Error: CSV missing required columns: {missing_columns}")
        print("   Required columns: name, protein pdb, top n types")
        print(f"   Found columns: {list(df.columns)}")
        sys.exit(1)
    
    # Check for test mode columns
    if test_mode:
        test_columns = ['Residue', 'ResNum', 'Chain']
        missing_test_cols = [col for col in test_columns if col not in df.columns]
        if missing_test_cols:
            print(f"❌ Error: Test mode enabled but missing columns: {missing_test_cols}")
            print(f"   Test mode requires: {test_columns}")
            sys.exit(1)
    
    # Process each row
    total_rows = len(df)
    successful = 0
    failed = 0
    batch_results = []  # Collect statistics for batch CSV output
    
    for idx, row in df.iterrows():
        row_num = idx + 1
        print("\n" + "=" * 80)
        print(f"📌 Processing row {row_num}/{total_rows}")
        print("=" * 80)
        
        name = row['name']
        pdb_file = row['protein pdb']
        electrophile_smiles = get_csv_value(row, 'electrophile smiles', 'electrophile_smiles', default='')
        top_n_types = int(row['top n types'])
        
        # Get test information if in test mode
        test_residue = None
        test_resnum = None
        test_chain = None
        if test_mode:
            test_residue = row.get('Residue')
            test_resnum_raw = row.get('ResNum')
            test_chain = row.get('Chain')
            # Convert ResNum to int if possible
            try:
                test_resnum = int(test_resnum_raw) if pd.notna(test_resnum_raw) else None
            except (ValueError, TypeError):
                print(f"⚠️  Warning: Invalid ResNum '{test_resnum_raw}', treating as None")
                test_resnum = None
        
        print(f"Name: {name}")
        print(f"PDB: {pdb_file}")
        
        # Check if we need to extract ligand from PDB
        if pd.isna(electrophile_smiles) or electrophile_smiles == '':
            # Try to get LigID column
            ligand_id = get_csv_value(row, 'LigID', 'lig_id', default='')
            if pd.isna(ligand_id) or ligand_id == '':
                print(f"❌ Error: No electrophile SMILES or LigID provided")
                failed += 1
                result_data = {
                    "name": name,
                    "pdb_file": pdb_file,
                    "electrophile_smiles": "",
                    "filter_stats": {},
                    "per_warhead_results": []
                }
                if test_mode:
                    result_data.update({
                        "test_residue": test_residue,
                        "test_resnum": test_resnum,
                        "test_chain": test_chain
                    })
                batch_results.append(result_data)
                continue
            
            print(f"LigID: {ligand_id} (will extract from PDB)")
        else:
            print(f"Electrophile: {electrophile_smiles}")
        
        print(f"Top N types: {top_n_types}")
        if test_mode:
            print(f"Test Site: {test_residue}{test_resnum}:{test_chain}")
        
        # Check if PDB file exists (flexible matching: case-insensitive, auto .pdb extension)
        pdb_path_to_use = find_pdb_file(pdb_file, pdb_dir)
        
        if pdb_path_to_use:
            print(f"   ✓ Found PDB: {pdb_path_to_use}")
        
        # If still not found, try download or skip
        if not pdb_path_to_use:
            if pdb_download_dir is not None:
                print(f"⚠️  PDB file not found: {pdb_file}")
                # Try to download (normalize filename)
                pdb_filename = os.path.basename(pdb_file)
                downloaded_path = download_pdb_file(pdb_filename, pdb_download_dir)
                if downloaded_path:
                    pdb_path_to_use = downloaded_path
                else:
                    print(f"   Failed to download. Skipping this row...\n")
                    failed += 1
                    # Record as no warhead found
                    result_data = {
                        "name": name,
                        "pdb_file": pdb_file,
                        "electrophile_smiles": electrophile_smiles,
                        "filter_stats": {}
                    }
                    if test_mode:
                        result_data.update({
                            "test_residue": test_residue,
                            "test_resnum": test_resnum,
                            "test_chain": test_chain
                        })
                    batch_results.append(result_data)
                    continue
            else:
                print(f"⚠️  Warning: PDB file not found: {pdb_file}")
                print(f"   Skipping this row...\n")
                failed += 1
                # Record as no warhead found
                result_data = {
                    "name": name,
                    "pdb_file": pdb_file,
                    "electrophile_smiles": electrophile_smiles,
                    "filter_stats": {}
                }
                if test_mode:
                    result_data.update({
                        "test_residue": test_residue,
                        "test_resnum": test_resnum,
                        "test_chain": test_chain
                    })
                batch_results.append(result_data)
                continue
        
        # Extract ligand if needed
        if pd.isna(electrophile_smiles) or electrophile_smiles == '':
            ligand_id = get_csv_value(row, 'LigID', 'lig_id', default='')
            extracted_smiles = extract_ligand_from_pdb(pdb_path_to_use, ligand_id)
            if extracted_smiles is None:
                print(f"❌ Error: Failed to extract ligand {ligand_id}")
                failed += 1
                result_data = {
                    "name": name,
                    "pdb_file": pdb_file,
                    "electrophile_smiles": "",
                    "filter_stats": {},
                    "per_warhead_results": []
                }
                if test_mode:
                    result_data.update({
                        "test_residue": test_residue,
                        "test_resnum": test_resnum,
                        "test_chain": test_chain
                    })
                batch_results.append(result_data)
                continue
            electrophile_smiles = extracted_smiles
            print(f"   Using extracted SMILES: {electrophile_smiles}")
        
        # Run Frankenstein for this row
        try:
            result = main(pdb_path_to_use, electrophile_smiles, name, top_n_types, n_workers, use_cache)
            if result is not None:
                # Extract results from dictionary
                filter_stats = result["filter_statistics"]
                output_prefix = result["output_prefix"]
                smiles = result["electrophile_smiles"]
                combined_protonated = result.get("combined_protonated")
                combined_deprotonated = result.get("combined_deprotonated")
                per_warhead_results = result.get("per_warhead_results", [])
                warhead_analysis = result.get("warhead_analysis", {})
                nucleophile_site_lookup = result.get("nucleophile_site_lookup", {})
                
                result_data = {
                    "name": name,
                    "pdb_file": pdb_file,
                    "electrophile_smiles": smiles,
                    "filter_stats": filter_stats,
                    "combined_protonated": combined_protonated,
                    "combined_deprotonated": combined_deprotonated,
                    "per_warhead_results": per_warhead_results,
                    "warhead_analysis": warhead_analysis,
                    "nucleophile_site_lookup": nucleophile_site_lookup,
                }
                if test_mode:
                    result_data.update({
                        "test_residue": test_residue,
                        "test_resnum": test_resnum,
                        "test_chain": test_chain
                    })
                batch_results.append(result_data)
            else:
                # No results (e.g., no warheads found)
                result_data = {
                    "name": name,
                    "pdb_file": pdb_file,
                    "electrophile_smiles": electrophile_smiles,
                    "filter_stats": {},
                    "per_warhead_results": []
                }
                if test_mode:
                    result_data.update({
                        "test_residue": test_residue,
                        "test_resnum": test_resnum,
                        "test_chain": test_chain
                    })
                batch_results.append(result_data)
            successful += 1
            print(f"✅ Successfully completed processing for '{name}'")
        except Exception as e:
            print(f"❌ Error processing row {row_num} ('{name}'): {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            # Record as failed
            result_data = {
                "name": name,
                "pdb_file": pdb_file,
                "electrophile_smiles": electrophile_smiles,
                "filter_stats": {},
                "per_warhead_results": []
            }
            if test_mode:
                result_data.update({
                    "test_residue": test_residue,
                    "test_resnum": test_resnum,
                    "test_chain": test_chain
                })
            batch_results.append(result_data)
            continue
    
    # Generate batch statistics CSV
    if len(batch_results) > 0:
        batch_csv_path = os.path.join(os.path.dirname(csv_file), "batch_filtering_statistics.csv")
        generate_batch_statistics_csv(batch_results, batch_csv_path, test_mode)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total rows: {total_rows}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("=" * 80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Check for batch mode or test mode
    if len(sys.argv) >= 2 and sys.argv[1] in ['--batch', '-b', '--test', '-test']:
        if len(sys.argv) < 3:
            print("Usage: python Frankenstein.py --batch <csv_file> [OPTIONS]")
            print("       python Frankenstein.py --test <csv_file> [OPTIONS]")
            print()
            print("Batch mode: Process multiple protein-electrophile pairs from a CSV file")
            print("Test mode: Same as batch mode, but expects Residue,ResNum,Chain columns to test known binding sites")
            print()
            print("CSV format (with header):")
            print("  Required: name,protein pdb,top n types")
            print("  Optional: electrophile smiles, LigID (ligand from PDB)")
            print("  Test mode: Residue,ResNum,Chain (known binding site)")
            print()
            print("Example CSV:")
            print("  name,protein pdb,electrophile smiles,top n types")
            print('  afatinib,4g5j.pdb,C=CC(=O)Nc1cc(F)cc(Cl)c1,3')
            print()
            print("Example CSV with ligand extraction:")
            print("  name,protein pdb,electrophile smiles,LigID,top n types")
            print('  test1,4g5j.pdb,,AFA,3  # Extract ligand AFA from PDB')
            print()
            print("Example CSV for testing:")
            print("  name,protein pdb,electrophile smiles,top n types,Residue,ResNum,Chain")
            print('  test1,4g5j.pdb,SMILES,3,Cys,285,A  # Check if Cys285:A passes all filters')
            print()
            print("Options:")
            print("  --pdb-dir <dir>            Directory containing PDB files (prepended to paths in CSV)")
            print("  --pdb-download-dir <dir>   Directory to download missing PDB files")
            print("  --workers N, -j N          Number of parallel workers for calculations (default: 1)")
            print("  --no-cache                 Force recalculation (ignore cache, but still save results)")
            print()
            print("Examples:")
            print('  python Frankenstein.py --batch batch_input.csv')
            print('  python Frankenstein.py --batch batch_input.csv --pdb-dir ../Existing_Structures')
            print('  python Frankenstein.py --test test_input.csv --pdb-dir ../Existing_Structures')
            print('  python Frankenstein.py --batch batch_input.csv --workers 4')
            sys.exit(1)
        
        test_mode = sys.argv[1] in ['--test', '-test']
        csv_file = sys.argv[2]
        pdb_dir = None
        pdb_download_dir = None
        n_workers = 1  # default sequential
        use_cache = True  # default use cache
        
        # Parse optional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--pdb-dir' and i + 1 < len(sys.argv):
                pdb_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--pdb-download-dir' and i + 1 < len(sys.argv):
                pdb_download_dir = sys.argv[i + 1]
                if not os.path.exists(pdb_download_dir):
                    print(f"Creating PDB download directory: {pdb_download_dir}")
                    os.makedirs(pdb_download_dir, exist_ok=True)
                i += 2
            elif sys.argv[i] in ['--workers', '-j'] and i + 1 < len(sys.argv):
                n_workers = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--no-cache':
                use_cache = False
                i += 1
            else:
                print(f"Unknown argument: {sys.argv[i]}")
                sys.exit(1)
        
        batch_process(csv_file, pdb_dir, pdb_download_dir, test_mode, n_workers, use_cache)
    
    # Single mode
    elif len(sys.argv) >= 3:
        pdb_file = sys.argv[1]
        electrophile_smiles = sys.argv[2]
        output_prefix = sys.argv[3] if len(sys.argv) > 3 else "Frankenstein"
        top_n_types = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        n_workers = 1  # default sequential
        use_cache = True  # default use cache
        
        # Check for --workers or -j flag and --no-cache flag
        i = 5
        while i < len(sys.argv):
            if sys.argv[i] in ['--workers', '-j'] and i + 1 < len(sys.argv):
                n_workers = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--no-cache':
                use_cache = False
                i += 1
            else:
                i += 1
        
        if not os.path.exists(pdb_file):
            print(f"❌ Error: PDB file not found: {pdb_file}")
            sys.exit(1)
        
        main(pdb_file, electrophile_smiles, output_prefix, top_n_types, n_workers, use_cache)
    
    # No valid arguments - show help
    else:
        print("FRANKENSTEIN - Covalent Binding Detector")
        print("=" * 80)
        print()
        print("SINGLE MODE:")
        print("  python Frankenstein.py <pdb_file> <electrophile_smiles> [output_prefix] [top_n_types] [--workers N] [--no-cache]")
        print()
        print("  Arguments:")
        print("    pdb_file           : Path to protein PDB file")
        print("    electrophile_smiles: SMILES string of electrophile")
        print("    output_prefix      : Prefix for output files (default: Frankenstein)")
        print("    top_n_types        : Number of top reactive AA types to consider (default: 3)")
        print("    --workers N, -j N  : Number of parallel workers for calculations (default: 1)")
        print("    --no-cache         : Force recalculation (ignore cache, but still save results)")
        print()
        print("  Example:")
        print('    python Frankenstein.py protein.pdb "C=CC(=O)Nc1cc(F)cc(Cl)c1" afatinib 3')
        print('    python Frankenstein.py protein.pdb "C=CC(=O)N" test 3 --workers 4')
        print()
        print("BATCH MODE:")
        print("  python Frankenstein.py --batch <csv_file>")
        print("  python Frankenstein.py -b <csv_file>")
        print()
        print("  CSV format (with header):")
        print("    name,protein pdb,electrophile smiles,top n types")
        print()
        print("  Example CSV:")
        print("    name,protein pdb,electrophile smiles,top n types")
        print('    afatinib,protein1.pdb,C=CC(=O)Nc1cc(F)cc(Cl)c1,3')
        print('    carfilzomib,4R67.pdb,CC(C)C[C@H]...,5')
        print()
        print("Workflow:")
        print("  1. Determines top N reactive amino acid types (quantum calculations once per type)")
        print("  2. Identifies accessible nucleophiles in protein (SASA/pKa analysis)")
        print("  3. Filters to only accessible residues matching top N reactive types")
        print("  4. Combines reactivity × accessibility scores")
        print("  5. Outputs ranked covalent binding predictions")
        sys.exit(1)
