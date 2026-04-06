from rdkit import Chem
from rdkit.Chem import rdchem

# --- classify atomic orbitals for simple paper check ---
def classify_atom_orbitals(atom):
    symbol = atom.GetSymbol()
    hyb_raw = atom.GetHybridization()
    charge = atom.GetFormalCharge()
    
    # Hybridization mapping for robustness (handles edge cases beyond SP/SP2/SP3)
    HYB_MAP = {
        rdchem.HybridizationType.SP3: "sp3",
        rdchem.HybridizationType.SP2: "sp2",
        rdchem.HybridizationType.SP:  "sp",
        rdchem.HybridizationType.S:   "s",
    }
    
    donor = []
    acceptor = []
    strained_ring = False
    is_anionic = (charge < 0)

    # Check if atom is in a strained 3-membered ring (epoxide, aziridine, cyclopropane)
    if atom.IsInRing():
        mol = atom.GetOwningMol()
        for ring_info in mol.GetRingInfo().AtomRings():
            if atom.GetIdx() in ring_info and len(ring_info) == 3:
                strained_ring = True
                break

    # donor orbitals: nucleophile
    if symbol in ["O", "N", "S"]:
        if charge > 0:
            pass  # Cationic atom (e.g. NH3+) has no available lone pair — occupied by extra N-H bond
        elif is_anionic:
            donor.append("anionic lone pair (enhanced nucleophilicity)")
        else:
            donor.append("lone pair")
    if symbol == "C" and charge < 0:
        donor.append("carbanion lone pair")
    if atom.GetIsAromatic():
        donor.append("pi electrons")

    # acceptor orbitals: electrophile
    if symbol == "C":
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
        # Only halogens (Cl, Br, I) are good leaving groups - not O
        if any(sym in ["Cl", "Br", "I"] for sym in neighbors):
            acceptor.append("sigma* orbital (alkyl halide)")
        if strained_ring and not atom.GetIsAromatic():
            acceptor.append("strained ring sigma* (high reactivity)")
        
        # Check for sp3 carbons with activated leaving groups (sulfonates, phosphonates)
        if atom.GetHybridization() == rdchem.HybridizationType.SP3:
            mol = atom.GetOwningMol()
            atom_idx = atom.GetIdx()
            activated_lg_patterns = [
                "[CX4][OX2][SX4](=O)=O",  # sulfonate ester (mesylate, tosylate)
                "[CX4][OX2][P]",          # phosphonate ester
            ]
            for pattern in activated_lg_patterns:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                for match in matches:
                    if atom_idx == match[0]:  # First atom in pattern is the carbon
                        acceptor.append("sigma* orbital (activated leaving group)")
                        break
        
        if atom.GetHybridization() == rdchem.HybridizationType.SP2:
            mol = atom.GetOwningMol()
            atom_idx = atom.GetIdx()
            
            # SMARTS-based Michael acceptor detection (conjugated with EWG)
            # Format: (pattern, reactive_atom_index_in_pattern)
            # Note: β-carbon (nucleophile attack site) is at index 0 in these patterns
            michael_patterns = [
                ("[CX3]=[CX3]-[CX3]=[OX1]", 0),        # enone, β-carbon at index 0
                ("[CX3]=[CX3]-[CX3](=O)[#7,#8]", 0),   # α,β-unsaturated amide/ester, β-carbon at index 0
                ("[CX3]=[CX3]-[#6]#[#7]", 0),          # acrylonitrile, β-carbon at index 0
                ("[CX3]=[CX3]-[SX4](=O)=O", 0),        # vinyl sulfone, β-carbon at index 0
                ("[CX3]=[CX3]-[#7+]", 0),              # α,β-unsaturated iminium, β-carbon at index 0
            ]
            is_michael = False
            for pattern, reactive_idx in michael_patterns:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                for match in matches:
                    if atom_idx == match[reactive_idx]:  # Only the β-carbon is reactive
                        is_michael = True
                        break
                if is_michael:
                    break
            
            if is_michael:
                acceptor.append("pi* orbital (Michael acceptor)")
            
            # Check bonds directly for electrophilic groups (more robust than neighbor hybridization)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                # C=N imine/oxime (exclude aromatic to avoid false positives on amides/pyridine)
                if (bond.GetBondType() == Chem.BondType.DOUBLE 
                    and other.GetSymbol() == "N"
                    and not other.GetIsAromatic()):
                    acceptor.append("pi* orbital (imine/Schiff base)")
                # C=O aldehyde/ketone/ester — carbonyl pi* is highly electrophilic
                # Exclude amides (C=O with N neighbor) - not reactive due to resonance
                if bond.GetBondType() == Chem.BondType.DOUBLE and other.GetSymbol() == "O":
                    is_amide = any(n.GetSymbol() == "N" for n in atom.GetNeighbors() if n.GetIdx() != other.GetIdx())
                    if not is_amide:
                        acceptor.append("pi* orbital (carbonyl)")
        
        # Nitrile: only label CARBON as electrophile, not nitrogen (critical fix)
        if atom.GetHybridization() == rdchem.HybridizationType.SP and symbol == "C":
            acceptor.append("pi* orbital (triple bond)")
    
    # Disulfide detection for S-S bonds
    if symbol == "S":
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == "S":
                acceptor.append("sigma* orbital (disulfide)")
                break

    if symbol in ["B", "Al"]:
        acceptor.append("empty p orbital")

    # Deduplicate acceptor orbitals (e.g., enones may trigger both Michael and carbonyl)
    acceptor = list(dict.fromkeys(acceptor))  # Preserves order while removing duplicates

    return {
        "atom": symbol,
        "hybridization": HYB_MAP.get(hyb_raw, "other"),
        "donor_orbitals": donor,
        "acceptor_orbitals": acceptor,
        "strained_ring": strained_ring,
        "is_anionic": is_anionic
    }

def covalent_orbital_score(nuc_info, elec_info):
    """
    Returns:
        covalent_possible (bool)
        orbital_score (float 0-1)
    """
    nuc_hyb = nuc_info.get("hybridization", "")
    elec_hyb = elec_info.get("hybridization", "")
    nuc_donor = bool(nuc_info.get("donor_orbitals"))
    elec_acceptor = bool(elec_info.get("acceptor_orbitals"))
    elec_strained = elec_info.get("strained_ring", False)
    nuc_anionic = nuc_info.get("is_anionic", False)

    possible = False
    score = 0.0

    # Debug: Check if donor/acceptor orbitals are detected
    print(f"DEBUG - Nuc donor orbitals: {nuc_info.get('donor_orbitals')} (has donor: {nuc_donor})")
    print(f"DEBUG - Elec acceptor orbitals: {elec_info.get('acceptor_orbitals')} (has acceptor: {elec_acceptor})")
    print(f"DEBUG - Nuc hyb: {nuc_hyb}, Elec hyb: {elec_hyb}")

    if nuc_donor and elec_acceptor:
        if nuc_hyb == "sp3" and elec_hyb == "sp3":
            possible = True
            # Boost score for strained rings (epoxides, aziridines)
            score = 0.9 if elec_strained else 0.3
        elif nuc_hyb == "sp3" and elec_hyb == "sp2":
            possible = True
            score = 1.0
        elif nuc_hyb == "sp3" and elec_hyb == "sp":
            possible = True
            score = 0.5  # nitrile carbon, less reactive than sp2
        elif nuc_hyb == "sp2" and elec_hyb == "sp2":
            possible = True
            score = 0.8
        else:
            possible = False
            score = 0.0
        
        # Apply bonus for anionic nucleophiles (more reactive)
        if nuc_anionic and possible:
            score = min(1.0, score * 1.15)  # 15% bonus, capped at 1.0

    return possible, score


# --- paper check for specific nucleophile/electrophile atoms ---
def paper_check_interaction(smiles_nuc, atom_idx_nuc, smiles_elec, atom_idx_elec):
    nuc = Chem.MolFromSmiles(smiles_nuc)
    elec = Chem.MolFromSmiles(smiles_elec)
    
    atom_nuc = nuc.GetAtomWithIdx(atom_idx_nuc)
    atom_elec = elec.GetAtomWithIdx(atom_idx_elec)

    info_nuc = classify_atom_orbitals(atom_nuc)
    info_elec = classify_atom_orbitals(atom_elec)

    # check donor/acceptor complementarity
    covalent_possible, orbital_score = covalent_orbital_score(info_nuc, info_elec)


    return {
        "nucleophile": info_nuc,
        "electrophile": info_elec,
        "covalent_bond_possible": covalent_possible,
        "orbital_score": orbital_score
    }


# --- Example usage ---
#ligand_smiles = "CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4" #Afatinib, 4
#ligand_smiles = "C[CH](NC(=O)[CH](Cc1ccc(O)cc1)NC(=O)OCc2ccccc2)C(=O)CF" #ZYA and Cruzain, 23
ligand_smiles = "CC(C)C(NC(=O)c1ccc2ccccc2c1)C(=O)N1CCCC1C(=O)NC(CC(=O)NS(C)(=O)=O)C=O" #1bmq and MNO, 24
ligand_reactive_index = 36
#ligand_smiles = "CCO"
#ligand_smiles = "N#Cc3nc1c(ncn1C2CCCC2)c(n3)Nc5ccccc5OCCCn4ccnc4" #IHI
#nuc_smiles = "c1cn[n-]c1"  # His surrogate, 4
#nuc_smiles = "CS" # Cys surrogate, 1
#nuc_smiles = "CO" # Ser surrogate, 1
#nuc_smiles = "CCO" #Thr surrogate, 2
#nuc_smiles = "c1ccc(O)cc1" # Tyr surrogate, 4
#nuc_smiles = "NCC" # Lys surrogate, 0
nuc_smiles_list = [
    "c1cn[n-]c1",  # His surrogate
    "CS",          # Cys surrogate
    "CO",          # Ser surrogate
    "CCO",         # Thr surrogate
    "c1ccc(O)cc1", # Tyr surrogate
    "NCC"          # Lys surrogate
]

# Atom indices for nucleophile (adjust if needed)
nuc_atom_index_dict = {
    "c1cn[n-]c1": 4,
    "CS": 1,
    "CO": 1,
    "CCO": 2,
    "c1ccc(O)cc1": 4,
    "NCC": 0
}

# Loop over nucleophiles
for nuc_smiles in nuc_smiles_list:
    nuc_idx = nuc_atom_index_dict[nuc_smiles]
    try:
        result = paper_check_interaction(nuc_smiles, nuc_idx, ligand_smiles, ligand_reactive_index)
        print(f"Nucleophile: {nuc_smiles}, Result: {result}")
    except Exception as e:
        print(f"Nucleophile: {nuc_smiles}, Error: {e}")
