from rdkit import Chem
from rdkit.Chem import rdchem

# --- classify atomic orbitals for a broad permissive check ---
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
    is_anionic = (charge < 0)

    # donor orbitals: nucleophile
    if symbol in ["O", "N", "S"]:
        if charge > 0:
            pass
        elif is_anionic:
            donor.append("anionic lone pair")
        else:
            donor.append("lone pair")
    if symbol == "C" and charge < 0:
        donor.append("carbanion lone pair")
    if atom.GetIsAromatic():
        donor.append("pi electrons")

    # acceptor orbitals: electrophile
    if hyb_raw in [rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP]:
        acceptor.append("pi* orbital")
    if hyb_raw == rdchem.HybridizationType.SP3:
        acceptor.append("sigma* orbital")
    if symbol == "P":
        acceptor.append("phosphorus-centered acceptor")
    
    if symbol in ["B", "Al"]:
        acceptor.append("empty p orbital")

    # Deduplicate while keeping the logic broad and permissive.
    acceptor = list(dict.fromkeys(acceptor))  # Preserves order while removing duplicates

    return {
        "atom": symbol,
        "hybridization": HYB_MAP.get(hyb_raw, "other"),
        "donor_orbitals": donor,
        "acceptor_orbitals": acceptor,
        "strained_ring": False,
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
    nuc_anionic = nuc_info.get("is_anionic", False)

    possible = False
    score = 0.0

    # Debug: Check if donor/acceptor orbitals are detected
    print(f"DEBUG - Nuc donor orbitals: {nuc_info.get('donor_orbitals')} (has donor: {nuc_donor})")
    print(f"DEBUG - Elec acceptor orbitals: {elec_info.get('acceptor_orbitals')} (has acceptor: {elec_acceptor})")
    print(f"DEBUG - Nuc hyb: {nuc_hyb}, Elec hyb: {elec_hyb}")

    if nuc_donor and elec_acceptor:
        possible = True
        if nuc_hyb == "sp3" and elec_hyb in ["sp3", "sp2", "sp"]:
            score = 0.8
        elif nuc_hyb == "sp2" and elec_hyb in ["sp2", "sp"]:
            score = 0.7
        elif nuc_hyb == "sp" and elec_hyb in ["sp", "sp2"]:
            score = 0.6
        else:
            score = 0.5
        
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
