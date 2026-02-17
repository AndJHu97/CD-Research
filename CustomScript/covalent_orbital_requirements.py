from rdkit import Chem
from rdkit.Chem import rdchem

# --- classify atomic orbitals for simple paper check ---
def classify_atom_orbitals(atom):
    symbol = atom.GetSymbol()
    hyb = atom.GetHybridization()
    charge = atom.GetFormalCharge()
    
    donor = []
    acceptor = []

    # donor orbitals: nucleophile
    if symbol in ["O", "N", "S"]:
        donor.append("lone pair")
    if symbol == "C" and charge < 0:
        donor.append("carbanion lone pair")
    if atom.GetIsAromatic():
        donor.append("pi electrons")

    # acceptor orbitals: electrophile
    if symbol == "C":
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
        if "O" in neighbors or "Cl" in neighbors or "Br" in neighbors or "I" in neighbors:
            acceptor.append("sigma* or pi* orbital")
        if atom.GetHybridization() == rdchem.HybridizationType.SP2:
            for n in atom.GetNeighbors():
                if n.GetHybridization() == rdchem.HybridizationType.SP2 and n.GetSymbol() == "C":
                    acceptor.append("pi* orbital (Michael acceptor)")
        # add this for sp C (nitrile, alkyne)
        if atom.GetHybridization() == rdchem.HybridizationType.SP:
            acceptor.append("pi* orbital (triple bond)")


    if symbol in ["B", "Al"]:
        acceptor.append("empty p orbital")

    return {
        "atom": symbol,
        "hybridization": str(hyb).replace("SP","sp"),
        "donor_orbitals": donor,
        "acceptor_orbitals": acceptor
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

    possible = False
    score = 0.0

    if nuc_donor and elec_acceptor:
        if nuc_hyb == "sp3" and elec_hyb == "sp3":
            possible = True
            score = 0.3
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
ligand_reactive_index = 24
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
