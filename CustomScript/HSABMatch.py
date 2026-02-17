from rdkit import Chem

# === Step 1: Define electrophilic warhead SMARTS and HSAB classification ===
WARHEAD_DB = {
    "Michael_acceptor": {
        "smarts": "[C]=[C]-[C](=O)[N,O,S]",
        "hsab": "soft",
        "examples": ["acrylamide", "maleimide", "vinyl sulfone"]
    },
        "Aldehyde": {
        "smarts": "[CX3H1](=O)[#6,#1]"
,  # any sp2 carbonyl with H or C attached
        "hsab": "hard",
        "examples": ["formaldehyde", "pyridyl aldehyde"]
    },

    "Ketone": {
        "smarts": "[CX3](=O)[#6][#6]",  # generic ketone
        "hsab": "soft",
        "examples": ["FMK", "acetyl"]
    },
    "Fluoromethyl_ketone": {
        "smarts": "C(=O)C(F)",
        "hsab": "soft",
        "examples": ["ZYA", "FMK inhibitors"]
    },
    "Epoxide": {
        "smarts": "C1OC1",
        "hsab": "hard",
        "examples": ["glycidyl ether", "oxirane"]
    },
    "Sulfonyl_fluoride": {
        "smarts": "S(=O)(=O)F",
        "hsab": "borderline",
        "examples": ["fluorosulfonyl group"]
    },
    "Acrylonitrile": {
        "smarts": "C=CC#N",
        "hsab": "soft",
        "examples": ["cyano-vinyl"]
    },
    "Isothiocyanate": {
        "smarts": "N=C=S",
        "hsab": "soft",
        "examples": ["phenyl isothiocyanate"]
    },
    "β-Lactam": {
        "smarts": "C1C(=O)NC1",
        "hsab": "hard",
        "examples": ["penicillin-type"]
    },
}

# === Step 2: Define AA preferences based on HSAB ===
HSAB_TO_AA = {
    "soft": ["Cys", "Sec"],              # thiols, selenols
    "borderline": ["His", "Tyr"],        # imidazole, phenol
    "hard": ["Lys", "Ser", "Thr", "Asp", "Glu"]  # amine, alcohol, carboxylate
}

def identify_warhead(smiles):
    mol = Chem.MolFromSmiles(smiles)
    matched_classes = []
    for warhead, info in WARHEAD_DB.items():
        patt = Chem.MolFromSmarts(info["smarts"])
        if mol.HasSubstructMatch(patt):
            matched_classes.append((warhead, info["hsab"]))
    return matched_classes

def suggest_targets(warhead_matches):
    targets = set()
    for _, hsab in warhead_matches:
        targets.update(HSAB_TO_AA.get(hsab, []))
    return list(targets)

# === Example usage ===
ligands = {
    "afatinib": "CN(C)CC=C(C#N)C(=O)Nc1cc(F)cc(Cl)c1",
    "Acetaldehyde": "CC=O",
    "CLIK": "CN(C)C(=O)[C@H](Cc1ccccc1)NC(=O)C(=O)CC(=O)NCCc2ccccn2"
}

for name, smi in ligands.items():
    matches = identify_warhead(smi)
    if matches:
        print(f"\n{name} →")
        for w, h in matches:
            print(f"  Warhead: {w} | HSAB: {h}")
        print(f"  Suggested AA targets: {', '.join(suggest_targets(matches))}")
    else:
        print(f"\n{name}: No known warhead found")
