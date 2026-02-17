#!/usr/bin/env python3
"""
Test script to check warhead detection for a specific SMILES string
"""

from rdkit import Chem

# SMILES to test
test_smiles = "CC(C)CCNC(=O)[C@@H](CCc1ccccc1)NC(=O)[C@@H](O)CC(O)=O"

# Warhead patterns from Frankenstein.py
ELECTROPHILE_WARHEADS = [
    # Michael acceptors
    ("Alpha-beta unsaturated carbonyl (Michael acceptor)", "[CX3](=O)[CX3]=[CX3]", 3, "soft"),
    ("Acrylamide warhead", "C=CC(=O)N", 0, "soft"),
    ("Vinyl sulfone", "C=CS(=O)(=O)", 0, "soft"),

    # Carbonyl-based electrophiles
    ("Aldehyde", "[CX3H1](=O)[#6]", 0, "hard"),
    ("Ketone (reactive)", "[CX3](=O)([#6])[#6]", 0, "soft"),
    ("Activated ketone", "C(=O)C(=O)", 0, "soft"),

    # Halogenated carbonyls
    ("Alpha-halo carbonyl", "[CX4][F,Cl,Br,I]C(=O)", 0, "soft"),
    ("Fluoromethyl ketone", "C(=O)CF", 1, "soft"),

    # Epoxides and aziridines
    ("Epoxide", "C1OC1", 0, "hard"),
    ("Aziridine", "C1NC1", 0, "hard"),

    # Nitriles
    ("Nitrile (electrophilic)", "[C]#N", 0, "soft"),

    # Leaving group displacement
    ("Alkyl halide (good LG)", "[CX4][Br,I]", 0, "soft"),
    ("Alkyl chloride", "[CX4]Cl", 0, "borderline"),

    # Sulfonyl-based
    ("Sulfonyl fluoride", "S(=O)(=O)F", 0, "borderline"),
    ("Sulfonamide (activated)", "[NX3][S](=O)(=O)", 1, "borderline"),

    # Disulfides
    ("Disulfide", "[SX2][SX2]", 0, "soft"),

    # Lactones/lactams (strained rings)
    ("Beta-lactone", "C1OC(=O)C1", 2, "hard"),
    ("Beta-lactam", "C1NC(=O)C1", 2, "hard"),

    # Isocyanates/isothiocyanates
    ("Isocyanate", "N=C=O", 1, "hard"),
    ("Isothiocyanate", "N=C=S", 1, "soft"),

    # Boronic acids (reversible covalent)
    ("Boronic acid", "[BX3](O)O", 0, "hard"),
]

print(f"Testing SMILES: {test_smiles}")
print("=" * 80)

mol = Chem.MolFromSmiles(test_smiles)
if mol is None:
    print("❌ Error: Invalid SMILES")
else:
    print(f"✅ Valid molecule: {Chem.MolToSmiles(mol)}")
    print(f"   Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"   Number of atoms: {mol.GetNumAtoms()}")
    print()
    
    # Test each warhead pattern
    detected = []
    for name, smarts, reactive_atom_index, hsab_class in ELECTROPHILE_WARHEADS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            print(f"⚠️  Warning: Invalid SMARTS pattern for {name}: {smarts}")
            continue
        
        matches = mol.GetSubstructMatches(patt)
        if matches:
            for match in matches:
                if reactive_atom_index < len(match):
                    reactive_idx = match[reactive_atom_index]
                    atom = mol.GetAtomWithIdx(reactive_idx)
                    detected.append((name, smarts, reactive_idx, hsab_class, match))
                    print(f"✅ MATCH: {name}")
                    print(f"   SMARTS: {smarts}")
                    print(f"   Match atoms: {match}")
                    print(f"   Reactive atom index: {reactive_idx} ({atom.GetSymbol()})")
                    print(f"   HSAB class: {hsab_class}")
                    print()
    
    print("=" * 80)
    if detected:
        print(f"✅ Detected {len(detected)} warhead(s)")
    else:
        print("⚠️  NO WARHEADS DETECTED")
        print()
        print("This molecule does not contain any of the predefined electrophilic warheads.")
        print()
        print("The molecule contains:")
        # Analyze what's in the molecule
        amides = mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)N"))
        carboxylic_acids = mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)O"))
        alcohols = mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]"))
        
        if amides:
            print(f"   - {len(amides)} amide group(s) - these are NOT electrophilic")
        if carboxylic_acids:
            print(f"   - {len(carboxylic_acids)} carboxylic acid group(s) - typically not reactive warheads")
        if alcohols:
            print(f"   - {len(alcohols)} alcohol group(s) - nucleophilic, not electrophilic")
        
        print()
        print("💡 This appears to be a peptide-like molecule without reactive electrophilic centers.")
        print("   Common electrophilic warheads include:")
        print("   - Michael acceptors (e.g., C=CC=O)")
        print("   - Epoxides")
        print("   - Alkyl halides")
        print("   - Aldehydes")
        print("   - Activated carbonyls")
        print()
        print("   To use this with Frankenstein.py, you would need to add appropriate")
        print("   SMARTS patterns for any novel electrophilic groups in your molecule.")
