#!/usr/bin/env python3
from rdkit import Chem
from rdkit.Chem import Descriptors

# Test BOTH: Your SMILES vs. Correct Penicillin G
your_smiles = "CC1(C)S[CH](N[CH]1C(O)=O)[CH](NC(=O)Cc2ccccc2)C=O"
correct_pen_g = "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"

print("="*70)
print("YOUR SMILES:")
print("="*70)
mol = Chem.MolFromSmiles(your_smiles)

if mol is None:
    print("❌ Invalid SMILES")
else:
    print("✅ Valid SMILES")
    from rdkit.Chem import AllChem
    print(f"   Molecular Formula: {AllChem.CalcMolFormula(mol)}")
    print(f"   Number of atoms: {mol.GetNumAtoms()}")
    
    # Test for β-lactam ring (4-membered ring with N-C(=O)-C-C)
    beta_lactam_pattern = Chem.MolFromSmarts("C1NC(=O)C1")
    matches = mol.GetSubstructMatches(beta_lactam_pattern)
    
    print(f"\n🔍 Beta-lactam ring search:")
    print(f"   Pattern: C1NC(=O)C1")
    print(f"   Found: {len(matches) > 0}")
    print(f"   Number of matches: {len(matches)}")
    if matches:
        print(f"   Atom indices: {matches}")
    
    # Compare with canonical Penicillin G structure
    # True Penicillin G: CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O
    print(f"\n📊 Comparison with TRUE Penicillin G:")
    print(f"   Your formula: {AllChem.CalcMolFormula(mol)}")
    print(f"   True Pen G formula: C16H18N2O4S")
    
    # Look for thiazolidine ring (5-membered with S and N)
    thiazolidine = Chem.MolFromSmarts("C1SCNC1")
    thia_matches = mol.GetSubstructMatches(thiazolidine)
    print(f"\n🔍 Thiazolidine ring (5-membered S-N ring):")
    print(f"   Found: {len(thia_matches) > 0}")


print("\n" + "="*70)
print("CORRECT PENICILLIN G STRUCTURE:")
print("="*70)
correct_mol = Chem.MolFromSmiles(correct_pen_g)

if correct_mol is None:
    print("❌ Invalid SMILES")
else:
    print("✅ Valid SMILES")
    print(f"   Molecular Formula: {AllChem.CalcMolFormula(correct_mol)}")
    print(f"   Number of atoms: {correct_mol.GetNumAtoms()}")
    
    # Test for β-lactam ring
    beta_lactam_pattern = Chem.MolFromSmarts("C1NC(=O)C1")
    matches = correct_mol.GetSubstructMatches(beta_lactam_pattern)
    
    print(f"\n🎯 Beta-lactam ring search:")
    print(f"   Pattern: C1NC(=O)C1")
    print(f"   Found: {'YES ✅' if len(matches) > 0 else 'NO ❌'}")
    print(f"   Number of matches: {len(matches)}")
    if matches:
        print(f"   Atom indices: {matches}")
        print(f"   ⚡ THIS IS THE ELECTROPHILIC WARHEAD!")
    
    # Look for thiazolidine ring
    thiazolidine = Chem.MolFromSmarts("C1SCNC1")
    thia_matches = correct_mol.GetSubstructMatches(thiazolidine)
    print(f"\n🔍 Thiazolidine ring:")
    print(f"   Found: {'YES ✅' if len(thia_matches) > 0 else 'NO ❌'}")

print("\n" + "="*70)
print("MECHANISM OF ACTION:")
print("="*70)
print("Penicillin G inhibits bacterial cell wall synthesis by:")
print("1. β-lactam carbonyl acts as ELECTROPHILE")
print("2. Attacks Ser residue in Penicillin-Binding Proteins (PBPs)")
print("3. Forms IRREVERSIBLE covalent acyl-enzyme adduct")
print("4. Ring strain (~25 kcal/mol) drives reaction forward")
print("5. Bacteria cannot complete cell wall → cell lysis")
print("="*70)
