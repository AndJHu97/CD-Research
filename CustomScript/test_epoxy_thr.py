#!/usr/bin/env python3
"""
Test α,β-epoxyketone (epoxide carbon) vs deprotonated Thr
"""

import sys
import os

# Add CustomScript to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "CustomScript"))

import covalent_orbital_requirements as cor

# α,β-epoxyketone 
epoxyketone_smiles = "CC(=O)C1OC1"  # simple α,β-epoxyketone
epoxide_carbon_idx = 3  # The C1 carbon in the epoxide ring (index 3 in SMILES)

# Deprotonated Thr
thr_deprot_smiles = "CC[O-]"
thr_oxygen_idx = 2

print("=" * 70)
print("Testing: α,β-epoxyketone (epoxide) + deprotonated Thr-O⁻")
print("=" * 70)

result = cor.paper_check_interaction(
    thr_deprot_smiles, 
    thr_oxygen_idx,
    epoxyketone_smiles, 
    epoxide_carbon_idx
)

print(f"\nRESULT:")
print(f"  Covalent bond possible: {result['covalent_bond_possible']}")
print(f"  Orbital score: {result['orbital_score']}")
print(f"\nNucleophile (Thr-O⁻): {result['nucleophile']}")
print(f"\nElectrophile (epoxide): {result['electrophile']}")
