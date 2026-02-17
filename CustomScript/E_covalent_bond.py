from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile, shutil, os, subprocess, re

# ---------- helper functions ----------
def embed_opt_mol(mol):
    """Add Hs, embed 3D coordinates, and optimize geometry."""
    print("Before AddHs:", mol.GetNumAtoms())
    mol = Chem.AddHs(mol)
    print("After AddHs:", mol.GetNumAtoms())
    if mol.GetNumAtoms() < 4:  # skip embedding if too small
        print("Skipping embedding for small fragment")
        return mol
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        print("Embedding failed, skipping")
        return mol
    if AllChem.MMFFOptimizeMolecule(mol) != 0:
        AllChem.UFFOptimizeMolecule(mol)
    print("After embedding & optimization:", mol.GetNumAtoms())
    return mol


def extract_fragment_around_atom(mol, atom_idx, radius=1):
    """Extract fragment around a reactive atom, returns new Mol."""
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    amap = {}
    frag = Chem.PathToSubmol(mol, env, atomMap=amap)
    if frag.GetNumAtoms() == 0:
        raise ValueError("Fragment has 0 atoms. Check reactive atom index or radius")
    
    # Fix RingInfo / sanitization
    frag.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(frag)
    Chem.GetSymmSSSR(frag)  # ensures RingInfo is computed
    
    return frag, amap


def attach_nucleophile_fragment(e_frag, n_frag, reactive_atom_idx_in_frag):
    """
    Combine electrophile fragment and nucleophile fragment safely.
    - reactive_atom_idx_in_frag: index of the atom in e_frag to attach nucleophile to
    Returns an RWMol ready for geometry embedding.
    """
    # Embed & optimize fragments individually
    e_frag = embed_opt_mol(e_frag)
    n_frag = embed_opt_mol(n_frag)

    combined = Chem.CombineMols(e_frag, n_frag)
    combined = Chem.RWMol(combined)

    offset = e_frag.GetNumAtoms()
    n_idx = offset  # first atom of nucleophile

    # Remove a hydrogen from reactive atom to avoid valence error
    e_atom = combined.GetAtomWithIdx(reactive_atom_idx_in_frag)
    h_to_remove = None
    for neighbor in e_atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:  # hydrogen
            h_to_remove = neighbor.GetIdx()
            break
    if h_to_remove is not None:
        combined.RemoveAtom(h_to_remove)

    # Add single bond to nucleophile
    combined.AddBond(reactive_atom_idx_in_frag, n_idx, Chem.BondType.SINGLE)

    # Update & sanitize
    combined.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(combined)

    # Embed & optimize final adduct
    combined = embed_opt_mol(combined)
    return combined

def write_xyz(mol, path):
    """Write XYZ coordinates for xTB."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers. Embed first.")
    conf = mol.GetConformer()
    with open(path, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for i, a in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{a.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

def run_xtb_xyz(xyz_path, charge=0, gbsa="water"):
    """Run xTB single-point energy and return total energy in Hartree."""
    workdir = tempfile.mkdtemp(prefix="xtb_")
    shutil.copy(xyz_path, os.path.join(workdir, os.path.basename(xyz_path)))
    cmd = ["xtb", os.path.basename(xyz_path), "--gfn", "2", "--charge", str(charge)]
    if gbsa:
        cmd += ["--gbsa", gbsa]
    proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    energy = None
    for line in out.splitlines():
        if "TOTAL ENERGY" in line.upper():
            m = re.search(r"([-+]?\d*\.\d+|\d+)", line)
            if m:
                energy = float(m.group(1))
                break

    shutil.rmtree(workdir)
    if energy is None:
        raise RuntimeError("xTB failed to produce TOTAL ENERGY")
    return energy

# ---------- main workflow ----------
def compute_bond_deltaE(electrophile_smiles, reactive_atom_idx, nucleophile_smiles, radius=1):
    # Build molecules
    e_mol = Chem.MolFromSmiles(electrophile_smiles)
    n_mol = Chem.MolFromSmiles(nucleophile_smiles)
    if e_mol is None or n_mol is None:
        raise ValueError("Invalid SMILES input")
    e_mol = Chem.AddHs(e_mol)
    n_mol = Chem.AddHs(n_mol)

    # Extract electrophile fragment
    e_frag, _ = extract_fragment_around_atom(e_mol, reactive_atom_idx, radius=radius)
    if e_frag.GetNumAtoms() == 0:
        raise ValueError(f"Fragment is empty. Check reactive_atom_idx={reactive_atom_idx}")
    # Temporary directory
    tmpdir = tempfile.mkdtemp()
    try:
        # Write fragments
        e_xyz = os.path.join(tmpdir, "efrag.xyz")
        n_xyz = os.path.join(tmpdir, "nfrag.xyz")
        e_frag = embed_opt_mol(e_frag)
        print("before n_mol")
        n_mol = embed_opt_mol(n_mol)
        write_xyz(e_frag, e_xyz)
        write_xyz(n_mol, n_xyz)
        
        # xTB energies
        e_energy = run_xtb_xyz(e_xyz, charge=0)
        n_energy = run_xtb_xyz(n_xyz, charge=0)

        # Build adduct fragment
        e_frag, amap = extract_fragment_around_atom(e_mol, REACTIVE_IDX, radius=3)
        reactive_atom_idx_in_frag = amap[REACTIVE_IDX]

        # Then pass it
        adduct_frag = attach_nucleophile_fragment(e_frag, n_mol, reactive_atom_idx_in_frag)
        ad_xyz = os.path.join(tmpdir, "adduct.xyz")
        write_xyz(adduct_frag, ad_xyz)
        ad_energy = run_xtb_xyz(ad_xyz, charge=0)

        deltaE = ad_energy - (e_energy + n_energy)
    finally:
        shutil.rmtree(tmpdir)

    return deltaE

# ---------- example usage ----------
if __name__ == "__main__":
    ELECTROPHILE_SMILES = "CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4"
    NUCLEOPHILE_SMILES = "CS"  # Cys surrogate
    REACTIVE_IDX = 5

    deltaE = compute_bond_deltaE(ELECTROPHILE_SMILES, REACTIVE_IDX, NUCLEOPHILE_SMILES, radius=3)
    print(f"Intrinsic bond ΔE (Hartree): {deltaE}")
    print(f"Intrinsic bond ΔE (kcal/mol): {deltaE * 627.509:.2f}")
