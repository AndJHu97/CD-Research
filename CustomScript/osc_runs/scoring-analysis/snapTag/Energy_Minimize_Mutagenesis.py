"""Batch restrained energy minimization of mutagenesis PDBs with OpenMM.

Mutation sites are parsed from the file name (e.g. Combo10_C145A_E159C_Single.pdb
-> residues 145 and 159). Heavy atoms farther than --radius from every atom of
the mutated residues are harmonically restrained; everything near the
mutations is free to relax.

Files without mutation tokens in their name (e.g. 3L00_cys_modified.pdb) are
treated as references: they get the same cleanup and protocol, with
--reference-sites as the flexible region, and the cleaned, protonated,
pre-minimization structure is also written as <name>_clean.pdb.

Cleanup applied to every input:
  - HETATM records (waters, ions, ligands) removed
  - ANISOU / CONECT / remarks / Rosetta score tables dropped
  - Only the first alternate location (blank or 'A') kept
  - Existing hydrogens removed and rebuilt by OpenMM at --ph
  - Chain breaks split with TER so no bond is built across missing loops

Usage:
    python Energy_Minimize_Mutagenesis.py Mutagenesis_PDBs
    python Energy_Minimize_Mutagenesis.py Mutagenesis_PDBs -o Minimized --radius 8
"""

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np
from openmm import CustomExternalForce, VerletIntegrator, unit
from openmm.app import HBonds, Modeller, NoCutoff, PDBFile, Simulation, ForceField
from pdbfixer import PDBFixer

MUTATION_TOKEN = re.compile(r"([A-Z])(\d+)([A-Z])")

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

PEPTIDE_BOND_BREAK_A = 2.0


def parse_mutations(stem):
    """Return {resid: expected 3-letter residue name} from tokens like C145A."""
    sites = {}
    for token in re.split(r"[_\-]", stem):
        match = MUTATION_TOKEN.fullmatch(token)
        if match:
            _, resid, new_aa = match.groups()
            sites[resid] = ONE_TO_THREE[new_aa]
    return sites


def _atom_xyz(lines, atom_name):
    for line in lines:
        if line[12:16].strip() == atom_name:
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return None


def _is_chain_break(prev_lines, cur_lines):
    c_xyz = _atom_xyz(prev_lines, "C")
    n_xyz = _atom_xyz(cur_lines, "N")
    if c_xyz is None or n_xyz is None:
        return int(cur_lines[0][22:26]) - int(prev_lines[0][22:26]) > 1
    return np.linalg.norm(c_xyz - n_xyz) > PEPTIDE_BOND_BREAK_A


def clean_pdb(path):
    """Strip a PDB down to protein heavy atoms, with TER at every chain break."""
    residues = []
    removed_het = set()
    n_altloc = 0
    n_hydrogen = 0

    for line in path.read_text().splitlines():
        record = line[:6]
        if record == "HETATM":
            removed_het.add((line[17:20].strip(), line[21], line[22:27].strip()))
            continue
        if record != "ATOM  ":
            continue

        if line[16] not in (" ", "A"):
            n_altloc += 1
            continue

        element = line[76:78].strip() if len(line) >= 78 else ""
        name = line[12:16].strip()
        if element == "H" or (not element and name.lstrip("0123456789").startswith("H")):
            n_hydrogen += 1
            continue

        line = line[:16] + " " + line[17:]
        key = (line[21], line[22:27])
        if not residues or residues[-1][0] != key:
            residues.append((key, []))
        residues[-1][1].append(line)

    out = []
    breaks = []
    prev = None
    for (chain, resid), lines in residues:
        if prev is not None:
            prev_chain, prev_resid, prev_lines = prev
            if prev_chain != chain:
                out.append("TER")
            elif _is_chain_break(prev_lines, lines):
                out.append("TER")
                breaks.append(f"{prev_resid.strip()}/{resid.strip()}")
        out.extend(lines)
        prev = (chain, resid, lines)
    out += ["TER", "END"]

    het_counts = {}
    for resname, _, _ in removed_het:
        het_counts[resname] = het_counts.get(resname, 0) + 1

    info = {
        "removed_het": het_counts,
        "removed_altloc_atoms": n_altloc,
        "removed_hydrogens": n_hydrogen,
        "chain_breaks": breaks,
    }
    return "\n".join(out) + "\n", info


def prepare_structure(path, forcefield, ph):
    text, info = clean_pdb(path)

    fixer = PDBFixer(pdbfile=io.StringIO(text))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        names = sorted({res.name for res, _ in fixer.nonstandardResidues})
        raise ValueError(f"Nonstandard residues present: {names}")
    fixer.findMissingAtoms()
    info["missing_heavy_atoms_added"] = sum(len(v) for v in fixer.missingAtoms.values())
    fixer.addMissingAtoms()

    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=ph)
    return modeller, info


def site_atom_indices(topology, chain_id, sites):
    indices = []
    found = set()
    for residue in topology.residues():
        if residue.chain.id != chain_id or residue.id not in sites:
            continue
        expected = sites[residue.id]
        if expected is not None and residue.name != expected:
            raise ValueError(
                f"Residue {chain_id}{residue.id} is {residue.name}, "
                f"but the file name says it should be {expected}"
            )
        found.add(residue.id)
        indices.extend(atom.index for atom in residue.atoms())

    missing = set(sites) - found
    if missing:
        raise ValueError(f"Residues not found in chain {chain_id}: {sorted(missing, key=int)}")
    return np.array(indices)


def heavy_rmsd(a, b, mask):
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(np.sum((a[mask] - b[mask]) ** 2, axis=1))))


def minimize_file(path, sites, is_reference, args, forcefield, out_dir):
    modeller, info = prepare_structure(path, forcefield, args.ph)
    topology = modeller.topology

    if is_reference:
        clean_path = out_dir / f"{path.stem}_clean.pdb"
        with open(clean_path, "w") as f:
            PDBFile.writeFile(topology, modeller.positions, f, keepIds=True)

    xyz = np.array(modeller.positions.value_in_unit(unit.nanometer))
    heavy = np.array([
        atom.element is not None and atom.element.symbol != "H"
        for atom in topology.atoms()
    ])

    site_idx = site_atom_indices(topology, args.chain, sites)
    dist_to_sites = np.min(
        np.linalg.norm(xyz[:, None, :] - xyz[None, site_idx, :], axis=2), axis=1
    )
    restrained = heavy & (dist_to_sites > args.radius / 10.0)
    free_heavy = heavy & ~restrained

    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=HBonds)

    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", args.k)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    for idx in np.flatnonzero(restrained):
        restraint.addParticle(int(idx), xyz[idx].tolist())
    restraint.setForceGroup(1)
    system.addForce(restraint)

    simulation = Simulation(topology, system, VerletIntegrator(0.001 * unit.picoseconds))
    simulation.context.setPositions(modeller.positions)

    def potential(groups):
        state = simulation.context.getState(getEnergy=True, groups=groups)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    e_initial = potential({0})
    simulation.minimizeEnergy(
        tolerance=args.tolerance * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=args.max_iterations,
    )
    e_final = potential({0})
    e_restraint = potential({1})

    state = simulation.context.getState(getPositions=True)
    final_xyz = np.array(state.getPositions().value_in_unit(unit.nanometer))

    out_path = out_dir / f"{path.stem}_minimized.pdb"
    with open(out_path, "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f, keepIds=True)

    return {
        "n_atoms": topology.getNumAtoms(),
        "n_restrained_heavy": int(restrained.sum()),
        "n_free_heavy": int(free_heavy.sum()),
        "E_initial_kJmol": round(e_initial, 2),
        "E_final_kJmol": round(e_final, 2),
        "dE_kJmol": round(e_final - e_initial, 2),
        "E_restraint_kJmol": round(e_restraint, 2),
        "rmsd_heavy_A": round(10 * heavy_rmsd(xyz, final_xyz, heavy), 3),
        "rmsd_free_heavy_A": round(10 * heavy_rmsd(xyz, final_xyz, free_heavy), 3),
        "chain_breaks": ";".join(info["chain_breaks"]),
        "removed_het": ";".join(f"{k}x{v}" for k, v in sorted(info["removed_het"].items())),
        "removed_altloc_atoms": info["removed_altloc_atoms"],
        "missing_heavy_atoms_added": info["missing_heavy_atoms_added"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch restrained OpenMM minimization of mutagenesis PDBs."
    )
    parser.add_argument("input_dir", type=Path, help="Folder of PDBs to minimize")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output folder (default: <input_dir>_minimized next to the input folder)",
    )
    parser.add_argument("--chain", default="A", help="Chain containing the mutations")
    parser.add_argument(
        "--reference-sites", default="145",
        help="Comma-separated flexible residues for files without mutations in their name",
    )
    parser.add_argument("--radius", type=float, default=8.0,
                        help="Free-region radius around mutated residues (Angstrom)")
    parser.add_argument("--k", type=float, default=1000.0,
                        help="Restraint force constant (kJ/mol/nm^2)")
    parser.add_argument("--tolerance", type=float, default=10.0,
                        help="Minimizer force tolerance (kJ/mol/nm)")
    parser.add_argument("--max-iterations", type=int, default=5000,
                        help="Minimizer iteration cap (0 = until converged)")
    parser.add_argument("--ph", type=float, default=7.0, help="pH for protonation")
    parser.add_argument("--vacuum", action="store_true",
                        help="Minimize in vacuum instead of OBC2 implicit solvent")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run files that already have a minimized output")
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        sys.exit(f"Input folder not found: {input_dir}")

    out_dir = (args.output_dir or input_dir.parent / f"{input_dir.name}_minimized").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(input_dir.glob("*.pdb"))
    if not pdbs:
        sys.exit(f"No .pdb files in {input_dir}")

    ff_files = ["amber14-all.xml"] if args.vacuum else ["amber14-all.xml", "implicit/obc2.xml"]
    forcefield = ForceField(*ff_files)
    reference_sites = {s.strip(): None for s in args.reference_sites.split(",") if s.strip()}

    print(f"Input:  {input_dir}")
    print(f"Output: {out_dir}")
    print(f"Force field: {', '.join(ff_files)} | radius {args.radius} A | k {args.k} kJ/mol/nm^2\n")

    rows = []
    n_failed = 0
    for path in pdbs:
        sites = parse_mutations(path.stem)
        is_reference = not sites
        if is_reference:
            sites = reference_sites

        label = "reference" if is_reference else "mutant"
        site_str = ",".join(sorted(sites, key=int))
        out_path = out_dir / f"{path.stem}_minimized.pdb"

        if out_path.exists() and not args.overwrite:
            print(f"[skip] {path.name} (output exists; use --overwrite)")
            continue

        print(f"[{label}] {path.name}  sites {site_str}")
        row = {"file": path.name, "type": label, "sites": site_str}
        try:
            row.update(minimize_file(path, sites, is_reference, args, forcefield, out_dir))
            row["status"] = "ok"
            print(
                f"    E {row['E_initial_kJmol']:.1f} -> {row['E_final_kJmol']:.1f} kJ/mol | "
                f"free heavy atoms {row['n_free_heavy']} | "
                f"heavy RMSD {row['rmsd_heavy_A']:.3f} A (free region {row['rmsd_free_heavy_A']:.3f} A)"
            )
            if row["chain_breaks"]:
                print(f"    chain breaks split at {row['chain_breaks']}")
            if row["removed_het"]:
                print(f"    removed HETATM residues: {row['removed_het']}")
        except Exception as exc:
            n_failed += 1
            row["status"] = f"failed: {exc}"
            print(f"    FAILED: {exc}")
        rows.append(row)

    if rows:
        fieldnames = []
        for row in rows:
            fieldnames += [k for k in row if k not in fieldnames]
        summary_path = out_dir / "minimization_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary: {summary_path}")

    print(f"Done: {len(rows) - n_failed} succeeded, {n_failed} failed.")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
