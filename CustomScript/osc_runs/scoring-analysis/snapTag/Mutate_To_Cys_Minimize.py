"""Mutate specified residues to cysteine, restrained-minimize, and save new PDBs.

Reads a CSV with columns: pdb, chain, resnum   (header names are case-insensitive)
One row = one single-site mutation -> one output PDB.

    pdb,chain,resnum
    3L00_cys_modified.pdb,A,145
    3L00_cys_modified,A,41
    Combo10_C145A_E159C_Single.pdb,A,166

A background mutation (default C145A, see --background) is applied to every structure
together with the new Cys, so outputs match the Combo C145A_xxxC structures. If the
background residue is already the target (e.g. 145 is already Ala), it is left alone.

Output files: <outdir>/<pdbstem>_<background>_<chain><WT><resnum>C_minimized.pdb
A summary CSV (energies, RMSDs, status) is written to the output folder.

Requires Energy_Minimize_Mutagenesis.py in the same folder (reuses its cleanup helpers).

Usage:
    python Mutate_To_Cys_Minimize.py mutations.csv Mutagenesis_PDBs -o Cys_Minimized
    python Mutate_To_Cys_Minimize.py mutations.csv Mutagenesis_PDBs --radius 10 --vacuum
    python Mutate_To_Cys_Minimize.py mutations.csv Mutagenesis_PDBs --background none
"""

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np
from openmm import CustomExternalForce, VerletIntegrator, unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer

from Energy_Minimize_Mutagenesis import clean_pdb, heavy_rmsd

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_background(text):
    """'C145A,H41A' -> [(expected_3letter, resnum, new_3letter), ...]; 'none' -> []."""
    if not text or text.strip().lower() == "none":
        return []
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", tok)
        if not m:
            raise ValueError(f"Bad background mutation '{tok}' (expected like C145A)")
        old, num, new = m.groups()
        out.append((ONE_TO_THREE[old], num, ONE_TO_THREE[new], tok))
    return out


def find_pdb(input_dir, name):
    name = name.strip()
    for candidate in (input_dir / name, input_dir / f"{name}.pdb"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} not found in {input_dir}")


def mutate_and_minimize(path, chain, resnum, args, forcefield, out_dir):
    text, info = clean_pdb(path)

    fixer = PDBFixer(pdbfile=io.StringIO(text))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        names = sorted({res.name for res, _ in fixer.nonstandardResidues})
        raise ValueError(f"Nonstandard residues present: {names}")

    # Find the wild-type residue
    wt = None
    for residue in fixer.topology.residues():
        if residue.chain.id == chain and residue.id == resnum:
            wt = residue.name
            break
    if wt is None:
        raise ValueError(f"Residue {chain}{resnum} not found")
    if wt == "CYS":
        raise ValueError(f"{chain}{resnum} is already CYS")

    # Background mutations (e.g. C145A) applied together with the new Cys
    mutations = []
    bg_applied = []
    bg_resnums = []
    for old3, bg_num, new3, tok in parse_background(args.background):
        if bg_num == resnum:
            raise ValueError(f"{chain}{resnum} is also a background site ({tok})")
        cur = None
        for residue in fixer.topology.residues():
            if residue.chain.id == chain and residue.id == bg_num:
                cur = residue.name
                break
        if cur is None:
            raise ValueError(f"Background residue {chain}{bg_num} not found")
        bg_resnums.append(bg_num)
        if cur == new3:
            continue  # already the target residue
        if cur != old3:
            raise ValueError(f"Background {tok}: {chain}{bg_num} is {cur}, expected {old3}")
        mutations.append(f"{cur}-{bg_num}-{new3}")
        bg_applied.append(tok)

    tag = f"{chain}{THREE_TO_ONE.get(wt, wt)}{resnum}C"
    bg_tag = "_".join(bg_applied)
    out_stem = f"{path.stem}_{bg_tag}_{tag}" if bg_tag else f"{path.stem}_{tag}"
    out_path = out_dir / f"{out_stem}_minimized.pdb"
    if out_path.exists() and not args.overwrite:
        return None, out_path

    # Mutate: PDBFixer strips side-chain atoms and rebuilds them as CYS
    mutations.append(f"{wt}-{resnum}-CYS")
    fixer.applyMutations(mutations, chain)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=args.ph)
    topology = modeller.topology

    xyz = np.array(modeller.positions.value_in_unit(unit.nanometer))
    heavy = np.array([
        a.element is not None and a.element.symbol != "H" for a in topology.atoms()
    ])

    site_idx = np.array([
        a.index
        for r in topology.residues()
        if r.chain.id == chain and (r.id == resnum or r.id in bg_resnums)
        for a in r.atoms()
    ])
    dist = np.min(
        np.linalg.norm(xyz[:, None, :] - xyz[None, site_idx, :], axis=2), axis=1
    )
    restrained = heavy & (dist > args.radius / 10.0)
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

    sim = Simulation(topology, system, VerletIntegrator(0.001 * unit.picoseconds))
    sim.context.setPositions(modeller.positions)

    def potential(groups):
        s = sim.context.getState(getEnergy=True, groups=groups)
        return s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    e_initial = potential({0})
    sim.minimizeEnergy(
        tolerance=args.tolerance * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=args.max_iterations,
    )
    e_final = potential({0})

    state = sim.context.getState(getPositions=True)
    final_xyz = np.array(state.getPositions().value_in_unit(unit.nanometer))
    with open(out_path, "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f, keepIds=True)

    return {
        "mutation": f"{wt}{resnum}CYS",
        "background_applied": ";".join(bg_applied),
        "output": out_path.name,
        "n_free_heavy": int(free_heavy.sum()),
        "n_restrained_heavy": int(restrained.sum()),
        "E_initial_kJmol": round(e_initial, 2),
        "E_final_kJmol": round(e_final, 2),
        "rmsd_heavy_A": round(10 * heavy_rmsd(xyz, final_xyz, heavy), 3),
        "rmsd_free_heavy_A": round(10 * heavy_rmsd(xyz, final_xyz, free_heavy), 3),
        "chain_breaks": ";".join(info["chain_breaks"]),
    }, out_path


def parse_args():
    p = argparse.ArgumentParser(description="Mutate residues to Cys and minimize.")
    p.add_argument("csv", type=Path, help="CSV with columns: pdb, chain, resnum")
    p.add_argument("input_dir", type=Path, help="Folder containing the PDBs")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("Cys_Minimized"))
    p.add_argument("--background", default="C145A",
                   help="Extra mutation(s) applied to every structure, comma-separated "
                        "(e.g. C145A or C145A,H41A); 'none' to disable. Default: C145A")
    p.add_argument("--radius", type=float, default=8.0,
                   help="Free-region radius around the mutated residue (Angstrom)")
    p.add_argument("--k", type=float, default=1000.0, help="Restraint k (kJ/mol/nm^2)")
    p.add_argument("--tolerance", type=float, default=10.0)
    p.add_argument("--max-iterations", type=int, default=5000)
    p.add_argument("--ph", type=float, default=7.0)
    p.add_argument("--vacuum", action="store_true", help="No implicit solvent")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ff_files = ["amber14-all.xml"] if args.vacuum else ["amber14-all.xml", "implicit/obc2.xml"]
    forcefield = ForceField(*ff_files)

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        jobs = list(reader)

    rows, n_failed = [], 0
    for job in jobs:
        pdb, chain, resnum = job["pdb"].strip(), job["chain"].strip(), str(job["resnum"]).strip()
        print(f"[{pdb}] {chain}{resnum} -> CYS")
        row = {"pdb": pdb, "chain": chain, "resnum": resnum}
        try:
            result, out_path = mutate_and_minimize(
                find_pdb(input_dir, pdb), chain, resnum, args, forcefield, out_dir
            )
            if result is None:
                print(f"    [skip] {out_path.name} exists (use --overwrite)")
                continue
            row.update(result)
            row["status"] = "ok"
            print(f"    E {row['E_initial_kJmol']:.1f} -> {row['E_final_kJmol']:.1f} kJ/mol | "
                  f"free region RMSD {row['rmsd_free_heavy_A']:.3f} A -> {out_path.name}")
        except Exception as exc:
            n_failed += 1
            row["status"] = f"failed: {exc}"
            print(f"    FAILED: {exc}")
        rows.append(row)

    if rows:
        fieldnames = []
        for r in rows:
            fieldnames += [k for k in r if k not in fieldnames]
        summary = out_dir / "cys_mutation_summary.csv"
        with open(summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary: {summary}")
    print(f"Done: {len(rows) - n_failed} succeeded, {n_failed} failed.")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()