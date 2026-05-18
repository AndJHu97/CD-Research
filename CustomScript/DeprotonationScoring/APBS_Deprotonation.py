"""
apbs_site_analysis.py

Calculates APBS electrostatics around a user-specified reactive site residue.
Identifies the nucleophilic atom via SMARTS surrogate matching, extracts an
8 A sphere of atoms for context, assigns protonation states with pdb2pqr/PROPKA,
and reports the electrostatic potential at the reactive atom.

Dependencies (all should be in covalent_env):
    pdb2pqr     (pip install pdb2pqr  OR  conda install -c conda-forge pdb2pqr)
    apbs        (conda install -c conda-forge apbs)
    numpy
    biopython   (pip install biopython)

Usage:
    python apbs_site_analysis.py 4g5j.pdb A:CYS:797
    python apbs_site_analysis.py 4g5j.pdb A:SER:768 --radius 10.0 --ph 7.4
    python apbs_site_analysis.py 4g5j.pdb A:TYR:1016 --ph 6.0
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import numpy as np
import re

# ---------------------------------------------------------------------------
# Target atom lookup (reactive where applicable, otherwise a central atom).
# For non-reactive residues we use CA as a neutral center point.
# ---------------------------------------------------------------------------

TARGET_ATOM = {
    "ALA": "CA",  # Non-reactive: use CA as center
    "ARG": "CA",  # Non-reactive: use CA as center
    "ASN": "CA",  # Non-reactive: use CA as center
    "ASP": "CA",  # Non-reactive: use CA as center
    "CYS": "SG",  # Reactive sulfur
    "GLN": "CA",  # Non-reactive: use CA as center
    "GLU": "CA",  # Non-reactive: use CA as center
    "GLY": "CA",  # Non-reactive: use CA as center
    "HIS": "ND1", # Reactive N (pyridine-type)
    "ILE": "CA",  # Non-reactive: use CA as center
    "LEU": "CA",  # Non-reactive: use CA as center
    "LYS": "NZ",  # Reactive N (protonated)
    "MET": "CA",  # Non-reactive: use CA as center
    "PHE": "CA",  # Non-reactive: use CA as center
    "PRO": "CA",  # Non-reactive: use CA as center
    "SER": "OG",  # Reactive O
    "THR": "OG1", # Reactive O
    "TRP": "CA",  # Non-reactive: use CA as center
    "TYR": "OH",  # Reactive O
    "VAL": "CA",  # Non-reactive: use CA as center
}

# PDB atom names for standard residues — used to validate atom presence
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


# ---------------------------------------------------------------------------
# PDB parsing (pure Python, no biopython required for basic ops)
# ---------------------------------------------------------------------------

def parse_pdb_atoms(pdb_path):
    """
    Returns a list of atom dicts from ATOM/HETATM records.
    Keys: serial, name, alt, resname, chain, resseq, icode, x, y, z, occ, bfac, element
    """
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atoms.append({
                    "record":  line[0:6].strip(),
                    "serial":  int(line[6:11]),
                    "name":    line[12:16].strip(),
                    "alt":     line[16].strip(),
                    "resname": line[17:20].strip(),
                    "chain":   line[21].strip(),
                    "resseq":  int(line[22:26]),
                    "icode":   line[26].strip(),
                    "x":       float(line[30:38]),
                    "y":       float(line[38:46]),
                    "z":       float(line[46:54]),
                    "occ":     float(line[54:60]) if line[54:60].strip() else 1.0,
                    "bfac":    float(line[60:66]) if line[60:66].strip() else 0.0,
                    "element": line[76:78].strip() if len(line) > 76 else "",
                    "raw":     line,
                })
            except (ValueError, IndexError):
                continue
    return atoms


def parse_residue_spec(spec):
    """
    Parses 'A:CYS:797' or 'CYS:797' or 'cys797' into (chain, resname, resseq).
    """
    parts = spec.upper().replace("-", ":").split(":")
    if len(parts) == 3:
        chain, resname, resseq = parts
        return chain, resname, int(resseq)
    elif len(parts) == 2:
        # Could be 'CYS:797' or 'A:797' — disambiguate by length
        if parts[0].isalpha() and len(parts[0]) == 1:
            # 'A:797' — no resname, can't proceed
            raise ValueError(f"Residue spec '{spec}' needs residue name, e.g. A:CYS:797")
        resname, resseq = parts
        return None, resname, int(resseq)
    elif len(parts) == 1:
        # 'cys797' style
        token = parts[0]
        resname = token[:3]
        resseq = int(token[3:])
        return None, resname, resseq
    else:
        raise ValueError(f"Cannot parse residue spec '{spec}'. Use format A:CYS:797")


def find_target_atom(atoms, chain, resname, resseq, atom_name):
    """Returns the atom dict for the specified nucleophilic atom."""
    for a in atoms:
        chain_match = (chain is None) or (a["chain"] == chain)
        if (chain_match and
                a["resname"].upper() == resname.upper() and
                a["resseq"] == resseq and
                a["name"].upper() == atom_name.upper()):
            return a
    return None


def get_sphere_atoms(atoms, center_xyz, radius):
    """Returns all atoms within radius Angstroms of center_xyz."""
    cx, cy, cz = center_xyz
    result = []
    for a in atoms:
        dx = a["x"] - cx
        dy = a["y"] - cy
        dz = a["z"] - cz
        if (dx*dx + dy*dy + dz*dz) <= radius*radius:
            result.append(a)
    return result


# ---------------------------------------------------------------------------
# PDB writing
# ---------------------------------------------------------------------------

def write_pdb(atoms, out_path, header="REMARK  Written by apbs_site_analysis.py\n"):
    """Writes a list of atom dicts to a PDB file."""
    with open(out_path, "w") as f:
        f.write(header)
        for a in atoms:
            f.write(
                f"{a['record']:<6}{a['serial']:5d} {a['name']:<4}{a['alt']:1}"
                f"{a['resname']:<3} {a['chain']:1}{a['resseq']:4d}{a['icode']:1}   "
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
                f"{a['occ']:6.2f}{a['bfac']:6.2f}          "
                f"{a['element']:>2}\n"
            )
        f.write("END\n")

def split_pqr_line(line: str):
    """Split PQR line handling concatenated negative numbers like -11.936-100.539."""
    line = re.sub(r'([\d\.])-', r'\1 -', line)
    return line.split()


def sanitize_pqr(pqr_path: str) -> None:
    """
    Rewrite PQR with guaranteed whitespace between all fields.
    Fixes concatenated coordinate fields from large negative values.
    PQR format: ATOM serial name resname chain resseq x y z charge radius
    """
    fixed_lines = []
    fixed_count = 0

    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                fixed_lines.append(line)
                continue
            parts = split_pqr_line(line.strip())
            # Standard PQR has 10 or 11 fields depending on whether chain is present
            if len(parts) < 10:
                fixed_lines.append(line)
                continue
            try:
                record = parts[0]
                serial = int(parts[1])
                atom_name = parts[2]
                resname = parts[3]
                # Detect whether chain field is present
                if len(parts) == 10:
                    chain_id = ""
                    resseq   = int(parts[4])
                    x, y, z  = float(parts[5]), float(parts[6]), float(parts[7])
                    charge   = float(parts[8])
                    radius   = float(parts[9])
                else:
                    chain_id = parts[4]
                    resseq   = int(parts[5])
                    x, y, z  = float(parts[6]), float(parts[7]), float(parts[8])
                    charge   = float(parts[9])
                    radius   = float(parts[10])

                fixed_line = (
                    f"{record:<6}{serial:5d} {atom_name:<4} {resname:<4}"
                    f"{chain_id:1}{resseq:4d}    "
                    f"{x:8.3f} {y:8.3f} {z:8.3f} "
                    f"{charge:8.4f} {radius:7.4f}\n"
                )
                fixed_lines.append(fixed_line)
                fixed_count += 1
            except (ValueError, IndexError):
                fixed_lines.append(line)

    with open(pqr_path, "w") as f:
        f.writelines(fixed_lines)

    print(f"[pqr] Sanitized {fixed_count} ATOM/HETATM lines in PQR.")

def clean_pdb_for_pdb2pqr(pdb_path, out_path):
    MOD_TO_CANON = {
        "CSD": "CYS",
        "CYM": "CYS",
        "CSO": "CYS",
        "SEP": "SER",
        "TPO": "THR",
        "PTR": "TYR",
        "HIE": "HIS",
        "HID": "HIS",
        "HIP": "HIS",
    }

    with open(pdb_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith(("ANISOU",)):
                continue
            if line.startswith(("HETATM",)):
                # optionally skip ligands entirely OR remap residues only if protein-like
                continue

            if line.startswith("ATOM"):
                resname = line[17:20].strip()

                if resname in MOD_TO_CANON:
                    line = line[:17] + MOD_TO_CANON[resname].ljust(3) + line[20:]

                # optionally strip hydrogens
                if line[12:16].strip().startswith("H"):
                    continue

                fout.write(line)
            else:
                fout.write(line)


# ---------------------------------------------------------------------------
# pdb2pqr: assign protonation states
# ---------------------------------------------------------------------------

def run_pdb2pqr(pdb_path, pqr_path, ph=7.4, force_field="AMBER"):
    """
    Runs pdb2pqr to assign protonation states.
    First tries with PROPKA at the given pH.
    If that fails (e.g. non-standard residues), falls back to standard
    physiologic protonation states without PROPKA.
    """
    cleaned = pdb_path.replace(".pdb", "_clean.pdb")
    clean_pdb_for_pdb2pqr(pdb_path, cleaned)


    # Attempt 1: PROPKA at specified pH
    cmd = [
        "pdb2pqr",
        "--ff", force_field,
        "--with-ph", str(ph),
        "--titration-state-method", "propka",
        "--drop-water",
        "--pdb-output", pdb_path.replace(".pdb", "_propka.pdb"),
        cleaned,
        pqr_path,
    ]
    print(f"\n[pdb2pqr] Running with PROPKA at pH {ph}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"[pdb2pqr] Success → {pqr_path}")
            return True
        print(f"[pdb2pqr] PROPKA failed, falling back to physiologic defaults...")
    except FileNotFoundError:
        print("[pdb2pqr] ERROR: pdb2pqr not found. Install with: pip install pdb2pqr")
        return False
    except subprocess.TimeoutExpired:
        print("[pdb2pqr] ERROR: timed out after 120s")
        return False

   # Attempt 2: No PROPKA, just standard protonation
    cmd_fallback = [
        "pdb2pqr",
        "--ff", force_field,
        "--drop-water",
        cleaned,
        pqr_path,
    ]
    print(f"[pdb2pqr] Running without PROPKA (physiologic defaults)...")
    try:
        result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"[pdb2pqr] Success with physiologic defaults → {pqr_path}")
            return True
        print(f"[pdb2pqr] Fallback also failed, trying stripped PDB...")
    except subprocess.TimeoutExpired:
        print("[pdb2pqr] ERROR: fallback timed out, trying stripped PDB...")

    # Attempt 3: Strip non-standard residues and retry
    print(f"[pdb2pqr] Stripping non-standard residues and retrying...")
    stripped_path = pqr_path.replace(".pqr", "_stripped.pdb")
    with open(cleaned) as f_in, open(stripped_path, "w") as f_out:
        for line in f_in:
            if not line.startswith(("ATOM", "TER", "END")):
                continue

            f_out.write(line)

    cmd_stripped = [
        "pdb2pqr",
        "--ff", force_field,
        "--drop-water",
        stripped_path,
        pqr_path,
    ]

    print(f"[pdb2pqr] Using cleaned structure: {cleaned}")
    try:
        result = subprocess.run(cmd_stripped, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"[pdb2pqr] Success after stripping → {pqr_path}")
            return True
        print(f"[pdb2pqr] All attempts failed:\n{result.stderr[-500:]}")
        return False
    except subprocess.TimeoutExpired:
        print("[pdb2pqr] ERROR: stripped attempt timed out")
        return False


def neutralize_target_residue_pqr(pqr_path, chain, resname, resseq):
    """
    Zeroes charges for all atoms in the target residue within a PQR file.
    This removes the target's self-field contribution in APBS.
    """
    updated_lines = []
    neutralized = 0

    with open(pqr_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                updated_lines.append(line)
                continue

            # PQR format: fields are whitespace-delimited with charge and radius at the end
            parts = line.split()
            if len(parts) < 10:
                updated_lines.append(line)
                continue

            record, serial, atom_name, resname_field, chain_id, resseq_field = parts[:6]
            if (resname_field.upper() == resname.upper() and
                    int(resseq_field) == resseq and
                    (chain is None or chain_id == chain)):
                charge = "0.0000"
                radius = parts[-1]
                x, y, z = parts[6], parts[7], parts[8]
                updated_line = (
                    f"{record:<6}{int(serial):5d} {atom_name:<4} {resname_field:<3} {chain_id:1}"
                    f"{int(resseq_field):4d}    {float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
                    f"{float(charge):8.4f}{float(radius):7.4f}\n"
                )
                updated_lines.append(updated_line)
                neutralized += 1
            else:
                updated_lines.append(line)

    with open(pqr_path, "w") as f:
        f.writelines(updated_lines)

    if neutralized == 0:
        print("[pqr] WARNING: target residue not found for neutralization.")
    else:
        print(f"[pqr] Neutralized charges on {neutralized} atoms in {resname}{resseq}.")


def compute_cglen(pqr_path, padding=20.0):
    """Compute coarse grid size from actual atom coordinates in the PQR."""
    xs, ys, zs = [], [], []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                try:
                    xs.append(float(parts[-5]))
                    ys.append(float(parts[-4]))
                    zs.append(float(parts[-3]))
                except (ValueError, IndexError):
                    continue
    if not xs:
        return 150.0
    xlen = max(xs) - min(xs) + padding
    ylen = max(ys) - min(ys) + padding
    zlen = max(zs) - min(zs) + padding
    # Use the largest dimension for all three so the box is always cubic and safe
    return max(xlen, ylen, zlen, 150.0)


# ---------------------------------------------------------------------------
# APBS input file generation
# ---------------------------------------------------------------------------

def write_apbs_input(pqr_path, dx_prefix, apbs_input_path, fine_grid_center, fine_grid_length=20.0):
    """
    Writes an APBS input file using focusing: coarse grid covers the whole
    molecule, fine grid is centred on the reactive site (fine_grid_length Å cube).
    """
    cx, cy, cz = fine_grid_center
    pqr_path  = os.path.abspath(pqr_path)   
    dx_prefix = os.path.abspath(dx_prefix)   

    cglen = compute_cglen(pqr_path)  # ← dynamic instead of hardcoded
    print(f"[apbs] Coarse grid size: {cglen:.1f} Å")

    apbs_input = f"""# APBS input — generated by apbs_site_analysis.py
read
    mol pqr {pqr_path}
end

elec name whole_protein
    mg-auto
    mol 1

    # Grid dimensions (must be odd)
    dime 97 97 97

    # Coarse grid: covers the whole protein
    cglen {cglen:.1f} {cglen:.1f} {cglen:.1f}
    cgcent mol 1

    # Fine grid: centred on the reactive site
    fglen {fine_grid_length:.1f} {fine_grid_length:.1f} {fine_grid_length:.1f}
    fgcent {cx:.3f} {cy:.3f} {cz:.3f}

    # Solvent model
    lpbe                   # linearised Poisson-Boltzmann
    bcfl sdh               # single Debye-Hückel boundary conditions
    ion charge +1 conc 0.150 radius 2.0   # 150 mM monovalent salt
    ion charge -1 conc 0.150 radius 1.8

    # Dielectric constants
    pdie 2.0               # protein interior
    sdie 78.54             # solvent (water, 298 K)

    # Molecular surface
    srfm smol              # smoothed molecular surface
    chgm spl0              # charge mapping
    sdens 10.0             # surface density

    # Radii
    srad 1.4               # probe radius (water)
    swin 0.3               # spline window

    temp 298.15

    # Output volumetric data
    write pot dx {dx_prefix}
end

quit
"""
    with open(apbs_input_path, "w") as f:
        f.write(apbs_input)
    print(f"[apbs] Input written → {apbs_input_path}")


# ---------------------------------------------------------------------------
# APBS execution
# ---------------------------------------------------------------------------

def run_apbs(apbs_input_path, work_dir):
    """Runs APBS. Returns True on success."""

    # Running based on this APBS input file, with the working directory set to where the input is located.
    apbs_input_path = os.path.abspath(apbs_input_path)
    cmd = ["apbs", apbs_input_path]
    print(f"\n[apbs] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=300, cwd=work_dir
        )
        if result.returncode != 0:
            print(f"[apbs] STDERR:\n{result.stderr[-3000:]}")
            return False
        print("[apbs] Completed successfully.")
        return True
    except FileNotFoundError:
        print("[apbs] ERROR: apbs not found. Install with: conda install -c conda-forge apbs")
        return False
    except subprocess.TimeoutExpired:
        print("[apbs] ERROR: timed out after 300s")
        return False


# ---------------------------------------------------------------------------
# DX file parsing — read electrostatic potential at a point
# ---------------------------------------------------------------------------

def read_dx_potential_at_point(dx_path, query_xyz, sample_radius=1.4, n_samples=26):
    """
    Reads an OpenDX file and returns the mean electrostatic potential
    averaged over a shell of points around query_xyz at sample_radius Angstroms.
    Avoids sampling at the atom center itself where the potential diverges.
    
    sample_radius: distance from atom center to sample (default 1.4 A = ~water probe)
    n_samples: number of points on the shell (26 = face+edge+corner of a cube shell)
    """
    with open(dx_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Parse header
    origin = None
    delta = []
    counts = None
    data_start = 0

    for i, line in enumerate(lines):
        if line.startswith("object 1"):
            parts = line.split()
            counts = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
        elif line.startswith("origin"):
            parts = line.split()
            origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("delta"):
            parts = line.split()
            delta.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
        elif line.startswith("object 3"):
            data_start = i + 1
            break

    if origin is None or len(delta) < 3 or counts is None:
        raise ValueError(f"Could not parse DX header in {dx_path}")

    raw_values = []
    for line in lines[data_start:]:
        if line.startswith("object") or line.startswith("attribute"):
            break
        raw_values.extend(float(v) for v in line.split())

    nx, ny, nz = counts
    grid = np.array(raw_values[:nx*ny*nz]).reshape((nx, ny, nz))

    dx_vec = delta[0][0]
    dy_vec = delta[1][1]
    dz_vec = delta[2][2]

    def sample_at(xyz):
        """Trilinear interpolation at a single point."""
        qx, qy, qz = xyz
        fx = (qx - origin[0]) / dx_vec
        fy = (qy - origin[1]) / dy_vec
        fz = (qz - origin[2]) / dz_vec

        # Return None if outside grid
        if not (0 <= fx < nx-1 and 0 <= fy < ny-1 and 0 <= fz < nz-1):
            return None

        ix, iy, iz = int(fx), int(fy), int(fz)
        ix = min(ix, nx - 2)
        iy = min(iy, ny - 2)
        iz = min(iz, nz - 2)
        tx, ty, tz = fx - ix, fy - iy, fz - iz

        return (
            grid[ix,   iy,   iz  ] * (1-tx)*(1-ty)*(1-tz) +
            grid[ix+1, iy,   iz  ] * tx*(1-ty)*(1-tz) +
            grid[ix,   iy+1, iz  ] * (1-tx)*ty*(1-tz) +
            grid[ix,   iy,   iz+1] * (1-tx)*(1-ty)*tz +
            grid[ix+1, iy+1, iz  ] * tx*ty*(1-tz) +
            grid[ix+1, iy,   iz+1] * tx*(1-ty)*tz +
            grid[ix,   iy+1, iz+1] * (1-tx)*ty*tz +
            grid[ix+1, iy+1, iz+1] * tx*ty*tz
        )

    # Generate shell sample points using the golden spiral / uniform sphere
    # Simple approach: use the 26 face/edge/corner directions of a unit cube
    # normalized to the sample radius
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                norm = np.sqrt(dx**2 + dy**2 + dz**2)
                offsets.append(np.array([dx, dy, dz]) / norm * sample_radius)

    qx, qy, qz = query_xyz
    potentials = []
    for off in offsets:
        val = sample_at((qx + off[0], qy + off[1], qz + off[2]))
        if val is not None:
            # Reject singularity spikes — anything beyond ±50 kT/e is a grid artifact
            if abs(val) < 50.0:
                potentials.append(val)

    if not potentials:
        print("[analysis] WARNING: No valid sample points found — returning NaN.")
        return np.nan
    
    mean_pot = np.mean(potentials)
    print(f"[analysis] Sampled {len(potentials)}/26 shell points at r={sample_radius} Å")
    print(f"[analysis] Shell potential range: {min(potentials):.3f} to {max(potentials):.3f} kT/e")

    return mean_pot


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

KT_PER_E_TO_MV = 25.7  # at 298 K, 1 kT/e = 25.7 mV
#TO-DO: THIS IS NOT RIGHT
def interpret_potential(potential_kt_e, resname):
    """
    Interprets the electrostatic potential at the nucleophilic atom.
    A negative potential stabilises a positive charge (protonated form).
    A positive potential stabilises a negative charge (deprotonated form).
    """
    potential_mv = potential_kt_e * KT_PER_E_TO_MV

    # pKa shift: ΔpKa ≈ potential (in kT/e units) / ln(10)
    # Positive potential → lowers pKa (favours deprotonation)
    # Reference pKa values for protonated forms
    ref_pka = {
        "CYS": 8.3,
        "SER": 13.0,
        "THR": 13.6,
        "TYR": 10.5,
        "LYS": 10.5,
        "HIS": 6.0,
    }
    dpka = potential_kt_e / np.log(10)  # ΔpKa
    base_pka = ref_pka.get(resname.upper(), 10.0)
    predicted_pka = base_pka - dpka  # positive potential lowers pKa

    print(f"\n{'='*60}")
    print(f"  ELECTROSTATIC ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Residue:               {resname}")
    print(f"  Electrostatic potential at nucleophile:")
    print(f"    {potential_kt_e:+.3f} kT/e  ({potential_mv:+.1f} mV)")
    print(f"  Reference pKa (solution): {base_pka:.1f}")
    print(f"  Estimated pKa shift (ΔpKa): {-dpka:+.2f}")
    print(f"  Predicted pKa in this environment: {predicted_pka:.1f}")
    print(f"{'='*60}")

    if potential_kt_e > 0.5:
        verdict = "FAVOURS DEPROTONATION — positive potential stabilises the anion."
    elif potential_kt_e < -0.5:
        verdict = "DISFAVOURS DEPROTONATION — negative potential stabilises the protonated form."
    else:
        verdict = "NEUTRAL electrostatic environment (|φ| < 0.5 kT/e)."

    print(f"  Verdict: {verdict}")
    print(f"{'='*60}\n")
    print("  NOTE: This is a qualitative estimate. The ΔpKa formula")
    print("  (Born model) is approximate. For quantitative pKa,")
    print("  use thermodynamic integration or free energy perturbation.")
    print(f"{'='*60}\n")

    return predicted_pka

def neutralize_target_residue_pqr(pqr_path, chain, resname, resseq):
    updated_lines = []
    neutralized = 0

    with open(pqr_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                updated_lines.append(line)
                continue

            parts = line.split()

            if len(parts) < 10:
                updated_lines.append(line)
                continue

            try:
                record = parts[0]
                atom_name = parts[2]
                resname_field = parts[3]
                # Sometimes doesn't include the chain id
                if len(parts) == 10:
                    chain_id = None
                    resseq_field = int(parts[4])
                else:
                    chain_id = parts[4]
                    resseq_field = int(parts[5])

                is_target = (
                    resname_field.upper() == resname.upper()
                    and resseq_field == resseq
                    and (chain is None or chain_id is None or chain_id == chain)
                )

                if is_target:
                    parts[-2] = "0.0000"
                    neutralized += 1

                updated_lines.append(" ".join(parts) + "\n")

            except Exception:
                updated_lines.append(line)

    with open(pqr_path, "w") as f:
        f.writelines(updated_lines)

    print(f"[pqr] Neutralized {neutralized} atoms.")


def extract_and_renumber_model1(pdb_path: str, out_path: str) -> None:
    """Extract MODEL 1 only and renumber atom serials from 1."""
    in_model1 = False
    serial = 1
    found_models = False

    with open(pdb_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if line.startswith("MODEL"):
                found_models = True
                model_num = line.split()[1] if len(line.split()) > 1 else ""
                in_model1 = (model_num == "1")
                continue
            if line.startswith("ENDMDL"):
                if in_model1:
                    break
                continue
            if not found_models or in_model1:
                if line.startswith(("ATOM", "HETATM")):
                    line = f"{line[:6]}{serial:5d}{line[11:]}"
                    serial += 1
                f_out.write(line)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_apbs_site_analysis(pdb_path, residue_spec, radius=8.0, ph=7.4,
                            work_dir=None, keep_files=False):
    """
    Full pipeline:
      1. Parse residue spec and locate nucleophilic atom
      2. Run pdb2pqr on the full protein (protonation + charges)
      3. Write APBS input focused on the reactive site
      4. Run APBS
      5. Read potential at the nucleophilic atom from DX output
      6. Report and interpret
    """
    # --- 1. Setup ---
    pdb_path = os.path.abspath(pdb_path)
    if not os.path.exists(pdb_path):
        print(f"ERROR: PDB file not found: {pdb_path}")
        sys.exit(1)

    chain, resname, resseq = parse_residue_spec(residue_spec)
    resname = resname.upper()

    if resname not in TARGET_ATOM:
        print(f"ERROR: Residue '{resname}' not in supported residues.")
        print(f"  Supported: {', '.join(TARGET_ATOM.keys())}")
        sys.exit(1)

    nuc_atom_name = TARGET_ATOM[resname]
    print(f"\n[setup] Target: chain={chain or 'any'} {resname}{resseq}")
    print(f"[setup] Target atom: {nuc_atom_name}")
    print(f"[setup] Sphere radius: {radius} Å  |  pH: {ph}")

    # --- 2. Parse PDB and find nucleophilic atom ---
    # --- 2. Parse PDB and find nucleophilic atom ---
    # Extract model 1 first so atom lookup and coordinates match what pdb2pqr will see
    clean_pdb = pdb_path.replace(".pdb", "_model1.pdb")
    if not os.path.exists(clean_pdb):
        extract_and_renumber_model1(pdb_path, clean_pdb)
        print(f"[preprocess] Extracted MODEL 1 → {clean_pdb}")
    atoms = parse_pdb_atoms(clean_pdb)
    if not atoms:
        print(f"ERROR: No ATOM records found in {pdb_path}")
        sys.exit(1)

    # Filter out alternate conformations (keep blank or 'A')
    atoms = [a for a in atoms if a["alt"] in ("", "A")]

    # This is the specific atom of interest from the residue (from the reactive atoms)
    target = find_target_atom(atoms, chain, resname, resseq, nuc_atom_name)
    if target is None:
        print(f"ERROR: Could not find atom {nuc_atom_name} in {resname}{resseq}"
              f"{' chain '+chain if chain else ''}")
        print(f"  Available CYS/SER/TYR/etc atoms — check your residue spec.")
        # Print what's there for that residue to help debugging
        candidates = [a for a in atoms
                      if a["resseq"] == resseq and a["resname"].upper() == resname]
        if candidates:
            print(f"  Atoms found in {resname}{resseq}:")
            for c in candidates:
                print(f"    chain={c['chain']} name={c['name']}")
        sys.exit(1)

    # This is the coordinates of the nucleophilic atom — the centre of our APBS fine grid
    nuc_xyz = (target["x"], target["y"], target["z"])
    print(f"[setup] Nucleophile coordinates: {nuc_xyz[0]:.3f}, {nuc_xyz[1]:.3f}, {nuc_xyz[2]:.3f}")

    # --- 3. Working directory ---
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="apbs_")
        _cleanup = not keep_files
    else:
        os.makedirs(work_dir, exist_ok=True)
        _cleanup = False

    print(f"[setup] Working directory: {work_dir}")

    basename = os.path.splitext(os.path.basename(pdb_path))[0]
    pqr_path      = os.path.join(work_dir, f"{basename}.pqr")
    apbs_in_path  = os.path.join(work_dir, f"{basename}.in")
    dx_prefix     = os.path.join(work_dir, f"{basename}_pot")

    # --- 4. pdb2pqr on full protein ---
    # pdb2pqr needs the full protein for accurate PROPKA titration
    success = run_pdb2pqr(clean_pdb, pqr_path, ph=ph)
    if not success:
        print("\n[fallback] pdb2pqr failed. Attempting to continue with raw PDB...")
        print("  WARNING: No partial charges — APBS output will be unreliable.")
        print("  Install pdb2pqr:  pip install pdb2pqr")
        sys.exit(1)

    #clean this to prevent any errors or formatting errors
    sanitize_pqr(pqr_path)
    # Neutralize the target residue so it does not bias the local potential.
    neutralize_target_residue_pqr(pqr_path, chain, resname, resseq)

    # --- 5. Write APBS input ---
    # Fine grid: cube centred on nucleophilic atom, 2*radius on each side
    # Save the input file to run the APBS
    fine_grid_len = radius * 2.5  # slightly larger than sphere
    write_apbs_input(pqr_path, dx_prefix, apbs_in_path,
                     fine_grid_center=nuc_xyz,
                     fine_grid_length=fine_grid_len)

    # --- 6. Run APBS ---
    success = run_apbs(apbs_in_path, work_dir)
    if not success:
        print("ERROR: APBS failed. Check the input file and logs above.")
        sys.exit(1)

    # --- 7. Read potential at nucleophilic atom ---
    # APBS writes <prefix>0.dx (zero-indexed)
    dx_path = dx_prefix + "0.dx"
    if not os.path.exists(dx_path):
        # Try without the 0
        dx_path_alt = dx_prefix + ".dx"
        if os.path.exists(dx_path_alt):
            dx_path = dx_path_alt
        else:
            print(f"ERROR: DX output not found at {dx_prefix}0.dx or {dx_prefix}.dx")
            print(f"  Files in work dir: {os.listdir(work_dir)}")
            sys.exit(1)

    print(f"\n[analysis] Reading potential from: {dx_path}")
    potential = read_dx_potential_at_point(dx_path, nuc_xyz)
    print(f"[analysis] Potential at {nuc_atom_name} ({resname}{resseq}): {potential:+.4f} kT/e")

    # --- 8. Report sphere context ---
    sphere_atoms = get_sphere_atoms(atoms, nuc_xyz, radius)
    sphere_residues = set((a["chain"], a["resname"], a["resseq"]) for a in sphere_atoms)
    print(f"\n[context] Residues within {radius} Å of {nuc_atom_name}:")
    for chain_id, rname, rseq in sorted(sphere_residues, key=lambda x: x[2]):
        marker = " ← TARGET" if (rname == resname and rseq == resseq) else ""
        print(f"  {chain_id}:{rname}{rseq}{marker}")

    residue_counts = {"ARG": 0, "LYS": 0, "ASP": 0, "GLU": 0}
    for _, rname, _ in sphere_residues:
        rname_u = rname.upper()
        if rname_u in residue_counts:
            residue_counts[rname_u] += 1

    print("\n[context] Charged residue totals in sphere:")
    print(f"  ARG: {residue_counts['ARG']}")
    print(f"  LYS: {residue_counts['LYS']}")
    print(f"  ASP: {residue_counts['ASP']}")
    print(f"  GLU: {residue_counts['GLU']}")

    # --- 9. Interpret ---
    predicted_pka = interpret_potential(potential, resname)

    # --- 10. Cleanup ---
    if _cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)

    return predicted_pka


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="APBS electrostatic analysis of a reactive site for deprotonation likelihood.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python apbs_site_analysis.py 4g5j.pdb A:CYS:797
  python apbs_site_analysis.py 4g5j.pdb A:SER:768 --radius 10.0 --ph 7.4
  python apbs_site_analysis.py 4g5j.pdb A:TYR:1016 --ph 6.0 --keep-files

Supported nucleophiles: CYS (SG), SER (OG), THR (OG1), TYR (OH), LYS (NZ), HIS (ND1)
        """
    )
    parser.add_argument("pdb_path",      help="Input PDB file path")
    parser.add_argument("residue_spec",  help="Residue to analyse, e.g. A:CYS:797 or cys797")
    parser.add_argument("--radius",      type=float, default=8.0,
                        help="Sphere radius in Angstroms for context reporting (default: 8.0)")
    parser.add_argument("--ph",          type=float, default=7.4,
                        help="pH for pdb2pqr/PROPKA protonation assignment (default: 7.4)")
    parser.add_argument("--work-dir",    default=None,
                        help="Directory for intermediate files (default: temp dir, auto-deleted)")
    parser.add_argument("--keep-files",  action="store_true",
                        help="Keep intermediate files (PQR, DX, APBS input) after run")
    args = parser.parse_args()

    run_apbs_site_analysis(
        pdb_path=args.pdb_path,
        residue_spec=args.residue_spec,
        radius=args.radius,
        ph=args.ph,
        work_dir=args.work_dir,
        keep_files=args.keep_files,
    )