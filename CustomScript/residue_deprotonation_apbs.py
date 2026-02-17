import subprocess
from Bio.PDB import PDBParser
import numpy as np


# -------------------
# PARAMETERS
# -------------------
pdb_file = "4g5j_cleaned.pdb"       # Input PDB
residue_number = 797            # Residue to analyze
chain_id = "A"
pqr_file = "protein.pqr"
dx_file = "protein.dx"
apbs_input_file = "apbs.in"

# -------------------
# STEP 1: Convert PDB -> PQR
# -------------------
subprocess.run(["pdb2pqr", "--ff", "AMBER", pdb_file, pqr_file], check=True)

# -------------------
# STEP 2: Write APBS input
# -------------------
apbs_input = f"""
read:
  - mol:
      format: pqr
      file: protein.pqr

elec:
  - mg-auto:
      dime: [65, 65, 65]
      cglen: [40.0, 40.0, 40.0]
      fglen: [40.0, 40.0, 40.0]
      cgcent: mol 1
      fgcent: mol 1
      lpbe: true
      bcfl: sdh
      pdie: 2.0
      sdie: 78.54
      srfm: smol
      chgm: spl2
      sdens: 10.0
      srad: 1.4
      swin: 0.3
      temp: 298.15
      calcenergy: total
      calcforce: no
      write:
        pot: dx protein
"""



with open(apbs_input_file, "w") as f:
    f.write(apbs_input)

# -------------------
# STEP 3: Run APBS
# -------------------
subprocess.run(["apbs", apbs_input_file], check=True)

# -------------------
# STEP 4: Parse .dx file
# -------------------
def read_dx(filename):
    """Read a DX file and return the origin, spacing, and 3D grid as numpy array."""
    with open(filename) as f:
        lines = f.readlines()

    nx = ny = nz = 0
    origin = None
    delta = []

    grid_data = []

    read_grid = False
    for line in lines:
        tokens = line.split()
        if len(tokens) == 0:
            continue
        if tokens[0] == "object" and tokens[1] == "1":
            origin = list(map(float, lines[1].split()[1:]))
            nx, ny, nz = map(int, [lines[2].split()[1], lines[3].split()[1], lines[4].split()[1]])
            delta = [list(map(float, lines[5].split()[1:])),
                     list(map(float, lines[6].split()[1:])),
                     list(map(float, lines[7].split()[1:]))]
            read_grid = True
            grid_data_start = 9
            break

    if not read_grid:
        raise RuntimeError("Could not parse DX file header")

    # Flatten all remaining numbers into a list
    values = []
    for line in lines[grid_data_start:]:
        values.extend([float(x) for x in line.split()])

    # Convert to numpy array
    grid = np.array(values).reshape((nx, ny, nz), order='F')  # Fortran-style ordering

    return origin, delta, grid

origin, delta, grid = read_dx(dx_file)

# -------------------
# STEP 5: Load residue coordinates
# -------------------
parser = PDBParser()
structure = parser.get_structure("protein", pdb_file)

residue = None
for res in structure[0][chain_id]:
    if res.id[1] == residue_number:
        residue = res
        break

if residue is None:
    raise ValueError(f"Residue {residue_number} not found in chain {chain_id}")

# -------------------
# STEP 6: Trilinear interpolation
# -------------------
def trilinear_interp(x, y, z, origin, delta, grid):
    nx, ny, nz = grid.shape
    dx = np.linalg.norm(delta[0])
    dy = np.linalg.norm(delta[1])
    dz = np.linalg.norm(delta[2])

    # Map coordinates to grid indices
    ix = (x - origin[0]) / dx
    iy = (y - origin[1]) / dy
    iz = (z - origin[2]) / dz

    i0, j0, k0 = int(ix), int(iy), int(iz)
    i1, j1, k1 = min(i0+1, nx-1), min(j0+1, ny-1), min(k0+1, nz-1)
    xd, yd, zd = ix - i0, iy - j0, iz - k0

    # Corners
    c000 = grid[i0, j0, k0]
    c001 = grid[i0, j0, k1]
    c010 = grid[i0, j1, k0]
    c011 = grid[i0, j1, k1]
    c100 = grid[i1, j0, k0]
    c101 = grid[i1, j0, k1]
    c110 = grid[i1, j1, k0]
    c111 = grid[i1, j1, k1]

    # Interpolation
    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd

    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    c = c0*(1-zd) + c1*zd
    return c

# -------------------
# STEP 7: Compute average potential at heavy atoms
# -------------------
potentials = []
for atom in residue:
    if atom.element != 'H':
        x, y, z = atom.coord
        potentials.append(trilinear_interp(x, y, z, origin, delta, grid))

avg_potential = sum(potentials) / len(potentials)
print(f"Average electrostatic potential at residue {residue_number}: {avg_potential:.2f} kT/e")

# Rough interpretation
if avg_potential > 0:
    print("Positive potential → favors deprotonation")
else:
    print("Negative potential → disfavors deprotonation")
