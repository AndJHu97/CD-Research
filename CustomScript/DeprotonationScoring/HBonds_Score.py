"""
HBonds_Score.py

Counts and scores hydrogen-bond-like interactions in the forward cone of a
reactive residue. The forward direction is defined by the sidechain vector
CA -> reactive atom (e.g., CYS CA->SG). Interactions are weighted by distance
and angle, and binned into strict and weak proximity counts.

Usage:
	python HBonds_Score.py 4g5j.pdb A:CYS:797
	python HBonds_Score.py 4g5j.pdb CYS:797 --cone-cos 0.5 --radius 6.0
"""

import argparse
import math


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

DONOR_ATOMS = {
	"N", "NZ", "ND1", "NE2", "OG", "OG1", "OH", "SG"
}

ACCEPTOR_ATOMS = {
	"O", "OXT", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SG", "ND1", "NE2"
}


def parse_pdb_atoms(pdb_path):
	"""Return atom dicts from ATOM/HETATM records."""
	atoms = []
	with open(pdb_path, "r") as f:
		for line in f:
			if not (line.startswith("ATOM") or line.startswith("HETATM")):
				continue
			try:
				atoms.append({
					"record": line[0:6].strip(),
					"serial": int(line[6:11]),
					"name": line[12:16].strip(),
					"alt": line[16].strip(),
					"resname": line[17:20].strip(),
					"chain": line[21].strip(),
					"resseq": int(line[22:26]),
					"icode": line[26].strip(),
					"x": float(line[30:38]),
					"y": float(line[38:46]),
					"z": float(line[46:54]),
					"element": line[76:78].strip() if len(line) > 76 else "",
				})
			except (ValueError, IndexError):
				continue
	return atoms


def parse_residue_spec(spec):
	"""Parse A:CYS:797 or CYS:797 or cys797 into (chain, resname, resseq)."""
	parts = spec.upper().replace("-", ":").split(":")
	if len(parts) == 3:
		chain, resname, resseq = parts
		return chain, resname, int(resseq)
	if len(parts) == 2:
		if parts[0].isalpha() and len(parts[0]) == 1:
			raise ValueError(f"Residue spec '{spec}' needs residue name, e.g. A:CYS:797")
		resname, resseq = parts
		return None, resname, int(resseq)
	if len(parts) == 1:
		token = parts[0]
		return None, token[:3], int(token[3:])
	raise ValueError(f"Cannot parse residue spec '{spec}'. Use format A:CYS:797")


def find_atom(atoms, chain, resname, resseq, atom_name):
	for a in atoms:
		chain_match = chain is None or a["chain"] == chain
		if chain_match and a["resname"].upper() == resname.upper() and a["resseq"] == resseq:
			if a["name"].upper() == atom_name.upper():
				return a
	return None


def get_residue_atoms(atoms, chain, resname, resseq):
	result = []
	for a in atoms:
		chain_match = chain is None or a["chain"] == chain
		if chain_match and a["resname"].upper() == resname.upper() and a["resseq"] == resseq:
			result.append(a)
	return result


def vec(a, b):
	return (b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"])


def norm(v):
	return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v):
	n = norm(v)
	if n == 0:
		return (0.0, 0.0, 0.0)
	return (v[0] / n, v[1] / n, v[2] / n)


def dot(a, b):
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def angle_deg(v1, v2):
	n1 = norm(v1)
	n2 = norm(v2)
	if n1 == 0 or n2 == 0:
		return 0.0
	c = max(-1.0, min(1.0, dot(v1, v2) / (n1 * n2)))
	return math.degrees(math.acos(c))


def nearest_anchor(atom, residue_atoms):
	"""Find nearest heavy atom in same residue for angle approximation."""
	best = None
	best_d = 999.0
	for a in residue_atoms:
		if a["serial"] == atom["serial"]:
			continue
		if a["element"].upper() == "H":
			continue
		dx = a["x"] - atom["x"]
		dy = a["y"] - atom["y"]
		dz = a["z"] - atom["z"]
		d = math.sqrt(dx * dx + dy * dy + dz * dz)
		if d < best_d:
			best_d = d
			best = a
	return best


def in_forward_cone(center, direction, point, cone_cos):
	v = (point[0] - center[0], point[1] - center[1], point[2] - center[2])
	v_n = normalize(v)
	return dot(direction, v_n) >= cone_cos


def compute_hbond_scores(atoms, target_chain, target_resname, target_resseq, radius,
						 cone_cos, r0, sigma, strict_max, weak_max):
	reactive_atom_name = TARGET_ATOM[target_resname]
	reactive = find_atom(atoms, target_chain, target_resname, target_resseq, reactive_atom_name)
	ca = find_atom(atoms, target_chain, target_resname, target_resseq, "CA")
	if reactive is None or ca is None:
		raise ValueError("Target residue is missing CA or reactive atom.")

	reactive_xyz = (reactive["x"], reactive["y"], reactive["z"])
	reactive_vec = normalize(vec(ca, reactive))

	donors = []
	acceptors = []
	for a in atoms:
		if a["alt"] not in ("", "A"):
			continue
		dx = a["x"] - reactive_xyz[0]
		dy = a["y"] - reactive_xyz[1]
		dz = a["z"] - reactive_xyz[2]
		if dx * dx + dy * dy + dz * dz > radius * radius:
			continue
		name_u = a["name"].upper()
		if name_u in DONOR_ATOMS:
			donors.append(a)
		if name_u in ACCEPTOR_ATOMS:
			acceptors.append(a)

	strict_count = 0
	weak_count = 0
	weighted_score = 0.0
	strict_flexible_score = 0.0

	for d in donors:
		d_res_atoms = get_residue_atoms(atoms, d["chain"], d["resname"], d["resseq"])
		anchor = nearest_anchor(d, d_res_atoms)
		if anchor is None:
			continue
		for a in acceptors:
			if d["serial"] == a["serial"]:
				continue
			if d["resname"] == a["resname"] and d["resseq"] == a["resseq"] and d["chain"] == a["chain"]:
				continue

			midpoint = ((d["x"] + a["x"]) * 0.5, (d["y"] + a["y"]) * 0.5, (d["z"] + a["z"]) * 0.5)
			if not in_forward_cone(reactive_xyz, reactive_vec, midpoint, cone_cos):
				continue

			v_da = (a["x"] - d["x"], a["y"] - d["y"], a["z"] - d["z"])
			dist = norm(v_da)
			if dist > weak_max:
				continue

			v_dx = (anchor["x"] - d["x"], anchor["y"] - d["y"], anchor["z"] - d["z"])
			ang = angle_deg(v_dx, v_da)

			if 1.5 <= dist <= strict_max and ang >= 120.0:
				strict_count += 1
			elif strict_max < dist <= 5.0 and ang >= 90.0:
				weak_count += 1

			w_d = math.exp(-((dist - r0) ** 2) / (sigma ** 2))
			w_theta = max(0.0, math.cos(math.radians(ang) - math.pi))
			weighted_score += w_d * w_theta

			if 1.5 <= dist <= strict_max:
				strict_flexible_score += w_theta

	return strict_count, weak_count, weighted_score, strict_flexible_score, reactive_vec


def main():
	parser = argparse.ArgumentParser(description="Directional hydrogen bond scoring.")
	parser.add_argument("pdb_path")
	parser.add_argument("residue_spec")
	parser.add_argument("--radius", type=float, default=6.0, help="Search radius in Angstroms")
	parser.add_argument("--cone-cos", type=float, default=0.0, help="Cosine cutoff for forward cone")
	parser.add_argument("--r0", type=float, default=2.9, help="Distance center for weighting")
	parser.add_argument("--sigma", type=float, default=0.5, help="Distance weighting width")
	parser.add_argument("--strict-max", type=float, default=3.5, help="Strict H-bond max distance")
	parser.add_argument("--weak-max", type=float, default=6.0, help="Weak/proximity max distance")
	args = parser.parse_args()

	chain, resname, resseq = parse_residue_spec(args.residue_spec)
	resname = resname.upper()
	if resname not in TARGET_ATOM:
		raise SystemExit(f"Unsupported residue: {resname}. Use one of {', '.join(TARGET_ATOM.keys())}")

	atoms = parse_pdb_atoms(args.pdb_path)
	if not atoms:
		raise SystemExit("No atoms found in PDB.")

	strict_count, weak_count, weighted_score, strict_flexible_score, reactive_vec = compute_hbond_scores(
		atoms,
		chain,
		resname,
		resseq,
		args.radius,
		args.cone_cos,
		args.r0,
		args.sigma,
		args.strict_max,
		args.weak_max,
	)

	print("\n=== HBond Directional Score ===")
	print(f"Target: {chain or 'any'}:{resname}{resseq}")
	print(f"Reactive vector (CA->atom): ({reactive_vec[0]:.3f}, {reactive_vec[1]:.3f}, {reactive_vec[2]:.3f})")
	print(f"Forward cone cos cutoff: {args.cone_cos:.2f}")
	print(f"Strict H-bonds (1.5-{args.strict_max:.1f} A, angle >= 120): {strict_count}")
	print(f"Strict flexible score (1.5-{args.strict_max:.1f} A, angle weighted): {strict_flexible_score:.4f}")
	print(f"Weak proximity (>{args.strict_max:.1f} A to 5.0 A, angle >= 90): {weak_count}")
	print(f"Weighted score (<= {args.weak_max:.1f} A): {weighted_score:.4f}")


if __name__ == "__main__":
	main()
