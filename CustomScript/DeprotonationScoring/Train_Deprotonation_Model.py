"""
Deprotonation_Model.py

Train an XGBoost regressor to predict deprotonation probability from features:
- Physiologic pH (default 7.4)
- H-bond weighted score (from HBonds_Score.py)
- Charged residue counts (ARG/LYS/ASP/GLU/HIS) near the site (from APBS_Deprotonation.py)
- Electrostatic potential (from APBS_Deprotonation.py)
- SASA (from pdb_sasa file or computed if missing)

Label: P(deprot) = 1 / (1 + 10^(pKa - pH)) from PKAD-R Expt. pKa
"""

import argparse
import math
import os
import re
import subprocess
import urllib.request
from typing import Dict, Optional, Tuple
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score

try:
	from xgboost import XGBRegressor
except ImportError as exc:
	raise SystemExit("xgboost is required. Install with: pip install xgboost") from exc


PHYSIOLOGIC_PH = 7.4

REFERENCE_PKA = {
    "ASP": 3.9,
    "GLU": 4.1,
    "HIS": 6.0,
    "CYS": 8.3,
    "LYS": 10.5,
    "TYR": 10.5,
    "ARG": 12.5,
    "SER": 13.0,
    "THR": 13.6,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APBS_SCRIPT = os.path.join(SCRIPT_DIR, "APBS_Deprotonation.py")
HBOND_SCRIPT = os.path.join(SCRIPT_DIR, "HBonds_Score.py")

TARGET_ATOM = {
	"ALA": "CA",
	"ARG": "CA",
	"ASN": "CA",
	"ASP": "CA",
	"CYS": "SG",
	"GLN": "CA",
	"GLU": "CA",
	"GLY": "CA",
	"HIS": "ND1",
	"ILE": "CA",
	"LEU": "CA",
	"LYS": "NZ",
	"MET": "CA",
	"PHE": "CA",
	"PRO": "CA",
	"SER": "OG",
	"THR": "OG1",
	"TRP": "CA",
	"TYR": "OH",
	"VAL": "CA",
}


def compute_deprot_probability(pka: float, ph: float) -> float:
	return 1.0 / (1.0 + 10 ** (pka - ph))


def find_pdb_path(pdb_id_or_path: str, pdb_dir: str) -> str:
	if os.path.isfile(pdb_id_or_path):
		return pdb_id_or_path

	pdb_id = pdb_id_or_path.lower().replace(".pdb", "")
	candidate = os.path.join(pdb_dir, f"{pdb_id}.pdb")
	if os.path.isfile(candidate):
		return candidate

	url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
	print(f"[pdb] Downloading {pdb_id} from RCSB...")
	try:
		urllib.request.urlretrieve(url, candidate)
		if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
			print(f"[pdb] Saved to {candidate}")
			return candidate
	except Exception as exc:
		raise FileNotFoundError(
			f"PDB file not found locally and download failed for '{pdb_id}'."
		) from exc

	raise FileNotFoundError(f"PDB file not found for '{pdb_id_or_path}' in {pdb_dir}")


def parse_pkadr_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
	# Heuristic column lookup
	colmap = {c.lower(): c for c in df.columns}
	pdb_col = colmap.get("pdb", None)
	chain_col = colmap.get("chain", None)
	resid_col = colmap.get("resid in pdb", None) or colmap.get("resid", None) or colmap.get("resnum", None)
	resname_col = colmap.get("resname", None) or colmap.get("res name", None)
	pka_col = colmap.get("expt. pka", None) or colmap.get("expt pka", None) or colmap.get("pka", None)

	missing = [name for name, col in [
		("PDB", pdb_col),
		("Chain", chain_col),
		("ResID", resid_col),
		("ResName", resname_col),
		("Expt. pKa", pka_col),
	] if col is None]

	if missing:
		raise ValueError(f"Missing required columns in PKAD-R CSV: {', '.join(missing)}")

	return pdb_col, chain_col, resid_col, resname_col, pka_col


def run_hbonds_score(pdb_path: str, residue_spec: str, radius: float) -> Dict[str, float]:
	cmd = [
		"python",
		HBOND_SCRIPT,
		pdb_path,
		residue_spec,
		"--radius",
		str(radius),
	]
	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"HBonds_Score failed: {result.stderr.strip()}")

	text = result.stdout
	weighted = _extract_float(text, r"Weighted score .*?:\s*([\d.+-]+)")
	strict_flex = _extract_float(text, r"Strict flexible score .*?:\s*([\d.+-]+)")

	return {
		"hbonds_weighted": weighted,
		"hbonds_strict_flexible": strict_flex,
	}


def run_apbs_deprotonation(
	pdb_path: str,
	residue_spec: str,
	radius: float,
	ph: float,
	pdb2pqr_timeout: int,
	apbs_timeout: int,
	reuse_apbs_cache: bool,
) -> Dict[str, float]:
	base_name = os.path.splitext(os.path.basename(pdb_path))[0]
	safe_residue = residue_spec.replace(":", "_")
	apbs_work_dir = os.path.join(os.path.dirname(pdb_path), "apbs_cache", f"{base_name}_{safe_residue}")
	os.makedirs(apbs_work_dir, exist_ok=True)

	cmd = [
		"python",
		APBS_SCRIPT,
		pdb_path,
		residue_spec,
		"--radius",
		str(radius),
		"--ph",
		str(ph),
		"--work-dir",
		apbs_work_dir,
		"--keep-files",
		"--pdb2pqr-timeout",
		str(pdb2pqr_timeout),
		"--apbs-timeout",
		str(apbs_timeout),
	]
	if reuse_apbs_cache:
		cmd.append("--reuse-existing")
	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		print(f"[APBS stdout]:\n{result.stdout[-3000:]}") 
		print(f"[APBS stderr]:\n{result.stderr[-3000:]}")  
		raise RuntimeError(f"APBS_Deprotonation failed: {result.stderr.strip()}")

	text = result.stdout
	potential = _extract_float(text, r"Potential at .*?:\s*([\d.+-]+)\s*kT/e")
	counts = {
		"arg_count": _extract_int(text, r"ARG:\s*(\d+)"),
		"lys_count": _extract_int(text, r"LYS:\s*(\d+)"),
		"asp_count": _extract_int(text, r"ASP:\s*(\d+)"),
		"glu_count": _extract_int(text, r"GLU:\s*(\d+)"),
		"his_count": _extract_int(text, r"HIS:\s*(\d+)"),
	}

	return {
		"electrostatic_potential": potential,
		**counts,
	}


def parse_pdb_atoms(pdb_path: str):
	atoms = []
	with open(pdb_path) as f:
		for line in f:
			if not line.startswith(("ATOM", "HETATM")):
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
				})
			except (ValueError, IndexError):
				continue
	return atoms


def find_target_atom(atoms, chain: str, resname: str, resseq: int):
	target_atom = TARGET_ATOM.get(resname.upper())
	if target_atom is None:
		target_atom = "CA"
	for atom in atoms:
		chain_match = (chain is None) or (atom["chain"] == chain)
		if (
			chain_match
			and atom["resname"].upper() == resname.upper()
			and atom["resseq"] == resseq
			and atom["name"].upper() == target_atom.upper()
		):
			return atom
	return None


def get_sphere_atoms(atoms, center_xyz, radius: float):
	cx, cy, cz = center_xyz
	result = []
	for atom in atoms:
		dx = atom["x"] - cx
		dy = atom["y"] - cy
		dz = atom["z"] - cz
		if (dx * dx + dy * dy + dz * dz) <= radius * radius:
			result.append(atom)
	return result


def compute_charged_residue_counts(pdb_path: str, chain: str, resname: str, resseq: int, radius: float) -> Dict[str, float]:
	atoms = parse_pdb_atoms(pdb_path)
	if not atoms:
		raise RuntimeError(f"No ATOM records found in {pdb_path}")
	atoms = [atom for atom in atoms if atom["alt"] in ("", "A")]
	target = find_target_atom(atoms, chain, resname, resseq)
	if target is None:
		raise RuntimeError(f"Could not find target atom for {chain}:{resname}:{resseq}")
	nuc_xyz = (target["x"], target["y"], target["z"])
	sphere_atoms = get_sphere_atoms(atoms, nuc_xyz, radius)
	sphere_residues = {(atom["chain"], atom["resname"], atom["resseq"]) for atom in sphere_atoms}
	counts = {"arg_count": 0, "lys_count": 0, "asp_count": 0, "glu_count": 0, "his_count": 0}
	for _, residue_name, _ in sphere_residues:
		residue_name_u = residue_name.upper()
		if residue_name_u in ("HIE", "HID", "HIP"):
			residue_name_u = "HIS"
		if residue_name_u == "ARG":
			counts["arg_count"] += 1
		elif residue_name_u == "LYS":
			counts["lys_count"] += 1
		elif residue_name_u == "ASP":
			counts["asp_count"] += 1
		elif residue_name_u == "GLU":
			counts["glu_count"] += 1
		elif residue_name_u == "HIS":
			counts["his_count"] += 1
	return counts


def _extract_float(text: str, pattern: str) -> float:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return np.nan
    val = m.group(1).strip()
    if val.lower() == "nan":
        return np.nan
    return float(val)


def _extract_int(text: str, pattern: str) -> int:
	m = re.search(pattern, text)
	if not m:
		raise ValueError(f"Pattern not found: {pattern}")
	return int(m.group(1))


def parse_pka_value(value) -> float:
	"""Parse pKa values that may include qualifiers like '>11' or '<4.5'."""
	if pd.isna(value):
		return np.nan
	text = str(value).strip()
	text = text.replace(">", "").replace("<", "")
	try:
		return float(text)
	except ValueError:
		return np.nan
	
def compute_sasa_freesasa(pdb_path: str, chain: str, resseq: int, resname: str) -> Optional[float]:
	try:
		import freesasa
	except ImportError:
		print("[sasa] freesasa not installed. Run: pip install freesasa")
		return np.nan

	try:
		structure = freesasa.Structure(pdb_path)
		result = freesasa.calc(structure)
		rsa = result.residueAreas()  # method on result, not module

		rows = []
		for chain_id, residues in rsa.items():
			for res_id, areas in residues.items():
				res_num_str = re.sub(r'[^0-9-]', '', res_id.strip())
				if not res_num_str:
					continue
				r_resseq = int(res_num_str)
				r_resname = areas.residueType.strip()
				# relativeSideChain is 0-1, multiply by 100 to match Total-Side REL
				side_rel = areas.relativeSideChain
				side_rel = np.nan if side_rel is None else side_rel * 100.0
				rows.append((chain_id, r_resseq, r_resname, side_rel))

		sasa_df = pd.DataFrame(rows, columns=["chain", "resnum", "resname", "side_sasa"])

		pdb_base = os.path.splitext(pdb_path)[0]
		out_path = f"{pdb_base}_deprot_sasa.csv"
		sasa_df.to_csv(out_path, index=False)
		print(f"[sasa] Saved computed SASA to {out_path}")

		match = sasa_df[
			(sasa_df["chain"].astype(str) == str(chain)) &
			(sasa_df["resnum"].astype(int) == int(resseq)) &
			(sasa_df["resname"].str.upper() == resname.upper())
		]
		if match.empty:
			print(f"[sasa] Residue {chain}:{resname}:{resseq} not found after freesasa computation")
			return np.nan
		return float(match.iloc[0]["side_sasa"])

	except Exception as exc:
		print(f"[sasa] freesasa computation failed for {chain}:{resname}:{resseq}: {exc}")
		return np.nan


def load_sasa_from_csv(pdb_path: str, chain: str, resseq: int, resname: str) -> Optional[float]:
    pdb_dir = os.path.dirname(pdb_path)
    pdb_base = os.path.splitext(os.path.basename(pdb_path))[0]
    candidates = [
		os.path.join(pdb_dir, f"{pdb_base}_deprot_sasa.csv"),
        os.path.join(pdb_dir, f"{pdb_base}_sasa.csv"),
        os.path.join(pdb_dir, f"{pdb_base}_pdb_sasa.csv"),
        os.path.join(pdb_dir, "pdb_sasa.csv"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        chain_col = cols.get("chain")
        resnum_col = cols.get("resnum") or cols.get("resseq") or cols.get("resid")
        resname_col = cols.get("resname")
        # Accept side_sasa (computed) or Total-Side REL (downloaded freesasa output)
        sasa_col = (cols.get("side_sasa") or
                    cols.get("total-side rel") or
                    cols.get("total_side_rel"))
        if not all([chain_col, resnum_col, resname_col, sasa_col]):
            continue

        match = df[
            (df[chain_col].astype(str) == str(chain)) &
            (df[resnum_col].astype(int) == int(resseq)) &
            (df[resname_col].str.upper() == resname.upper())
        ]
        if not match.empty:
            return float(match.iloc[0][sasa_col])

    # No CSV found — fall back to freesasa
    print(f"[sasa] No CSV found for {chain}:{resname}:{resseq}, computing via freesasa...")
    return compute_sasa_freesasa(pdb_path, chain, resseq, resname)

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

def build_feature_row(
	row: pd.Series,
	pdb_dir: str,
	apbs_radius: float,
	hbond_radius: float,
	ph: float,
	pdb2pqr_timeout: int,
	apbs_timeout: int,
	reuse_apbs_cache: bool,
	use_apbs: bool,
) -> Optional[Dict[str, float]]:
	pdb_path = find_pdb_path(row["pdb"], pdb_dir)
	chain = str(row["chain"]).strip()
	resseq = int(row["resseq"])
	resname = str(row["resname"]).strip().upper()
	residue_spec = f"{chain}:{resname}:{resseq}"

	sasa = load_sasa_from_csv(pdb_path, chain, resseq, resname)

	base, ext = os.path.splitext(pdb_path)
	clean_pdb = base + "_model1.pdb"
	if not os.path.exists(clean_pdb):
		extract_and_renumber_model1(pdb_path, clean_pdb)
	if use_apbs:
		try:
			apbs = run_apbs_deprotonation(
				clean_pdb,
				residue_spec,
				apbs_radius,
				ph,
				pdb2pqr_timeout,
				apbs_timeout,
				reuse_apbs_cache,
			)
		except RuntimeError as exc:
			print(f"[APBS] FAILED for {residue_spec}: {exc} -- using count-only fallback")
			apbs = {
				"electrostatic_potential": np.nan,
				**compute_charged_residue_counts(clean_pdb, chain, resname, resseq, apbs_radius),
			}
	else:
		apbs = {
			"electrostatic_potential": np.nan,
			**compute_charged_residue_counts(clean_pdb, chain, resname, resseq, apbs_radius),
		}
	hbonds = run_hbonds_score(clean_pdb, residue_spec, hbond_radius)

	return {
		"resname": resname,
		"ref_pka": REFERENCE_PKA.get(resname, np.nan),
		"sasa": sasa,
		**apbs,
		**hbonds,
	}

def log1p_signed(x):
    """log1p for positive values, pass-through for negative (e.g. electrostatic potential)."""
    return np.where(x < 0, x, np.log1p(x))


def safe_pearson(y_true, y_pred):
	try:
		if len(y_true) < 2 or np.allclose(y_true, y_true.iloc[0]) or np.allclose(y_pred, y_pred[0]):
			return np.nan, np.nan
		return stats.pearsonr(y_true, y_pred)
	except Exception:
		return np.nan, np.nan


def safe_spearman(y_true, y_pred):
	try:
		if len(y_true) < 2 or np.allclose(y_true, y_true.iloc[0]) or np.allclose(y_pred, y_pred[0]):
			return np.nan, np.nan
		return stats.spearmanr(y_true, y_pred)
	except Exception:
		return np.nan, np.nan


def safe_auc(y_true_bin, y_pred):
	try:
		if len(np.unique(y_true_bin)) < 2:
			return np.nan
		return roc_auc_score(y_true_bin, y_pred)
	except Exception:
		return np.nan


def make_group_splits(groups, n_splits: int, shuffle_groups: bool, random_state: int):
	group_series = pd.Series(groups).reset_index(drop=True)
	group_counts = group_series.value_counts()
	unique_groups = list(group_counts.index)
	rng = np.random.default_rng(random_state)

	if shuffle_groups:
		random_tiebreak = {group: rng.random() for group in unique_groups}
		ordered_groups = sorted(
			unique_groups,
			key=lambda group: (-group_counts[group], random_tiebreak[group]),
		)
	else:
		ordered_groups = sorted(unique_groups, key=lambda group: (-group_counts[group], str(group)))

	fold_group_lists = [[] for _ in range(n_splits)]
	fold_sizes = np.zeros(n_splits, dtype=int)

	for group in ordered_groups:
		fold_idx = int(np.argmin(fold_sizes))
		fold_group_lists[fold_idx].append(group)
		fold_sizes[fold_idx] += int(group_counts[group])

	splits = []
	for fold_groups in fold_group_lists:
		test_mask = group_series.isin(fold_groups).to_numpy()
		test_idx = np.flatnonzero(test_mask)
		train_idx = np.flatnonzero(~test_mask)
		splits.append((train_idx, test_idx))

	return splits

def main():
	parser = argparse.ArgumentParser(description="Train deprotonation probability model.")
	parser.add_argument("pkadr_csv", help="Path to PKAD-R CSV")
	parser.add_argument("--pdb-dir", default=".", help="Directory containing PDB files")
	parser.add_argument("--apbs-radius", type=float, default=12.0, help="APBS sphere radius")
	parser.add_argument("--hbond-radius", type=float, default=6.0, help="H-bond search radius")
	parser.add_argument("--ph", type=float, default=PHYSIOLOGIC_PH, help="Physiologic pH")
	parser.add_argument("--pdb2pqr-timeout", type=int, default=600, help="Timeout in seconds per pdb2pqr attempt")
	parser.add_argument("--apbs-timeout", type=int, default=900, help="Timeout in seconds for APBS solve")
	parser.add_argument("--reuse-apbs-cache", action="store_true", help="Reuse cached APBS intermediates per residue")
	parser.add_argument("--no-apbs", action="store_true", help="Skip the APBS solve and train on count-only features")
	parser.add_argument("--use-cached-fold-features", action="store_true", help="Reuse previously saved per-fold train/test feature CSVs instead of rebuilding features")
	parser.add_argument("--feature-cache-stem", default=None, help="Stem used to locate cached fold feature CSVs (defaults to the current model stem)")
	parser.add_argument("--shuffle-groups", action="store_true", help="Randomize group assignment before building folds")
	parser.add_argument("--random-state", type=int, default=7, help="Random seed used when shuffling groups")
	parser.add_argument("--model-out", default="deprot_xgb.json", help="Output model file")
	args = parser.parse_args()

	df = pd.read_csv(args.pkadr_csv)
	# Preserve original PKAD row position for traceability after filtering.
	df["pkadr_rownum"] = np.arange(len(df))
	pdb_col, chain_col, resid_col, resname_col, pka_col = parse_pkadr_columns(df)

	df = df.rename(columns={
		pdb_col: "pdb",
		chain_col: "chain",
		resid_col: "resseq",
		resname_col: "resname",
		pka_col: "pka",
	})

	df = df.dropna(subset=["pdb", "chain", "resseq", "resname", "pka"])
	df["pka"] = df["pka"].apply(parse_pka_value)
	df = df.dropna(subset=["pka"])

	# Skip any row with a warning
	if "Warning" in df.columns:
		df["Warning"] = df["Warning"].astype(str)
		df = df[df["Warning"].isna() | (df["Warning"].str.strip() == "") | (df["Warning"].str.strip() == "nan")].reset_index(drop=True)

	print(f"[filter] {len(df)} samples remaining after filtering warnings.")


	output_dir = os.path.dirname(args.model_out) or "."
	model_stem = os.path.splitext(os.path.basename(args.model_out))[0]
	results_csv = os.path.join(output_dir, f"{model_stem}_cv_results.csv")
	oof_csv = os.path.join(output_dir, f"{model_stem}_oof_predictions.csv")
	metadata_csv = os.path.join(output_dir, f"{model_stem}_row_metadata.csv")
	cache_stem = args.feature_cache_stem or model_stem

	if args.use_cached_fold_features:
		print(f"[build] Reusing cached fold features from stem '{cache_stem}'")
		cached_test_frames = []
		for fold_num in range(1, 11):
			train_cache = os.path.join(output_dir, f"{cache_stem}_fold{fold_num:02d}_train_features.csv")
			test_cache = os.path.join(output_dir, f"{cache_stem}_fold{fold_num:02d}_test_features.csv")
			if not os.path.isfile(train_cache):
				raise FileNotFoundError(f"Missing cached training features: {train_cache}")
			if not os.path.isfile(test_cache):
				raise FileNotFoundError(f"Missing cached test features: {test_cache}")
			cached_test_frames.append(pd.read_csv(test_cache))

		feat_df = pd.concat(cached_test_frames, ignore_index=True)
		if "row_id" not in feat_df.columns:
			raise ValueError("Cached fold feature CSVs must include a row_id column.")
		feat_df = feat_df.sort_values("row_id").drop_duplicates(subset=["row_id"], keep="first").reset_index(drop=True)
		if "row_id" not in feat_df.columns:
			feat_df["row_id"] = np.arange(len(feat_df))
		if "label" not in feat_df.columns:
			raise ValueError("Cached fold feature CSVs must include a label column.")
		print(f"[build] Loaded cached fold feature rows: {len(feat_df)}")
	else:
		feature_rows = []
		kept_indices = []
		skipped = 0
		for idx, row in df.iterrows():
			feature_row = build_feature_row(
				row,
				args.pdb_dir,
				args.apbs_radius,
				args.hbond_radius,
				args.ph,
				args.pdb2pqr_timeout,
				args.apbs_timeout,
				args.reuse_apbs_cache,
				not args.no_apbs,
			)
			if feature_row is None:
				skipped += 1
				continue
			feature_rows.append(feature_row)
			kept_indices.append(idx)

		print(f"[build] {len(feature_rows)} rows built, {skipped} skipped due to APBS errors.")
		if not feature_rows:
			raise SystemExit("No features could be computed. Check APBS/pdb2pqr installation and timeouts.")

		feat_df = pd.DataFrame(feature_rows)
		kept_df = df.loc[kept_indices].reset_index(drop=True)
		labels = kept_df["pka"].apply(lambda x: compute_deprot_probability(x, args.ph))
		feat_df["label"] = labels.values
		feat_df["pdb"] = kept_df["pdb"].values  # for grouped splitting
		feat_df["chain"] = kept_df["chain"].astype(str).str.strip().values
		feat_df["resseq"] = kept_df["resseq"].values
		feat_df["expt_pka"] = kept_df["pka"].values
		feat_df["pkadr_rownum"] = kept_df["pkadr_rownum"].values
		if "Index" in kept_df.columns:
			feat_df["pkadr_index"] = kept_df["Index"].values
		feat_df["row_id"] = np.arange(len(feat_df))
      
	feature_cols = [
		"ref_pka",
		"sasa",
		"arg_count",
		"lys_count",
		"asp_count",
		"glu_count",
		"his_count",
		"hbonds_weighted",
		"hbonds_strict_flexible",
		"resname",
	]
	if not args.no_apbs:
		feature_cols.insert(2, "electrostatic_potential")

	X = feat_df[feature_cols]
	y = feat_df["label"]

	numeric_features = [
		"ref_pka",
		"sasa",
		"arg_count",
		"lys_count",
		"asp_count",
		"glu_count",
		"his_count",
		"hbonds_weighted",
		"hbonds_strict_flexible",
	]
	if not args.no_apbs:
		numeric_features.insert(2, "electrostatic_potential")

	categorical_features = ["resname"]

	numeric_transform = Pipeline(steps=[
		("log1p_counts", FunctionTransformer(log1p_signed, feature_names_out="one-to-one")),
		("scaler", StandardScaler()),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transform, numeric_features),
			("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
		],
		remainder="drop",
	)

	model = XGBRegressor(
		n_estimators=300,
		max_depth=4,
		learning_rate=0.05,
		subsample=0.9,
		colsample_bytree=0.9,
		objective="reg:squarederror",
		random_state=7,
	)

	n_splits = 10
	group_counts = feat_df.groupby("pdb").size()
	if group_counts.size < n_splits:
		raise ValueError(
			f"Need at least {n_splits} unique PDB groups for {n_splits}-fold CV; found {group_counts.size}."
		)
	if args.shuffle_groups:
		print(f"[cv] Shuffling group assignment with random_state={args.random_state}")
	else:
		print("[cv] Using deterministic group assignment (no shuffle)")

	metadata_cols = [
		"row_id",
		"pdb",
		"chain",
		"resseq",
		"resname",
		"expt_pka",
		"label",
		"pkadr_rownum",
	]
	if "pkadr_index" in feat_df.columns:
		metadata_cols.append("pkadr_index")
	if not args.use_cached_fold_features:
		feat_df[metadata_cols].to_csv(metadata_csv, index=False)
		print(f"Saved row metadata to {metadata_csv}")
	elif os.path.isfile(metadata_csv):
		print(f"[build] Using existing metadata file {metadata_csv}")

	if args.use_cached_fold_features:
		splits = []
		for fold_num in range(1, n_splits + 1):
			test_cache = os.path.join(output_dir, f"{cache_stem}_fold{fold_num:02d}_test_features.csv")
			test_df = pd.read_csv(test_cache)
			test_row_ids = set(test_df["row_id"].tolist()) if "row_id" in test_df.columns else set()
			if test_row_ids:
				test_mask = feat_df["row_id"].isin(test_row_ids).to_numpy()
			else:
				test_mask = np.zeros(len(feat_df), dtype=bool)
			train_idx = np.flatnonzero(~test_mask)
			test_idx = np.flatnonzero(test_mask)
			splits.append((train_idx, test_idx))
	else:
		splits = make_group_splits(feat_df["pdb"], n_splits, args.shuffle_groups, args.random_state)
	fold_results = []
	oof_rows = []

	for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
		X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
		y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
		train_cache = os.path.join(output_dir, f"{model_stem}_fold{fold_num:02d}_train_features.csv")
		test_cache = os.path.join(output_dir, f"{model_stem}_fold{fold_num:02d}_test_features.csv")

		feat_df.iloc[train_idx].to_csv(train_cache, index=False)
		feat_df.iloc[test_idx].to_csv(test_cache, index=False)

		pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
		pipeline.fit(X_train, y_train)
		preds = pipeline.predict(X_test)

		rmse = math.sqrt(mean_squared_error(y_test, preds))
		r2 = r2_score(y_test, preds)
		mae = np.mean(np.abs(y_test - preds))
		pearson_r, pearson_p = safe_pearson(y_test, preds)
		spearman_r, spearman_p = safe_spearman(y_test, preds)
		y_test_bin = (y_test >= 0.5).astype(int)
		preds_bin = (preds >= 0.5).astype(int)
		mcc = matthews_corrcoef(y_test_bin, preds_bin)
		auc = safe_auc(y_test_bin, preds)
		tn, fp, fn, tp = confusion_matrix(y_test_bin, preds_bin, labels=[0, 1]).ravel()

		fold_results.append({
			"fold": fold_num,
			"train_samples": len(train_idx),
			"test_samples": len(test_idx),
			"rmse": rmse,
			"mae": mae,
			"r2": r2,
			"pearson_r": pearson_r,
			"pearson_p": pearson_p,
			"spearman_r": spearman_r,
			"spearman_p": spearman_p,
			"auc_roc": auc,
			"mcc": mcc,
			"tn": tn,
			"fp": fp,
			"fn": fn,
			"tp": tp,
		})

		fold_oof = feat_df.iloc[test_idx].copy()
		fold_oof["fold"] = fold_num
		fold_oof["prediction"] = preds
		oof_rows.append(fold_oof)

		print(f"\n{'='*50}")
		print(f"  FOLD {fold_num:02d} / {n_splits}")
		print(f"{'='*50}")
		print(f"  Samples (train/test): {len(train_idx)} / {len(test_idx)}")
		print(f"  Regression metrics:")
		print(f"    RMSE:               {rmse:.4f}")
		print(f"    MAE:                {mae:.4f}")
		print(f"    R²:                 {r2:.4f}")
		print(f"    Pearson r:          {pearson_r:.4f}  (p={pearson_p:.2e})")
		print(f"    Spearman ρ:         {spearman_r:.4f}  (p={spearman_p:.2e})")
		print(f"  Classification metrics (threshold=0.5):")
		print(f"    AUC-ROC:            {auc:.4f}")
		print(f"    MCC:                {mcc:.4f}")
		print(f"    TP/TN/FP/FN:        {tp}/{tn}/{fp}/{fn}")

	results_df = pd.DataFrame(fold_results)
	results_df.to_csv(results_csv, index=False)
	print(f"\nSaved fold metrics to {results_csv}")

	oof_df = pd.concat(oof_rows, ignore_index=True).sort_values("row_id").reset_index(drop=True)
	oof_df.to_csv(oof_csv, index=False)
	print(f"Saved out-of-fold predictions to {oof_csv}")

	oof_y = oof_df["label"]
	oof_preds = oof_df["prediction"]
	oof_rmse = math.sqrt(mean_squared_error(oof_y, oof_preds))
	oof_r2 = r2_score(oof_y, oof_preds)
	oof_mae = np.mean(np.abs(oof_y - oof_preds))
	oof_pearson_r, oof_pearson_p = safe_pearson(oof_y, oof_preds)
	oof_spearman_r, oof_spearman_p = safe_spearman(oof_y, oof_preds)
	oof_y_bin = (oof_y >= 0.5).astype(int)
	oof_preds_bin = (oof_preds >= 0.5).astype(int)
	oof_mcc = matthews_corrcoef(oof_y_bin, oof_preds_bin)
	oof_auc = safe_auc(oof_y_bin, oof_preds)
	oof_tn, oof_fp, oof_fn, oof_tp = confusion_matrix(oof_y_bin, oof_preds_bin, labels=[0, 1]).ravel()

	print(f"\n{'='*50}")
	print(f"  CROSS-VALIDATION SUMMARY")
	print(f"{'='*50}")
	print(f"  Folds:              {n_splits}")
	print(f"  OOF samples:        {len(oof_df)}")
	print(f"  Regression metrics:")
	print(f"    RMSE:               {oof_rmse:.4f}")
	print(f"    MAE:                {oof_mae:.4f}")
	print(f"    R²:                 {oof_r2:.4f}")
	print(f"    Pearson r:          {oof_pearson_r:.4f}  (p={oof_pearson_p:.2e})")
	print(f"    Spearman ρ:         {oof_spearman_r:.4f}  (p={oof_spearman_p:.2e})")
	print(f"  Classification metrics (threshold=0.5):")
	print(f"    AUC-ROC:            {oof_auc:.4f}")
	print(f"    MCC:                {oof_mcc:.4f}")
	print(f"    TP/TN/FP/FN:        {oof_tp}/{oof_tn}/{oof_fp}/{oof_fn}")
	print(f"{'='*50}\n")

	# --- Save final model fit on all data --- #
	final_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
	final_pipeline.fit(X, y)
	final_pipeline.named_steps["model"].save_model(args.model_out)
	print(f"Saved XGBoost model (JSON) to {args.model_out}")

	pkl_out = args.model_out.replace(".json", ".pkl")
	joblib.dump(final_pipeline, pkl_out)
	print(f"Saved full pipeline (pkl)  to {pkl_out}")


if __name__ == "__main__":
	main()
