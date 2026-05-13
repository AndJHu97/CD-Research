"""
Deprotonation_Model.py

Train an XGBoost regressor to predict deprotonation probability from features:
- Physiologic pH (default 7.4)
- H-bond weighted score (from HBonds_Score.py)
- Charged residue counts (ARG/LYS/ASP/GLU) near the site (from APBS_Deprotonation.py)
- Electrostatic potential (from APBS_Deprotonation.py)
- SASA (from pdb_sasa file or computed if missing)

Label: P(deprot) = 1 / (1 + 10^(pKa - pH)) from PKAD-R Expt. pKa
"""

import argparse
import math
import os
import re
import subprocess
from unittest import result
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
	from xgboost import XGBRegressor
except ImportError as exc:
	raise SystemExit("xgboost is required. Install with: pip install xgboost") from exc


PHYSIOLOGIC_PH = 7.4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APBS_SCRIPT = os.path.join(SCRIPT_DIR, "APBS_Deprotonation.py")
HBOND_SCRIPT = os.path.join(SCRIPT_DIR, "HBonds_Score.py")


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


def run_apbs_deprotonation(pdb_path: str, residue_spec: str, radius: float, ph: float) -> Dict[str, float]:
	cmd = [
		"python",
		APBS_SCRIPT,
		pdb_path,
		residue_spec,
		"--radius",
		str(radius),
		"--ph",
		str(ph),
		"--keep-files",
	]
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
	}

	return {
		"electrostatic_potential": potential,
		**counts,
	}


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


def load_sasa_from_csv(pdb_path: str, chain: str, resseq: int, resname: str) -> Optional[float]:
	pdb_dir = os.path.dirname(pdb_path)
	pdb_base = os.path.splitext(os.path.basename(pdb_path))[0]
	candidates = [
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
		sasa_col = cols.get("sasa") or cols.get("total_sasa") or cols.get("side_sasa")
		if not all([chain_col, resnum_col, resname_col, sasa_col]):
			continue

		match = df[
			(df[chain_col].astype(str) == str(chain)) &
			(df[resnum_col].astype(int) == int(resseq)) &
			(df[resname_col].str.upper() == resname.upper())
		]
		if not match.empty:
			return float(match.iloc[0][sasa_col])
	return None


def build_feature_row(row: pd.Series, pdb_dir: str, apbs_radius: float, hbond_radius: float, ph: float) -> Dict[str, float]:
	pdb_path = find_pdb_path(row["pdb"], pdb_dir)
	chain = str(row["chain"]).strip()
	resseq = int(row["resseq"])
	resname = str(row["resname"]).strip().upper()
	residue_spec = f"{chain}:{resname}:{resseq}"

	sasa = load_sasa_from_csv(pdb_path, chain, resseq, resname)
	if sasa is None:
		# If SASA isn't available, leave NaN; the model will ignore missing values.
		sasa = np.nan

	apbs = run_apbs_deprotonation(pdb_path, residue_spec, apbs_radius, ph)
	hbonds = run_hbonds_score(pdb_path, residue_spec, hbond_radius)

	return {
		"resname": resname,
		"ph": ph,
		"sasa": sasa,
		**apbs,
		**hbonds,
	}


def main():
	parser = argparse.ArgumentParser(description="Train deprotonation probability model.")
	parser.add_argument("pkadr_csv", help="Path to PKAD-R CSV")
	parser.add_argument("--pdb-dir", default=".", help="Directory containing PDB files")
	parser.add_argument("--apbs-radius", type=float, default=12.0, help="APBS sphere radius")
	parser.add_argument("--hbond-radius", type=float, default=6.0, help="H-bond search radius")
	parser.add_argument("--ph", type=float, default=PHYSIOLOGIC_PH, help="Physiologic pH")
	parser.add_argument("--model-out", default="deprot_xgb.json", help="Output model file")
	args = parser.parse_args()

	df = pd.read_csv(args.pkadr_csv)
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
		df = df[df["Warning"].isna() | (df["Warning"].str.strip() == "")].reset_index(drop=True)

	print(f"[filter] {len(df)} samples remaining after filtering warnings.")


	labels = df["pka"].apply(lambda x: compute_deprot_probability(x, args.ph))

	feature_rows = []
	for _, row in df.iterrows():
		feature_rows.append(build_feature_row(row, args.pdb_dir, args.apbs_radius, args.hbond_radius, args.ph))

	
	feat_df = pd.DataFrame(feature_rows)
	feat_df["label"] = labels.values

	feature_cols = [
		"ph",
		"sasa",
		"electrostatic_potential",
		"arg_count",
		"lys_count",
		"asp_count",
		"glu_count",
		"hbonds_weighted",
		"hbonds_strict_flexible",
		"resname",
	]

	X = feat_df[feature_cols]
	y = feat_df["label"]

	numeric_features = [
		"ph",
		"sasa",
		"electrostatic_potential",
		"arg_count",
		"lys_count",
		"asp_count",
		"glu_count",
		"hbonds_weighted",
		"hbonds_strict_flexible",
	]

	categorical_features = ["resname"]

	numeric_transform = Pipeline(steps=[
		("log1p_counts", FunctionTransformer(lambda x: np.where(x < 0, x, np.log1p(x)), feature_names_out="one-to-one")),
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

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

	pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)
	print(f"RMSE: {math.sqrt(mean_squared_error(y_test, preds)):.4f}")
	print(f"R2: {r2_score(y_test, preds):.4f}")

	pipeline.named_steps["model"].save_model(args.model_out)
	print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
	main()
