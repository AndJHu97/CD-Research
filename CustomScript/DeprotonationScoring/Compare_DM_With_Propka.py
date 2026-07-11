"""
Compare deprotonation model outputs with PROPKA-derived pKa values.

This utility looks for the files produced by Train_Deprotonation_Model.py:
- <prefix>_cv_results.csv
- <prefix>_oof_predictions.csv
- <prefix>_foldXX_test_features.csv

It then tries to locate matching PROPKA .pka files under a user-specified root,
joins residue-level PROPKA pKa values onto the OOF table, and writes augmented
comparison CSVs alongside a fold-level summary.

Typical usage:
	python Compare_DM_With_Propka.py --prefix deprot_xgb --propka-root apbs_cache
	python Compare_DM_With_Propka.py --prefix deprot_xgb --propka-root . --ph 7.4
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_curve
try:
	from scipy.ndimage import uniform_filter1d
except ImportError:
	uniform_filter1d = None

NUCLEOPHILIC_RESIDUES = {"CYS", "SER", "THR", "TYR", "LYS", "HIS"}


def compute_deprot_probability(pka: float, ph: float) -> float:
	return 1.0 / (1.0 + 10 ** (pka - ph))


def deprot_probability_to_pka(probability: float, ph: float) -> float:
	"""Invert deprotonation probability back to pKa at a given pH.

	P(deprot) = 1 / (1 + 10^(pKa - pH))
	pKa = pH + log10((1 - P) / P)
	"""
	if pd.isna(probability):
		return np.nan
	p = float(probability)
	if p <= 0.0 or p >= 1.0:
		return np.nan
	return float(ph + np.log10((1.0 - p) / p))


def parse_pka_value(value) -> float:
	"""Parse pKa values that may include qualifiers like '>11' or '<4.5'."""
	if pd.isna(value):
		return np.nan
	text = str(value).strip()
	text = text.replace(">", "").replace("<", "").replace("~", "")
	try:
		return float(text)
	except ValueError:
		return np.nan


def parse_pkadr_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
	"""Heuristic lookup for PKAD-R-style column names."""
	colmap = {c.lower(): c for c in df.columns}
	pdb_col = colmap.get("pdb", None)
	chain_col = colmap.get("chain", None)
	resid_col = colmap.get("resid in pdb", None) or colmap.get("resid", None) or colmap.get("resnum", None)
	resname_col = colmap.get("resname", None) or colmap.get("res name", None)
	pka_col = colmap.get("expt. pka", None) or colmap.get("expt pka", None) or colmap.get("pka", None)
	return pdb_col, chain_col, resid_col, resname_col, pka_col


def normalize_pdb_stem(value: str) -> str:
	if pd.isna(value):
		return ""
	return os.path.splitext(os.path.basename(str(value)))[0].upper()


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
	if len(x) < 2:
		return np.nan
	if np.allclose(x, x.iloc[0]) or np.allclose(y, y.iloc[0]):
		return np.nan
	return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
	if len(x) < 2:
		return np.nan
	if np.allclose(x, x.iloc[0]) or np.allclose(y, y.iloc[0]):
		return np.nan
	return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def build_state_column(df: pd.DataFrame, pka_column: str, threshold: float) -> pd.Series:
	return df[pka_column].apply(lambda value: bool(value < threshold) if pd.notna(value) else np.nan)


def filter_nucleophilic_residues(df: pd.DataFrame) -> pd.DataFrame:
	if "resname" not in df.columns:
		return df.copy()
	mask = df["resname"].astype(str).str.upper().isin(NUCLEOPHILIC_RESIDUES)
	return df.loc[mask].copy()


def compute_threshold_sweep_metrics(df: pd.DataFrame, thresholds: np.ndarray, subset_mask: Optional[pd.Series] = None) -> pd.DataFrame:
	if subset_mask is None:
		subset_mask = pd.Series(True, index=df.index)
	else:
		subset_mask = pd.Series(subset_mask, index=df.index)

	work = df.loc[subset_mask].copy()
	if work.empty:
		return pd.DataFrame()
	if "experimental_pka" not in work.columns and "expt_pka" in work.columns:
		work["experimental_pka"] = work["expt_pka"]
	if "experimental_pka" not in work.columns and "label" in work.columns:
		work["experimental_pka"] = work["label"].apply(lambda value: deprot_probability_to_pka(value, 7.4) if pd.notna(value) else np.nan)
	if "propka_pka" not in work.columns and "propka_deprot_probability" in work.columns:
		work["propka_pka"] = work["propka_deprot_probability"].apply(lambda value: deprot_probability_to_pka(value, 7.4) if pd.notna(value) else np.nan)

	rows = []
	for threshold in thresholds:
		truth = build_state_column(work, "experimental_pka", threshold)
		model_probability_threshold = compute_deprot_probability(threshold, 7.4)
		for method_name, pka_col in (("model", "model_predicted_pka"), ("propka", "propka_pka"), ("ref", "ref_pka")):
			if method_name == "model":
				if "prediction" not in work.columns:
					continue
				pred = work["prediction"].apply(lambda value: bool(value >= model_probability_threshold) if pd.notna(value) else np.nan)
			else:
				if pka_col not in work.columns:
					continue
				pred = build_state_column(work, pka_col, threshold)
			valid = truth.notna() & pred.notna()
			if valid.sum() == 0:
				metrics = {"threshold": threshold, "method": method_name, "n": 0, "accuracy": np.nan, "sensitivity": np.nan, "specificity": np.nan, "mcc": np.nan}
			else:
				y_true = truth.loc[valid].astype(int)
				y_pred = pred.loc[valid].astype(int)
				tp = int(((y_true == 1) & (y_pred == 1)).sum())
				tn = int(((y_true == 0) & (y_pred == 0)).sum())
				fp = int(((y_true == 0) & (y_pred == 1)).sum())
				fn = int(((y_true == 1) & (y_pred == 0)).sum())
				accuracy = float((tp + tn) / len(y_true))
				sensitivity = float(tp / (tp + fn)) if (tp + fn) else np.nan
				specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
				mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else np.nan
				metrics = {"threshold": threshold, "method": method_name, "n": int(len(y_true)), "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, "mcc": mcc}
			rows.append(metrics)
	return pd.DataFrame(rows)


def build_residue_accuracy_sweep(df: pd.DataFrame, thresholds: np.ndarray, residues: Iterable[str]) -> pd.DataFrame:
	work = df.copy()
	if "experimental_pka" not in work.columns and "expt_pka" in work.columns:
		work["experimental_pka"] = work["expt_pka"]
	if "experimental_pka" not in work.columns and "label" in work.columns:
		work["experimental_pka"] = work["label"].apply(lambda value: deprot_probability_to_pka(value, 7.4) if pd.notna(value) else np.nan)
	if "propka_pka" not in work.columns and "propka_deprot_probability" in work.columns:
		work["propka_pka"] = work["propka_deprot_probability"].apply(lambda value: deprot_probability_to_pka(value, 7.4) if pd.notna(value) else np.nan)

	rows = []
	for residue in residues:
		res_df = work[work["resname"].astype(str).str.upper() == residue.upper()].copy()
		if res_df.empty:
			continue
		for threshold in thresholds:
			truth = build_state_column(res_df, "experimental_pka", threshold)
			model_probability_threshold = compute_deprot_probability(threshold, 7.4)
			for method_name, pka_col in (("model", "model_predicted_pka"), ("propka", "propka_pka"), ("ref", "ref_pka")):
				if method_name == "model":
					if "prediction" not in res_df.columns:
						continue
					pred = res_df["prediction"].apply(lambda value: bool(value >= model_probability_threshold) if pd.notna(value) else np.nan)
				else:
					if pka_col not in res_df.columns:
						continue
					pred = build_state_column(res_df, pka_col, threshold)
				valid = truth.notna() & pred.notna()
				if valid.sum() == 0:
					accuracy = np.nan
				else:
					accuracy = float((truth.loc[valid].astype(int) == pred.loc[valid].astype(int)).mean())
				rows.append({"residue": residue, "threshold": threshold, "method": method_name, "accuracy": accuracy, "n": int(valid.sum())})
	return pd.DataFrame(rows)


def compute_roc_curve_metrics(df: pd.DataFrame, subset_mask: Optional[pd.Series] = None, ph: float = 7.4) -> pd.DataFrame:
	work = df.copy() if subset_mask is None else df.loc[pd.Series(subset_mask, index=df.index)].copy()
	if work.empty:
		return pd.DataFrame()
	if "experimental_pka" not in work.columns and "expt_pka" in work.columns:
		work["experimental_pka"] = work["expt_pka"]
	if "experimental_pka" not in work.columns and "label" in work.columns:
		work["experimental_pka"] = work["label"].apply(lambda value: deprot_probability_to_pka(value, ph) if pd.notna(value) else np.nan)
	if "propka_pka" not in work.columns and "propka_deprot_probability" in work.columns:
		work["propka_pka"] = work["propka_deprot_probability"].apply(lambda value: deprot_probability_to_pka(value, ph) if pd.notna(value) else np.nan)

	truth = work["experimental_pka"].apply(lambda value: bool(value < ph) if pd.notna(value) else np.nan)
	rows = []
	for method_name, score_col in (("model", "prediction"), ("propka", "propka_deprot_probability"), ("ref", "ref_pka")):
		if method_name == "model":
			if score_col not in work.columns:
				continue
			scores = work[score_col].astype(float)
		elif method_name == "propka":
			if score_col in work.columns:
				scores = work[score_col].apply(lambda value: compute_deprot_probability(value, ph) if pd.notna(value) else np.nan)
			elif "propka_pka" in work.columns:
				scores = work["propka_pka"].apply(lambda value: compute_deprot_probability(value, ph) if pd.notna(value) else np.nan)
			else:
				continue
		else:
			if score_col not in work.columns:
				continue
			scores = work[score_col].apply(lambda value: compute_deprot_probability(value, ph) if pd.notna(value) else np.nan)

		valid = truth.notna() & scores.notna()
		if valid.sum() < 2:
			continue
		y_true = truth.loc[valid].astype(int)
		y_score = scores.loc[valid].astype(float)
		if len(np.unique(y_true)) < 2:
			continue
		fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
		specificity = 1.0 - fpr
		for idx in range(len(thresholds)):
			rows.append(
				{
					"method": method_name,
					"threshold": float(thresholds[idx]),
					"specificity": float(specificity[idx]),
					"sensitivity": float(tpr[idx]),
					"n": int(valid.sum()),
				}
			)
	return pd.DataFrame(rows)


def plot_threshold_sweeps(df: pd.DataFrame, out_dir: str, prefix: str, ph_min: float = 5.0, ph_max: float = 9.0, num_points: int = 81) -> None:
	thresholds = np.linspace(ph_min, ph_max, num_points)
	sweep = compute_threshold_sweep_metrics(df, thresholds)
	if sweep.empty:
		print("[compare] WARNING: no data available for threshold sweep plots")
		return

	def _plot_side_by_side(metric: str, title: str, ylabel: str, filename: str, subset: Optional[pd.Series] = None):
		fig, ax = plt.subplots(figsize=(8, 5))
		plot_df = sweep if subset is None else sweep.loc[sweep["subset"] == subset]
		for method, style in (("model", "-"), ("propka", "--"), ("ref", ":")):
			method_df = plot_df[plot_df["method"] == method]
			if method_df.empty:
				continue
			ax.plot(method_df["threshold"], method_df[metric], style, linewidth=2, label=method.upper())
		ax.set_title(title)
		ax.set_xlabel("pH threshold")
		ax.set_ylabel(ylabel)
		ax.set_xlim(ph_min, ph_max)
		ax.grid(True, alpha=0.25)
		ax.legend()
		fig.tight_layout()
		fig.savefig(os.path.join(out_dir, filename), dpi=200)
		plt.close(fig)

	# Plot 1: overall accuracy and MCC
	fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
	for method, style in (("model", "-"), ("propka", "--"), ("ref", ":")):
		method_df = sweep[sweep["method"] == method]
		axes[0, 0].plot(method_df["threshold"], method_df["accuracy"], style, linewidth=2, label=method.upper())
		axes[0, 1].plot(method_df["threshold"], method_df["mcc"], style, linewidth=2, label=method.upper())
		axes[1, 0].plot(method_df["threshold"], method_df["sensitivity"], style, linewidth=2, label=method.upper())
		axes[1, 1].plot(method_df["threshold"], method_df["specificity"], style, linewidth=2, label=method.upper())
	axes[0, 0].set_title("Accuracy vs pH threshold")
	axes[0, 0].set_ylabel("Accuracy")
	axes[0, 1].set_title("MCC vs pH threshold")
	axes[0, 1].set_ylabel("MCC")
	axes[1, 0].set_title("Sensitivity vs pH threshold")
	axes[1, 0].set_ylabel("Sensitivity")
	axes[1, 1].set_title("Specificity vs pH threshold")
	axes[1, 1].set_ylabel("Specificity")
	for ax in axes.flat:
		ax.set_xlabel("pH threshold")
		ax.set_xlim(ph_min, ph_max)
		ax.grid(True, alpha=0.25)
		ax.legend()
	for ax in axes[0, :]:
		ax.tick_params(labelbottom=True)
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, f"{prefix}_threshold_overall.png"), dpi=200)
	plt.close(fig)

	# Plot 2: flipped subset sensitivity/specificity
	if "flip_state_group" in df.columns:
		flip_sweep = compute_threshold_sweep_metrics(df, thresholds, subset_mask=(df["flip_state_group"] == "flipped"))
		if not flip_sweep.empty:
			fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
			for method, style in (("model", "-"), ("propka", "--"), ("ref", ":")):
				method_df = flip_sweep[flip_sweep["method"] == method]
				axes[0].plot(method_df["threshold"], method_df["sensitivity"], style, linewidth=2, label=method.upper())
				axes[1].plot(method_df["threshold"], method_df["specificity"], style, linewidth=2, label=method.upper())
			axes[0].set_title("Flipped subset sensitivity")
			axes[0].set_ylabel("Sensitivity")
			axes[1].set_title("Flipped subset specificity")
			axes[1].set_ylabel("Specificity")
			for ax in axes:
				ax.set_xlabel("pH threshold")
				ax.set_xlim(ph_min, ph_max)
				ax.grid(True, alpha=0.25)
				ax.legend()
			fig.tight_layout()
			fig.savefig(os.path.join(out_dir, f"{prefix}_threshold_flipped_sens_spec.png"), dpi=200)
			plt.close(fig)

			# Plot 2b: overall and flipped ROC curves from native scores
			fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
			panel_specs = [
				("overall", compute_roc_curve_metrics(df, ph=7.4), "Overall ROC curve"),
				("flipped", compute_roc_curve_metrics(df, subset_mask=(df["flip_state_group"] == "flipped"), ph=7.4), "Flipped subset ROC curve"),
			]
			for ax, (panel_name, panel_df, panel_title) in zip(axes, panel_specs):
				for method, color in (("model", "tab:blue"), ("propka", "tab:orange"), ("ref", "tab:green")):
					method_df = panel_df[panel_df["method"] == method]
					if method_df.empty:
						continue
					x = method_df["specificity"].to_numpy(dtype=float)
					y = method_df["sensitivity"].to_numpy(dtype=float)
					mask = np.isfinite(x) & np.isfinite(y)
					if mask.sum() == 0:
						continue
					x = x[mask]
					y = y[mask]
					order = np.argsort(x)
					x = x[order]
					y = y[order]
					if method == "model" and len(x) >= 3:
						if uniform_filter1d is not None:
							x = uniform_filter1d(x, size=3, mode="nearest")
							y = uniform_filter1d(y, size=3, mode="nearest")
						else:
							kernel = np.ones(3, dtype=float) / 3.0
							x = np.convolve(np.pad(x, (1, 1), mode="edge"), kernel, mode="valid")
							y = np.convolve(np.pad(y, (1, 1), mode="edge"), kernel, mode="valid")
					line_style = "-" if panel_name == "overall" else "--"
					label = f"{method.upper()} {panel_name}"
					ax.plot(x, y, color=color, linewidth=2.5, linestyle=line_style, label=label)
					step = max(1, len(x) // 8)
					ax.scatter(x[::step], y[::step], color=color, s=18, alpha=0.55 if panel_name == "overall" else 0.8)
				ax.set_title(panel_title)
				ax.set_xlabel("Specificity")
				ax.set_ylabel("Sensitivity")
				ax.set_xlim(0.0, 1.0)
				ax.set_ylim(0.0, 1.0)
				ax.grid(True, alpha=0.25)
				ax.legend(fontsize=8)
			fig.tight_layout()
			fig.savefig(os.path.join(out_dir, f"{prefix}_threshold_flipped_roc.png"), dpi=200)
			plt.close(fig)

	# Plot 3: residue-specific accuracy
	residue_df = build_residue_accuracy_sweep(df, thresholds, residues=["CYS", "SER", "THR", "TYR", "LYS", "HIS"])
	if not residue_df.empty:
		available_residues = [residue for residue in ["CYS", "SER", "THR", "TYR", "LYS", "HIS"] if residue in set(residue_df["residue"].astype(str))]
		if not available_residues:
			return
		num_plots = len(available_residues)
		cols = min(3, num_plots)
		rows = int(np.ceil(num_plots / cols))
		fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
		axes_flat = np.atleast_1d(axes).flat
		for ax, residue in zip(axes_flat, available_residues):
			res_sub = residue_df[residue_df["residue"] == residue]
			for method, style in (("model", "-"), ("propka", "--"), ("ref", "-.")):
				method_df = res_sub[res_sub["method"] == method]
				ax.plot(method_df["threshold"], method_df["accuracy"], style, linewidth=2, label=method.upper())
			ax.set_title(f"{residue} accuracy")
			ax.set_xlabel("pH threshold")
			ax.grid(True, alpha=0.25)
			ax.legend(fontsize=8)
		for ax in np.atleast_1d(axes).flat:
			if hasattr(ax, "set_ylabel"):
				ax.set_ylabel("Accuracy")
		for ax in np.atleast_1d(axes).flat:
			ax.set_xlim(ph_min, ph_max)
		for ax in list(np.atleast_1d(axes).flat)[num_plots:]:
			ax.set_visible(False)
		fig.tight_layout()
		fig.savefig(os.path.join(out_dir, f"{prefix}_threshold_residue_accuracy.png"), dpi=200)
		plt.close(fig)


def locate_file(prefix: str, suffix: str) -> Optional[str]:
	candidate = f"{prefix}{suffix}"
	if os.path.exists(candidate):
		return os.path.abspath(candidate)
	return None


def locate_fold_feature_files(prefix: str) -> List[str]:
	pattern = f"{prefix}_fold*_test_features.csv"
	return sorted(os.path.abspath(path) for path in glob.glob(pattern))


def fold_number_from_path(path: str) -> Optional[int]:
	match = re.search(r"_fold(\d+)_test_features\.csv$", os.path.basename(path))
	if not match:
		return None
	return int(match.group(1))


def parse_propka_file(path: str) -> pd.DataFrame:
	"""Parse a PROPKA .pka file into a residue-level dataframe.

	The parser is intentionally tolerant. It accepts the common compact format
	used by the existing project, where a data line looks roughly like:
		CYS 33 A 8.12
	"""
	rows = []
	with open(path, "r") as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			parts = stripped.split()
			if len(parts) < 4:
				continue
			try:
				pka_value = float(parts[-1])
			except ValueError:
				continue

			resname = None
			resseq = None
			chain = ""

			# Preferred format used elsewhere in this repo: RESNAME RESSEQ CHAIN PKA
			if len(parts) >= 4 and parts[0].isalpha() and parts[1].isdigit():
				resname = parts[0].upper()
				resseq = int(parts[1])
				chain = parts[2] if len(parts) > 2 else ""
			# Alternative fallback: chain resname resseq pka
			elif len(parts) >= 4 and len(parts[0]) == 1 and parts[1].isalpha() and parts[2].isdigit():
				chain = parts[0]
				resname = parts[1].upper()
				resseq = int(parts[2])
			# Another fallback: resname chain resseq pka
			elif len(parts) >= 4 and parts[0].isalpha() and len(parts[1]) == 1 and parts[2].isdigit():
				resname = parts[0].upper()
				chain = parts[1]
				resseq = int(parts[2])

			if resname is None or resseq is None:
				continue

			rows.append(
				{
					"resname": resname,
					"chain": str(chain).strip(),
					"resseq": str(int(resseq)),
					"propka_pka": pka_value,
				}
			)

	return pd.DataFrame(rows)


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
	"""Pick the most specific join keys available in both frames."""
	keys = ["resname", "resseq"]
	if "chain" not in left.columns or "chain" not in right.columns:
		return keys

	left_chain = left["chain"].astype(str).str.strip().replace({"": pd.NA})
	right_chain = right["chain"].astype(str).str.strip().replace({"": pd.NA})
	if left_chain.dropna().empty or right_chain.dropna().empty:
		return keys
	return keys + ["chain"]


def parse_pdb_residue_order(pdb_path: str) -> List[Dict[str, str]]:
	"""Extract protein residue identifiers in file order from a PDB file."""
	residues: List[Dict[str, str]] = []
	seen = set()
	with open(pdb_path, "r") as handle:
		for line in handle:
			if not line.startswith(("ATOM", "HETATM")):
				continue
			resname = line[17:20].strip().upper()
			chain = line[21].strip() or "_"
			resseq = line[22:26].strip()
			icode = line[26].strip()
			if not resname or not resseq:
				continue
			key = (chain, resseq, icode, resname)
			if key in seen:
				continue
			seen.add(key)
			residues.append({"resname": resname, "chain": chain, "resseq": resseq, "icode": icode})
	return residues


def attach_residue_metadata(df: pd.DataFrame, pdb_lookup: Dict[str, str], reference_df: Optional[pd.DataFrame] = None, ph: float = 7.4) -> pd.DataFrame:
	"""Backfill residue identifiers from the source PDB when they are absent.

	The fold feature files only carry the PDB identifier and row order. For
	comparison against PROPKA, we recover residue IDs by walking the PDB in file
	order and matching the row's residue name sequence.
	"""
	if {"chain", "resseq"}.issubset(df.columns):
		return df
	if "pdb" not in df.columns or "resname" not in df.columns:
		return df

	# If a reference frame is provided (for example the original test/features
	# file that contains explicit `chain` and `resseq` columns), use it to
	# backfill missing identifiers by `row_id` before attempting PDB-order
	# heuristics.
	if reference_df is not None and "row_id" in df.columns and "row_id" in reference_df.columns:
		ref = reference_df.set_index("row_id")[[]]
		# Only keep relevant columns if present
		for col in ("chain", "resseq", "icode"):
			if col in reference_df.columns:
				ref[col] = reference_df.set_index("row_id")[col]
		# Merge by row_id, prefer existing df values and only fill missing ones
		merged = df.set_index("row_id").copy()
		for col in ("chain", "resseq", "icode"):
			if col in ref.columns:
				merged[col] = merged[col].fillna(ref[col])
		merged = merged.reset_index()
		df = merged

	# Fallback: if a PKAD-R style reference is provided (e.g. filtered_pkadr.csv),
	# match rows by (pdb, resname, reconstructed expt_pka). This allows metadata
	# backfill even when row_id is unrelated to the original PKAD indexing.
	if reference_df is not None and "label" in df.columns and "pdb" in df.columns and "resname" in df.columns:
		pdb_col, chain_col, resid_col, resname_col, pka_col = parse_pkadr_columns(reference_df)
		if pdb_col and resname_col and pka_col and chain_col and resid_col:
			ref = reference_df.copy()
			ref["_pdb_norm"] = ref[pdb_col].apply(normalize_pdb_stem)
			ref["_resname_norm"] = ref[resname_col].astype(str).str.upper().str.strip()
			ref["_expt_pka_num"] = ref[pka_col].apply(parse_pka_value)
			ref = ref.dropna(subset=["_expt_pka_num"]).copy()
			ref["_pka_key"] = ref["_expt_pka_num"].round(3)

			# Build key -> rows map so duplicates can be reported explicitly.
			ref_key_map: Dict[Tuple[str, str, float], List[Tuple[str, str]]] = {}
			for _, rr in ref.iterrows():
				chain_val = str(rr[chain_col]).strip()
				resseq_val = str(rr[resid_col]).strip()
				key = (rr["_pdb_norm"], rr["_resname_norm"], float(rr["_pka_key"]))
				ref_key_map.setdefault(key, []).append((chain_val, resseq_val))

			df = df.copy()
			if "chain" not in df.columns:
				df["chain"] = pd.NA
			if "resseq" not in df.columns:
				df["resseq"] = pd.NA
			df["label_expt_pka_reconstructed"] = df["label"].apply(lambda p: deprot_probability_to_pka(p, ph))
			df["label_expt_pka_key"] = df["label_expt_pka_reconstructed"].round(3)

			status = []
			dup_count = []
			for idx, row in df.iterrows():
				current_chain = row["chain"] if "chain" in df.columns else pd.NA
				current_resseq = row["resseq"] if "resseq" in df.columns else pd.NA
				if pd.notna(current_chain) and str(current_chain).strip() != "" and pd.notna(current_resseq) and str(current_resseq).strip() != "":
					status.append("already_present")
					dup_count.append(0)
					continue

				pka_key = row["label_expt_pka_key"]
				if pd.isna(pka_key):
					status.append("missing_label_pka")
					dup_count.append(0)
					continue

				key = (normalize_pdb_stem(row["pdb"]), str(row["resname"]).upper().strip(), float(pka_key))
				matches = ref_key_map.get(key, [])
				if len(matches) == 1:
					m_chain, m_resseq = matches[0]
					df.at[idx, "chain"] = m_chain
					df.at[idx, "resseq"] = m_resseq
					status.append("reference_pka_key")
					dup_count.append(1)
				elif len(matches) > 1:
					status.append("duplicate_reference_pka_key")
					dup_count.append(len(matches))
				else:
					status.append("unmatched_reference_pka_key")
					dup_count.append(0)

			df["reference_pka_key_status"] = status
			df["reference_pka_key_match_count"] = dup_count
			dup_rows = int((df["reference_pka_key_status"] == "duplicate_reference_pka_key").sum())
			if dup_rows:
				print(f"[compare] WARNING: {dup_rows} row(s) had duplicate PKAD key matches (same pdb+resname+expt_pka)")

	filled_parts = []
	for pdb_name, group in df.groupby("pdb", sort=False):
		pdb_stem = os.path.splitext(os.path.basename(str(pdb_name)))[0]
		pdb_path = pdb_lookup.get(pdb_stem)
		group = group.copy()
		if not pdb_path or not os.path.exists(pdb_path):
			group["chain"] = group.get("chain", pd.Series([pd.NA] * len(group), index=group.index))
			group["resseq"] = group.get("resseq", pd.Series([pd.NA] * len(group), index=group.index))
			group["icode"] = group.get("icode", pd.Series([pd.NA] * len(group), index=group.index))
			group["residue_metadata_source"] = "missing_pdb"
			filled_parts.append(group)
			continue

		residue_order = parse_pdb_residue_order(pdb_path)
		group = group.sort_values("row_id") if "row_id" in group.columns else group.copy()
		group_resnames = group["resname"].astype(str).str.upper().tolist()
		assignments = []
		cursor = 0
		for target_resname in group_resnames:
			match_index = None
			for idx in range(cursor, len(residue_order)):
				if residue_order[idx]["resname"] == target_resname:
					match_index = idx
					break
			if match_index is None:
				assignments.append({"chain": pd.NA, "resseq": pd.NA, "icode": pd.NA, "residue_metadata_source": "unmatched"})
				continue
			entry = residue_order[match_index]
			assignments.append(
				{
					"chain": entry["chain"],
					"resseq": str(entry["resseq"]),
					"icode": entry["icode"],
					"residue_metadata_source": "pdb_order",
				}
			)
			cursor = match_index + 1

		assign_df = pd.DataFrame(assignments, index=group.index)
		for column in ["chain", "resseq", "icode", "residue_metadata_source"]:
			group[column] = assign_df[column]
		filled_parts.append(group)

	result = pd.concat(filled_parts, ignore_index=True)
	if "row_id" in result.columns:
		result = result.sort_values("row_id").reset_index(drop=True)
	return result


def attach_predictions(fold_features: pd.DataFrame, oof_df: pd.DataFrame) -> pd.DataFrame:
	"""Bring OOF predictions back onto the fold feature rows.

	The OOF CSV only stores model outputs, while the fold feature CSVs keep the
	residue identifiers needed for the PROPKA join. Row IDs are the stable link.
	"""
	if "row_id" not in fold_features.columns or "row_id" not in oof_df.columns:
		merged = fold_features.copy()
		if "prediction" in oof_df.columns and len(oof_df) == len(merged):
			merged["prediction"] = oof_df["prediction"].to_numpy()
		return merged

	base_cols = ["row_id"]
	if "fold" in oof_df.columns:
		base_cols.append("fold")
	if "label" in oof_df.columns:
		base_cols.append("label")
	if "prediction" in oof_df.columns:
		base_cols.append("prediction")

	oof_base = oof_df[base_cols].copy()
	merged = fold_features.merge(oof_base, on="row_id", how="left", suffixes=("", "_oof"))
	if "fold_oof" in merged.columns and "fold" not in fold_features.columns:
		merged.rename(columns={"fold_oof": "fold"}, inplace=True)
	elif "fold_oof" in merged.columns:
		merged.drop(columns=["fold_oof"], inplace=True)
	if "label_oof" in merged.columns and "label" not in fold_features.columns:
		merged.rename(columns={"label_oof": "label"}, inplace=True)
	elif "label_oof" in merged.columns:
		merged.drop(columns=["label_oof"], inplace=True)
	return merged


def build_propka_index(propka_root: str, pdb_stems: Iterable[str]) -> Dict[str, str]:
	"""Map a PDB stem to the best matching .pka file under propka_root."""
	propka_root = os.path.abspath(propka_root)
	candidates_by_stem: Dict[str, List[str]] = {stem: [] for stem in pdb_stems}

	if not os.path.exists(propka_root):
		return {}

	all_pka = glob.glob(os.path.join(propka_root, "**", "*.pka"), recursive=True)
	for stem in pdb_stems:
		stem_lower = stem.lower()
		for path in all_pka:
			base = os.path.splitext(os.path.basename(path))[0].lower()
			if base == stem_lower or base == f"{stem_lower}_propka":
				candidates_by_stem[stem].append(path)

	result: Dict[str, str] = {}
	for stem, paths in candidates_by_stem.items():
		if not paths:
			continue
		result[stem] = max(paths, key=os.path.getmtime)
	return result


def build_pdb_index(pdb_root: str, pdb_stems: Iterable[str]) -> Dict[str, str]:
	"""Map a PDB stem to the best matching .pdb file under pdb_root."""
	pdb_root = os.path.abspath(pdb_root)
	if not os.path.exists(pdb_root):
		return {}

	all_pdb = glob.glob(os.path.join(pdb_root, "**", "*.pdb"), recursive=True)
	result: Dict[str, str] = {}
	for stem in pdb_stems:
		stem_lower = stem.lower()
		matches = []
		for path in all_pdb:
			base = os.path.splitext(os.path.basename(path))[0].lower()
			if base == stem_lower or base == f"{stem_lower}_model1" or base == f"{stem_lower}_clean":
				matches.append(path)
		if matches:
			result[stem] = max(matches, key=os.path.getmtime)
	return result


def generate_propka_pka_from_pdb(pdb_path: str, output_dir: str) -> str:
	"""Create a .pka file using PROPKA's Python API, not the CLI."""
	try:
		from propka.run import single as propka_single
	except ImportError as exc:
		raise RuntimeError(
			"PROPKA Python package is not installed; cannot generate missing .pka files"
		) from exc

	os.makedirs(output_dir, exist_ok=True)
	stem = os.path.splitext(os.path.basename(pdb_path))[0]
	out_path = os.path.join(output_dir, f"{stem}.pka")

	cwd = os.getcwd()
	try:
		# Run PROPKA through its Python API and write the pKa file explicitly.
		molecule = propka_single(pdb_path, write_pka=False)
		molecule.write_pka(filename=out_path)
	finally:
		os.chdir(cwd)

	if not os.path.exists(out_path):
		raise RuntimeError(f"PROPKA finished but did not create expected file: {out_path}")

	return out_path


def enrich_frame_with_propka(df: pd.DataFrame, propka_lookup: Dict[str, str], ph: float) -> pd.DataFrame:
	enriched_parts = []
	for pdb_name, group in df.groupby("pdb", sort=False):
		pdb_stem = os.path.splitext(os.path.basename(str(pdb_name)))[0]
		propka_path = propka_lookup.get(pdb_stem)
		group = group.copy()
		if not propka_path:
			group["propka_pka"] = np.nan
			group["propka_deprot_probability"] = np.nan
			group["propka_delta_vs_label"] = np.nan
			group["propka_delta_vs_prediction"] = np.nan
			enriched_parts.append(group)
			continue

		propka_df = parse_propka_file(propka_path)
		if propka_df.empty:
			group["propka_pka"] = np.nan
			group["propka_deprot_probability"] = np.nan
			group["propka_delta_vs_label"] = np.nan
			group["propka_delta_vs_prediction"] = np.nan
			group["propka_file"] = propka_path
			enriched_parts.append(group)
			continue


		# Normalize merge-key column types to avoid int/str conflicts
		if "resname" in propka_df.columns:
			propka_df["resname"] = propka_df["resname"].astype(str).str.upper().str.strip()
		if "chain" in propka_df.columns:
			propka_df["chain"] = propka_df["chain"].astype(str).str.strip()
		if "resseq" in propka_df.columns:
			propka_df["resseq"] = propka_df["resseq"].astype(str).str.strip()

		group["resname"] = group["resname"].astype(str).str.upper().str.strip()
		if "chain" in group.columns:
			group["chain"] = group["chain"].astype(str).str.strip()
		group["resseq"] = group["resseq"].astype(str).str.strip()

		merge_keys = choose_merge_keys(group, propka_df)
		merged = group.merge(
			propka_df,
			how="left",
			on=merge_keys,
		)
		merged["propka_file"] = propka_path
		merged["propka_merge_keys"] = "+".join(merge_keys)
		merged["propka_deprot_probability"] = merged["propka_pka"].apply(
			lambda value: compute_deprot_probability(value, ph) if pd.notna(value) else np.nan
		)
		if "label" in merged.columns:
			merged["propka_delta_vs_label"] = merged["propka_deprot_probability"] - merged["label"]
		else:
			merged["propka_delta_vs_label"] = np.nan
		if "prediction" in merged.columns:
			merged["propka_delta_vs_prediction"] = merged["propka_deprot_probability"] - merged["prediction"]
		else:
			merged["propka_delta_vs_prediction"] = np.nan
		enriched_parts.append(merged)

	if not enriched_parts:
		return df.copy()

	result = pd.concat(enriched_parts, ignore_index=True)
	if "row_id" in result.columns:
		result = result.sort_values("row_id").reset_index(drop=True)
	if "propka_merge_keys" in result.columns:
		counts = result["propka_merge_keys"].value_counts(dropna=False).to_dict()
		print(f"[compare] PROPKA merge keys used: {counts}")
	return result


def summarize_by_fold(oof_df: pd.DataFrame, cv_df: pd.DataFrame) -> pd.DataFrame:
	summary_rows = []
	for _, cv_row in cv_df.iterrows():
		fold = int(cv_row["fold"])
		fold_df = oof_df[oof_df["fold"] == fold].copy()
		row = cv_row.to_dict()

		# Defaults when no PROPKA data available
		if fold_df.empty or "propka_deprot_probability" not in fold_df.columns:
			row["propka_rows"] = 0
			row["propka_mean_pka"] = np.nan
			row["propka_mean_deprot_probability"] = np.nan
			row["propka_pearson_r_vs_model"] = np.nan
			row["propka_spearman_r_vs_model"] = np.nan
			row["propka_mae_vs_model"] = np.nan
			row["propka_pearson_r_vs_label"] = np.nan
			row["propka_spearman_r_vs_label"] = np.nan
			row["propka_mae_vs_label"] = np.nan
			summary_rows.append(row)
			continue

		propka_subset = fold_df.dropna(subset=["propka_deprot_probability"])  # rows with PROPKA
		row["propka_rows"] = len(propka_subset)
		row["propka_mean_pka"] = propka_subset["propka_pka"].mean()
		row["propka_mean_deprot_probability"] = propka_subset["propka_deprot_probability"].mean()

		# PROPKA vs model
		if "prediction" in propka_subset.columns and propka_subset["prediction"].notna().sum() >= 2:
			row["propka_pearson_r_vs_model"] = safe_pearson(propka_subset["propka_deprot_probability"], propka_subset["prediction"])
			row["propka_spearman_r_vs_model"] = safe_spearman(propka_subset["propka_deprot_probability"], propka_subset["prediction"])
			row["propka_mae_vs_model"] = float(np.mean(np.abs(propka_subset["propka_deprot_probability"] - propka_subset["prediction"])))
		else:
			row["propka_pearson_r_vs_model"] = np.nan
			row["propka_spearman_r_vs_model"] = np.nan
			row["propka_mae_vs_model"] = np.nan

		# PROPKA vs label
		if "label" in propka_subset.columns and propka_subset.dropna(subset=["propka_deprot_probability", "label"]).shape[0] >= 2:
			psub = propka_subset.dropna(subset=["propka_deprot_probability", "label"]).copy()
			row["propka_pearson_r_vs_label"] = safe_pearson(psub["propka_deprot_probability"], psub["label"])
			row["propka_spearman_r_vs_label"] = safe_spearman(psub["propka_deprot_probability"], psub["label"])
			row["propka_mae_vs_label"] = float(np.mean(np.abs(psub["propka_deprot_probability"] - psub["label"])))
		else:
			row["propka_pearson_r_vs_label"] = np.nan
			row["propka_spearman_r_vs_label"] = np.nan
			row["propka_mae_vs_label"] = np.nan

		summary_rows.append(row)

	return pd.DataFrame(summary_rows)

def summarize_state_flips(df: pd.DataFrame, ph: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Summarize protonation-state flips relative to the experimental transition.

	A residue is treated as deprotonated when pKa < pH.
	"flipped" means reference pKa and experimental pKa are on opposite sides of pH.
	The returned tuple is (summary_table, detailed_rows).
	"""
	required = {"prediction", "propka_pka", "ref_pka"}
	if not required.issubset(df.columns):
		missing = ", ".join(sorted(required - set(df.columns)))
		raise ValueError(f"State-flip analysis requires columns: {missing}")

	state_df = df.copy()
	if "expt_pka" in state_df.columns:
		state_df["experimental_pka"] = state_df["expt_pka"]
	elif "label" in state_df.columns:
		state_df["experimental_pka"] = state_df["label"].apply(
			lambda value: deprot_probability_to_pka(value, ph) if pd.notna(value) else np.nan
		)
	else:
		raise ValueError("State-flip analysis needs either `expt_pka` or `label`.")

	state_df["reference_state_deprot"] = state_df["ref_pka"].apply(
		lambda value: bool(value < ph) if pd.notna(value) else np.nan
	)
	state_df["experimental_state_deprot"] = state_df["experimental_pka"].apply(
		lambda value: bool(value < ph) if pd.notna(value) else np.nan
	)
	state_df["flip_state_group"] = np.where(
		state_df["reference_state_deprot"] != state_df["experimental_state_deprot"],
		"flipped",
		"unflipped",
	)

	state_df["model_predicted_pka"] = state_df["prediction"].apply(
		lambda value: deprot_probability_to_pka(value, ph) if pd.notna(value) else np.nan
	)
	state_df["model_state_deprot"] = state_df["model_predicted_pka"].apply(
		lambda value: bool(value < ph) if pd.notna(value) else np.nan
	)
	state_df["propka_state_deprot"] = state_df["propka_pka"].apply(
		lambda value: bool(value < ph) if pd.notna(value) else np.nan
	)

	rows = state_df.dropna(subset=["prediction", "propka_pka", "experimental_state_deprot", "reference_state_deprot"]).copy()
	if rows.empty:
		return pd.DataFrame(), rows

	rows["model_state_correct"] = rows["model_state_deprot"] == rows["experimental_state_deprot"]
	rows["propka_state_correct"] = rows["propka_state_deprot"] == rows["experimental_state_deprot"]
	rows["model_flipped_vs_experiment"] = rows["model_state_deprot"] != rows["experimental_state_deprot"]
	rows["propka_flipped_vs_experiment"] = rows["propka_state_deprot"] != rows["experimental_state_deprot"]
	rows["pka_delta_model_vs_expt"] = rows["model_predicted_pka"] - rows["experimental_pka"]
	rows["pka_delta_propka_vs_expt"] = rows["propka_pka"] - rows["experimental_pka"]

	def _accuracy_stats(subset: pd.DataFrame, label: str) -> Dict[str, float]:
		metrics = {
			f"{label}_rows": len(subset),
			f"{label}_model_state_accuracy": np.nan,
			f"{label}_propka_state_accuracy": np.nan,
			f"{label}_model_state_mae_pka": np.nan,
			f"{label}_propka_state_mae_pka": np.nan,
			f"{label}_model_only_correct_rows": 0,
			f"{label}_propka_only_correct_rows": 0,
			f"{label}_both_correct_rows": 0,
			f"{label}_both_wrong_rows": 0,
		}
		if subset.empty:
			return metrics
		metrics[f"{label}_model_state_accuracy"] = float(subset["model_state_correct"].mean())
		metrics[f"{label}_propka_state_accuracy"] = float(subset["propka_state_correct"].mean())
		metrics[f"{label}_model_state_mae_pka"] = float(np.mean(np.abs(subset["model_predicted_pka"] - subset["experimental_pka"])))
		metrics[f"{label}_propka_state_mae_pka"] = float(np.mean(np.abs(subset["propka_pka"] - subset["experimental_pka"])))
		return metrics

	summary = {
		"rows_analyzed": len(rows),
		"reference_deprot_rows": int((rows["reference_state_deprot"] == True).sum()),
		"reference_prot_rows": int((rows["reference_state_deprot"] == False).sum()),
		"experimental_deprot_rows": int((rows["experimental_state_deprot"] == True).sum()),
		"experimental_prot_rows": int((rows["experimental_state_deprot"] == False).sum()),
		"model_state_accuracy": float(rows["model_state_correct"].mean()),
		"propka_state_accuracy": float(rows["propka_state_correct"].mean()),
		"model_state_mae_pka": float(np.mean(np.abs(rows["model_predicted_pka"] - rows["experimental_pka"]))),
		"propka_state_mae_pka": float(np.mean(np.abs(rows["propka_pka"] - rows["experimental_pka"]))),
	}
	summary.update(_accuracy_stats(rows[rows["flip_state_group"] == "flipped"], "flipped"))
	summary.update(_accuracy_stats(rows[rows["flip_state_group"] == "unflipped"], "unflipped"))

	summary_df = pd.DataFrame([summary])
	return summary_df, rows

def main() -> None:
	parser = argparse.ArgumentParser(description="Compare deprotonation model outputs against PROPKA pKa values.")
	parser.add_argument("--prefix", help="Output prefix used by Train_Deprotonation_Model.py")
	parser.add_argument(
		"--propka-root",
		default=".",
		help="Root directory to search recursively for PROPKA .pka files (default: current directory)",
	)
	parser.add_argument(
		"--pdb-root",
		default=".",
		help="Root directory to search recursively for source PDB files when a .pka is missing",
	)
	parser.add_argument(
		"--reference-features",
		help="Path to a reference features CSV (e.g. the test/features file) that contains `row_id`, `chain`, and `resseq` to backfill missing identifiers",
	)
	parser.add_argument(
		"--generate-missing-propka",
		action="store_true",
		help="If a matching .pka is missing, generate it via the PROPKA Python API from a matching PDB",
	)
	parser.add_argument(
		"--state-csv",
		help="Path to an existing OOF+PROPKA CSV to run a protonation-state flip analysis directly",
	)
	parser.add_argument(
		"--flip-analysis",
		action="store_true",
		help="Print and save a protonation-state flip analysis for the merged OOF/PROPKA rows",
	)
	parser.add_argument(
		"--nucleophiles-only",
		action="store_true",
		help="Restrict analysis to nucleophilic residues only (CYS, SER, THR, TYR, LYS, HIS)",
	)
	parser.add_argument(
		"--ph",
		type=float,
		default=7.4,
		help="pH used to convert PROPKA pKa to deprotonation probability (default: 7.4)",
	)
	args = parser.parse_args()

	if args.state_csv:
		state_df = pd.read_csv(args.state_csv)
		if args.nucleophiles_only:
			state_df = filter_nucleophilic_residues(state_df)
		summary_df, detail_df = summarize_state_flips(state_df, args.ph)
		base = os.path.splitext(os.path.basename(args.state_csv))[0]
		if args.nucleophiles_only:
			base = f"{base}_nucleophiles_only"
		out_dir = os.path.dirname(os.path.abspath(args.state_csv))
		summary_out = os.path.join(out_dir, f"{base}_flip_summary.csv")
		detail_out = os.path.join(out_dir, f"{base}_flip_rows.csv")
		summary_df.to_csv(summary_out, index=False)
		detail_df.to_csv(detail_out, index=False)
		print(f"[compare] Wrote {summary_out}")
		print(f"[compare] Wrote {detail_out}")
		row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
		print(f"[compare] State-flip analysis at pH {args.ph:.2f}:")
		print(f"  Rows analyzed: {int(row.get('rows_analyzed', 0))}")
		print(f"  Reference deprot rows:     {int(row.get('reference_deprot_rows', 0))}")
		print(f"  Reference prot rows:       {int(row.get('reference_prot_rows', 0))}")
		print(f"  Experimental deprot rows:  {int(row.get('experimental_deprot_rows', 0))}")
		print(f"  Experimental prot rows:    {int(row.get('experimental_prot_rows', 0))}")
		print(f"  Overall model accuracy:     {row.get('model_state_accuracy', np.nan):.4f}")
		print(f"  Overall PROPKA accuracy:    {row.get('propka_state_accuracy', np.nan):.4f}")
		print(f"  Model pKa MAE vs expt:      {row.get('model_state_mae_pka', np.nan):.4f}")
		print(f"  PROPKA pKa MAE vs expt:     {row.get('propka_state_mae_pka', np.nan):.4f}")
		print(f"  Flipped rows:               {int(row.get('flipped_rows', 0))}")
		print(f"  Unflipped rows:             {int(row.get('unflipped_rows', 0))}")
		print(f"  Flipped model accuracy:     {row.get('flipped_model_state_accuracy', np.nan):.4f}")
		print(f"  Flipped PROPKA accuracy:    {row.get('flipped_propka_state_accuracy', np.nan):.4f}")
		print(f"  Unflipped model accuracy:   {row.get('unflipped_model_state_accuracy', np.nan):.4f}")
		print(f"  Unflipped PROPKA accuracy:  {row.get('unflipped_propka_state_accuracy', np.nan):.4f}")
		print(f"  Model only correct:         {int(row.get('model_only_correct_rows', 0))}")
		print(f"  PROPKA only correct:        {int(row.get('propka_only_correct_rows', 0))}")
		print(f"  Both correct:               {int(row.get('both_correct_rows', 0))}")
		print(f"  Both wrong:                 {int(row.get('both_wrong_rows', 0))}")
		plot_threshold_sweeps(detail_df, out_dir, base)
		print(f"[compare] Wrote {os.path.join(out_dir, f'{base}_threshold_overall.png')}")
		print(f"[compare] Wrote {os.path.join(out_dir, f'{base}_threshold_flipped_sens_spec.png')}")
		print(f"[compare] Wrote {os.path.join(out_dir, f'{base}_threshold_flipped_roc.png')}")
		print(f"[compare] Wrote {os.path.join(out_dir, f'{base}_threshold_residue_accuracy.png')}")
		return

	if not args.prefix:
		raise SystemExit("--prefix is required unless --state-csv is provided")

	prefix = args.prefix
	cv_path = locate_file(prefix, "_cv_results.csv")
	oof_path = locate_file(prefix, "_oof_predictions.csv")
	fold_paths = locate_fold_feature_files(prefix)

	if cv_path is None:
		raise FileNotFoundError(f"Could not find {prefix}_cv_results.csv")
	if oof_path is None:
		raise FileNotFoundError(f"Could not find {prefix}_oof_predictions.csv")
	if not fold_paths:
		raise FileNotFoundError(f"Could not find any {prefix}_foldXX_test_features.csv files")

	cv_df = pd.read_csv(cv_path)
	oof_df = pd.read_csv(oof_path)

	fold_dfs = []
	for path in fold_paths:
		fold_num = fold_number_from_path(path)
		fold_df = pd.read_csv(path)
		if fold_num is not None and "fold" not in fold_df.columns:
			fold_df["fold"] = fold_num
		fold_dfs.append(fold_df)

	all_fold_features = pd.concat(fold_dfs, ignore_index=True)

	# Optionally load a reference features CSV to use as a source of truth for
	# residue identifiers. This is typically the original feature/test CSVs that
	# contain explicit `chain` and `resseq` columns.
	reference_df = None
	if getattr(args, "reference_features", None):
		ref_path = args.reference_features
		if os.path.exists(ref_path):
			reference_df = pd.read_csv(ref_path)
			print(f"[compare] Loaded reference features from {ref_path} (rows={len(reference_df)})")
		else:
			print(f"[compare] WARNING: reference features path not found: {ref_path}")
	pdb_stems = sorted({os.path.splitext(os.path.basename(str(pdb_name)))[0] for pdb_name in all_fold_features["pdb"].dropna().unique()})
	propka_lookup = build_propka_index(args.propka_root, pdb_stems)
	pdb_lookup = build_pdb_index(args.pdb_root, pdb_stems)

	if args.generate_missing_propka:
		missing_stems = [stem for stem in pdb_stems if stem not in propka_lookup]
		if missing_stems:
			print(f"[compare] Generating missing PROPKA files for {len(missing_stems)} stem(s) via Python API")
		for stem in missing_stems:
			pdb_path = pdb_lookup.get(stem)
			if not pdb_path:
				continue
			try:
				generated = generate_propka_pka_from_pdb(pdb_path, os.path.join(args.propka_root, "generated_pka"))
				propka_lookup[stem] = generated
				print(f"[compare] Generated {generated}")
			except Exception as exc:
				print(f"[compare] WARNING: could not generate PROPKA for {stem}: {exc}")

	if not propka_lookup:
		print(f"[compare] WARNING: no matching .pka files found under {args.propka_root}")
	else:
		print(f"[compare] Matched {len(propka_lookup)} PDB stem(s) to PROPKA files")

	oof_with_features = attach_predictions(all_fold_features, oof_df)
	oof_with_features = attach_residue_metadata(oof_with_features, pdb_lookup, reference_df=reference_df, ph=args.ph)
	enriched_oof = enrich_frame_with_propka(oof_with_features, propka_lookup, args.ph)
	if args.nucleophiles_only:
		enriched_oof = filter_nucleophilic_residues(enriched_oof)
		print(f"[compare] Nucleophiles-only mode enabled: {len(enriched_oof)} row(s) retained")
	enriched_cv = summarize_by_fold(enriched_oof, cv_df)

	out_dir = os.path.dirname(os.path.abspath(cv_path))
	oof_out = os.path.join(out_dir, f"{os.path.basename(prefix)}_oof_predictions_with_propka.csv")
	cv_out = os.path.join(out_dir, f"{os.path.basename(prefix)}_cv_results_with_propka.csv")
	fold_out = os.path.join(out_dir, f"{os.path.basename(prefix)}_fold_test_features_with_propka.csv")

	enriched_oof.to_csv(oof_out, index=False)
	enriched_cv.to_csv(cv_out, index=False)
	all_fold_enriched = enrich_frame_with_propka(oof_with_features, propka_lookup, args.ph)
	all_fold_enriched.to_csv(fold_out, index=False)

	print(f"[compare] Wrote {oof_out}")
	print(f"[compare] Wrote {cv_out}")
	print(f"[compare] Wrote {fold_out}")

	if "propka_deprot_probability" in enriched_oof.columns:
		propka_rows = enriched_oof["propka_deprot_probability"].notna().sum()
		print(f"[compare] PROPKA rows merged: {propka_rows}/{len(enriched_oof)}")
	if "row_id" in enriched_oof.columns and "propka_pka" in enriched_oof.columns:
		matched = enriched_oof["propka_pka"].notna().sum()
		print(f"[compare] Matched PROPKA pKa rows after row_id rejoin: {matched}/{len(enriched_oof)}")
	if "residue_metadata_source" in enriched_oof.columns:
		print(f"[compare] Residue metadata sources: {enriched_oof['residue_metadata_source'].value_counts(dropna=False).to_dict()}")

	if args.flip_analysis:
		summary_df, detail_df = summarize_state_flips(enriched_oof, args.ph)
		flip_prefix = os.path.basename(prefix)
		if args.nucleophiles_only:
			flip_prefix = f"{flip_prefix}_nucleophiles_only"
		flip_summary_out = os.path.join(out_dir, f"{flip_prefix}_flip_summary.csv")
		flip_detail_out = os.path.join(out_dir, f"{flip_prefix}_flip_rows.csv")
		summary_df.to_csv(flip_summary_out, index=False)
		detail_df.to_csv(flip_detail_out, index=False)
		print(f"[compare] Wrote {flip_summary_out}")
		print(f"[compare] Wrote {flip_detail_out}")
		if not summary_df.empty:
			row = summary_df.iloc[0].to_dict()
			print(f"[compare] State-flip analysis at pH {args.ph:.2f}:")
			print(f"  Flipped rows:               {int(row.get('flipped_rows', 0))}")
			print(f"  Unflipped rows:             {int(row.get('unflipped_rows', 0))}")
			print(f"  Flipped model accuracy:     {row.get('flipped_model_state_accuracy', np.nan):.4f}")
			print(f"  Flipped PROPKA accuracy:    {row.get('flipped_propka_state_accuracy', np.nan):.4f}")
			print(f"  Unflipped model accuracy:   {row.get('unflipped_model_state_accuracy', np.nan):.4f}")
			print(f"  Unflipped PROPKA accuracy:  {row.get('unflipped_propka_state_accuracy', np.nan):.4f}")
		plot_threshold_sweeps(detail_df, out_dir, flip_prefix)
		print(f"[compare] Wrote {os.path.join(out_dir, f'{flip_prefix}_threshold_overall.png')}")
		print(f"[compare] Wrote {os.path.join(out_dir, f'{flip_prefix}_threshold_flipped_sens_spec.png')}")
		print(f"[compare] Wrote {os.path.join(out_dir, f'{flip_prefix}_threshold_flipped_roc.png')}")
		print(f"[compare] Wrote {os.path.join(out_dir, f'{flip_prefix}_threshold_residue_accuracy.png')}")

	# Overall comparison: PROPKA vs label and model vs label on intersecting rows
	if {
		"propka_deprot_probability",
		"prediction",
		"label",
	}.issubset(enriched_oof.columns):
		both = enriched_oof.dropna(subset=["propka_deprot_probability", "prediction", "label"]).copy()
		if not both.empty:
			mae_model = float(np.mean(np.abs(both["prediction"] - both["label"])))
			mae_propka = float(np.mean(np.abs(both["propka_deprot_probability"] - both["label"])))
			pearson_propka_label = safe_pearson(both["propka_deprot_probability"], both["label"])
			spearman_propka_label = safe_spearman(both["propka_deprot_probability"], both["label"])
			pearson_model_label = safe_pearson(both["prediction"], both["label"])
			spearman_model_label = safe_spearman(both["prediction"], both["label"])
			print("[compare] Overall comparison on rows with propka+prediction+label:")
			print(f"  Rows: {len(both)}")
			print(f"  MAE model vs label:   {mae_model:.4f}")
			print(f"  MAE propka vs label:   {mae_propka:.4f}")
			print(f"  Pearson propka vs label: {pearson_propka_label:.4f}")
			print(f"  Pearson model vs label: {pearson_model_label:.4f}")
			print(f"  Spearman propka vs label: {spearman_propka_label:.4f}")
			print(f"  Spearman model vs label: {spearman_model_label:.4f}")
			if mae_propka < mae_model:
				print("[compare] PROPKA is closer to labels by MAE on these rows.")
			elif mae_propka > mae_model:
				print("[compare] Model is closer to labels by MAE on these rows.")
			else:
				print("[compare] PROPKA and model have equal MAE on these rows.")


if __name__ == "__main__":
	main()
