#!/usr/bin/env python3
"""Create labeled training CSVs from a label sheet and run output folders.

Usage:
    python create_training_label.py <labels_csv> <runs_root> [--labels-output PATH]
        [--training-output PATH] [--missing-output PATH]

Outputs:
    labels CSV
        Copies the input CSV and adds a first-column Name field derived from
        the PDB column. For repeated PDB entries, the latest row becomes the
        base name and earlier unique SMILES values get suffixes like -2, -3,
        etc.

    training CSV
        Concatenates every
        deprotonated_ranked_covalent_targets_all_warheads_deprot_predictions.csv
        found under run*/<name>_output/ folders and inserts Name as the first
        column using the folder name without the _output suffix.

    training_missing_warheads.csv
        Contains one Name column listing folders that do not contain a usable
        deprotonated prediction CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Any, Optional
from collections import defaultdict


PREDICTION_FILENAME = "deprotonated_ranked_covalent_targets_all_warheads_deprot_predictions.csv"

_FRANKENSTEIN_TO_RAW: Dict[str, List[str]] = {
 
    # --- Michael acceptors ---
    "Alpha-beta unsaturated carbonyl (Michael acceptor)": [
        "Michael Acceptor", "Vinyl Nitroalkane", "Enamine",
    ],
    "Acrylamide warhead": [
        "Michael Acceptor",
    ],
    "Propiolamide warhead": [
        "Alkyne", "Alkynyl", "Michael Acceptor",
    ],
    "Ketoamide (Michael acceptor)": [
        "Ketoamide", "Michael Acceptor",
    ],
 
    # --- Vinyl sulfones ---
    "Vinyl sulfone": [
        "Vinyl Sulfone", "Vinylsulfone", "Aryl Sulfone",
    ],
    "Vinyl sulfonate ester": [
        "Vinyl Sulfone", "Vinylsulfone",
    ],
    "Vinylsulfonamide": [
        "Vinyl Sulfone", "Vinylsulfone",
    ],
 
    # --- Carbonyls ---
    "Aldehyde": [
        "Aldehyde", "Aldehydic carbonyl",
    ],
    "Ketone (reactive)": [
        "Carbonyl", "alpha-acyloxymethyl_Ketone",
    ],
    "Activated ketone": [
        "Carbonyl", "alpha-acyloxymethyl_Ketone",
    ],
    "Fluoromethyl ketone": [
        "Carbonyl",
    ],
 
    # --- Esters / acyl transfer ---
    "Phenyl ester": [
        "Ester", "Aryloxymethyl Carbonyl",
    ],
    "Activated ester": [
        "Ester", "Carbonate", "O-Acyl Hydroxamic Acid",
        "Disulfanyl-Ester", "Thiosulfonate",
    ],
    "Carbamate (amide-like)": [
        "Carbamate", "Amide",
    ],
    "Thioester": [
        "Thioster", "Thiocarboxylic Acid",
    ],
 
    # --- Alpha-halo carbonyls ---
    "Alpha-halo carbonyl": [
        "Halohydrocarbon", "Diazomethyl Carbonyl", "Diazo",
    ],
 
    # --- Alkyl / aryl halides ---
    "Alkyl halide (good LG)": [
        "Halohydrocarbon", "Alkyne",          # propargyl halides
    ],
    "Alkyl chloride": [
        "Halohydrocarbon",
    ],
    "Alkyl halide (Cl,Br,I)": [
        "Halohydrocarbon",
    ],
    "Nitro-activated aryl halide (SNAr)": [
        "Aryl Halide", "Vinyl Halide",
    ],
    "Heteroaryl halide (SNAr)": [
        "Aryl Halide", "Vinyl Halide",
    ],
    "Haloacetamidine": [
        "Halohydrocarbon(C=N)", "Amidine",
    ],
 
    # --- Epoxides ---
    "Epoxide": [
        "Epoxide",
    ],
    "Alpha-beta epoxyketone (epoxide)": [
        "Epoxide",
    ],
    "Alpha-beta epoxyketone (carbonyl)": [
        "Epoxide",
    ],
 
    # --- Aziridine ---
    "Aziridine": [
        "Aziridine",
    ],
 
    # --- Nitriles ---
    "Nitrile (electrophilic)": [
        "Nitrile",
    ],
    "Aromatic nitrile (cathepsin-like)": [
        "Nitrile",
    ],
    "Cyanamide": [
        "Nitrile",                            # cyanamide is N-CN, a specific nitrile subtype
    ],
 
    # --- Aminium / amines ---
    "Aminium ion": [
        "Aminium Ion",
    ],
    "Protonated amine": [
        "Aminium Ion",
    ],
 
    # --- Sulfonyl-based ---
    "Sulfonyl fluoride": [
        "Sulfonyl Fluorine", "Sulfonyl Halide", "Alpha-Cyclosulfate",
    ],
    "Sulfonyl chloride": [
        "Sulfonyl Chloride", "Sulfonyl chloride",   # both casings present in CSV
        "Sulfonyl Halide",
    ],
    "Sulfonamide (activated)": [
        "Sulfonyl", "Beta sultams",
    ],
    "Sulfonyl-sulfide": [
        "Thiosulfonate", "Disulfanyl-Ester",
    ],
 
    # --- Phosphorus ---
    "Phosphonyl fluoride (phosphonofluoridate)": [
        "Phosphonate", "Acyl Phosphonate",
    ],
    "Phosphonate (general)": [
        "Phosphonate", "Acyl Phosphonate", "Thiophosphonate_Ester",
    ],
 
    # --- Disulfides ---
    "Disulfide": [
        "Disulfide", "Disulfanyl-Ester",
    ],
 
    # --- Lactones / lactams ---
    "Beta-lactone": [
        "Beta Lacton", "Lactone",
    ],
    "Beta-lactam": [
        "Beta Lactam", "Gamma Lactam", "Beta sultams",
    ],
 
    # --- Isocyanate / isothiocyanate / carbodiimide ---
    "Isocyanate": [
        "Carbodiimide",                       # N=C=N is structurally related
    ],
    "Isothiocyanate": [
        "Isothiocyanate",
    ],
    "Carbodiimide": [
        "Carbodiimide",
    ],
 
    # --- Urea ---
    "Urea carbonyl": [
        "Urea carbonyl",
    ],
 
    # --- Boronic acid ---
    "Boronic acid": [
        "Boronic Acid",
    ],
 
    # --- Strained sulfur ring ---
    "Thiirane (episulfide)": [
        "Thiirane",
    ],
 
    # --- Imines (generally reversible) ---
    "Aliphatic imine (Schiff base former)": [
        "Mannich Base",
    ],
    "Vinyl imine": [
        "Enamine",
    ],
}
 
# ---------------------------------------------------------------------------
# Build reverse lookup:  normalised CSV label  →  list of Frankenstein names
# ---------------------------------------------------------------------------
 
def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())
 
 
_CSV_TO_FRANKENSTEIN: Dict[str, List[str]] = defaultdict(list)
 
for frank_name, csv_labels in _FRANKENSTEIN_TO_RAW.items():
    for label in csv_labels:
        _CSV_TO_FRANKENSTEIN[_norm(label)].append(frank_name)
 
_UNCLASSIFIED = "Unclassified"
 
 
def _to_frankenstein(raw_warhead: Optional[str]) -> str:
    """
    Given a raw CSV warhead label, return the matching Frankenstein
    ELECTROPHILE_WARHEADS name(s) as a comma-separated string,
    or 'Unclassified' if no match exists.
    """
    if not raw_warhead:
        return _UNCLASSIFIED
    matches = _CSV_TO_FRANKENSTEIN.get(_norm(raw_warhead))
    if not matches:
        return _UNCLASSIFIED
    # Deduplicate while preserving order
    seen = set()
    deduped = [m for m in matches if not (m in seen or seen.add(m))]
    return ", ".join(deduped)


def _normalize_text(value: object) -> str:
	return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _clean_value(value: object) -> str:
	if value is None:
		return ""
	return str(value).strip()


def _find_column(fieldnames: Sequence[str], candidates: Iterable[str]) -> str | None:
	normalized = {_normalize_text(name): name for name in fieldnames}
	for candidate in candidates:
		key = _normalize_text(candidate)
		if key in normalized:
			return normalized[key]
	return None


def _folder_has_data_file(folder: Path) -> bool:
	prediction_path = folder / PREDICTION_FILENAME
	if not prediction_path.is_file():
		return False

	with prediction_path.open("r", newline="", encoding="utf-8-sig") as handle:
		reader = csv.DictReader(handle)
		for _ in reader:
			return True

	return False


def _discover_output_folders(runs_root: Path) -> List[Path]:
	if not runs_root.is_dir():
		raise FileNotFoundError(f"Runs root is not a directory: {runs_root}")

	output_folders: List[Path] = []
	for run_dir in sorted(runs_root.iterdir()):
		if not run_dir.is_dir():
			continue
		if not run_dir.name.lower().startswith("run"):
			continue

		for candidate in sorted(run_dir.rglob("*")):
			if candidate.is_dir() and candidate.name.lower().endswith("_output"):
				output_folders.append(candidate)

	return output_folders


def _build_label_groups(rows: List[dict], pdb_col: str) -> Tuple[List[Tuple[str, List[dict]]], Dict[str, str]]:
	groups: List[Tuple[str, List[dict]]] = []
	group_index: Dict[str, int] = {}
	first_seen_pdb: Dict[str, str] = {}

	for row in rows:
		pdb_value = _clean_value(row.get(pdb_col))
		normalized = _normalize_text(pdb_value)
		if normalized not in group_index:
			group_index[normalized] = len(groups)
			groups.append((normalized, []))
			first_seen_pdb[normalized] = pdb_value
		groups[group_index[normalized]][1].append(row)

	return groups, first_seen_pdb


def _assign_label_names(
    group_rows: List[Dict[str, Any]],
    pdb_value: str,
    smiles_col: str,
    warhead_col: str,
) -> List[Dict[str, Any]]:
    """
    Assigns Name labels per SMILES group and adds a Frankenstein_Warhead column
    that maps each raw CSV warhead label to the canonical ELECTROPHILE_WARHEADS
    name(s) from the Frankenstein covalent screening pipeline.
 
    New column added to every output row:
        Frankenstein_Warhead : str
            Comma-separated Frankenstein warhead name(s), or 'Unclassified'.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    last_index: Dict[str, int] = {}
 
    for i, row in enumerate(group_rows):
        smi = _clean_value(row.get(smiles_col))
        groups[smi].append(row)
        last_index[smi] = i
 
    ranked_smiles = sorted(groups.keys(), key=lambda s: last_index[s], reverse=True)
 
    smiles_to_name: Dict[str, str] = {}
    suffix = 2
 
    for rank, smi in enumerate(ranked_smiles):
        if rank == 0:
            smiles_to_name[smi] = pdb_value
        else:
            smiles_to_name[smi] = f"{pdb_value}-{suffix}"
            suffix += 1
 
    selected: List[Dict[str, Any]] = []
    for row in group_rows:
        row_copy = dict(row)
 
        smi = _clean_value(row.get(smiles_col))
        row_copy["Name"] = smiles_to_name[smi]
 
        raw_warhead = _clean_value(row.get(warhead_col))
        row_copy[warhead_col] = raw_warhead
 
        # Map to Frankenstein canonical name(s)
        row_copy["Frankenstein_Warhead"] = _to_frankenstein(raw_warhead)
 
        selected.append(row_copy)
 
    return selected

def _write_csv(path: Path, fieldnames: Sequence[str], rows: List[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


def process_labels(labels_csv: Path, output_path: Path) -> Tuple[Path, int]:
	with labels_csv.open("r", newline="", encoding="utf-8-sig") as handle:
		reader = csv.DictReader(handle)
		if not reader.fieldnames:
			raise ValueError(f"Input CSV has no header row: {labels_csv}")
		fieldnames = list(reader.fieldnames)
		rows = list(reader)

	pdb_col = _find_column(fieldnames, ["pdb", "name"])
	smiles_col = _find_column(fieldnames, ["smiles", "electrophile_smiles", "electrophile smiles"])
	warhead_col = _find_column(fieldnames, ["warhead", "Warhead"])
	if pdb_col is None:
		raise ValueError("Missing required column: PDB or name")
	if smiles_col is None:
		raise ValueError("Missing required column: SMILES")
	if warhead_col is None:
		raise ValueError("Missing required column: Warhead")

	groups, first_seen_pdb = _build_label_groups(rows, pdb_col)
	labeled_rows: List[dict] = []

	for normalized_pdb, group_rows in groups:
		pdb_value = first_seen_pdb[normalized_pdb]
		labeled_rows.extend(_assign_label_names(group_rows, pdb_value, smiles_col, warhead_col))

	output_fieldnames = ["Name"] + [name for name in fieldnames if name not in {"Name", pdb_col}] + ["Frankenstein_Warhead"]
	for row in labeled_rows:
		if "Name" not in row:
			row["Name"] = _clean_value(row.get(pdb_col))

	_write_csv(output_path, output_fieldnames, labeled_rows)
	return output_path, len(labeled_rows)


def process_training_rows(runs_root: Path, output_path: Path, missing_output_path: Path) -> Tuple[Path, Path, int, int]:
	output_folders = _discover_output_folders(runs_root)
	training_rows: List[dict] = []
	missing_names: List[dict] = []
	fieldnames: List[str] = []

	for folder in output_folders:
		folder_name = folder.name[:-7]
		prediction_path = folder / PREDICTION_FILENAME
		if not _folder_has_data_file(folder):
			missing_names.append({"Name": folder_name})
			continue

		with prediction_path.open("r", newline="", encoding="utf-8-sig") as handle:
			reader = csv.DictReader(handle)
			if not reader.fieldnames:
				missing_names.append({"Name": folder_name})
				continue

			for field in reader.fieldnames:
				if field not in fieldnames and _normalize_text(field) != "name":
					fieldnames.append(field)

			for row in reader:
				row_copy = dict(row)
				row_copy["Name"] = folder_name
				training_rows.append(row_copy)

	if not training_rows:
		output_fieldnames = ["Name"]
	else:
		output_fieldnames = ["Name"] + fieldnames

	_write_csv(output_path, output_fieldnames, training_rows)
	_write_csv(missing_output_path, ["Name"], missing_names)
	return output_path, missing_output_path, len(training_rows), len(missing_names)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create labeled CSVs for training and missing folders.")
	parser.add_argument("labels_csv", help="Input CSV that will receive a Name column")
	parser.add_argument("runs_root", help="Folder containing run*/<name>_output directories")
	parser.add_argument(
		"--labels-output",
		default=None,
		help="Output path for the labeled CSV (default: <labels_csv>_with_name.csv)",
	)
	parser.add_argument(
		"--training-output",
		default=None,
		help="Output path for the combined training CSV (default: training.csv next to labels_csv)",
	)
	parser.add_argument(
		"--missing-output",
		default=None,
		help="Output path for folders missing usable predictions (default: training_missing_warheads.csv next to labels_csv)",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	labels_csv = Path(args.labels_csv)
	runs_root = Path(args.runs_root)

	if not labels_csv.is_file():
		print(f"Error: labels CSV not found: {labels_csv}", file=sys.stderr)
		return 1
	if not runs_root.is_dir():
		print(f"Error: runs root not found: {runs_root}", file=sys.stderr)
		return 1

	labels_output = Path(args.labels_output) if args.labels_output else labels_csv.with_name(f"{labels_csv.stem}_with_name.csv")
	training_output = Path(args.training_output) if args.training_output else labels_csv.with_name("training.csv")
	missing_output = Path(args.missing_output) if args.missing_output else labels_csv.with_name("training_missing_warheads.csv")

	try:
		labels_path, label_count = process_labels(labels_csv, labels_output)
		training_path, missing_path, training_count, missing_count = process_training_rows(
			runs_root,
			training_output,
			missing_output,
		)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(f"Wrote {label_count} labeled rows to {labels_path}")
	print(f"Wrote {training_count} training rows to {training_path}")
	print(f"Wrote {missing_count} missing warhead names to {missing_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
