#!/usr/bin/env python3
"""Apply electrophile_smiles corrections from one CSV to a master CSV.

Expected behavior:
- Match rows by `name`.
- Replace master `electrophile_smiles` with corrected values from the corrections CSV.
- Clear `LigID` for each updated row.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple


def _normalize_col(col: str) -> str:
	return "".join(ch for ch in col.lower() if ch.isalnum())


def _find_column(fieldnames: List[str], candidates: List[str]) -> str | None:
	normalized = {_normalize_col(name): name for name in fieldnames}
	for candidate in candidates:
		if candidate in normalized:
			return normalized[candidate]
	return None


def _find_corrections_smiles_column(fieldnames: List[str]) -> str | None:
	# Prefer explicit `electrophile_smiles` if present.
	explicit = _find_column(fieldnames, ["electrophilesmiles"])
	if explicit:
		return explicit

	# Fall back to any column that looks like it stores electrophile smiles.
	for col in fieldnames:
		ncol = _normalize_col(col)
		if "electrophil" in ncol and "smile" in ncol:
			return col

	return None


def _find_special_actual_electrophile_column(fieldnames: List[str]) -> str | None:
	# Special mode expects "Actual Electrophile" (name matching is normalized).
	return _find_column(fieldnames, ["actualelectrophile"])


def _build_corrections_map(
	corrections_path: str,
	name_col: str,
	smiles_col: str,
) -> Tuple[Dict[str, str], int]:
	corrections: Dict[str, str] = {}
	overwritten = 0

	with open(corrections_path, "r", newline="", encoding="utf-8-sig") as f:
		reader = csv.DictReader(f)
		for row in reader:
			name_val = (row.get(name_col) or "").strip()
			smiles_val = (row.get(smiles_col) or "").strip()

			if not name_val or not smiles_val:
				continue

			if name_val in corrections:
				overwritten += 1
			corrections[name_val] = smiles_val

	return corrections, overwritten


def _match_special_smiles(master_name: str, corrections_map: Dict[str, str]) -> str | None:
	"""Return smiles for special mode via partial name matching.

	A correction name matches if it appears within the master name (case-insensitive).
	If multiple correction names match, use the longest match.
	"""
	master_name_lower = master_name.lower()
	best_key = None

	for corr_name in corrections_map:
		if corr_name.lower() in master_name_lower:
			if best_key is None or len(corr_name) > len(best_key):
				best_key = corr_name

	if best_key is None:
		return None

	return corrections_map[best_key]


def apply_corrections(
	master_path: str,
	corrections_path: str,
	output_path: str,
	special_mode: bool = False,
) -> Tuple[int, int, int]:
	with open(master_path, "r", newline="", encoding="utf-8-sig") as fm:
		master_reader = csv.DictReader(fm)
		if not master_reader.fieldnames:
			raise ValueError("Master CSV has no header row.")
		master_fields = master_reader.fieldnames
		master_rows = list(master_reader)

	with open(corrections_path, "r", newline="", encoding="utf-8-sig") as fc:
		corrections_reader = csv.DictReader(fc)
		if not corrections_reader.fieldnames:
			raise ValueError("Corrections CSV has no header row.")
		corrections_fields = corrections_reader.fieldnames

	master_name_col = _find_column(master_fields, ["name"])
	master_smiles_col = _find_column(master_fields, ["electrophilesmiles"])
	master_ligid_col = _find_column(master_fields, ["ligid"])

	corr_name_col = _find_column(corrections_fields, ["name"])
	if special_mode:
		corr_smiles_col = _find_special_actual_electrophile_column(corrections_fields)
	else:
		corr_smiles_col = _find_corrections_smiles_column(corrections_fields)

	missing = []
	if master_name_col is None:
		missing.append("master.name")
	if master_smiles_col is None:
		missing.append("master.electrophile_smiles")
	if master_ligid_col is None:
		missing.append("master.LigID")
	if corr_name_col is None:
		missing.append("corrections.name")
	if corr_smiles_col is None:
		if special_mode:
			missing.append("corrections.Actual Electrophile")
		else:
			missing.append("corrections.electrophile_smiles")
	if missing:
		raise ValueError(
			"Missing required columns: " + ", ".join(missing)
		)

	corrections_map, duplicate_overwrites = _build_corrections_map(
		corrections_path,
		corr_name_col,
		corr_smiles_col,
	)

	updated = 0
	names_not_found = 0

	master_names = [(row.get(master_name_col) or "").strip() for row in master_rows]
	if special_mode:
		master_names_lower = [name.lower() for name in master_names]
		for corr_name in corrections_map:
			corr_name_lower = corr_name.lower()
			if not any(corr_name_lower in master_name for master_name in master_names_lower):
				names_not_found += 1
	else:
		master_name_set = set(master_names)
		for name in corrections_map:
			if name not in master_name_set:
				names_not_found += 1

	for row in master_rows:
		name_val = (row.get(master_name_col) or "").strip()
		if special_mode:
			corrected_smiles = _match_special_smiles(name_val, corrections_map)
		else:
			corrected_smiles = corrections_map.get(name_val)
		if corrected_smiles:
			row[master_smiles_col] = corrected_smiles
			row[master_ligid_col] = ""
			updated += 1

	with open(output_path, "w", newline="", encoding="utf-8") as fo:
		writer = csv.DictWriter(fo, fieldnames=master_fields)
		writer.writeheader()
		writer.writerows(master_rows)

	return updated, names_not_found, duplicate_overwrites


def _default_output_path(master_path: str) -> str:
	root, ext = os.path.splitext(master_path)
	ext = ext or ".csv"
	return f"{root}.corrected{ext}"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Update master CSV electrophile_smiles using a corrections CSV matched by name.",
	)
	parser.add_argument("master_csv", help="Path to the master CSV")
	parser.add_argument("corrections_csv", help="Path to the corrections CSV")
	parser.add_argument(
		"-o",
		"--output",
		default=None,
		help="Path to output CSV (default: <master>.corrected.csv)",
	)
	parser.add_argument(
		"--special",
		action="store_true",
		help=(
			"Use corrections CSV column 'Actual Electrophile' and partial name "
			"matching (correction name contained in master name)"
		),
	)
	parser.add_argument(
		"--in-place",
		action="store_true",
		help="Overwrite the master CSV directly",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	master_path = args.master_csv
	corrections_path = args.corrections_csv

	if args.in_place and args.output:
		print("Error: use either --in-place or --output, not both.", file=sys.stderr)
		return 2

	output_path = master_path if args.in_place else (args.output or _default_output_path(master_path))

	try:
		updated, names_not_found, duplicate_overwrites = apply_corrections(
			master_path,
			corrections_path,
			output_path,
			special_mode=args.special,
		)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(f"Updated rows: {updated}")
	print(f"Correction names not found in master: {names_not_found}")
	print(f"Duplicate correction names (last value used): {duplicate_overwrites}")
	print(f"Wrote output: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
