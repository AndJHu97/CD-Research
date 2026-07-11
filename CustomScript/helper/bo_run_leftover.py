#!/usr/bin/env python3
"""Split a BO batch CSV into leftover and missing-warhead exports.

Usage:
    python bo_run_leftover.py <input_csv> <runs_root> [--output-dir OUTDIR]

Outputs:
    bo_leftover_input.csv
        Rows whose name does not have a matching completed output folder,
        plus duplicate-name rows that represent additional unique
        electrophile_smiles values.

    bo_missing_warheads.csv
        Rows whose matching output folder exists but contains no files.

Matching rules:
    - A folder matches when its name is <name>_output, case-insensitive.
    - Matching folders are searched under run*/ directories inside runs_root.
    - A folder counts as complete only if it contains at least one file.
    - Duplicate names are collapsed by keeping the latest row as the base
      name and assigning earlier unique electrophile_smiles values suffixes
      like -2, -3, etc.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _normalize_text(value: str) -> str:
	return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_column(fieldnames: List[str], candidates: Iterable[str]) -> str | None:
	normalized = {_normalize_text(name): name for name in fieldnames}
	for candidate in candidates:
		if candidate in normalized:
			return normalized[candidate]
	return None


def _clean_value(value: object) -> str:
	if value is None:
		return ""
	return str(value).strip()


def _folder_has_files(folder: Path) -> bool:
	for child in folder.rglob("*"):
		if child.is_file():
			return True
	return False


def _discover_output_dirs(runs_root: Path) -> Dict[str, List[Path]]:
	"""Return normalized base-name -> matching *_output folders."""
	matches: Dict[str, List[Path]] = defaultdict(list)

	if not runs_root.is_dir():
		raise FileNotFoundError(f"Runs root is not a directory: {runs_root}")

	for run_dir in sorted(runs_root.iterdir()):
		if not run_dir.is_dir():
			continue
		if not run_dir.name.lower().startswith("run"):
			continue

		for candidate in run_dir.rglob("*"):
			if not candidate.is_dir():
				continue
			if not candidate.name.lower().endswith("_output"):
				continue

			base_name = candidate.name[:-7]  # strip "_output"
			normalized = _normalize_text(base_name)
			if normalized:
				matches[normalized].append(candidate)

	return matches


def _build_group_emissions(
	group_rows: List[dict],
	include_base_row: bool,
	electrophile_col: str,
) -> List[Tuple[int, dict, str | None]]:
	"""Select rows to emit from a single name group.

	The latest row in the group is treated as the canonical base row. Earlier
	rows are emitted only when they introduce a unique electrophile_smiles
	value that has not already appeared in a later row.
	"""
	selected: List[Tuple[int, dict, str | None]] = []
	seen_smiles = set()
	suffix = 2
	last_index = len(group_rows) - 1

	for idx in range(last_index, -1, -1):
		row = group_rows[idx]
		smiles_key = _clean_value(row.get(electrophile_col))
		if smiles_key in seen_smiles:
			continue
		seen_smiles.add(smiles_key)

		if idx == last_index:
			if include_base_row:
				selected.append((idx, row, None))
			continue

		selected.append((idx, row, f"-{suffix}"))
		suffix += 1

	selected.sort(key=lambda item: item[0])
	return selected


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def process_csv(
	input_csv: Path,
	runs_root: Path,
	output_dir: Path,
) -> Tuple[Path, Path, int, int]:
	with input_csv.open("r", newline="", encoding="utf-8-sig") as f:
		reader = csv.DictReader(f)
		if not reader.fieldnames:
			raise ValueError(f"Input CSV has no header row: {input_csv}")
		fieldnames = reader.fieldnames
		rows = list(reader)

	name_col = _find_column(fieldnames, ["name"])
	electrophile_col = _find_column(fieldnames, ["electrophilesmiles", "electrophile smiles"])

	missing_columns = []
	if name_col is None:
		missing_columns.append("name")
	if electrophile_col is None:
		missing_columns.append("electrophile_smiles")
	if missing_columns:
		raise ValueError("Missing required columns: " + ", ".join(missing_columns))

	output_dirs = _discover_output_dirs(runs_root)

	groups: List[Tuple[str, List[dict]]] = []
	group_index: Dict[str, int] = {}
	for row in rows:
		name_value = _clean_value(row.get(name_col))
		normalized = _normalize_text(name_value)
		if normalized not in group_index:
			group_index[normalized] = len(groups)
			groups.append((normalized, []))
		groups[group_index[normalized]][1].append(row)

	leftover_rows: List[dict] = []
	missing_warhead_rows: List[dict] = []

	for normalized_name, group_rows in groups:
		matching_dirs = output_dirs.get(normalized_name, [])
		folder_exists = bool(matching_dirs)
		folder_has_files = any(_folder_has_files(folder) for folder in matching_dirs)
		include_base_row = not folder_has_files

		emissions = _build_group_emissions(
			group_rows,
			include_base_row=include_base_row,
			electrophile_col=electrophile_col,
		)

		for _, row, suffix in emissions:
			row_copy = dict(row)
			base_name = _clean_value(row_copy.get(name_col))
			if suffix:
				row_copy[name_col] = f"{base_name}{suffix}"
			else:
				row_copy[name_col] = base_name

			if folder_exists and not folder_has_files:
				missing_warhead_rows.append(row_copy)
				leftover_rows.append(row_copy)
			elif folder_exists and folder_has_files:
				if suffix:
					leftover_rows.append(row_copy)
			else:
				leftover_rows.append(row_copy)

	leftover_path = output_dir / "bo_leftover_input.csv"
	missing_path = output_dir / "bo_missing_warheads.csv"

	_write_csv(leftover_path, fieldnames, leftover_rows)
	_write_csv(missing_path, fieldnames, missing_warhead_rows)

	return leftover_path, missing_path, len(leftover_rows), len(missing_warhead_rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Split BO batch rows into leftover-input and missing-warhead CSVs.",
	)
	parser.add_argument("input_csv", help="Path to the input CSV")
	parser.add_argument("runs_root", help="Folder containing run*/<name>_output folders")
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Directory for the exported CSVs (default: input CSV folder)",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	input_csv = Path(args.input_csv)
	runs_root = Path(args.runs_root)
	output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent

	if not input_csv.is_file():
		print(f"Error: input CSV not found: {input_csv}", file=sys.stderr)
		return 1
	if not runs_root.is_dir():
		print(f"Error: runs root not found: {runs_root}", file=sys.stderr)
		return 1

	try:
		leftover_path, missing_path, leftover_count, missing_count = process_csv(
			input_csv,
			runs_root,
			output_dir,
		)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(f"Wrote {leftover_count} rows to {leftover_path}")
	print(f"Wrote {missing_count} rows to {missing_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
