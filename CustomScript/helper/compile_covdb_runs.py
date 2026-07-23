#!/usr/bin/env python3
"""Compile covsite CSVs from run* folders into combined outputs.

Walks a parent directory for subfolders named run* (e.g. run1, run2),
reads each folder's covsite_candidates.csv and covsite_labels.csv, and
concatenates each into a single output with one header row.

Usage:
    python compile_covdb_runs.py <directory>

    # Default outputs under <directory>:
    #   covsite_candidates_compiled.csv
    #   covsite_labels_compiled.csv
    python compile_covdb_runs.py /path/to/runs

    python compile_covdb_runs.py /path/to/runs \\
        -o all_candidates.csv --labels-output all_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CANDIDATES_NAME = "covsite_candidates.csv"
LABELS_NAME = "covsite_labels.csv"
DEFAULT_CANDIDATES_OUTPUT = "covsite_candidates_compiled.csv"
DEFAULT_LABELS_OUTPUT = "covsite_labels_compiled.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Concatenate covsite_candidates.csv and covsite_labels.csv "
			"from run* subfolders into two CSVs (single header each)."
		),
	)
	parser.add_argument(
		"directory",
		type=Path,
		help="Parent directory containing run* folders",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help=(
			"Compiled candidates CSV path "
			f"(default: <directory>/{DEFAULT_CANDIDATES_OUTPUT})"
		),
	)
	parser.add_argument(
		"--labels-output",
		type=Path,
		default=None,
		help=(
			"Compiled labels CSV path "
			f"(default: <directory>/{DEFAULT_LABELS_OUTPUT})"
		),
	)
	return parser.parse_args()


def find_run_dirs(root: Path) -> list[Path]:
	"""Return sorted run* directories under root."""
	runs: list[Path] = []
	for child in root.iterdir():
		if child.is_dir() and child.name.startswith("run"):
			runs.append(child)

	def sort_key(path: Path) -> tuple:
		suffix = path.name[len("run") :]
		if suffix.isdigit():
			return (0, int(suffix), path.name)
		return (1, suffix, path.name)

	return sorted(runs, key=sort_key)


def collect_csv_paths(
	run_dirs: list[Path],
	filename: str,
) -> tuple[list[Path], list[str]]:
	"""Return (existing paths, run folder names missing the file)."""
	found: list[Path] = []
	missing: list[str] = []
	for run_dir in run_dirs:
		path = run_dir / filename
		if path.is_file():
			found.append(path)
		else:
			missing.append(run_dir.name)
	return found, missing


def compile_csvs(csv_paths: list[Path], out_path: Path, label: str) -> int:
	"""Concatenate CSVs into out_path with a single header. Return row count."""
	if not csv_paths:
		print(f"[WARN] No {label} files to compile; skipping {out_path}", file=sys.stderr)
		return 0

	header: list[str] | None = None
	total_rows = 0
	files_used = 0

	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="", encoding="utf-8") as out_fh:
		writer: csv.writer | None = None

		for path in csv_paths:
			with path.open("r", newline="", encoding="utf-8") as in_fh:
				reader = csv.reader(in_fh)
				try:
					file_header = next(reader)
				except StopIteration:
					print(f"[WARN] Empty file, skipping: {path}", file=sys.stderr)
					continue

				if header is None:
					header = file_header
					writer = csv.writer(out_fh)
					writer.writerow(header)
				elif file_header != header:
					sys.exit(
						f"[ERROR] Header mismatch in {path}\n"
						f"  expected: {header}\n"
						f"  got:      {file_header}"
					)

				assert writer is not None
				n = 0
				for row in reader:
					if not row or all(not cell.strip() for cell in row):
						continue
					writer.writerow(row)
					n += 1

			files_used += 1
			total_rows += n
			print(f"[INFO] {path.parent.name}/{path.name}: {n:,} row(s)")

	print(
		f"[INFO] Wrote {total_rows:,} {label} row(s) from {files_used} file(s) "
		f"to {out_path}"
	)
	return total_rows


def main() -> None:
	args = parse_args()
	root = args.directory.resolve()
	if not root.is_dir():
		sys.exit(f"[ERROR] Not a directory: {root}")

	candidates_out = (
		args.output.resolve()
		if args.output is not None
		else root / DEFAULT_CANDIDATES_OUTPUT
	)
	labels_out = (
		args.labels_output.resolve()
		if args.labels_output is not None
		else root / DEFAULT_LABELS_OUTPUT
	)

	run_dirs = find_run_dirs(root)
	if not run_dirs:
		sys.exit(f"[ERROR] No run* folders found under {root}")

	candidates_paths, candidates_missing = collect_csv_paths(
		run_dirs, CANDIDATES_NAME
	)
	labels_paths, labels_missing = collect_csv_paths(run_dirs, LABELS_NAME)

	if candidates_missing:
		print(
			f"[WARN] Missing {CANDIDATES_NAME} in: {', '.join(candidates_missing)}",
			file=sys.stderr,
		)
	if labels_missing:
		print(
			f"[WARN] Missing {LABELS_NAME} in: {', '.join(labels_missing)}",
			file=sys.stderr,
		)
	if not candidates_paths and not labels_paths:
		sys.exit(
			f"[ERROR] No {CANDIDATES_NAME} or {LABELS_NAME} files found "
			f"under run* folders"
		)

	compile_csvs(candidates_paths, candidates_out, "candidates")
	compile_csvs(labels_paths, labels_out, "labels")


if __name__ == "__main__":
	main()
