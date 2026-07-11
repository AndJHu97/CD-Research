import argparse
import csv
from pathlib import Path


def split_csv(input_csv: Path, lines_per_file: int = 10, start_index: int = 1) -> list[Path]:
	"""
	Split a CSV into multiple files.

	Each output file contains the original header plus up to
	(lines_per_file - 1) data rows.
	"""
	if lines_per_file < 2:
		raise ValueError("lines_per_file must be at least 2 (header + 1 data row)")

	with input_csv.open("r", newline="", encoding="utf-8-sig") as f:
		reader = csv.reader(f)
		try:
			header = next(reader)
		except StopIteration as exc:
			raise ValueError(f"Input CSV is empty: {input_csv}") from exc

		rows = list(reader)

	data_rows_per_file = lines_per_file - 1
	stem = input_csv.stem
	suffix = input_csv.suffix or ".csv"
	parent = input_csv.parent

	if start_index < 0:
		raise ValueError("start_index must be >= 0")

	written_files: list[Path] = []
	for i, start in enumerate(range(0, len(rows), data_rows_per_file), start=start_index):
		chunk = rows[start : start + data_rows_per_file]
		out_path = parent / f"{stem}_{i}{suffix}"

		with out_path.open("w", newline="", encoding="utf-8") as out_f:
			writer = csv.writer(out_f)
			writer.writerow(header)
			writer.writerows(chunk)

		written_files.append(out_path)

	return written_files


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Split a CSV into multiple files with a fixed total line count per file. "
			"Default: 10 lines per file (1 header + 9 data rows)."
		)
	)
	parser.add_argument("input_csv", help="Path to the input CSV file")
	parser.add_argument(
		"--lines-per-file",
		type=int,
		default=10,
		help="Total lines per output file including header (default: 10)",
	)

	parser.add_argument(
		"--start-index",
		"-s",
		type=int,
		default=1,
		help="Starting index used when numbering output files (default: 1)",
	)

	args = parser.parse_args()
	input_csv = Path(args.input_csv)

	if not input_csv.exists():
		raise FileNotFoundError(f"Input file not found: {input_csv}")

	output_files = split_csv(
		input_csv, lines_per_file=args.lines_per_file, start_index=args.start_index
	)

	if not output_files:
		print("No data rows found. No split files were created.")
		return

	print(f"Created {len(output_files)} files:")
	for p in output_files:
		print(p)


if __name__ == "__main__":
	main()
