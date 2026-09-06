#!/usr/bin/env python3
"""Combine covsite_candidates.csv from named kinase folders.

Usage:
    python combine_kinase_candidates.py <directory>
    python combine_kinase_candidates.py <directory> -o all_candidates.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compile_covdb_runs import CANDIDATES_NAME, compile_csvs

DEFAULT_FOLDERS = (
    "BMX",
    "BTK",
    "EGFR",
    "FGFR1",
    "FGFR4-477",
    "FGFR4-552",
    "ITK",
    "JAK3",
    "KRAS",
    "MAP3K7",
)
DEFAULT_OUTPUT = "covsite_candidates_combined.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate covsite_candidates.csv from kinase subfolders "
            "under a parent directory (single header)."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Parent directory containing the kinase folders",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Combined CSV path (default: <directory>/{DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=list(DEFAULT_FOLDERS),
        metavar="NAME",
        help="Folder names to include (default: BMX BTK EGFR ... MAP3K7)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.directory.resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] Not a directory: {root}")

    out_path = (
        args.output.resolve()
        if args.output is not None
        else root / DEFAULT_OUTPUT
    )

    found: list[Path] = []
    missing_dir: list[str] = []
    missing_csv: list[str] = []
    for name in args.folders:
        folder = root / name
        if not folder.is_dir():
            missing_dir.append(name)
            continue
        csv_path = folder / CANDIDATES_NAME
        if csv_path.is_file():
            found.append(csv_path)
        else:
            missing_csv.append(name)

    if missing_dir:
        print(
            f"[WARN] Folder not found: {', '.join(missing_dir)}",
            file=sys.stderr,
        )
    if missing_csv:
        print(
            f"[WARN] Missing {CANDIDATES_NAME} in: {', '.join(missing_csv)}",
            file=sys.stderr,
        )
    if not found:
        sys.exit(
            f"[ERROR] No {CANDIDATES_NAME} files found under {root}"
        )

    compile_csvs(found, out_path, "candidates")


if __name__ == "__main__":
    main()
