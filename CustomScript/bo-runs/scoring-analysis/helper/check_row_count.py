#!/usr/bin/env python3
"""Print row and column counts for a CSV file."""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count rows in a CSV file.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument(
        "--sep",
        default=",",
        help="Column separator (default: comma)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        df = pd.read_csv(args.csv_path, sep=args.sep, low_memory=False)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {args.csv_path}")
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to read CSV: {exc}")

    print(f"File: {args.csv_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")


if __name__ == "__main__":
    main()
