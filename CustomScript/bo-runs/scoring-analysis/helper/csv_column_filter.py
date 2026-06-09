import argparse
import pandas as pd
import sys


def filter_by_csv(
    base_csv: str,
    filter_csv: str,
    output_csv: str,
    base_column: str,
    filter_column: str,
):
    try:
        base_df = pd.read_csv(base_csv)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read base CSV: {e}")

    try:
        filter_df = pd.read_csv(filter_csv)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read filter CSV: {e}")

    if base_column not in base_df.columns:
        sys.exit(
            f"[ERROR] Base column '{base_column}' not found.\n"
            f"Available: {list(base_df.columns)}"
        )

    if filter_column not in filter_df.columns:
        sys.exit(
            f"[ERROR] Filter column '{filter_column}' not found.\n"
            f"Available: {list(filter_df.columns)}"
        )

    # Convert both columns to strings and remove whitespace
    allowed_values = set(
        filter_df[filter_column]
        .astype(str)
        .str.strip()
    )

    filtered_df = base_df[
        base_df[base_column]
        .astype(str)
        .str.strip()
        .isin(allowed_values)
    ]

    print("\n[INFO] FILTER SUMMARY")
    print(f"  Base rows          : {len(base_df):,}")
    print(f"  Filter values      : {len(allowed_values):,}")
    print(f"  Matching rows      : {len(filtered_df):,}")
    print(f"  Base column        : {base_column}")
    print(f"  Filter column      : {filter_column}")

    filtered_df.to_csv(output_csv, index=False)

    print(f"\n[SUCCESS] Exported → {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter one CSV using values from another CSV."
    )

    parser.add_argument(
        "--base",
        required=True,
        help="CSV containing rows to keep/filter"
    )

    parser.add_argument(
        "--filter",
        required=True,
        help="CSV containing allowed values"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV"
    )

    parser.add_argument(
        "--base-column",
        required=True,
        help="Column in base CSV"
    )

    parser.add_argument(
        "--filter-column",
        required=True,
        help="Column in filter CSV"
    )

    args = parser.parse_args()

    filter_by_csv(
        base_csv=args.base,
        filter_csv=args.filter,
        output_csv=args.output,
        base_column=args.base_column,
        filter_column=args.filter_column,
    )


if __name__ == "__main__":
    main()