import argparse
import pandas as pd
import sys

_KEY_SEP = "\x00"


def _composite_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    normalized = df[columns].astype(str).apply(lambda col: col.str.strip())
    return normalized.agg(_KEY_SEP.join, axis=1)


def filter_by_csv(
    base_csv: str,
    filter_csv: str,
    output_csv: str,
    base_columns: list[str],
    filter_columns: list[str],
):
    if len(base_columns) != len(filter_columns):
        sys.exit(
            "[ERROR] --base-column and --filter-column must be provided "
            "the same number of times (paired by order)."
        )

    try:
        base_df = pd.read_csv(base_csv)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read base CSV: {e}")

    try:
        filter_df = pd.read_csv(filter_csv)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read filter CSV: {e}")

    for base_column, filter_column in zip(base_columns, filter_columns):
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

    filter_keys_df = filter_df[filter_columns].copy()
    filter_keys_df.columns = base_columns
    allowed_keys = set(_composite_key(filter_keys_df, base_columns))

    base_keys = _composite_key(base_df, base_columns)
    filtered_df = base_df[base_keys.isin(allowed_keys)]

    print("\n[INFO] FILTER SUMMARY")
    print(f"  Base rows          : {len(base_df):,}")
    print(f"  Filter row combos  : {len(allowed_keys):,}")
    print(f"  Matching rows      : {len(filtered_df):,}")
    for base_column, filter_column in zip(base_columns, filter_columns):
        print(f"  Pair               : {base_column} <- {filter_column}")

    filtered_df.to_csv(output_csv, index=False)

    print(f"\n[SUCCESS] Exported -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter one CSV using rows from another CSV. "
            "Pass multiple --base-column / --filter-column pairs (by order); "
            "a base row is kept only when all pairs match on the same filter row."
        )
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
        action="append",
        required=True,
        metavar="COL",
        help="Column in base CSV (repeat per pair; matched to same-position --filter-column)",
    )

    parser.add_argument(
        "--filter-column",
        action="append",
        required=True,
        metavar="COL",
        help="Column in filter CSV (repeat per pair; matched to same-position --base-column)",
    )

    args = parser.parse_args()

    filter_by_csv(
        base_csv=args.base,
        filter_csv=args.filter,
        output_csv=args.output,
        base_columns=args.base_column,
        filter_columns=args.filter_column,
    )


if __name__ == "__main__":
    main()