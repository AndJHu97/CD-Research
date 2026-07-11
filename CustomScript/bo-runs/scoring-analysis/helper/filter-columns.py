import argparse
import pandas as pd
import sys


def filter_csv(input_path: str,
               output_path: str,
               column: str,
               value: str,
               numeric: bool = False):
    """
    Filter a CSV by column == value and export result.
    """

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read input CSV: {e}")

    if column not in df.columns:
        sys.exit(f"[ERROR] Column '{column}' not found in CSV. Available columns: {list(df.columns)}")

    # Optional numeric conversion
    if numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        try:
            value = float(value)
        except ValueError:
            sys.exit("[ERROR] numeric flag set but value is not numeric")

    # Filtering
    filtered = df[
    df[column].astype(str).str.upper() == value.upper()
    ]

    print("\n[INFO] FILTER SUMMARY")
    print(f"  Input rows    : {len(df):,}")
    print(f"  Output rows   : {len(filtered):,}")
    print(f"  Filter column : {column}")
    print(f"  Filter value  : {value}")

    # Export
    try:
        filtered.to_csv(output_path, index=False)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to write output CSV: {e}")

    print(f"\n[SUCCESS] Exported filtered CSV → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter a CSV by column value and export result.")

    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--column", required=True, help="Column/header name to filter on")
    parser.add_argument("--value", required=True, help="Value to match")
    parser.add_argument("--numeric", action="store_true",
                        help="Treat column and value as numeric")

    args = parser.parse_args()

    filter_csv(
        input_path=args.input,
        output_path=args.output,
        column=args.column,
        value=args.value,
        numeric=args.numeric
    )


if __name__ == "__main__":
    main()