#!/usr/bin/env python3

import pandas as pd
import sys


def transfer_column(
    csv1_path,
    csv2_path,
    csv1_match_column,
    csv1_value_column,
    csv2_match_column,
    output_path,
    new_column_name=None,
):
    """
    Copy values from CSV 1 into CSV 2 based on matching column values.

    Parameters
    ----------
    csv1_path : str
        Path to the lookup CSV.
    csv2_path : str
        Path to the target CSV.
    csv1_match_column : str
        Column in CSV 1 used for matching.
    csv1_value_column : str
        Column in CSV 1 whose values will be copied.
    csv2_match_column : str
        Column in CSV 2 used for matching.
    output_path : str
        Path to save the output CSV.
    new_column_name : str, optional
        Name of the new column added to CSV 2.
        Defaults to csv1_value_column.
    """

    if new_column_name is None:
        new_column_name = csv1_value_column

    # Read CSV files
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Check that columns exist
    for col in [csv1_match_column, csv1_value_column]:
        if col not in df1.columns:
            raise ValueError(
                f"Column '{col}' not found in {csv1_path}.\n"
                f"Available columns: {list(df1.columns)}"
            )

    if csv2_match_column not in df2.columns:
        raise ValueError(
            f"Column '{csv2_match_column}' not found in {csv2_path}.\n"
            f"Available columns: {list(df2.columns)}"
        )

    # Warn about duplicate lookup keys
    duplicates = df1[csv1_match_column].duplicated().sum()
    if duplicates > 0:
        print(
            f"WARNING: Found {duplicates} duplicate key(s) in "
            f"'{csv1_match_column}'. The last occurrence will be used."
        )

    # Create lookup dictionary
    lookup_dict = dict(
        zip(
            df1[csv1_match_column],
            df1[csv1_value_column]
        )
    )

    # Add new column to CSV 2
    df2[new_column_name] = df2[csv2_match_column].map(lookup_dict)

    # Save output
    df2.to_csv(output_path, index=False)

    # Summary
    matches = df2[new_column_name].notna().sum()

    print(f"Output written to: {output_path}")
    print(f"Matched {matches} of {len(df2)} rows.")


if __name__ == "__main__":

    if len(sys.argv) < 7:
        print(
            "\nUsage:\n"
            "python transfer_column.py "
            "<csv1> <csv2> "
            "<csv1_match_column> <csv1_value_column> "
            "<csv2_match_column> <output_csv> "
            "[new_column_name]\n"
        )
        print("Example:")
        print(
            "python transfer_column.py "
            "lookup.csv data.csv "
            "ID Score MatchID merged.csv"
        )
        sys.exit(1)

    csv1_path = sys.argv[1]
    csv2_path = sys.argv[2]
    csv1_match_column = sys.argv[3]
    csv1_value_column = sys.argv[4]
    csv2_match_column = sys.argv[5]
    output_path = sys.argv[6]

    new_column_name = None
    if len(sys.argv) >= 8:
        new_column_name = sys.argv[7]

    transfer_column(
        csv1_path=csv1_path,
        csv2_path=csv2_path,
        csv1_match_column=csv1_match_column,
        csv1_value_column=csv1_value_column,
        csv2_match_column=csv2_match_column,
        output_path=output_path,
        new_column_name=new_column_name,
    )