#!/usr/bin/env python3

"""
Calculate the mean and median residue counts for each label_type,
plus an overall summary across all rows.

Output:
    residue_summary.csv
"""

import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "final_data_and_draft/residue_specificity/residue_composition_detail.csv"
OUTPUT_CSV = "final_data_and_draft/residue_specificity/residue_summary.csv"

def main():
    # Load input CSV
    df = pd.read_csv(INPUT_CSV)

    required_columns = ["label_type", "n_candidates"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Clean label_type values
    df["label_type"] = (
        df["label_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Convert candidate counts to numeric values
    df["n_candidates"] = pd.to_numeric(
        df["n_candidates"], errors="coerce"
    )

    # Remove rows without a label_type
    df = df.dropna(subset=["label_type"])
    df = df[df["label_type"] != ""]

    # Summary by label_type
    summary = (
        df.groupby("label_type")["n_candidates"]
        .agg(mean="mean", median="median")
        .reset_index()
    )

    # Overall summary across all rows, regardless of label_type
    overall = pd.DataFrame([{
        "label_type": "OVERALL",
        "mean": df["n_candidates"].mean(),
        "median": df["n_candidates"].median()
    }])

    # Append overall row
    summary = pd.concat([summary, overall], ignore_index=True)

    # Round to 2 decimal places
    summary[["mean", "median"]] = summary[
        ["mean", "median"]
    ].round(2)

    # Save results
    summary.to_csv(OUTPUT_CSV, index=False)

    # Display results
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()