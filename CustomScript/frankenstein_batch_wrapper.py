#!/usr/bin/env python3
"""
Wrapper for Frankenstein batch test runs.

Runs Frankenstein in test-batch mode from a CSV and writes a consolidated summary folder:
- batch_filtering_statistics.csv (raw Frankenstein batch stats)
- positive_findings.csv
- negative_findings.csv
- no_warhead_found.csv
- summary.json
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

import Frankenstein as frankenstein


def _to_bool(value):
    """Convert mixed string/bool values to bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Frankenstein in test-batch mode and generate a summary folder "
            "with positive and negative findings."
        )
    )
    parser.add_argument("--csv", required=True, help="Input CSV (test-mode format).")
    parser.add_argument("--pdb-dir", default=None, help="Optional PDB directory.")
    parser.add_argument(
        "--pdb-download-dir",
        default=None,
        help="Optional download directory for missing PDB files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output summary folder. Default: <csv_dir>/batch_filtering_statistics"
        ),
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Number of workers passed to Frankenstein (default: 1).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache reading in Frankenstein (cache still updates).",
    )
    return parser


def run_wrapper(args):
    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    csv_dir = os.path.dirname(csv_path)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(csv_dir, "batch_filtering_statistics")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Always run in test mode because per-site pass/fail feedback is required.
    frankenstein.batch_process(
        csv_file=csv_path,
        pdb_dir=args.pdb_dir,
        pdb_download_dir=args.pdb_download_dir,
        test_mode=True,
        n_workers=args.workers,
        use_cache=not args.no_cache,
    )

    raw_batch_csv = os.path.join(csv_dir, "batch_filtering_statistics.csv")
    if not os.path.exists(raw_batch_csv):
        raise FileNotFoundError(
            "Expected Frankenstein output not found: "
            f"{raw_batch_csv}. Ensure Frankenstein batch run completed successfully."
        )

    df = pd.read_csv(raw_batch_csv)
    if "found_site" not in df.columns:
        raise ValueError(
            "batch_filtering_statistics.csv is missing 'found_site'. "
            "Ensure Frankenstein was executed in test mode."
        )

    df["found_site"] = df["found_site"].apply(_to_bool)

    required_feedback_cols = [
        "step0_nucleophilic_pass",
        "step1_accessible_pass",
        "step2_reactivity_pass",
        "step3_orbital_pass",
        "failed_step",
        "succeeded_steps",
    ]
    for col in required_feedback_cols:
        if col not in df.columns:
            df[col] = ""

    positive_columns = [
        "name",
        "pdb_file",
        "electrophile_smiles",
        "warhead_type",
        "is_protonated",
        "found_site",
    ]

    negative_columns = [
        "name",
        "pdb_file",
        "electrophile_smiles",
        "warhead_type",
        "is_protonated",
        "found_site",
        "step0_nucleophilic_pass",
        "step1_accessible_pass",
        "step2_reactivity_pass",
        "step3_orbital_pass",
        "failed_step",
        "succeeded_steps",
    ]

    no_warhead_df = df[df["warhead_type"].astype(str) == "No warhead found"].copy()
    positive_df = df[df["found_site"] == True].copy()
    negative_df = df[
        (df["found_site"] == False)
        & (df["warhead_type"].astype(str) != "No warhead found")
    ].copy()

    raw_copy = os.path.join(output_dir, "batch_filtering_statistics.csv")
    shutil.copy2(raw_batch_csv, raw_copy)

    positive_out = os.path.join(output_dir, "positive_findings.csv")
    negative_out = os.path.join(output_dir, "negative_findings.csv")
    no_warhead_out = os.path.join(output_dir, "no_warhead_found.csv")

    positive_df.reindex(columns=positive_columns).to_csv(positive_out, index=False)
    negative_df.reindex(columns=negative_columns).to_csv(negative_out, index=False)
    no_warhead_df.to_csv(no_warhead_out, index=False)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_csv": csv_path,
        "raw_batch_csv": raw_copy,
        "positive_findings_csv": positive_out,
        "negative_findings_csv": negative_out,
        "no_warhead_found_csv": no_warhead_out,
        "counts": {
            "total_rows": int(len(df)),
            "positive_rows": int(len(positive_df)),
            "negative_rows": int(len(negative_df)),
            "no_warhead_rows": int(len(no_warhead_df)),
            "unique_positive_pairs": int(
                positive_df[["name", "pdb_file"]].drop_duplicates().shape[0]
                if not positive_df.empty
                else 0
            ),
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nWrapper run complete.")
    print(f"Summary folder: {output_dir}")
    print(f"Raw batch statistics: {raw_copy}")
    print(f"Positive findings: {positive_out} ({len(positive_df)} rows)")
    print(f"Negative findings: {negative_out} ({len(negative_df)} rows)")
    print(f"No warhead found: {no_warhead_out} ({len(no_warhead_df)} rows)")
    print(f"Summary JSON: {summary_path}")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    run_wrapper(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
