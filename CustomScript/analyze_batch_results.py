#!/usr/bin/env python3
"""
analyze_batch_results.py — Analyzes batch filtering statistics CSV and exports categorized results.

Usage:
    python analyze_batch_results.py <input_csv> [--output-folder FOLDER] [--name NAME]
                                   [--target-site-csv TARGET_SITE_CSV]

Exports produced
----------------
  1. {name}_no_warhead_found.csv
       Rows where warhead_type contains "No warhead found".
      Columns: name, pdb_file, Residue, ResNum, Chain, warhead_type
      Target-site columns are included only when --target-site-csv is provided.

  2. {name}_found_site_protonated.csv
       One row per name where is_protonated == True AND at least one row has found_site == True.
       Columns: name, pdb_file, warhead_types, search_space_reduction,
             found_site, found_site_with_HSAB, hit_rate
      Target-site columns are included only when --target-site-csv is provided.

  3. {name}_found_site_unprotonated.csv
       Same as above but for is_protonated == False.

  4. {name}_no_found_site_protonated.csv
       One row per name where is_protonated == True AND NO row has found_site == True.
      Columns: name, pdb_file, electrophile_smiles, warhead_types,
             search_space_reduction, found_site
      Target-site columns are included only when --target-site-csv is provided.

  5. {name}_no_found_site_unprotonated.csv
       Same as above but for is_protonated == False.
"""

import os
import sys
import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_true(val) -> bool:
    """Return True when val represents a positive found-site result."""
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() == "true":
        return True
    return False


def protonation_filter(df: pd.DataFrame, protonated: bool) -> pd.DataFrame:
    """Return rows matching the given protonation state (handles bool and str)."""
    target = "true" if protonated else "false"
    return df[df["is_protonated"].astype(str).str.strip().str.lower() == target].copy()


def load_target_sites(target_site_csv: str) -> pd.DataFrame:
    """Load target-site data and aggregate Residue/ResNum/Chain by name."""
    target_df = pd.read_csv(target_site_csv, dtype=str)

    required_columns = {"name", "Residue", "ResNum", "Chain"}
    missing_columns = required_columns.difference(target_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Target-site CSV is missing required columns: {missing_list}"
        )

    def join_unique(values: pd.Series) -> str:
        seen = []
        for value in values.fillna(""):
            cleaned = str(value).strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return ", ".join(seen)

    return (
        target_df.groupby("name", sort=False, as_index=False)
        .agg({
            "Residue": join_unique,
            "ResNum": join_unique,
            "Chain": join_unique,
        })
    )


def merge_target_sites(export_df: pd.DataFrame, target_sites_df: pd.DataFrame | None) -> pd.DataFrame:
    """Merge optional target-site columns into an export and place them after pdb_file."""
    if target_sites_df is None or export_df.empty:
        return export_df

    merged_df = export_df.merge(target_sites_df, on="name", how="left")
    target_columns = [
        column for column in ["Residue", "ResNum", "Chain"]
        if column in merged_df.columns
    ]
    base_columns = [column for column in merged_df.columns if column not in target_columns]

    if "pdb_file" not in base_columns:
        return merged_df

    pdb_file_index = base_columns.index("pdb_file") + 1
    ordered_columns = base_columns[:pdb_file_index] + target_columns + base_columns[pdb_file_index:]
    return merged_df.loc[:, ordered_columns]


def build_found_site_rows(prot_df: pd.DataFrame) -> list[dict]:
    """
    For each unique name in prot_df, build one output row if at least one
    row has found_site == True.  Only rows with found_site == True contribute
    to warhead_types, search_space_reduction, and found_site_with_HSAB.
    """
    rows = []
    for name, group in prot_df.groupby("name", sort=False):
        hit_rows = group[group["found_site"].apply(is_true)]
        if hit_rows.empty:
            continue

        pdb_file = group["pdb_file"].iloc[0]
        electrophile_smiles = group["electrophile_smiles"].iloc[0]

        # Align values across warhead_types that produced a hit
        warhead_list = []
        reduction_list = []
        hsab_list = []
        seen_warheads = set()
        for _, row in hit_rows.iterrows():
            wt = str(row["warhead_type"]).strip()
            if wt not in seen_warheads:
                seen_warheads.add(wt)
                warhead_list.append(wt)
                reduction_list.append(str(row["step3_absolute_reduction_pct"]).strip())
                hsab_list.append(str(row["found_site_with_HSAB"]).strip())

        rows.append({
            "name": name,
            "pdb_file": pdb_file,
            "warhead_types": ", ".join(warhead_list),
            "search_space_reduction": ", ".join(reduction_list),
            "found_site": True,
            "found_site_with_HSAB": ", ".join(hsab_list),
        })
    return rows


def build_not_found_rows(prot_df: pd.DataFrame) -> list[dict]:
    """
    For each unique name in prot_df, build one output row if NO row has
    found_site == True (all values are False or 'no matches').
    """
    rows = []
    for name, group in prot_df.groupby("name", sort=False):
        hit_rows = group[group["found_site"].apply(is_true)]
        if not hit_rows.empty:
            continue  # At least one true – skip

        pdb_file = group["pdb_file"].iloc[0]
        electrophile_smiles = group["electrophile_smiles"].iloc[0]

        # Collect unique warhead_types (preserving first-seen order), aligned
        # with their step3 reduction values and actual found_site values.
        warhead_list = []
        reduction_list = []
        found_site_list = []
        seen_warheads = set()
        for _, row in group.iterrows():
            wt = str(row["warhead_type"]).strip()
            if wt not in seen_warheads:
                seen_warheads.add(wt)
                warhead_list.append(wt)
                reduction_list.append(str(row["step3_absolute_reduction_pct"]).strip())
                found_site_list.append(str(row["found_site"]).strip())

        rows.append({
            "name": name,
            "pdb_file": pdb_file,
            "electrophile_smiles": electrophile_smiles,
            "warhead_types": ", ".join(warhead_list),
            "search_space_reduction": ", ".join(reduction_list),
            "found_site": ", ".join(found_site_list),
        })
    return rows


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_csv(
    input_csv: str,
    output_folder: str,
    output_name: str,
    target_site_csv: str | None = None,
) -> None:
    df = pd.read_csv(input_csv, dtype=str)  # Read everything as str to avoid mixed-type surprises
    target_sites_df = load_target_sites(target_site_csv) if target_site_csv else None

    # Restore proper booleans where needed (is_protonated may be read as 'True'/'False')
    # We keep them as strings and use the helper comparisons throughout.

    os.makedirs(output_folder, exist_ok=True)
    prefix = f"{output_name}_" if output_name else ""

    total_unique_names = df["name"].nunique()

    # -------------------------------------------------------------------------
    # Export 1: No warhead found
    # -------------------------------------------------------------------------
    no_warhead_mask = df["warhead_type"].str.contains("No warhead found", case=False, na=False)
    no_warhead_df = df.loc[no_warhead_mask, ["name", "pdb_file", "warhead_type"]].copy()
    no_warhead_df = merge_target_sites(no_warhead_df, target_sites_df)

    path1 = os.path.join(output_folder, f"{prefix}no_warhead_found.csv")
    no_warhead_df.to_csv(path1, index=False)
    print(f"[1] No warhead found         → {path1}  ({len(no_warhead_df)} rows, "
          f"{no_warhead_df['name'].nunique()} unique names)")

    # -------------------------------------------------------------------------
    # Exports 2–5: Split by protonation state
    # -------------------------------------------------------------------------
    for label, protonated in [("protonated", True), ("unprotonated", False)]:
        prot_df = protonation_filter(df, protonated)
        export_num_found = "2" if protonated else "3"
        export_num_notfound = "4" if protonated else "5"

        # --- Export 2 / 3: found_site == True ---
        found_rows = build_found_site_rows(prot_df)
        found_df = pd.DataFrame(found_rows)

        if not found_df.empty:
            hit_count = found_df["name"].nunique()
            hit_rate = hit_count / total_unique_names if total_unique_names > 0 else 0.0
            found_df["hit_rate"] = (
                f"{hit_count}/{total_unique_names} ({hit_rate:.2%})"
            )

        found_df = merge_target_sites(found_df, target_sites_df)

        path_found = os.path.join(output_folder, f"{prefix}found_site_{label}.csv")
        found_df.to_csv(path_found, index=False)
        print(f"[{export_num_found}] Found site ({label:<12}) → {path_found}  "
              f"({len(found_df)} rows)")

        # --- Export 4 / 5: no found_site == True ---
        not_found_rows = build_not_found_rows(prot_df)
        not_found_df = pd.DataFrame(not_found_rows)
        not_found_df = merge_target_sites(not_found_df, target_sites_df)

        path_notfound = os.path.join(output_folder, f"{prefix}no_found_site_{label}.csv")
        not_found_df.to_csv(path_notfound, index=False)
        print(f"[{export_num_notfound}] No found site ({label:<12}) → {path_notfound}  "
              f"({len(not_found_df)} rows)")

    print(f"\nDone. All exports written to: {os.path.abspath(output_folder)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze batch filtering statistics CSV and export categorized results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_csv",
                        help="Path to the input CSV file")
    parser.add_argument("--output-folder", "-o",
                        default="results",
                        help="Destination folder for all exports (default: results/)")
    parser.add_argument("--name", "-n",
                        default="",
                        help="Prefix added to every output filename (e.g. 'run1' → run1_found_site_protonated.csv)")
    parser.add_argument("--target-site-csv",
                        default=None,
                        help="Optional CSV containing target-site columns name, Residue, ResNum, and Chain")

    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"ERROR: Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    if args.target_site_csv and not os.path.isfile(args.target_site_csv):
        print(f"ERROR: Target-site CSV not found: {args.target_site_csv}", file=sys.stderr)
        sys.exit(1)

    process_csv(args.input_csv, args.output_folder, args.name, args.target_site_csv)


if __name__ == "__main__":
    main()
