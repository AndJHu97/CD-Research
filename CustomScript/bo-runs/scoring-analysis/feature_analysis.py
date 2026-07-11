"""
feature_threshold_eval.py

Evaluates any numeric training feature as a binary classifier for reactive
residue identification, using labeled covalent binding sites as ground truth.

Usage:
    python feature_threshold_eval.py \
        --training training.csv \
        --labels batch_pdbs_bo_with_name.csv \
        [--feature Rel_Side_SASA] \
        [--threshold 0.25] \
        [--direction above]      # above = feature >= threshold is positive (default)
                              # below = feature <= threshold is positive
    [--match-warheads]        # also require training Warhead to match label
                              # Frankenstein_Warhead (comma-separated OR logic)
    [--export-sweep PATH]     # CSV of recall/threshold sweep (0.00–1.00, step 0.01)

    python feature_analysis.py --training training_bo.csv --labels batch_pdbs_bo_fixed_with_name.csv --feature deprotonation_prob --threshold 0.14 --direction above --match-warheads --export-sweep sweep.csv

Training CSV expected headers:
    Residue, Chain, ResNum, Warhead, pKa, Abs_Side_SASA, Rel_Side_SASA,
    Accessible, Deprotonated, Accessibility_Score
    (plus a column identifying the PDB name — see --name-col)

Labels CSV expected headers:
    Name, Residue, ResNum, Chain, Frankenstein_Warhead
"""

import argparse
import math
import sys
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_csv(path: str, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Cannot find {label} file: {path}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read {label} file: {e}")
    df.columns = df.columns.str.strip()
    return df


def make_key(name, residue, chain, resnum) -> tuple:
    return (
        str(name).strip().upper(),
        str(residue).strip().upper(),
        str(chain).strip(),
        int(resnum),
    )


def parse_frankenstein_warheads(raw: str) -> set[str]:
    """Split a comma-separated Frankenstein_Warhead string into a normalised set."""
    return {w.strip().lower() for w in str(raw).split(",") if w.strip()}

def _metrics_at_threshold(df, feature, labeled_col, direction, thr):
    if direction == "above":
        pred = df[feature] >= thr
    else:
        pred = df[feature] <= thr

    TP = int(((df[labeled_col]) & pred).sum())
    FN = int(((df[labeled_col]) & ~pred).sum())
    FP = int(((~df[labeled_col]) & pred).sum())
    TN = int(((~df[labeled_col]) & ~pred).sum())

    recall = TP / (TP + FN) if (TP + FN) else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    selected_frac = float(pred.mean())

    return {
        "threshold": thr,
        "recall": recall,
        "specificity": spec,
        "precision": precision,
        "selected_fraction": selected_frac,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }


def _threshold_for_recall_target(labeled_vals, target_recall: float, direction: str) -> float:
    """Smallest threshold change that achieves recall >= target_recall on labeled sites."""
    vals = sorted(labeled_vals)
    n = len(vals)
    if n == 0:
        return float("nan")

    if target_recall <= 0:
        if direction == "above":
            return vals[-1] + 1e-9
        return vals[0] - 1e-9

    if target_recall >= 1:
        if direction == "above":
            return vals[0]
        return vals[-1]

    k = math.ceil(target_recall * n)
    if direction == "above":
        return vals[n - k]
    return vals[k - 1]


def threshold_sweep(df, feature, labeled_col, direction, export_sweep_fn: str | None = None):

    print("\n[THRESHOLD SWEEP]")
    print(
        f"{'Thr':>8} "
        f"{'Recall':>8} "
        f"{'Spec':>8} "
        f"{'Selected%':>10}"
    )

    quantiles = [0.50,0.60,0.70,0.80,0.90,0.95,0.99]

    for q in quantiles:

        thr = df[feature].quantile(q)
        m = _metrics_at_threshold(df, feature, labeled_col, direction, thr)

        print(
            f"{m['threshold']:8.4f} "
            f"{m['recall']:8.3f} "
            f"{m['specificity']:8.3f} "
            f"{m['selected_fraction']*100:9.1f}%"
        )

    if export_sweep_fn is not None:
        labeled_vals = df.loc[df[labeled_col], feature].tolist()
        rows = []
        for i in range(101):
            target_recall = i / 100.0
            thr = _threshold_for_recall_target(labeled_vals, target_recall, direction)
            m = _metrics_at_threshold(df, feature, labeled_col, direction, thr)
            rows.append({
                "recall_target": target_recall,
                "threshold": m["threshold"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "selected_fraction": m["selected_fraction"],
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TN": m["TN"],
            })

        sweep_df = pd.DataFrame(rows)
        sweep_df.to_csv(export_sweep_fn, index=False)
        print(f"\n[INFO] Recall/threshold sweep (0.00–1.00, step 0.01) exported to: {export_sweep_fn}")


# ---------------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------------

def run(
    training_path: str,
    labels_path: str,
    name_col: str,
    feature: str,
    threshold: float | None,
    direction: str,          # "above" or "below"
    verbose: bool,
    export_fn: str | None = None,
    export_sweep_fn: str | None = None,
    match_warheads: bool = False,
) -> None:

    # ---- load data -------------------------------------------------------
    train = load_csv(training_path, "training")
    labels = load_csv(labels_path, "labels")

    # ---- validate required columns --------------------------------------
    required_train = {name_col, "Residue", "Chain", "ResNum", feature}
    missing = required_train - set(train.columns)
    if missing:
        sys.exit(
            f"[ERROR] Training CSV is missing column(s): {missing}\n"
            f"  Found: {list(train.columns)}\n"
            f"  Use --name-col to specify the PDB name column if it differs from '{name_col}'.\n"
            f"  Use --feature to specify the feature column (default: Rel_Side_SASA)."
        )

    required_labels = {"Name", "Residue", "ResNum", "Chain"}
    if match_warheads:
        required_labels.add("Frankenstein_Warhead")
    missing = required_labels - set(labels.columns)
    if missing:
        sys.exit(f"[ERROR] Labels CSV is missing column(s): {missing}")

    if match_warheads and "Warhead" not in train.columns:
        sys.exit("[ERROR] --match-warheads requires a 'Warhead' column in the training CSV.")

    # ---- coerce types ----------------------------------------------------
    train["ResNum"]   = pd.to_numeric(train["ResNum"],   errors="coerce")
    train[feature]    = pd.to_numeric(train[feature],    errors="coerce")
    labels["ResNum"]  = pd.to_numeric(labels["ResNum"],  errors="coerce")

    train.dropna(subset=["ResNum", feature], inplace=True)
    train["ResNum"]  = train["ResNum"].astype(int)
    labels["ResNum"] = labels["ResNum"].astype(int)

    # ---- build ground-truth label set ------------------------------------
    # label_keys: set of (name, residue, chain, resnum) for basic matching
    # label_warheads: key → set of allowed warhead strings (if --match-warheads)
    label_keys: set[tuple] = set()
    label_warheads: dict[tuple, set[str]] = {}

    for _, row in labels.iterrows():
        key = make_key(row["Name"], row["Residue"], row["Chain"], row["ResNum"])
        label_keys.add(key)
        if match_warheads:
            allowed = parse_frankenstein_warheads(row.get("Frankenstein_Warhead", ""))
            # merge in case the same site appears in multiple label rows
            if key in label_warheads:
                label_warheads[key] |= allowed
            else:
                label_warheads[key] = allowed

    print(f"[INFO] Unique labeled residue sites : {len(label_keys)}")
    print(f"[INFO] Feature column               : {feature}")
    print(f"[INFO] Direction                    : {'≥ threshold (above)' if direction == 'above' else '≤ threshold (below)'}")
    print(f"[INFO] Warhead matching             : {'enabled (training Warhead must match Frankenstein_Warhead)' if match_warheads else 'disabled'}")

    # ---- build key column (vectorised) ----------------------------------
    train["_key"] = list(zip(
        train[name_col].str.strip().str.upper(),
        train["Residue"].str.strip().str.upper(),
        train["Chain"].str.strip(),
        train["ResNum"],
    ))
    train["_labeled"] = train["_key"].isin(label_keys)

    # ---- warhead filter (optional) --------------------------------------
    # A labeled row is only truly positive if its training Warhead also
    # matches at least one entry in the label's Frankenstein_Warhead list.
    if match_warheads:
        def _warhead_matches(row) -> bool:
            if not row["_labeled"]:
                return False
            allowed = label_warheads.get(row["_key"], set())
            if not allowed:
                return True   # label had no warhead info — don't penalise
            return row["Warhead"].strip().lower() in allowed

        train["_labeled"] = train.apply(_warhead_matches, axis=1)
        n_warhead_filtered = train["_key"].isin(label_keys).sum() - train["_labeled"].sum()
        if n_warhead_filtered > 0:
            print(f"[INFO] {n_warhead_filtered} label-matched rows excluded due to warhead mismatch")

    labeled_feature_vals = train.loc[train["_labeled"], feature].tolist()

    print("\n[FEATURE DISTRIBUTION]")
    print("All residues:")
    print(train[feature].quantile([
        0.01, 0.05, 0.10,
        0.25, 0.50,
        0.75, 0.90,
        0.95, 0.99
    ]).to_string())

    print("\nLabeled residues:")
    print(train.loc[train["_labeled"], feature].quantile([
        0.01, 0.05, 0.10,
        0.25, 0.50,
        0.75, 0.90,
        0.95, 0.99
    ]).to_string())

    # Warn about unmatched labels
    matched_keys   = set(train.loc[train["_labeled"], "_key"])
    unmatched      = label_keys - matched_keys
    if unmatched:
        print(f"[WARN] {len(unmatched)} label key(s) had no match in training CSV:")
        for k in sorted(unmatched)[:10]:
            print(f"       {k}")
        if len(unmatched) > 10:
            print(f"       ... and {len(unmatched) - 10} more")

    if not labeled_feature_vals:
        sys.exit("[ERROR] No labeled residues could be matched to training data. "
                 "Check --name-col and residue identifiers.")
        
    threshold_sweep(
        train,
        feature=feature,
        labeled_col="_labeled",
        direction=direction,
        export_sweep_fn=export_sweep_fn,
    )

    # ---- determine threshold --------------------------------------------
    if direction == "above":
        auto_threshold = min(labeled_feature_vals)
        auto_label     = "minimum"
    else:
        auto_threshold = max(labeled_feature_vals)
        auto_label     = "maximum"

    if threshold is None:
        threshold = auto_threshold
        print(f"\n[INFO] Threshold set to {auto_label} {feature} of labeled residues: {threshold:.6f}")
    else:
        print(f"\n[INFO] {auto_label.capitalize()} {feature} of labeled residues: {auto_threshold:.6f}")
        print(f"[INFO] Using user-specified threshold: {threshold:.6f}")

    # ---- classify (vectorised) ------------------------------------------
    if direction == "above":
        train["_pred_pos"] = train[feature] >= threshold
        direction_symbol   = "≥"
    else:
        train["_pred_pos"] = train[feature] <= threshold
        direction_symbol   = "≤"

    TP = int(( train["_labeled"] &  train["_pred_pos"]).sum())
    FN = int(( train["_labeled"] & ~train["_pred_pos"]).sum())
    FP = int((~train["_labeled"] &  train["_pred_pos"]).sum())
    TN = int((~train["_labeled"] & ~train["_pred_pos"]).sum())

    fp_rows = train[~train["_labeled"] &  train["_pred_pos"]]
    fn_rows = train[ train["_labeled"] & ~train["_pred_pos"]]

    # ---- metrics --------------------------------------------------------
    total       = TP + TN + FP + FN
    recall      = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    sensitivity = recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    precision   = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else float("nan"))
    accuracy    = (TP + TN) / total if total > 0 else float("nan")

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FEATURE THRESHOLD CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"  Feature                     : {feature}")
    print(f"  Threshold ({direction_symbol}):             {threshold:.6f}")
    print(f"  Total residues evaluated    : {total}")
    print("-" * 60)
    print(f"  True  Positives (TP):  {TP:>7}")
    print(f"  True  Negatives (TN):  {TN:>7}")
    print(f"  False Positives (FP):  {FP:>7}  ← not labeled, but selected by threshold")
    print(f"  False Negatives (FN):  {FN:>7}  ← labeled, but rejected by threshold")
    print("-" * 60)
    print(f"  Sensitivity / Recall:  {recall:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    print(f"  Precision (PPV):       {precision:.4f}")
    print(f"  F1 Score:              {f1:.4f}")
    print(f"  Accuracy:              {accuracy:.4f}")
    print("=" * 60)

    if verbose and not fp_rows.empty:
        print(f"\n[VERBOSE] False Positive rows (sample, up to 20):")
        disp_cols = [c for c in [name_col, "Residue", "Chain", "ResNum", feature] if c in fp_rows.columns]
        print(fp_rows[disp_cols].head(20).to_string(index=False))

    if verbose and not fn_rows.empty:
        print(f"\n[VERBOSE] False Negative rows (labeled but rejected by threshold):")
        disp_cols = [c for c in [name_col, "Residue", "Chain", "ResNum", feature] if c in fn_rows.columns]
        print(fn_rows[disp_cols].to_string(index=False))

    # ---- export false negatives -----------------------------------------
    if export_fn is not None:
        base_cols    = [name_col, "Residue", "Chain", "ResNum"]
        feature_cols = [feature] + [c for c in [
            "Rel_Side_SASA", "Abs_Side_SASA", "pKa",
            "Accessible", "Deprotonated", "Accessibility_Score",
        ] if c in fn_rows.columns and c != feature]
        export_cols  = [c for c in base_cols + feature_cols if c in fn_rows.columns]
        fn_rows[export_cols].to_csv(export_fn, index=False)
        print(f"\n[INFO] False negatives ({FN} rows) exported to: {export_fn}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate any numeric feature as a threshold classifier for reactive residue sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--training",  required=True,
                        help="Path to training CSV.")
    parser.add_argument("--labels",    required=True,
                        help="Path to batch_pdbs labels CSV.")
    parser.add_argument("--name-col",  default="Name",
                        help="Column in training CSV containing PDB name (default: 'Name').")
    parser.add_argument("--feature",   default="Rel_Side_SASA",
                        help="Numeric feature column to threshold on (default: Rel_Side_SASA).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold value. Omit to auto-use min (above) or max (below) "
                             "of the feature across labeled residues.")
    parser.add_argument("--direction", choices=["above", "below"], default="above",
                        help="'above': predict positive when feature >= threshold (default). "
                             "'below': predict positive when feature <= threshold.")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print FP/FN residue details.")
    parser.add_argument("--export-fn", default=None, metavar="PATH",
                        help="Export false negative rows to this CSV path.")
    parser.add_argument("--export-sweep", default=None, metavar="PATH",
                        help="Export recall/threshold sweep (recall 0.00–1.00, step 0.01) "
                             "to this CSV path for plotting.")
    parser.add_argument("--match-warheads", action="store_true",
                        help="Require training 'Warhead' column to match at least one entry "
                             "in the label 'Frankenstein_Warhead' column (comma-separated OR). "
                             "Rows with a key match but wrong warhead are excluded from TP/FN.")
    args = parser.parse_args()

    run(
        training_path=args.training,
        labels_path=args.labels,
        name_col=args.name_col,
        feature=args.feature,
        threshold=args.threshold,
        direction=args.direction,
        verbose=args.verbose,
        export_fn=args.export_fn,
        export_sweep_fn=args.export_sweep,
        match_warheads=args.match_warheads,
    )


if __name__ == "__main__":
    main()