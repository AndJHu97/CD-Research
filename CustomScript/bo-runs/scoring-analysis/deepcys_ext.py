"""
Cysteine modification predictor — ExtraTreesClassifier (trail5__2_.pkl)

Requirements:
    pip install pycaret==3.x  (Python 3.9–3.11 only)

Usage:
    python predict.py --input your_data.csv
    python predict.py --input your_data.csv --output results.csv
    python predict.py --input your_data.csv --model path/to/trail5__2_
"""

import argparse
import sys
import pandas as pd

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run ExtraTrees predictions on cysteine feature data.")
parser.add_argument("--input",  required=True,  help="Path to input CSV file")
parser.add_argument("--output", default="predictions.csv", help="Path for output CSV (default: predictions.csv)")
parser.add_argument("--model",  default="trail5__2_", help="Path to model file without .pkl extension")
args = parser.parse_args()

# ── Load PyCaret ─────────────────────────────────────────────────────────────

try:
    from pycaret.classification import load_model, predict_model
except ImportError:
    sys.exit(
        "ERROR: PyCaret is not installed, or you're on Python 3.12+.\n"
        "  Install with: pip install pycaret\n"
        "  PyCaret requires Python 3.9, 3.10, or 3.11."
    )

# ── Load data ────────────────────────────────────────────────────────────────

print(f"Loading input: {args.input}")
try:
    df = pd.read_csv(args.input)
except FileNotFoundError:
    sys.exit(f"ERROR: Input file not found: {args.input}")

print(f"  {len(df)} rows, {len(df.columns)} columns")

# Add missing derived columns expected by the model
df["total_side"] = df["all_atoms"] - df["main_chain"]
df["sg_pocket_d1"] = df["d_pocket"]  # SG≈CA at d1 threshold; best available proxy

# Replace sentinel missing values (999) with NaN so the pipeline imputes them
sentinel_count = (df == 999).sum().sum()
if sentinel_count > 0:
    print(f"  Replacing {sentinel_count} sentinel 999 values with NaN")
    df.replace(999, float("nan"), inplace=True)

# ── Load model ───────────────────────────────────────────────────────────────

print(f"Loading model: {args.model}.pkl")
try:
    model = load_model(args.model)
except FileNotFoundError:
    sys.exit(f"ERROR: Model file not found: {args.model}.pkl")

# ── Predict ──────────────────────────────────────────────────────────────────

print("Running predictions...")
results = predict_model(model, data=df)


# Keep only ID column (if present) + prediction columns
id_cols = [c for c in ["id", "pdb", "chain", "resid", "uniprot_id"] if c in results.columns]
output_df = results[id_cols + ["Label", "Score"]].copy()

output_df = output_df.rename(columns={
    "Label": "predicted_cov_modified",
    "Score": "confidence",
})
# ── Save & summarise ─────────────────────────────────────────────────────────

output_df.to_csv(args.output, index=False)
print(f"\nSaved {len(output_df)} predictions to: {args.output}")

# Quick summary
counts = output_df["predicted_cov_modified"].value_counts()
print("\nPrediction summary:")
for label, count in counts.items():
    pct = count / len(output_df) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

avg_conf = output_df["confidence"].mean()
print(f"\nMean confidence: {avg_conf:.3f}")