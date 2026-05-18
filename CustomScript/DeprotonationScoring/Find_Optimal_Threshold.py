"""
Find_Optimal_Threshold.py
Loads the saved pipeline and test data to find the best classification threshold.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

def log1p_signed(x):
    """log1p for positive values, pass-through for negative."""
    return np.where(x < 0, x, np.log1p(x))

def main():
    feature_cols = [
        "ref_pka",
        "sasa",
        "electrostatic_potential",
        "arg_count",
        "lys_count",
        "asp_count",
        "glu_count",
        "hbonds_weighted",
        "hbonds_strict_flexible",
        "resname",
        ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained pipeline (.pkl)")
    parser.add_argument("features_csv", help="Path to cached features CSV")
    parser.add_argument("--fpr-tolerance", type=float, default=0.30,
                        help="Max acceptable false positive rate (default: 0.30)")
    args = parser.parse_args()

    # Load pipeline
    pipeline = joblib.load(args.model)

    # Load cached features
    feat_df = pd.read_csv(args.features_csv)
    X = feat_df[feature_cols]
    y = feat_df["label"]
    # No need for GroupShuffleSplit — already the test set
    preds = pipeline.predict(X)
    y_bin = (y >= 0.5).astype(int)

    # Sweep thresholds
    thresholds = np.arange(0.05, 0.95, 0.01)
    sensitivities, specificities, fprs, f1s = [], [], [], []

    for t in thresholds:
        p_bin = (preds >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin, p_bin).ravel()
        sensitivities.append(tp / (tp + fn))
        specificities.append(tn / (tn + fp))
        fprs.append(fp / (fp + tn))
        f1s.append(f1_score(y_bin, p_bin))

    # Find best threshold within FPR tolerance
    valid = [(t, s, fpr, f) for t, s, fpr, f
             in zip(thresholds, sensitivities, fprs, f1s)
             if fpr <= args.fpr_tolerance]

    print(f"\n{'='*50}")
    print(f"  THRESHOLD ANALYSIS")
    print(f"  FPR tolerance: {args.fpr_tolerance:.0%}")
    print(f"{'='*50}")

    if valid:
        best = max(valid, key=lambda x: x[1])  # max sensitivity
        print(f"  Optimal threshold: {best[0]:.2f}")
        print(f"  Sensitivity:       {best[1]:.4f}")
        print(f"  FPR:               {best[2]:.4f}")
        print(f"  F1:                {best[3]:.4f}")
        t_bin = (preds >= best[0]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin, t_bin).ravel()
        print(f"  TP/TN/FP/FN:       {tp}/{tn}/{fp}/{fn}")
    else:
        print("  No threshold found within FPR tolerance.")
    print(f"{'='*50}\n")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, sensitivities, label="Sensitivity (TPR)")
    plt.plot(thresholds, specificities, label="Specificity (TNR)")
    plt.plot(thresholds, fprs, label="FPR")
    plt.plot(thresholds, f1s, label="F1", linestyle="--")
    plt.axvline(x=0.5, color="gray", linestyle=":", label="Default 0.5")
    if valid:
        plt.axvline(x=best[0], color="red", linestyle="--",
                    label=f"Optimal {best[0]:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sensitivity Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig("threshold_analysis.png", dpi=150)
    print("[threshold] Plot saved to threshold_analysis.png")

if __name__ == "__main__":
    main()