"""
Plot LightGBM validation curves (binary_logloss / auc vs iteration)
for one GroupKFold fold — diagnose early-stopping / plateau behavior.

Usage (from scoring-analysis/):
    python plot_classifier.py \\
        --training training_bo_extended_full.csv \\
        --labels labels_bo.csv \\
        --model ./cov_classifier_output_noncovalent_deep_cluster_same_test/lgbm_classifier.pkl \\
        --clusters-tsv mmseqs_output/clusters.tsv \\
        --split-csv query_group_splits.csv \\
        --deep-cluster \\
        --pdb_folder ../Existing_Structures \\
        --output lgbm_eval_curves_fold0.png

If --model is omitted, uses --max-depth (default 6) and the feature lists below.
"""

from __future__ import annotations

import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold

from Training_Cov_Classifier import fit_lgbm_classifier
from Training_Cov_Screen import (
    assign_splits,
    build_cluster_map,
    build_features,
    load_and_merge,
)

# Defaults matching the noncovalent classifier training command
DEFAULT_FEATURES = [
    "Rel_Side_SASA",
    "deprotonation_prob",
    "N_Terminal",
    "Geo_Fit",
    "Hydrophobic_Fit",
    "Hydrogen_DPAL_Fit",
    "Hydrogen_APDL_Fit",
    "Electro_Fit",
    "TPSA_Ligand",
    "Flatness_Ligand",
]
DEFAULT_RESIDUE_SPECIFIC = [
    "HOMO_LUMO_Gap_Deprotonated",
    "Partial_Charge_Deprotonated",
    "Fukui_Deprotonated",
    "Nucleophilicity_Index_Deprotonated",
    "Electrophile_LUMO_Deprotonated",
    "Nucleophile_HOMO_Deprotonated",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot LGBMClassifier eval curves for one CV fold",
    )
    p.add_argument("--training", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument(
        "--model", default=None,
        help="Optional lgbm_classifier.pkl (reads features + max_depth)",
    )
    p.add_argument("--pdb_folder", default=None)
    p.add_argument("--mmseqs_dir", default="mmseqs_output")
    p.add_argument("--clusters-tsv", default=None)
    p.add_argument("--split-csv", default=None,
                   help="Reuse fixed splits (recommended for same-test runs)")
    p.add_argument("--deep-cluster", action="store_true")
    p.add_argument("--deep-cluster-min-chain-length", type=int, default=30)
    p.add_argument("--seq_identity", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0,
                   help="0-based fold index to plot (default 0 = first fold)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Used if --model is omitted (default 6)")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument(
        "--residue_specific_features", nargs="+",
        default=DEFAULT_RESIDUE_SPECIFIC,
    )
    p.add_argument("--no-residue-type", action="store_true")
    p.add_argument("--normalize_within_protein", action="store_true")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--output", default="lgbm_eval_curves_fold0.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold < 0 or args.fold >= args.n_folds:
        sys.exit(f"[ERROR] --fold must be in [0, {args.n_folds - 1}]")

    bundle = None
    if args.model:
        with open(args.model, "rb") as fh:
            bundle = pickle.load(fh)
        print(f"Loaded model bundle: {args.model}")
        print(f"  model_type={bundle.get('model_type')}  "
              f"max_depth={bundle.get('max_depth')}")

    # 1. Load & merge
    merged = load_and_merge(args.training, args.labels)

    # 2. Clusters
    cluster_map = build_cluster_map(
        merged,
        args.pdb_folder,
        args.mmseqs_dir,
        args.seq_identity,
        args.coverage,
        deep_cluster=args.deep_cluster,
        min_chain_length=args.deep_cluster_min_chain_length,
        clusters_tsv=args.clusters_tsv,
    )

    # 3. Splits (creates 'split' column)
    merged = assign_splits(
        merged,
        cluster_map,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_state=args.random_state,
        split_csv=args.split_csv,
    )

    # 4. Features (creates res_* one-hots, etc.)
    residue_specific = list(args.residue_specific_features)
    all_req = list(dict.fromkeys(list(args.features) + residue_specific))
    merged, built_cols = build_features(
        merged,
        feature_cols=all_req,
        residue_specific_cols=residue_specific,
        normalize_within_protein=args.normalize_within_protein,
        resnum_equals=None,
        include_residue_type=not args.no_residue_type,
    )

    if bundle is not None:
        feature_cols = list(bundle["features"])
        missing = [c for c in feature_cols if c not in merged.columns]
        if missing:
            sys.exit(
                "[ERROR] After build_features, pkl features still missing:\n  "
                + ", ".join(missing)
                + "\nCheck --features / --residue_specific_features / "
                "--no-residue-type / --normalize_within_protein match training."
            )
        max_depth = int(bundle.get("max_depth") or args.max_depth or 6)
    else:
        feature_cols = built_cols
        max_depth = int(args.max_depth or 6)

    train_df = (
        merged[merged["split"] == "train"]
        .sort_values("query_group")
        .reset_index(drop=True)
    )
    print(f"\nTrain rows: {len(train_df):,}  features: {len(feature_cols)}  "
          f"max_depth={max_depth}")

    X = train_df[feature_cols].values
    y = train_df["relevance"].values.astype(int)
    groups = train_df["cluster_id"].values

    splits = list(GroupKFold(n_splits=args.n_folds).split(X, y, groups=groups))
    tr_idx, val_idx = splits[args.fold]
    print(
        f"Fold {args.fold}: train={len(tr_idx):,}  val={len(val_idx):,}  "
        f"is_unbalance=True  val_pos={int(y[val_idx].sum())}"
    )

    model = fit_lgbm_classifier(
        X[tr_idx], y[tr_idx],
        X[val_idx], y[val_idx],
        max_depth,
    )

    if not hasattr(model, "evals_result_") or not model.evals_result_:
        sys.exit("[ERROR] Model has no evals_result_ (fit without eval_set?).")

    er = model.evals_result_["valid_0"]
    # LightGBM may name metrics slightly differently across versions
    logloss_key = next(
        (k for k in er if "logloss" in k.lower() or "binary_logloss" in k.lower()),
        None,
    )
    auc_key = next((k for k in er if k.lower() == "auc"), None)
    if logloss_key is None or auc_key is None:
        sys.exit(f"[ERROR] Unexpected evals_result_ keys: {list(er.keys())}")

    logloss = np.asarray(er[logloss_key], dtype=float)
    auc = np.asarray(er[auc_key], dtype=float)
    iters = np.arange(1, len(logloss) + 1)
    best_iter = int(getattr(model, "best_iteration_", 0) or 0)

    print(f"\nbest_iteration_: {best_iter}")
    print(f"min {logloss_key} at iter {int(np.argmin(logloss) + 1)}: "
          f"{float(np.min(logloss)):.6f}")
    print(f"max {auc_key} at iter {int(np.argmax(auc) + 1)}: "
          f"{float(np.max(auc)):.6f}")
    print(f"n_boosting_rounds recorded: {len(logloss)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(iters, logloss, label=logloss_key)
    if best_iter > 0:
        axes[0].axvline(best_iter, ls="--", color="C1",
                        label=f"best_iteration_={best_iter}")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(logloss_key)
    axes[0].set_title(f"Fold {args.fold} valid {logloss_key}")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, auc, label=auc_key, color="C2")
    if best_iter > 0:
        axes[1].axvline(best_iter, ls="--", color="C1",
                        label=f"best_iteration_={best_iter}")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(auc_key)
    axes[1].set_title(f"Fold {args.fold} valid {auc_key}")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"LGBMClassifier eval curves (depth={max_depth}, early stop on logloss)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.output}")

    # Also write CSV for inspection without replotting
    csv_path = args.output.rsplit(".", 1)[0] + "_evals.csv"
    import pandas as pd
    pd.DataFrame({
        "iteration": iters,
        logloss_key: logloss,
        auc_key: auc,
    }).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
