"""
feature_filter_analysis.py

Evaluates which feature columns from a training CSV are suitable as hard filters
for a covalent residue screening pipeline, using sequence-identity-clustered
cross-validation for generalizability across protein families.

After overall analysis, runs optional per-residue-type threshold analysis for
features with sufficient data per residue type.

Features listed under --features-residue-type are analyzed separately at the
(Name, Warhead, Residue) combo level (one row per protein x warhead x residue
type; ResNum/Chain ignored). Use this for residue-type-level descriptors that
are constant across sites of the same type on a given protein-warhead combo.

Usage:
    python feature_filter_analysis.py \\
        --training training.csv \\
        --labels labels.csv \\
        [--clusters mmseqs_clusters.tsv] \\
        [--features feat1 feat2 ...] \\
        [--features-residue-type feat3 feat4 ...] \\
        [--output-dir ./feature_analysis_output/] \\
        [--min-residue-positives 50] \\
        [--skip-per-residue]

        python feature_filter_analysis.py   --training training_bo.csv   --labels labels_bo.csv   --features deprotonation_prob Rel_Side_SASA   --features-residue-type Nucleophilicity_Index_Deprotonated HOMO_LUMO_Gap_Deprotonated Fukui_Deprotonated Partial_Charge_Deprotonated Electrophile_LUMO_Deprotonated Nucleophile_HOMO_Deprotonated   --residue-types CYS SER HIS THR LYS   --output-dir ./feature_analysis_output2/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

METADATA_COLS = {
    "Name", "Residue", "Chain", "ResNum", "Warhead",
    "Accessible", "Deprotonated", "Absolute_Deprotonation_State",
    "Deprotonation_State", "Weak_Bond",
}

MMSEQS_INSTALL_MSG = """
MMseqs2 not found in PATH.

Install MMseqs2:
  conda install -c bioconda mmseqs2
  # or see https://github.com/soedinglab/MMseqs2

Then re-run with --run-mmseqs <fasta_file>
"""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "analysis.log"
    logger = logging.getLogger("feature_filter_analysis")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Data loading helpers
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


def base_pdb_id(name: str) -> str:
    s = str(name).strip()
    return s.split("-")[0] if "-" in s else s


def parse_warheads(raw: str) -> set[str]:
    return {w.strip() for w in str(raw).split(",") if w.strip()}


def detect_protein_id_column(df: pd.DataFrame) -> str | None:
    for col in ("Proteins", "UniProt", "uniprot_id", "Uniprot", "protein pdb", "Protein"):
        if col in df.columns:
            return col
    return None


def build_protein_id_map(labels: pd.DataFrame, logger: logging.Logger) -> dict[str, str]:
    """Map label Name -> protein ID used for clustering."""
    protein_col = detect_protein_id_column(labels)
    if protein_col:
        logger.info("Using '%s' column from labels for Name->protein ID lookup", protein_col)
    else:
        logger.info("No protein ID column in labels; Name->protein lookup will use base PDB ID")

    mapping: dict[str, str] = {}
    for _, row in labels.iterrows():
        name = str(row["Name"]).strip()
        if protein_col and pd.notna(row.get(protein_col)) and str(row[protein_col]).strip():
            mapping[name] = str(row[protein_col]).strip()
        else:
            mapping[name] = base_pdb_id(name)
    return mapping


def resolve_training_protein_ids(
    train: pd.DataFrame,
    protein_id_map: dict[str, str],
    logger: logging.Logger,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Resolve per-row protein IDs for clustering.

    Priority:
      1. Protein ID column on training CSV (Proteins, UniProt, etc.)
      2. labels Name -> protein ID map
      3. base PDB ID from training Name
    """
    train_protein_col = detect_protein_id_column(train)
    names = train["Name"].astype(str).str.strip()
    stats = {"from_training_column": 0, "from_labels_map": 0, "from_base_pdb": 0}

    if train_protein_col:
        logger.info(
            "Using '%s' column from training CSV as primary protein ID",
            train_protein_col,
        )
        col_vals = train[train_protein_col].astype(str).str.strip()
        has_train_id = col_vals.notna() & (col_vals != "") & (col_vals.str.lower() != "nan")
        protein_ids = pd.Series(index=train.index, dtype=str)
        protein_ids[has_train_id] = col_vals[has_train_id]
        missing = ~has_train_id
        if missing.any():
            mapped = names[missing].map(lambda n: protein_id_map.get(n, base_pdb_id(n)))
            from_map = names[missing].isin(protein_id_map)
            stats["from_labels_map"] = int(from_map.sum())
            stats["from_base_pdb"] = int((~from_map).sum())
            protein_ids[missing] = mapped
        stats["from_training_column"] = int(has_train_id.sum())
    else:
        mapped = names.map(lambda n: protein_id_map.get(n, base_pdb_id(n)))
        from_map = names.isin(protein_id_map)
        stats["from_labels_map"] = int(from_map.sum())
        stats["from_base_pdb"] = int((~from_map).sum())
        protein_ids = mapped

    logger.info(
        "Protein ID resolution: %d from training column, %d from labels map, %d base-PDB fallback",
        stats["from_training_column"], stats["from_labels_map"], stats["from_base_pdb"],
    )
    if stats["from_base_pdb"] > 0:
        logger.warning(
            "%d training rows fell back to base PDB ID (Name not in labels protein map "
            "and no protein ID on training row)",
            stats["from_base_pdb"],
        )
    return protein_ids, stats


def assign_targets(train: pd.DataFrame, labels: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Add is_target column via Name/Residue/ResNum/Chain/Warhead matching."""
    required_train = {"Name", "Residue", "Chain", "ResNum", "Warhead"}
    required_labels = {"Name", "Residue", "ResNum", "Chain", "Frankenstein_Warhead"}
    missing_train = required_train - set(train.columns)
    missing_labels = required_labels - set(labels.columns)
    if missing_train:
        sys.exit(f"[ERROR] Training CSV missing columns: {missing_train}")
    if missing_labels:
        sys.exit(f"[ERROR] Labels CSV missing columns: {missing_labels}")

    train = train.copy()
    train["ResNum"] = pd.to_numeric(train["ResNum"], errors="coerce")

    label_lookup: dict[tuple, set[str]] = {}
    for _, row in labels.iterrows():
        key = (
            str(row["Name"]).strip(),
            str(row["Residue"]).strip().upper(),
            str(row["Chain"]).strip(),
            int(pd.to_numeric(row["ResNum"], errors="coerce")),
        )
        warheads = parse_warheads(row["Frankenstein_Warhead"])
        label_lookup.setdefault(key, set()).update(warheads)

    def _is_target(row) -> bool:
        key = (
            str(row["Name"]).strip(),
            str(row["Residue"]).strip().upper(),
            str(row["Chain"]).strip(),
            int(row["ResNum"]),
        )
        allowed = label_lookup.get(key)
        if not allowed:
            return False
        return str(row["Warhead"]).strip() in allowed

    train["is_target"] = train.apply(_is_target, axis=1)
    n_pos = int(train["is_target"].sum())
    logger.info("Assigned is_target=True to %d / %d training rows", n_pos, len(train))
    return train


def auto_detect_features(train: pd.DataFrame) -> list[str]:
    features = []
    for col in train.columns:
        if col in METADATA_COLS or col.startswith("Binary") or col in (
            "is_target", "_protein_id", "_cluster_group", "Proteins", "UniProt",
            "uniprot_id", "Uniprot", "protein pdb", "Protein",
        ):
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            features.append(col)
    return sorted(features)


def load_cluster_map(cluster_path: str, logger: logging.Logger) -> dict[str, str]:
    """Load MMseqs2 cluster TSV: member -> representative."""
    df = pd.read_csv(cluster_path, sep="\t", header=None, names=["representative", "member"])
    df.columns = df.columns.str.strip()
    if "representative" not in df.columns:
        # try with header row
        df = pd.read_csv(cluster_path, sep="\t")
        df.columns = df.columns.str.strip()
        if not {"representative", "member"}.issubset(df.columns):
            cols = list(df.columns)
            if len(cols) >= 2:
                df = df.rename(columns={cols[0]: "representative", cols[1]: "member"})
            else:
                sys.exit(f"[ERROR] Cluster CSV must have representative and member columns: {cluster_path}")

    mapping = dict(zip(df["member"].astype(str).str.strip(), df["representative"].astype(str).str.strip()))
    logger.info("Loaded %d cluster memberships from %s", len(mapping), cluster_path)
    return mapping


def assign_cluster_groups(
    train: pd.DataFrame,
    protein_id_map: dict[str, str],
    cluster_map: dict[str, str] | None,
    logger: logging.Logger,
) -> pd.DataFrame:
    train = train.copy()
    train["_protein_id"], id_stats = resolve_training_protein_ids(train, protein_id_map, logger)

    def _cluster_group(pid: str) -> str:
        if cluster_map and pid in cluster_map:
            return cluster_map[pid]
        return pid  # singleton cluster per protein ID

    train["_cluster_group"] = train["_protein_id"].map(_cluster_group)
    n_proteins = train["_protein_id"].nunique()
    n_groups = train["_cluster_group"].nunique()
    logger.info(
        "Assigned %d CV groups from %d unique protein IDs",
        n_groups, n_proteins,
    )
    if cluster_map:
        logger.info("MMseqs cluster file applied for sequence-identity grouping")
    else:
        logger.info(
            "No MMseqs cluster file: each protein ID is its own CV group "
            "(NOT sequence-identity clustered at 30%%)"
        )
    return train


def run_mmseqs_clustering(fasta_path: str, output_dir: Path, logger: logging.Logger) -> str:
    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        logger.error(MMSEQS_INSTALL_MSG.strip())
        print(MMSEQS_INSTALL_MSG)
        sys.exit(1)

    fasta_path = str(Path(fasta_path).resolve())
    if not Path(fasta_path).exists():
        sys.exit(f"[ERROR] FASTA file not found: {fasta_path}")

    work_dir = output_dir / "mmseqs_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    cluster_prefix = str(output_dir / "mmseqs_cluster")
    tmp_dir = str(work_dir / "tmp")

    cmd = [
        mmseqs, "easy-cluster",
        fasta_path, cluster_prefix, tmp_dir,
        "--min-seq-id", "0.3",
        "-c", "0.8",
    ]
    logger.info("Running: %s", " ".join(cmd))
    print(f"[INFO] Running MMseqs2 easy-cluster on {fasta_path} ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error("MMseqs2 failed: %s\n%s", e, e.stderr)
        sys.exit(f"[ERROR] MMseqs2 easy-cluster failed:\n{e.stderr}")

    cluster_tsv = f"{cluster_prefix}_cluster.tsv"
    if not Path(cluster_tsv).exists():
        sys.exit(f"[ERROR] Expected MMseqs2 output not found: {cluster_tsv}")

    logger.info("MMseqs2 clustering complete: %s", cluster_tsv)
    print(f"[INFO] Cluster file written to: {cluster_tsv}")
    return cluster_tsv


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_discriminative_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float, float, bool]:
    """
    Returns (auc, mw_pval, effect_size, higher_is_positive).
    If AUC < 0.5, flip direction and use 1 - AUC.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan"), True

    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan"), True

    auc_raw = roc_auc_score(y_true, scores)
    higher_is_positive = auc_raw >= 0.5
    auc = auc_raw if higher_is_positive else 1.0 - auc_raw

    try:
        _, mw_pval = mannwhitneyu(pos, neg, alternative="two-sided")
    except ValueError:
        mw_pval = float("nan")

    effect_size = 2.0 * (auc - 0.5)
    return auc, mw_pval, effect_size, higher_is_positive


def orient_scores(scores: np.ndarray, higher_is_positive: bool) -> np.ndarray:
    return scores if higher_is_positive else -scores


def oriented_threshold_to_raw(oriented_thr: float, higher_is_positive: bool) -> float:
    """Map an oriented-scale threshold back to the original feature scale."""
    if oriented_thr is None or (isinstance(oriented_thr, float) and np.isnan(oriented_thr)):
        return float("nan")
    return float(oriented_thr) if higher_is_positive else float(-oriented_thr)


def filter_direction_label(higher_is_positive: bool) -> str:
    """How to apply operating_threshold on raw feature values."""
    return "gte" if higher_is_positive else "lte"


def is_degenerate_filter(specificity: float, min_specificity: float) -> bool:
    """True when the cutoff retains (nearly) all residues — not useful as a filter."""
    return not np.isnan(specificity) and specificity < min_specificity


def apply_raw_threshold_reporting(
    thr_result: dict,
    higher_is_positive: bool,
    min_specificity: float,
) -> dict:
    """Store thresholds on raw feature scale; flag degenerate filters."""
    op_oriented = thr_result.get("operating_threshold", float("nan"))
    r1_oriented = thr_result.get("recall1_threshold", float("nan"))

    thr_result["operating_threshold_oriented"] = op_oriented
    thr_result["recall1_threshold_oriented"] = r1_oriented
    thr_result["higher_is_positive"] = higher_is_positive
    thr_result["filter_direction"] = filter_direction_label(higher_is_positive)
    thr_result["operating_threshold"] = oriented_threshold_to_raw(op_oriented, higher_is_positive)
    thr_result["recall1_threshold"] = oriented_threshold_to_raw(r1_oriented, higher_is_positive)

    spec = thr_result.get("specificity_at_operating_threshold", float("nan"))
    thr_result["degenerate_filter"] = is_degenerate_filter(spec, min_specificity)
    apply_scenario2_diagnostic_raw_thresholds(thr_result, higher_is_positive)
    return thr_result


def enrichment_ratio(n: int, N: int, k: int, K: int) -> float:
    """Observed / expected target rate in passing set: (k/K) / (n/N)."""
    if K == 0 or n == 0 or N == 0:
        return float("nan")
    return (k / K) / (n / N)


def hypergeom_enrichment_at_oriented_threshold(
    y_true: np.ndarray,
    oriented: np.ndarray,
    thr: float,
) -> tuple[int, int, float, float]:
    """Return (k, K, enrichment_ratio, hypergeom_pval) for oriented >= thr."""
    N = len(y_true)
    n = int(y_true.sum())
    K = int((oriented >= thr).sum())
    k = int((y_true & (oriented >= thr)).sum())
    if K == 0 or n == 0 or N == 0:
        return k, K, float("nan"), float("nan")
    pval = float(hypergeom.sf(k - 1, N, n, K))
    return k, K, enrichment_ratio(n, N, k, K), pval


def scenario2_diagnostic_defaults() -> dict:
    return {
        "enrichment_ratio_at_operating_threshold": float("nan"),
        "n_passing_at_operating_threshold": float("nan"),
        "n_targets_in_passing_at_operating_threshold": float("nan"),
        "scenario2_best_hypergeom_pval": float("nan"),
        "scenario2_best_recall_at_p_lt_0_05": float("nan"),
        "scenario2_best_recall_at_p_lt_0_05_threshold": float("nan"),
        "scenario2_best_recall_at_p_lt_0_05_threshold_oriented": float("nan"),
        "scenario2_hypergeom_pval_at_best_recall": float("nan"),
        "n_passing_at_best_recall_p_lt_0_05": float("nan"),
        "n_targets_in_passing_at_best_recall_p_lt_0_05": float("nan"),
        "enrichment_ratio_at_best_recall_p_lt_0_05": float("nan"),
    }


SCENARIO2_DIAGNOSTIC_KEYS = tuple(scenario2_diagnostic_defaults().keys())


def apply_scenario2_diagnostic_raw_thresholds(
    thr_result: dict,
    higher_is_positive: bool,
) -> None:
    """Convert oriented diagnostic thresholds to raw feature scale."""
    ori = thr_result.get("scenario2_best_recall_at_p_lt_0_05_threshold_oriented")
    if ori is not None and not (isinstance(ori, float) and np.isnan(ori)):
        thr_result["scenario2_best_recall_at_p_lt_0_05_threshold"] = oriented_threshold_to_raw(
            float(ori), higher_is_positive
        )


def attach_operating_enrichment_fields(
    result: dict,
    n: int,
    N: int,
    k: int,
    K: int,
    pval: float,
) -> None:
    result["n_passing_at_operating_threshold"] = K
    result["n_targets_in_passing_at_operating_threshold"] = k
    result["enrichment_ratio_at_operating_threshold"] = enrichment_ratio(n, N, k, K)
    result["hypergeom_pval_at_operating_threshold"] = pval


def apply_threshold_fields_to_row(row: dict, thr_result: dict) -> None:
    """Copy threshold/scenario fields from thr_result into a fold result row."""
    for key in (
        "operating_threshold", "operating_threshold_oriented",
        "recall_at_operating_threshold", "specificity_at_operating_threshold",
        "recall1_threshold", "recall1_threshold_oriented", "specificity_at_recall1",
        "hypergeom_pval_at_operating_threshold",
        "filter_direction", "higher_is_positive", "degenerate_filter",
        *SCENARIO2_DIAGNOSTIC_KEYS,
    ):
        if key in thr_result:
            row[key] = thr_result[key]


def metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    higher_is_positive: bool,
) -> tuple[float, float]:
    """Return (recall, specificity) for scores >= threshold (after orientation)."""
    oriented = orient_scores(scores, higher_is_positive)
    pred = oriented >= threshold
    tp = int((y_true & pred).sum())
    fn = int((y_true & ~pred).sum())
    fp = int((~y_true & pred).sum())
    tn = int((~y_true & ~pred).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return recall, specificity


def scenario1_roc_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    higher_is_positive: bool,
    min_recall: float,
) -> dict:
    """Find operating threshold via ROC (highest threshold with TPR >= min_recall)."""
    oriented = orient_scores(scores, higher_is_positive)
    result = {
        "operating_threshold": float("nan"),
        "recall_at_operating_threshold": float("nan"),
        "specificity_at_operating_threshold": float("nan"),
        "recall1_threshold": float("nan"),
        "specificity_at_recall1": float("nan"),
        "hypergeom_pval_at_operating_threshold": float("nan"),
        "scenario": 1,
        "passed": False,
        **scenario2_diagnostic_defaults(),
    }

    if len(np.unique(y_true)) < 2 or y_true.sum() == 0:
        return result

    fpr, tpr, thresholds = roc_curve(y_true, oriented)
    # roc_curve thresholds are in decreasing order: index 0 = highest threshold.
    # Among points with TPR >= min_recall, take the first (highest threshold,
    # best specificity while meeting the recall constraint).
    valid_idx = np.where(tpr >= min_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[0]
        if best_idx < len(thresholds):
            op_thr = float(thresholds[best_idx])
        else:
            op_thr = float(thresholds[-1])
        recall, spec = metrics_at_threshold(y_true, scores, op_thr, higher_is_positive)
        result.update({
            "operating_threshold": op_thr,
            "recall_at_operating_threshold": recall,
            "specificity_at_operating_threshold": spec,
            "passed": recall >= min_recall,
        })
        k, K, enrich, pval = hypergeom_enrichment_at_oriented_threshold(y_true, oriented, op_thr)
        attach_operating_enrichment_fields(result, int(y_true.sum()), len(y_true), k, K, pval)

    # recall=1.0 threshold: lowest oriented threshold retaining all positives
    pos_vals = oriented[y_true == 1]
    if len(pos_vals) > 0:
        recall1_thr = float(np.min(pos_vals))
        r1, s1 = metrics_at_threshold(y_true, scores, recall1_thr, higher_is_positive)
        result["recall1_threshold"] = recall1_thr
        result["specificity_at_recall1"] = s1

    return result


def scenario2_hypergeom_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    higher_is_positive: bool,
    min_recall: float,
    hypergeom_alpha: float = 0.05,
) -> dict:
    """Sweep percentile thresholds; hypergeometric test for enrichment.

    Operating threshold: among points with p < alpha and recall >= min_recall,
    pick the one with highest specificity (higher threshold on ties).
    Diagnostic fields still report max-recall among all p < alpha points.
    """
    oriented = orient_scores(scores, higher_is_positive)
    result = {
        "operating_threshold": float("nan"),
        "recall_at_operating_threshold": float("nan"),
        "specificity_at_operating_threshold": float("nan"),
        "recall1_threshold": float("nan"),
        "specificity_at_recall1": float("nan"),
        "hypergeom_pval_at_operating_threshold": float("nan"),
        "scenario": 2,
        "passed": False,
        **scenario2_diagnostic_defaults(),
    }

    N = len(y_true)
    n = int(y_true.sum())
    if n == 0 or N == 0:
        return result

    best_qualifying = None
    best_pval_point = None
    best_recall_significant = None

    for pct in range(5, 96, 5):
        thr = float(np.percentile(oriented, pct))
        K = int((oriented >= thr).sum())
        k = int((y_true & (oriented >= thr)).sum())
        if K == 0:
            continue
        pval = float(hypergeom.sf(k - 1, N, n, K))
        recall = k / n
        _, spec = metrics_at_threshold(y_true, scores, thr, higher_is_positive)
        enrich = enrichment_ratio(n, N, k, K)
        point = {
            "threshold": thr,
            "recall": recall,
            "specificity": spec,
            "pval": pval,
            "K": K,
            "k": k,
            "enrichment": enrich,
        }

        if best_pval_point is None or pval < best_pval_point["pval"]:
            best_pval_point = point

        if pval < hypergeom_alpha:
            if best_recall_significant is None or recall > best_recall_significant["recall"] or (
                recall == best_recall_significant["recall"]
                and thr > best_recall_significant["threshold"]
            ):
                best_recall_significant = point

        if pval < hypergeom_alpha and recall >= min_recall:
            if best_qualifying is None or spec > best_qualifying["specificity"] or (
                spec == best_qualifying["specificity"] and thr > best_qualifying["threshold"]
            ):
                best_qualifying = point

    if best_pval_point is not None:
        result["scenario2_best_hypergeom_pval"] = best_pval_point["pval"]

    if best_recall_significant is not None:
        result.update({
            "scenario2_best_recall_at_p_lt_0_05": best_recall_significant["recall"],
            "scenario2_best_recall_at_p_lt_0_05_threshold_oriented": best_recall_significant["threshold"],
            "scenario2_hypergeom_pval_at_best_recall": best_recall_significant["pval"],
            "n_passing_at_best_recall_p_lt_0_05": best_recall_significant["K"],
            "n_targets_in_passing_at_best_recall_p_lt_0_05": best_recall_significant["k"],
            "enrichment_ratio_at_best_recall_p_lt_0_05": best_recall_significant["enrichment"],
        })

    if best_qualifying is not None:
        recall, spec = metrics_at_threshold(
            y_true, scores, best_qualifying["threshold"], higher_is_positive
        )
        result.update({
            "operating_threshold": best_qualifying["threshold"],
            "recall_at_operating_threshold": recall,
            "specificity_at_operating_threshold": spec,
            "passed": True,
        })
        attach_operating_enrichment_fields(
            result, n, N,
            best_qualifying["k"], best_qualifying["K"], best_qualifying["pval"],
        )

    pos_vals = oriented[y_true == 1]
    if len(pos_vals) > 0:
        recall1_thr = float(np.min(pos_vals))
        _, s1 = metrics_at_threshold(y_true, scores, recall1_thr, higher_is_positive)
        result["recall1_threshold"] = recall1_thr
        result["specificity_at_recall1"] = s1

    return result


def fold_role_from_result(thr_result: dict) -> str:
    if not thr_result.get("passed"):
        return "rank_input"
    if thr_result.get("degenerate_filter"):
        return "rank_input"
    if thr_result.get("scenario") == 1:
        return "hard_filter"
    return "permissive_filter"


# ---------------------------------------------------------------------------
# Per-feature analysis
# ---------------------------------------------------------------------------

def analyze_feature_in_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    logger: logging.Logger,
    feature: str,
) -> dict:
    """Run full per-fold pipeline for one feature."""
    y_train = y[train_idx]
    s_train = scores[train_idx]
    y_val = y[val_idx]
    s_val = scores[val_idx]

    row = {
        "fold": fold,
        "n_train_residues": len(train_idx),
        "n_train_targets": int(y_train.sum()),
        "n_val_residues": len(val_idx),
        "n_val_targets": int(y_val.sum()),
        "auc": float("nan"),
        "mw_pval": float("nan"),
        "effect_size": float("nan"),
        "scenario": "failed",
        "operating_threshold": float("nan"),
        "recall_at_operating_threshold": float("nan"),
        "specificity_at_operating_threshold": float("nan"),
        "recall1_threshold": float("nan"),
        "specificity_at_recall1": float("nan"),
        "hypergeom_pval_at_operating_threshold": float("nan"),
        "operating_threshold_oriented": float("nan"),
        "recall1_threshold_oriented": float("nan"),
        "filter_direction": "",
        "higher_is_positive": True,
        "degenerate_filter": False,
        "role": "rank_input",
        **scenario2_diagnostic_defaults(),
    }

    # AUC/MW on training fold
    auc, mw_pval, effect_size, higher_is_positive = compute_discriminative_metrics(
        y_train.astype(int), s_train
    )
    row["auc"] = auc
    row["mw_pval"] = mw_pval
    row["effect_size"] = effect_size
    row["higher_is_positive"] = higher_is_positive

    passes_step1 = (
        not np.isnan(auc)
        and auc >= args.auc_threshold
        and not np.isnan(mw_pval)
        and mw_pval < 0.05
        and not np.isnan(effect_size)
        and effect_size >= args.effect_size
    )

    if row["n_train_targets"] < 10:
        logger.warning(
            "Feature '%s' fold %d: only %d training positives (<10); "
            "skipping threshold setting",
            feature, fold, row["n_train_targets"],
        )
        print(
            f"[WARN] {feature} fold {fold}: {row['n_train_targets']} training positives "
            f"(<10); skipping threshold setting"
        )
        return row

    if passes_step1:
        thr_result = scenario1_roc_threshold(
            y_train.astype(int), s_train, higher_is_positive, args.min_recall
        )
        row["scenario"] = 1
        logger.debug(
            "Feature '%s' fold %d: passed Step 1 (AUC=%.3f); Scenario 1 threshold=%.4f",
            feature, fold, auc, thr_result.get("operating_threshold", float("nan")),
        )
    else:
        thr_result = scenario2_hypergeom_threshold(
            y_train.astype(int), s_train, higher_is_positive, args.min_recall
        )
        row["scenario"] = 2 if thr_result.get("passed") else "failed"
        logger.debug(
            "Feature '%s' fold %d: failed Step 1 (AUC=%.3f, p=%.4g); Scenario 2 passed=%s",
            feature, fold, auc, mw_pval, thr_result.get("passed"),
        )

    thr_result = apply_raw_threshold_reporting(
        thr_result, higher_is_positive, args.min_specificity
    )
    apply_threshold_fields_to_row(row, thr_result)

    if thr_result.get("degenerate_filter"):
        logger.info(
            "Feature '%s' fold %d: degenerate filter (specificity=%.4f < %.4f); "
            "demoted to rank_input. Raw threshold=%s (%s)",
            feature, fold,
            thr_result.get("specificity_at_operating_threshold", float("nan")),
            args.min_specificity,
            thr_result.get("operating_threshold"),
            thr_result.get("filter_direction"),
        )

    row["role"] = fold_role_from_result(thr_result)
    return row


def _mean_numeric_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _scenario2_summary_means(eval_df: pd.DataFrame) -> dict:
    return {
        f"{key}_mean": _mean_numeric_column(eval_df, key)
        for key in SCENARIO2_DIAGNOSTIC_KEYS
    }


def _summarize_folds_common(
    fold_results: pd.DataFrame,
    valid_fold_mask: pd.Series | None,
    args: argparse.Namespace,
) -> dict:
    """Shared cross-fold statistics for overall and per-residue analysis."""
    if valid_fold_mask is not None:
        eval_df = fold_results.loc[valid_fold_mask]
    else:
        eval_df = fold_results

    passing = eval_df[eval_df["role"].isin(["hard_filter", "permissive_filter"])]
    n_valid = len(eval_df)
    pass_rate = len(passing) / n_valid if n_valid else 0.0

    auc_mean = eval_df["auc"].mean()
    auc_std = eval_df["auc"].std()
    auc_cv = (
        auc_std / auc_mean
        if auc_mean and not np.isnan(auc_mean) and auc_mean != 0
        else float("nan")
    )

    thr_passing = passing["operating_threshold"].dropna()
    threshold_mean = thr_passing.mean() if len(thr_passing) else float("nan")
    threshold_std = thr_passing.std() if len(thr_passing) else float("nan")
    threshold_cv = (
        abs(threshold_std / threshold_mean)
        if threshold_mean and not np.isnan(threshold_mean) and threshold_mean != 0
        else float("nan")
    )

    recall1_mean = eval_df["recall1_threshold"].mean()
    spec_recall1_mean = eval_df["specificity_at_recall1"].mean()
    recall_at_op_mean = passing["recall_at_operating_threshold"].mean() if len(passing) else float("nan")
    spec_at_op_mean = passing["specificity_at_operating_threshold"].mean() if len(passing) else float("nan")

    if "degenerate_filter" in eval_df.columns:
        degenerate_rate = float(eval_df["degenerate_filter"].fillna(False).mean())
    else:
        degenerate_rate = float("nan")

    if "filter_direction" in passing.columns and len(passing):
        dir_vals = passing["filter_direction"].replace("", np.nan).dropna()
        filter_direction_majority = dir_vals.mode().iloc[0] if len(dir_vals) else ""
    else:
        filter_direction_majority = ""

    scenario_vals = passing["scenario"].replace("failed", np.nan).dropna()
    scenario_majority = int(scenario_vals.mode().iloc[0]) if len(scenario_vals) > 0 else float("nan")

    if pass_rate >= args.pass_rate:
        role = "hard_filter" if scenario_majority == 1 else "permissive_filter"
    else:
        role = "rank_input"

    degenerate_filter = False
    if not np.isnan(spec_at_op_mean) and spec_at_op_mean < args.min_specificity:
        role = "rank_input"
        degenerate_filter = True
    elif not np.isnan(degenerate_rate) and degenerate_rate >= 0.5:
        role = "rank_input"
        degenerate_filter = True

    return {
        "pass_rate": pass_rate,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_cv": auc_cv,
        "threshold_mean": threshold_mean,
        "threshold_std": threshold_std,
        "threshold_cv": threshold_cv,
        "recall1_threshold_mean": recall1_mean,
        "specificity_at_recall1_mean": spec_recall1_mean,
        "recall_at_operating_threshold_mean": recall_at_op_mean,
        "specificity_at_operating_threshold_mean": spec_at_op_mean,
        "scenario_majority": scenario_majority,
        "filter_direction_majority": filter_direction_majority,
        "degenerate_filter": degenerate_filter,
        "degenerate_fold_rate": degenerate_rate,
        "role": role,
        "n_folds_passed": len(passing),
        "n_valid_folds": n_valid,
        **_scenario2_summary_means(eval_df),
    }


def summarize_feature_across_folds(
    fold_results: pd.DataFrame,
    feature: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict:
    """Cross-fold summarization and final role assignment."""
    stats = _summarize_folds_common(fold_results, None, args)
    pass_rate = stats["pass_rate"]
    threshold_cv = stats["threshold_cv"]
    scenario_majority = stats["scenario_majority"]
    role_final = stats["role"]
    thr_cv_str = f"{threshold_cv:.3f}" if pd.notna(threshold_cv) else "NA"

    if pass_rate >= args.pass_rate:
        reason = (
            f"pass_rate={pass_rate:.2f}>={args.pass_rate}, "
            f"threshold_cv={thr_cv_str} (reference<={args.threshold_cv}), "
            f"majority scenario={scenario_majority}"
        )
    else:
        reason = f"pass_rate={pass_rate:.2f}<{args.pass_rate} (too many fold failures)"

    if stats.get("degenerate_filter"):
        reason += (
            f"; degenerate filter (mean specificity="
            f"{stats.get('specificity_at_operating_threshold_mean', float('nan')):.4f} "
            f"< {args.min_specificity})"
        )

    logger.info("Feature '%s' final role=%s: %s", feature, role_final, reason)

    if role_final in ("hard_filter", "permissive_filter"):
        recommendation = "use_as_filter"
    else:
        recommendation = "use_in_ranker"

    if stats.get("degenerate_filter"):
        logger.info(
            "Feature '%s' demoted to rank_input: degenerate filter "
            "(mean specificity=%.4f, degenerate fold rate=%.2f)",
            feature,
            stats.get("specificity_at_operating_threshold_mean", float("nan")),
            stats.get("degenerate_fold_rate", float("nan")),
        )

    return {
        "feature": feature,
        "pass_rate": stats["pass_rate"],
        "auc_mean": stats["auc_mean"],
        "auc_std": stats["auc_std"],
        "auc_cv": stats["auc_cv"],
        "threshold_mean": stats["threshold_mean"],
        "threshold_std": stats["threshold_std"],
        "threshold_cv": stats["threshold_cv"],
        "recall1_threshold_mean": stats["recall1_threshold_mean"],
        "specificity_at_recall1_mean": stats["specificity_at_recall1_mean"],
        "scenario_majority": stats["scenario_majority"],
        "filter_direction_majority": stats.get("filter_direction_majority", ""),
        "degenerate_filter": stats.get("degenerate_filter", False),
        "degenerate_fold_rate": stats.get("degenerate_fold_rate", float("nan")),
        "role_final": role_final,
        "n_folds_passed": stats["n_folds_passed"],
        "recommendation": recommendation,
        "recommended_operating_threshold": stats["threshold_mean"],
        "recall_at_operating_threshold_mean": stats["recall_at_operating_threshold_mean"],
        "specificity_at_operating_threshold_mean": stats["specificity_at_operating_threshold_mean"],
        **{f"{key}_mean": stats.get(f"{key}_mean", float("nan")) for key in SCENARIO2_DIAGNOSTIC_KEYS},
    }


# ---------------------------------------------------------------------------
# Pairwise logistic regression
# ---------------------------------------------------------------------------

def run_pairwise_analysis(
    feature_fold_failures: dict[str, list[bool]],
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    train: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Test pairs of features that failed both scenarios in >=50% of folds.
    Returns (pairwise_results_df, feature->partner map).
    """
    n_folds = len(fold_splits)
    candidates = []
    for f in features:
        failures = feature_fold_failures.get(f, [True] * n_folds)
        if sum(failures) >= 0.5 * n_folds:
            candidates.append(f)

    logger.info(
        "Pairwise analysis: %d candidate features (failed both scenarios in >=50%% folds)",
        len(candidates),
    )

    rows = []
    rescued_partners: dict[str, str] = {}

    for f1, f2 in combinations(candidates, 2):
        joint_pass_count = 0
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            x1_train = train.iloc[train_idx][f1].values.astype(float)
            x2_train = train.iloc[train_idx][f2].values.astype(float)
            x1_val = train.iloc[val_idx][f1].values.astype(float)
            x2_val = train.iloc[val_idx][f2].values.astype(float)
            y_train = y[train_idx]
            y_val = y[val_idx]

            mask_train = ~(np.isnan(x1_train) | np.isnan(x2_train))
            mask_val = ~(np.isnan(x1_val) | np.isnan(x2_val))

            if mask_train.sum() < 10 or len(np.unique(y_train[mask_train])) < 2:
                continue
            if mask_val.sum() < 2 or len(np.unique(y_val[mask_val])) < 2:
                continue

            X_train = np.column_stack([x1_train[mask_train], x2_train[mask_train]])
            X_val = np.column_stack([x1_val[mask_val], x2_val[mask_val]])
            y_tr = y_train[mask_train]
            y_vl = y_val[mask_val]

            try:
                auc1 = roc_auc_score(y_vl, x1_val[mask_val])
                auc2 = roc_auc_score(y_vl, x2_val[mask_val])
                if auc1 < 0.5:
                    auc1 = 1.0 - auc1
                if auc2 < 0.5:
                    auc2 = 1.0 - auc2
            except ValueError:
                continue

            try:
                clf = LogisticRegression(max_iter=1000, random_state=random_seed)
                clf.fit(X_train, y_tr)
                joint_proba = clf.predict_proba(X_val)[:, 1]
                joint_auc = roc_auc_score(y_vl, joint_proba)
                if joint_auc < 0.5:
                    joint_auc = 1.0 - joint_auc
            except Exception as e:
                logger.debug("Pairwise %s+%s fold %d failed: %s", f1, f2, fold, e)
                continue

            best_ind = max(auc1, auc2)
            improvement = joint_auc - best_ind
            passes_joint = joint_auc >= args.auc_threshold
            if passes_joint:
                joint_pass_count += 1

            rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "fold": fold,
                "individual_auc_feature_1": auc1,
                "individual_auc_feature_2": auc2,
                "joint_auc": joint_auc,
                "improvement_over_best_individual": improvement,
                "passes_joint": passes_joint,
            })

        if joint_pass_count >= 9:
            logger.info(
                "Pair %s + %s: jointly discriminative in %d/%d folds",
                f1, f2, joint_pass_count, n_folds,
            )
            if f1 not in rescued_partners:
                rescued_partners[f1] = f2
            if f2 not in rescued_partners:
                rescued_partners[f2] = f1

    pairwise_dir = output_dir / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    pairwise_df = pd.DataFrame(rows)
    pairwise_path = pairwise_dir / "pairwise_logistic_results.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    logger.info("Saved pairwise results to %s (%d rows)", pairwise_path, len(pairwise_df))
    return pairwise_df, rescued_partners


# ---------------------------------------------------------------------------
# Orthogonality
# ---------------------------------------------------------------------------

def run_orthogonality_analysis(
    train: pd.DataFrame,
    filter_features: list[str],
    y: np.ndarray,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Pearson/Spearman correlations and VIF on non-target residues."""
    ortho_dir = output_dir / "orthogonality"
    ortho_dir.mkdir(parents=True, exist_ok=True)

    nontarget_mask = ~y.astype(bool)
    subset = train.loc[nontarget_mask, filter_features].dropna()
    if len(subset) < 2 or len(filter_features) < 2:
        logger.warning("Insufficient data for orthogonality analysis")
        return pd.DataFrame(columns=["feature", "vif", "redundant"])

    pearson = subset.corr(method="pearson")
    spearman = subset.corr(method="spearman")
    pearson.to_csv(ortho_dir / "correlation_matrix_nontarget_pearson.csv")
    spearman.to_csv(ortho_dir / "correlation_matrix_nontarget_spearman.csv")
    logger.info("Saved correlation matrices for %d filter features", len(filter_features))

    vif_rows = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = subset.values.astype(float)
        for i, feat in enumerate(filter_features):
            try:
                vif = float(variance_inflation_factor(X, i))
            except Exception:
                vif = float("nan")
            vif_rows.append({
                "feature": feat,
                "vif": vif,
                "redundant": vif > 5 if not np.isnan(vif) else False,
            })
    except ImportError:
        logger.warning("statsmodels not installed; skipping VIF calculation")
        for feat in filter_features:
            vif_rows.append({"feature": feat, "vif": float("nan"), "redundant": False})

    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(ortho_dir / "vif_scores.csv", index=False)
    logger.info("Saved VIF scores to %s", ortho_dir / "vif_scores.csv")
    return vif_df


# ---------------------------------------------------------------------------
# Per-residue-type threshold analysis
# ---------------------------------------------------------------------------

MIN_FOLD_TRAIN_POSITIVES = 10

COMBO_DEDUP_GROUP_COLS = ("Name", "Warhead", "Residue")


def min_valid_folds_required(n_folds: int) -> int:
    """Require at least 7 of 10 folds, scaled for smaller n_folds."""
    if n_folds >= 10:
        return 7
    return max(1, math.ceil(0.7 * n_folds))


def build_combo_dedup_frame(
    train: pd.DataFrame,
    feature: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict]:
    """
    Collapse site-level rows to one row per (Name, Warhead, Residue).

    is_target is True when any site in the combo is labeled. The feature value
    is taken from the first row in each group (expected constant for residue-type
    descriptors).
    """
    df = train.copy()
    df["_Residue_norm"] = df["Residue"].astype(str).str.strip().str.upper()
    df["_Name_norm"] = df["Name"].astype(str).str.strip()
    df["_Warhead_norm"] = df["Warhead"].astype(str).str.strip()
    df["_score"] = pd.to_numeric(df[feature], errors="coerce")

    grouped = df.groupby(["_Name_norm", "_Warhead_norm", "_Residue_norm"], dropna=False)
    n_groups = grouped.ngroups
    n_varying = int(grouped["_score"].nunique().gt(1).sum())
    constancy_pct = 100.0 * (n_groups - n_varying) / n_groups if n_groups else 100.0

    if n_varying > 0:
        logger.warning(
            "Feature '%s': %d / %d (Name, Warhead, Residue) groups have varying "
            "values — combo-level analysis may be inappropriate",
            feature, n_varying, n_groups,
        )

    dedup = grouped.agg(
        is_target=("is_target", "any"),
        _cluster_group=("_cluster_group", "first"),
        _protein_id=("_protein_id", "first"),
        score=("_score", "first"),
        n_sites=("is_target", "size"),
        n_target_sites=("is_target", "sum"),
    ).reset_index()
    dedup = dedup.rename(columns={
        "_Name_norm": "Name",
        "_Warhead_norm": "Warhead",
        "_Residue_norm": "Residue",
    })

    meta = {
        "n_groups": n_groups,
        "n_varying_groups": n_varying,
        "feature_constant_within_group_pct": constancy_pct,
    }
    return dedup, meta


def _analyze_combo_residue_type_feature(
    feature: str,
    residue_type: str,
    train: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    random_seed: int,
    logger: logging.Logger,
) -> dict:
    """Combo-deduplicated per-residue-type threshold analysis."""
    dedup, dedup_meta = build_combo_dedup_frame(train, feature, logger)
    residue_type = residue_type.upper()
    residue_mask = dedup["Residue"].astype(str).str.upper() == residue_type
    subset = dedup.loc[residue_mask]
    n_positives = int(subset["is_target"].sum())
    n_combos = len(subset)
    n_sites_total = int(subset["n_sites"].sum()) if n_combos else 0

    base_row = {
        "feature": feature,
        "residue_type": residue_type,
        "analysis_mode": "combo_dedup",
        "n_positives": n_positives,
        "n_combos": n_combos,
        "n_sites_total": n_sites_total,
        "feature_constant_within_group_pct": dedup_meta["feature_constant_within_group_pct"],
        "n_varying_groups": dedup_meta["n_varying_groups"],
    }

    if n_positives < args.min_residue_positives:
        msg = (
            f"Skipping combo-level {feature}/{residue_type}: {n_positives} positive combos "
            f"(<{args.min_residue_positives} required; {n_combos} combos, "
            f"{n_sites_total} site rows collapsed)"
        )
        logger.info(msg)
        print(f"[INFO] {msg}")
        return {
            **base_row,
            "sufficient_data": False,
            "role": "insufficient_data",
            "threshold_mean": float("nan"),
            "auc_mean": float("nan"),
            "pass_rate": float("nan"),
            "threshold_cv": float("nan"),
            "recall1_threshold_mean": float("nan"),
            "specificity_at_recall1_mean": float("nan"),
            "recall_at_operating_threshold_mean": float("nan"),
            "specificity_at_operating_threshold_mean": float("nan"),
            "filter_direction_majority": "",
            "degenerate_filter": False,
            "degenerate_fold_rate": float("nan"),
            "scenario_majority": float("nan"),
            "caveat": "insufficient positive combos for combo-level calibration",
        }

    scores = subset["score"].values.astype(float)
    y = subset["is_target"].values.astype(int)
    groups = subset["_cluster_group"].values
    valid_mask = ~np.isnan(scores)

    if valid_mask.sum() < 10 or np.nanstd(scores[valid_mask]) == 0:
        logger.info(
            "Skipping combo-level %s/%s: insufficient or zero-variance data after NaN removal",
            feature, residue_type,
        )
        return {
            **base_row,
            "sufficient_data": False,
            "role": "insufficient_data",
            "threshold_mean": float("nan"),
            "auc_mean": float("nan"),
            "pass_rate": float("nan"),
            "threshold_cv": float("nan"),
            "recall1_threshold_mean": float("nan"),
            "specificity_at_recall1_mean": float("nan"),
            "recall_at_operating_threshold_mean": float("nan"),
            "specificity_at_operating_threshold_mean": float("nan"),
            "filter_direction_majority": "",
            "degenerate_filter": False,
            "degenerate_fold_rate": float("nan"),
            "scenario_majority": float("nan"),
            "caveat": "insufficient or zero-variance combo-level data",
        }

    n_unique_groups = len(np.unique(groups[valid_mask]))
    n_folds = min(args.n_folds, n_unique_groups)
    if n_folds < args.n_folds:
        logger.warning(
            "Combo-level %s/%s: only %d cluster groups; reducing n_folds from %d to %d",
            feature, residue_type, n_unique_groups, args.n_folds, n_folds,
        )

    fold_splits = build_fold_splits(groups[valid_mask], n_folds, random_seed)
    min_valid = min_valid_folds_required(n_folds)
    y_valid = y[valid_mask]
    scores_valid = scores[valid_mask]

    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        if len(train_idx) == 0:
            fold_rows.append({
                "fold": fold,
                "residue_type": residue_type,
                "analysis_mode": "combo_dedup",
                "n_train_residues": 0,
                "n_train_targets": 0,
                "n_val_residues": len(val_idx),
                "n_val_targets": int(y_valid[val_idx].sum()) if len(val_idx) else 0,
                "fold_skipped_reason": "skipped_no_training_rows",
                "auc": float("nan"),
                "mw_pval": float("nan"),
                "effect_size": float("nan"),
                "scenario": "failed",
                "operating_threshold": float("nan"),
                "recall_at_operating_threshold": float("nan"),
                "specificity_at_operating_threshold": float("nan"),
                "recall1_threshold": float("nan"),
                "specificity_at_recall1": float("nan"),
                "hypergeom_pval_at_operating_threshold": float("nan"),
                "role": "rank_input",
            })
            continue

        row = analyze_residue_feature_in_fold(
            train_idx, val_idx, y_valid, scores_valid, fold, residue_type, args, logger, feature
        )
        row["analysis_mode"] = "combo_dedup"
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_path = output_dir / f"{feature}_{residue_type}_fold_results.csv"
    fold_df.to_csv(fold_path, index=False)

    valid_folds = fold_df[fold_df["fold_skipped_reason"].fillna("") == ""]
    n_valid = len(valid_folds)

    if n_valid < min_valid:
        msg = (
            f"Combo-level {feature}/{residue_type}: only {n_valid}/{n_folds} valid folds "
            f"(<{min_valid} required)"
        )
        logger.warning(msg)
        print(f"[WARN] {msg}")
        return {
            **base_row,
            "sufficient_data": False,
            "role": "insufficient_data",
            "threshold_mean": valid_folds["operating_threshold"].mean() if n_valid else float("nan"),
            "auc_mean": valid_folds["auc"].mean() if n_valid else float("nan"),
            "pass_rate": float("nan"),
            "threshold_cv": float("nan"),
            "recall1_threshold_mean": float("nan"),
            "specificity_at_recall1_mean": float("nan"),
            "recall_at_operating_threshold_mean": float("nan"),
            "specificity_at_operating_threshold_mean": float("nan"),
            "filter_direction_majority": "",
            "degenerate_filter": False,
            "degenerate_fold_rate": float("nan"),
            "scenario_majority": float("nan"),
            "caveat": f"only {n_valid} valid folds (need {min_valid})",
        }

    stats = summarize_residue_across_folds(fold_df, args)
    role = stats["role_residue"]
    degenerate = stats.get("degenerate_filter_residue", False)

    caveat = ""
    if dedup_meta["n_varying_groups"] > 0:
        caveat = (
            f"feature varies within {dedup_meta['n_varying_groups']} "
            f"(Name, Warhead, Residue) groups — not a pure residue-type descriptor"
        )
    if degenerate:
        caveat = (
            "degenerate filter — specificity too low for practical use "
            f"(mean={stats.get('specificity_at_operating_threshold_mean_residue', float('nan')):.4f})"
        )
    elif role == "rank_input" and not caveat:
        caveat = "combo-level filter did not pass fold stability criteria"

    combo_export = subset.loc[valid_mask].copy()
    combo_export["feature_value"] = scores_valid
    combo_export["role_combo_level"] = role
    combo_export["operating_threshold_mean"] = stats["threshold_mean_residue"]
    combo_export["filter_direction"] = stats.get("filter_direction_majority_residue", "")
    combo_export["degenerate_filter"] = degenerate
    combo_path = output_dir / f"{feature}_{residue_type}_combo_data.csv"
    combo_export.to_csv(combo_path, index=False)

    return {
        **base_row,
        "sufficient_data": True,
        "role": role,
        "threshold_mean": stats["threshold_mean_residue"],
        "auc_mean": stats["auc_mean_residue"],
        "pass_rate": stats["pass_rate_residue"],
        "threshold_cv": stats["threshold_cv_residue"],
        "recall1_threshold_mean": stats["recall1_threshold_mean_residue"],
        "specificity_at_recall1_mean": stats["specificity_at_recall1_mean_residue"],
        "recall_at_operating_threshold_mean": stats["recall_at_operating_threshold_mean_residue"],
        "specificity_at_operating_threshold_mean": stats[
            "specificity_at_operating_threshold_mean_residue"
        ],
        "filter_direction_majority": stats.get("filter_direction_majority_residue", ""),
        "degenerate_filter": degenerate,
        "degenerate_fold_rate": stats.get("degenerate_fold_rate_residue", float("nan")),
        "scenario_majority": stats["scenario_majority_residue"],
        **{
            f"{key}_mean": stats.get(f"{key}_mean_residue", float("nan"))
            for key in SCENARIO2_DIAGNOSTIC_KEYS
        },
        "caveat": caveat,
    }


def run_residue_type_level_analysis(
    train: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Analyze residue-type-level features at (Name, Warhead, Residue) granularity.
    """
    rtl_dir = output_dir / "residue_type_level"
    summary_dir = output_dir / "summary"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    if args.residue_types:
        residue_types = [r.upper() for r in args.residue_types]
    else:
        residue_types = sorted(train["Residue"].astype(str).str.strip().str.upper().unique())

    result_rows = []
    combos = [(feature, residue_type) for feature in features for residue_type in residue_types]

    for feature, residue_type in tqdm(combos, desc="Residue-type-level analysis", unit="combo"):
        try:
            row = _analyze_combo_residue_type_feature(
                feature=feature,
                residue_type=residue_type,
                train=train,
                args=args,
                output_dir=rtl_dir,
                random_seed=args.random_seed,
                logger=logger,
            )
            result_rows.append(row)
        except Exception as e:
            logger.error(
                "Combo-level analysis failed for %s / %s: %s",
                feature, residue_type, e, exc_info=True,
            )
            print(f"[ERROR] Combo-level analysis failed for {feature}/{residue_type}: {e}")
            result_rows.append({
                "feature": feature,
                "residue_type": residue_type,
                "analysis_mode": "combo_dedup",
                "sufficient_data": False,
                "role": "insufficient_data",
                "caveat": f"analysis error: {e}",
            })

    results_df = pd.DataFrame(result_rows)
    results_path = rtl_dir / "residue_type_level_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved combo-level results to %s", results_path)

    summary_rows = [_to_residue_type_level_summary_row(r) for r in result_rows]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = summary_dir / "residue_type_level_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved combo-level summary to %s", summary_path)

    return results_df


def _to_residue_type_level_summary_row(row: dict) -> dict:
    return {
        "feature": row.get("feature"),
        "residue_type": row.get("residue_type"),
        "analysis_mode": row.get("analysis_mode", "combo_dedup"),
        "role": row.get("role"),
        "recommended_threshold": row.get("threshold_mean", float("nan")),
        "filter_direction": row.get("filter_direction_majority", ""),
        "degenerate_filter": row.get("degenerate_filter", False),
        "auc_mean": row.get("auc_mean", float("nan")),
        "pass_rate": row.get("pass_rate", float("nan")),
        "threshold_cv": row.get("threshold_cv", float("nan")),
        "enrichment_ratio_at_operating_threshold_mean": row.get(
            "enrichment_ratio_at_operating_threshold_mean", float("nan")
        ),
        "scenario2_best_hypergeom_pval_mean": row.get(
            "scenario2_best_hypergeom_pval_mean", float("nan")
        ),
        "scenario2_best_recall_at_p_lt_0_05_mean": row.get(
            "scenario2_best_recall_at_p_lt_0_05_mean", float("nan")
        ),
        "enrichment_ratio_at_best_recall_p_lt_0_05_mean": row.get(
            "enrichment_ratio_at_best_recall_p_lt_0_05_mean", float("nan")
        ),
        "n_positives": row.get("n_positives", 0),
        "n_combos": row.get("n_combos", 0),
        "n_sites_total": row.get("n_sites_total", 0),
        "feature_constant_within_group_pct": row.get("feature_constant_within_group_pct", float("nan")),
        "sufficient_data": row.get("sufficient_data", False),
        "caveat": row.get("caveat", ""),
    }


def print_residue_type_level_summary_table(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    print("\n" + "=" * 120)
    print("  RESIDUE-TYPE-LEVEL (COMBO-DEDUP) ANALYSIS")
    print("  One row per (Name, Warhead, Residue); positive = any labeled site in combo")
    print("=" * 120)

    display_cols = [
        "feature", "residue_type", "role", "n_positives", "n_combos",
        "threshold_mean", "filter_direction_majority", "auc_mean", "pass_rate",
        "feature_constant_within_group_pct", "caveat",
    ]
    sub = results_df.reindex(columns=display_cols).copy()
    for col in ("threshold_mean", "auc_mean", "pass_rate", "feature_constant_within_group_pct"):
        if col in sub.columns:
            sub[col] = sub[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    print(sub.to_string(index=False))
    print("=" * 120)


def reconcile_roles(overall_role: str, per_residue_role: str) -> str:
    if per_residue_role == "insufficient_data":
        if overall_role in ("hard_filter", "permissive_filter"):
            return overall_role
        return "rank_input"

    table = {
        ("hard_filter", "hard_filter"): "hard_filter",
        ("hard_filter", "permissive_filter"): "permissive_filter",
        ("hard_filter", "rank_input"): "rank_input",
        ("permissive_filter", "hard_filter"): "hard_filter",
        ("permissive_filter", "permissive_filter"): "permissive_filter",
        ("permissive_filter", "rank_input"): "rank_input",
        ("rank_input", "hard_filter"): "hard_filter",
        ("rank_input", "permissive_filter"): "permissive_filter",
        ("rank_input", "rank_input"): "rank_input",
    }
    return table.get((overall_role, per_residue_role), per_residue_role)


def compare_threshold_recommendation(
    threshold_mean_overall: float,
    threshold_mean_residue: float,
    threshold_diff_pct_limit: float,
) -> str:
    if (
        np.isnan(threshold_mean_overall)
        or np.isnan(threshold_mean_residue)
        or threshold_mean_overall == 0
    ):
        return "use_overall"
    diff_pct = abs(threshold_mean_residue - threshold_mean_overall) / abs(threshold_mean_overall) * 100
    return "use_per_residue" if diff_pct > threshold_diff_pct_limit else "use_overall"


def compare_auc_note(auc_mean_overall: float, auc_mean_residue: float) -> str:
    if np.isnan(auc_mean_overall) or np.isnan(auc_mean_residue):
        return "comparable"
    diff = auc_mean_residue - auc_mean_overall
    if diff > 0.05:
        return "per_residue_stronger"
    if diff < -0.05:
        return "overall_stronger"
    return "comparable"


def is_rescued_by_residue_analysis(overall_role: str, reconciled_role: str) -> bool:
    return overall_role == "rank_input" and reconciled_role in ("hard_filter", "permissive_filter")


def analyze_residue_feature_in_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    fold: int,
    residue_type: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    feature: str,
) -> dict:
    """Per-residue fold analysis using pre-assigned fold indices."""
    y_train = y[train_idx]
    s_train = scores[train_idx]
    y_val = y[val_idx]
    s_val = scores[val_idx]

    row = {
        "fold": fold,
        "residue_type": residue_type,
        "n_train_residues": len(train_idx),
        "n_train_targets": int(y_train.sum()),
        "n_train_residues_this_type": len(train_idx),
        "n_train_targets_this_type": int(y_train.sum()),
        "n_val_residues": len(val_idx),
        "n_val_targets": int(y_val.sum()),
        "fold_skipped_reason": "",
        "auc": float("nan"),
        "mw_pval": float("nan"),
        "effect_size": float("nan"),
        "scenario": "failed",
        "operating_threshold": float("nan"),
        "recall_at_operating_threshold": float("nan"),
        "specificity_at_operating_threshold": float("nan"),
        "recall1_threshold": float("nan"),
        "specificity_at_recall1": float("nan"),
        "hypergeom_pval_at_operating_threshold": float("nan"),
        "operating_threshold_oriented": float("nan"),
        "recall1_threshold_oriented": float("nan"),
        "filter_direction": "",
        "higher_is_positive": True,
        "degenerate_filter": False,
        "role": "rank_input",
        **scenario2_diagnostic_defaults(),
    }

    if row["n_train_targets"] < MIN_FOLD_TRAIN_POSITIVES:
        row["fold_skipped_reason"] = "skipped_insufficient_data"
        logger.warning(
            "Feature '%s' residue %s fold %d: %d training positives (<10); skipping fold",
            feature, residue_type, fold, row["n_train_targets"],
        )
        return row

    if len(np.unique(y_train)) < 2:
        row["fold_skipped_reason"] = "skipped_single_class"
        logger.warning(
            "Feature '%s' residue %s fold %d: single class in training; skipping fold",
            feature, residue_type, fold,
        )
        return row

    try:
        auc, mw_pval, effect_size, higher_is_positive = compute_discriminative_metrics(
            y_train.astype(int), s_train
        )
    except Exception as e:
        row["fold_skipped_reason"] = f"skipped_error: {e}"
        logger.warning(
            "Feature '%s' residue %s fold %d: metric computation failed: %s",
            feature, residue_type, fold, e,
        )
        return row

    row["auc"] = auc
    row["mw_pval"] = mw_pval
    row["effect_size"] = effect_size
    row["higher_is_positive"] = higher_is_positive

    passes_step1 = (
        not np.isnan(auc)
        and auc >= args.auc_threshold
        and not np.isnan(mw_pval)
        and mw_pval < 0.05
        and not np.isnan(effect_size)
        and effect_size >= args.effect_size
    )

    try:
        if passes_step1:
            thr_result = scenario1_roc_threshold(
                y_train.astype(int), s_train, higher_is_positive, args.min_recall
            )
            row["scenario"] = 1
        else:
            thr_result = scenario2_hypergeom_threshold(
                y_train.astype(int), s_train, higher_is_positive, args.min_recall
            )
            row["scenario"] = 2 if thr_result.get("passed") else "failed"
    except Exception as e:
        row["fold_skipped_reason"] = f"skipped_threshold_error: {e}"
        logger.warning(
            "Feature '%s' residue %s fold %d: threshold setting failed: %s",
            feature, residue_type, fold, e,
        )
        return row

    thr_result = apply_raw_threshold_reporting(
        thr_result, higher_is_positive, args.min_specificity
    )
    apply_threshold_fields_to_row(row, thr_result)

    if thr_result.get("degenerate_filter"):
        logger.info(
            "Feature '%s' residue %s fold %d: degenerate filter (specificity=%.4f); "
            "raw threshold=%s (%s)",
            feature, residue_type, fold,
            thr_result.get("specificity_at_operating_threshold", float("nan")),
            thr_result.get("operating_threshold"),
            thr_result.get("filter_direction"),
        )

    row["role"] = fold_role_from_result(thr_result)
    return row


def summarize_residue_across_folds(
    fold_results: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    """Summarize per-residue fold results using only non-skipped folds."""
    valid_mask = fold_results["fold_skipped_reason"].fillna("") == ""
    stats = _summarize_folds_common(fold_results, valid_mask, args)
    return {
        "pass_rate_residue": stats["pass_rate"],
        "auc_mean_residue": stats["auc_mean"],
        "auc_std_residue": stats["auc_std"],
        "auc_cv_residue": stats["auc_cv"],
        "threshold_mean_residue": stats["threshold_mean"],
        "threshold_std_residue": stats["threshold_std"],
        "threshold_cv_residue": stats["threshold_cv"],
        "recall1_threshold_mean_residue": stats["recall1_threshold_mean"],
        "specificity_at_recall1_mean_residue": stats["specificity_at_recall1_mean"],
        "recall_at_operating_threshold_mean_residue": stats["recall_at_operating_threshold_mean"],
        "specificity_at_operating_threshold_mean_residue": stats["specificity_at_operating_threshold_mean"],
        "scenario_majority_residue": stats["scenario_majority"],
        "filter_direction_majority_residue": stats.get("filter_direction_majority", ""),
        "degenerate_filter_residue": stats.get("degenerate_filter", False),
        "degenerate_fold_rate_residue": stats.get("degenerate_fold_rate", float("nan")),
        "role_residue": stats["role"],
        "n_valid_folds": stats["n_valid_folds"],
        "n_folds_passed_residue": stats["n_folds_passed"],
        **{
            f"{key}_mean_residue": stats.get(f"{key}_mean", float("nan"))
            for key in SCENARIO2_DIAGNOSTIC_KEYS
        },
    }


def run_per_residue_analysis(
    train: pd.DataFrame,
    features: list[str],
    summary_df: pd.DataFrame,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    fold_assignment: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
    exclude_features: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Run per-residue-type threshold analysis after overall analysis.
    Returns (reconciliation_df, per_residue_summary_df, pipeline_readable_df, filter_config).
    """
    per_residue_dir = output_dir / "per_residue"
    summary_dir = output_dir / "summary"
    per_residue_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    n_folds = len(fold_splits)
    min_valid = min_valid_folds_required(n_folds)

    if args.residue_types:
        residue_types = [r.upper() for r in args.residue_types]
    else:
        residue_types = sorted(train["Residue"].astype(str).str.strip().str.upper().unique())

    overall_by_feature = summary_df.set_index("feature").to_dict("index")
    skip_features = exclude_features or set()
    analysis_features = [
        f for f in features
        if overall_by_feature.get(f, {}).get("role_final", "drop") != "drop"
        and f not in skip_features
    ]
    if skip_features:
        skipped = [f for f in features if f in skip_features]
        if skipped:
            logger.info(
                "Skipping site-level per-residue analysis for combo-level features: %s",
                ", ".join(sorted(skipped)),
            )

    reconciliation_rows = []
    per_residue_summary_rows = []

    combos = [
        (feature, residue_type)
        for feature in analysis_features
        for residue_type in residue_types
    ]

    for feature, residue_type in tqdm(combos, desc="Per-residue analysis", unit="combo"):
        try:
            row = _analyze_feature_residue_combo(
                feature=feature,
                residue_type=residue_type,
                train=train,
                y=y,
                fold_splits=fold_splits,
                fold_assignment=fold_assignment,
                overall_info=overall_by_feature.get(feature, {}),
                args=args,
                per_residue_dir=per_residue_dir,
                min_valid=min_valid,
                n_folds=n_folds,
                logger=logger,
            )
            reconciliation_rows.append(row)
            per_residue_summary_rows.append(_to_per_residue_summary_row(row))
        except Exception as e:
            logger.error(
                "Per-residue analysis failed for %s / %s: %s",
                feature, residue_type, e, exc_info=True,
            )
            print(f"[ERROR] Per-residue analysis failed for {feature}/{residue_type}: {e}")
            overall_info = overall_by_feature.get(feature, {})
            reconciliation_rows.append({
                "feature": feature,
                "residue_type": residue_type,
                "n_positives": int(
                    ((train["Residue"].astype(str).str.upper() == residue_type) & train["is_target"]).sum()
                ),
                "sufficient_data": False,
                "overall_role": overall_info.get("role_final", "rank_input"),
                "overall_threshold_mean": overall_info.get("threshold_mean", float("nan")),
                "per_residue_role": "insufficient_data",
                "per_residue_threshold_mean": float("nan"),
                "threshold_difference_pct": float("nan"),
                "threshold_recommendation": "use_overall",
                "auc_mean_overall": overall_info.get("auc_mean", float("nan")),
                "auc_mean_residue": float("nan"),
                "auc_note": "comparable",
                "reconciled_role": reconcile_roles(
                    overall_info.get("role_final", "rank_input"), "insufficient_data"
                ),
                "rescued_by_residue_analysis": False,
                "pass_rate_residue": float("nan"),
                "threshold_cv_residue": float("nan"),
                "recall1_threshold_mean_residue": float("nan"),
                "specificity_at_recall1_mean_residue": float("nan"),
                "caveat": f"analysis error: {e}",
            })

    reconciliation_df = pd.DataFrame(reconciliation_rows)
    recon_path = per_residue_dir / "per_residue_reconciliation.csv"
    reconciliation_df.to_csv(recon_path, index=False)
    logger.info("Saved per-residue reconciliation to %s", recon_path)

    per_residue_summary_df = pd.DataFrame(per_residue_summary_rows)
    pr_summary_path = summary_dir / "per_residue_summary.csv"
    per_residue_summary_df.to_csv(pr_summary_path, index=False)
    logger.info("Saved per-residue summary to %s", pr_summary_path)

    filter_config = build_final_filter_config(summary_df, reconciliation_df)
    config_path = output_dir / "final_filter_config.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(filter_config, fh, indent=2, default=_json_default)
    logger.info("Saved final filter config to %s", config_path)

    pipeline_readable_df = build_pipeline_readable_df(filter_config, reconciliation_df)
    pipeline_path = summary_dir / "pipeline_config_readable.csv"
    pipeline_readable_df.to_csv(pipeline_path, index=False)
    logger.info("Saved pipeline config readable to %s", pipeline_path)

    return reconciliation_df, per_residue_summary_df, pipeline_readable_df, filter_config


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _analyze_feature_residue_combo(
    feature: str,
    residue_type: str,
    train: pd.DataFrame,
    y: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    fold_assignment: np.ndarray,
    overall_info: dict,
    args: argparse.Namespace,
    per_residue_dir: Path,
    min_valid: int,
    n_folds: int,
    logger: logging.Logger,
) -> dict:
    overall_role = overall_info.get("role_final", "rank_input")
    overall_threshold_mean = overall_info.get("threshold_mean", float("nan"))
    auc_mean_overall = overall_info.get("auc_mean", float("nan"))

    residue_mask = train["Residue"].astype(str).str.strip().str.upper() == residue_type
    subset = train.loc[residue_mask]
    n_positives = int(subset["is_target"].sum())

    base_row = {
        "feature": feature,
        "residue_type": residue_type,
        "n_positives": n_positives,
        "overall_role": overall_role,
        "overall_threshold_mean": overall_threshold_mean,
        "auc_mean_overall": auc_mean_overall,
    }

    if n_positives < args.min_residue_positives:
        msg = (
            f"Skipping {feature}/{residue_type}: {n_positives} positives "
            f"(<{args.min_residue_positives} required)"
        )
        logger.info(msg)
        print(f"[INFO] {msg}")
        reconciled = reconcile_roles(overall_role, "insufficient_data")
        caveat = "insufficient data for residue-specific calibration"
        if overall_role in ("hard_filter", "permissive_filter"):
            caveat = "insufficient data for residue-specific calibration"
        return {
            **base_row,
            "sufficient_data": False,
            "per_residue_role": "insufficient_data",
            "per_residue_threshold_mean": float("nan"),
            "threshold_difference_pct": float("nan"),
            "threshold_recommendation": "use_overall",
            "auc_mean_residue": float("nan"),
            "auc_note": "comparable",
            "reconciled_role": reconciled,
            "rescued_by_residue_analysis": is_rescued_by_residue_analysis(overall_role, reconciled),
            "pass_rate_residue": float("nan"),
            "threshold_cv_residue": float("nan"),
            "recall1_threshold_mean_residue": float("nan"),
            "specificity_at_recall1_mean_residue": float("nan"),
            "caveat": caveat,
        }

    scores = pd.to_numeric(train[feature], errors="coerce").values
    valid_mask = residue_mask & ~np.isnan(scores)

    if valid_mask.sum() < 10 or np.nanstd(scores[valid_mask]) == 0:
        logger.info(
            "Skipping %s/%s: insufficient or zero-variance data after NaN removal",
            feature, residue_type,
        )
        reconciled = reconcile_roles(overall_role, "insufficient_data")
        return {
            **base_row,
            "sufficient_data": False,
            "per_residue_role": "insufficient_data",
            "per_residue_threshold_mean": float("nan"),
            "threshold_difference_pct": float("nan"),
            "threshold_recommendation": "use_overall",
            "auc_mean_residue": float("nan"),
            "auc_note": "comparable",
            "reconciled_role": reconciled,
            "rescued_by_residue_analysis": is_rescued_by_residue_analysis(overall_role, reconciled),
            "pass_rate_residue": float("nan"),
            "threshold_cv_residue": float("nan"),
            "recall1_threshold_mean_residue": float("nan"),
            "specificity_at_recall1_mean_residue": float("nan"),
            "caveat": "insufficient data for residue-specific calibration",
        }

    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        tr_idx = train_idx[valid_mask[train_idx]]
        va_idx = val_idx[valid_mask[val_idx]]
        if len(tr_idx) == 0:
            fold_rows.append({
                "fold": fold,
                "residue_type": residue_type,
                "n_train_residues": 0,
                "n_train_targets": 0,
                "n_train_residues_this_type": 0,
                "n_train_targets_this_type": 0,
                "n_val_residues": len(va_idx),
                "n_val_targets": int(y[va_idx].sum()) if len(va_idx) else 0,
                "fold_skipped_reason": "skipped_no_training_rows",
                "auc": float("nan"),
                "mw_pval": float("nan"),
                "effect_size": float("nan"),
                "scenario": "failed",
                "operating_threshold": float("nan"),
                "recall_at_operating_threshold": float("nan"),
                "specificity_at_operating_threshold": float("nan"),
                "recall1_threshold": float("nan"),
                "specificity_at_recall1": float("nan"),
                "hypergeom_pval_at_operating_threshold": float("nan"),
                "role": "rank_input",
            })
            continue

        row = analyze_residue_feature_in_fold(
            tr_idx, va_idx, y, scores, fold, residue_type, args, logger, feature
        )
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_path = per_residue_dir / f"{feature}_{residue_type}_fold_results.csv"
    fold_df.to_csv(fold_path, index=False)

    valid_folds = fold_df[fold_df["fold_skipped_reason"].fillna("") == ""]
    n_valid = len(valid_folds)

    if n_valid < min_valid:
        msg = (
            f"{feature}/{residue_type}: only {n_valid}/{n_folds} valid folds "
            f"(<{min_valid} required); marking insufficient_data"
        )
        logger.warning(msg)
        print(f"[WARN] {msg}")
        reconciled = reconcile_roles(overall_role, "insufficient_data")
        return {
            **base_row,
            "sufficient_data": False,
            "per_residue_role": "insufficient_data",
            "per_residue_threshold_mean": float("nan"),
            "threshold_difference_pct": float("nan"),
            "threshold_recommendation": "use_overall",
            "auc_mean_residue": valid_folds["auc"].mean() if n_valid else float("nan"),
            "auc_note": "comparable",
            "reconciled_role": reconciled,
            "rescued_by_residue_analysis": is_rescued_by_residue_analysis(overall_role, reconciled),
            "pass_rate_residue": float("nan"),
            "threshold_cv_residue": float("nan"),
            "recall1_threshold_mean_residue": float("nan"),
            "specificity_at_recall1_mean_residue": float("nan"),
            "caveat": f"only {n_valid} valid folds (need {min_valid})",
        }

    residue_stats = summarize_residue_across_folds(fold_df, args)
    per_residue_role = residue_stats["role_residue"]
    per_residue_threshold_mean = residue_stats["threshold_mean_residue"]

    if not np.isnan(overall_threshold_mean) and overall_threshold_mean != 0:
        threshold_difference_pct = (
            abs(per_residue_threshold_mean - overall_threshold_mean)
            / abs(overall_threshold_mean) * 100
        )
    else:
        threshold_difference_pct = float("nan")

    threshold_recommendation = compare_threshold_recommendation(
        overall_threshold_mean, per_residue_threshold_mean, args.threshold_diff_pct
    )
    auc_note = compare_auc_note(auc_mean_overall, residue_stats["auc_mean_residue"])
    reconciled_role = reconcile_roles(overall_role, per_residue_role)
    rescued = is_rescued_by_residue_analysis(overall_role, reconciled_role)
    degenerate_residue = residue_stats.get("degenerate_filter_residue", False)

    caveat = ""
    if per_residue_role == "insufficient_data":
        caveat = "insufficient data for residue-specific calibration"
    elif degenerate_residue:
        caveat = (
            "degenerate filter — specificity too low for practical use "
            f"(mean={residue_stats.get('specificity_at_operating_threshold_mean_residue', float('nan')):.4f})"
        )
        rescued = False
    elif overall_role in ("hard_filter", "permissive_filter") and per_residue_role == "rank_input":
        caveat = "demoted to rank_input for this residue type"
    elif rescued:
        caveat = "rescued by per-residue analysis"

    result = {
        **base_row,
        "sufficient_data": True,
        "per_residue_role": per_residue_role,
        "per_residue_threshold_mean": per_residue_threshold_mean,
        "threshold_difference_pct": threshold_difference_pct,
        "threshold_recommendation": threshold_recommendation,
        "auc_mean_residue": residue_stats["auc_mean_residue"],
        "auc_note": auc_note,
        "reconciled_role": reconciled_role,
        "rescued_by_residue_analysis": rescued,
        "pass_rate_residue": residue_stats["pass_rate_residue"],
        "threshold_cv_residue": residue_stats["threshold_cv_residue"],
        "recall1_threshold_mean_residue": residue_stats["recall1_threshold_mean_residue"],
        "specificity_at_recall1_mean_residue": residue_stats["specificity_at_recall1_mean_residue"],
        "recall_at_operating_threshold_mean_residue": residue_stats[
            "recall_at_operating_threshold_mean_residue"
        ],
        "specificity_at_operating_threshold_mean_residue": residue_stats[
            "specificity_at_operating_threshold_mean_residue"
        ],
        "filter_direction_majority_residue": residue_stats.get("filter_direction_majority_residue", ""),
        "degenerate_filter_residue": degenerate_residue,
        "degenerate_fold_rate_residue": residue_stats.get("degenerate_fold_rate_residue", float("nan")),
        **{
            f"{key}_mean_residue": residue_stats.get(f"{key}_mean_residue", float("nan"))
            for key in SCENARIO2_DIAGNOSTIC_KEYS
        },
        "caveat": caveat,
    }
    y_valid = y[valid_mask]
    s_valid = scores[valid_mask]
    auc_all, mw_all, es_all, _ = compute_discriminative_metrics(y_valid, s_valid)

    residue_df = train.loc[valid_mask].copy()
    residue_df["fold_assignment"] = fold_assignment[valid_mask]
    residue_df["auc_overall"] = auc_all
    residue_df["mw_pval_overall"] = mw_all
    residue_df["effect_size_overall"] = es_all
    residue_df["operating_threshold_mean"] = overall_threshold_mean
    residue_df["recall1_threshold_mean"] = overall_info.get("recall1_threshold_mean", float("nan"))
    residue_df["specificity_at_recall1_mean"] = overall_info.get(
        "specificity_at_recall1_mean", float("nan")
    )
    residue_df["scenario_overall"] = overall_info.get("scenario_majority", float("nan"))
    residue_df["role_final"] = overall_role
    residue_df["per_residue_threshold_mean"] = per_residue_threshold_mean
    residue_df["per_residue_role"] = per_residue_role
    residue_df["reconciled_role"] = reconciled_role
    residue_df["threshold_recommendation"] = threshold_recommendation
    residue_df["filter_direction"] = residue_stats.get("filter_direction_majority_residue", "")
    residue_df["degenerate_filter"] = degenerate_residue
    residue_path = per_residue_dir / f"{feature}_{residue_type}_residue_data.csv"
    residue_df.to_csv(residue_path, index=False)

    if rescued:
        logger.info(
            "RESCUED: %s/%s upgraded from %s to %s",
            feature, residue_type, overall_role, reconciled_role,
        )
        print(f"[INFO] RESCUED: {feature}/{residue_type}: {overall_role} -> {reconciled_role}")

    return result


def _to_per_residue_summary_row(recon_row: dict) -> dict:
    overall_thr = recon_row.get("overall_threshold_mean", float("nan"))
    residue_thr = recon_row.get("per_residue_threshold_mean", float("nan"))
    thr_rec = recon_row.get("threshold_recommendation", "use_overall")

    if thr_rec == "use_per_residue" and not np.isnan(residue_thr):
        recommended_threshold = residue_thr
        threshold_source = "per_residue"
    elif not np.isnan(overall_thr):
        recommended_threshold = overall_thr
        threshold_source = "overall"
    else:
        recommended_threshold = float("nan")
        threshold_source = "none"

    auc = recon_row.get("auc_mean_residue", float("nan"))
    if np.isnan(auc):
        auc = recon_row.get("auc_mean_overall", float("nan"))

    return {
        "feature": recon_row["feature"],
        "residue_type": recon_row["residue_type"],
        "reconciled_role": recon_row["reconciled_role"],
        "recommended_threshold": recommended_threshold,
        "threshold_source": threshold_source,
        "filter_direction": recon_row.get("filter_direction_majority_residue", ""),
        "degenerate_filter": recon_row.get("degenerate_filter_residue", False),
        "auc_mean": auc,
        "pass_rate": recon_row.get("pass_rate_residue", float("nan")),
        "threshold_cv": recon_row.get("threshold_cv_residue", float("nan")),
        "enrichment_ratio_at_operating_threshold_mean": recon_row.get(
            "enrichment_ratio_at_operating_threshold_mean_residue", float("nan")
        ),
        "scenario2_best_hypergeom_pval_mean": recon_row.get(
            "scenario2_best_hypergeom_pval_mean_residue", float("nan")
        ),
        "scenario2_best_recall_at_p_lt_0_05_mean": recon_row.get(
            "scenario2_best_recall_at_p_lt_0_05_mean_residue", float("nan")
        ),
        "enrichment_ratio_at_best_recall_p_lt_0_05_mean": recon_row.get(
            "enrichment_ratio_at_best_recall_p_lt_0_05_mean_residue", float("nan")
        ),
        "sufficient_data": recon_row.get("sufficient_data", False),
        "caveat": recon_row.get("caveat", ""),
        "rescued_by_residue_analysis": recon_row.get("rescued_by_residue_analysis", False),
    }


def build_final_filter_config(
    summary_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
) -> dict:
    config: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "overall_thresholds": {},
        "per_residue_thresholds": {},
        "pipeline_recommendation": {},
    }

    for _, row in summary_df.iterrows():
        feature = row["feature"]
        if row.get("role_final") == "drop":
            continue
        config["overall_thresholds"][feature] = {
            "role": row.get("role_final"),
            "threshold": _safe_float(row.get("threshold_mean")),
            "filter_direction": row.get("filter_direction_majority") or None,
            "degenerate_filter": bool(row.get("degenerate_filter", False)),
            "recall_at_threshold": _safe_float(row.get("recall_at_operating_threshold_mean")),
            "specificity_at_threshold": _safe_float(row.get("specificity_at_operating_threshold_mean")),
            "recall1_threshold": _safe_float(row.get("recall1_threshold_mean")),
            "specificity_at_recall1": _safe_float(row.get("specificity_at_recall1_mean")),
            "auc_mean": _safe_float(row.get("auc_mean")),
            "pass_rate": _safe_float(row.get("pass_rate")),
        }

    for feature in reconciliation_df["feature"].unique():
        config["per_residue_thresholds"][feature] = {}
        config["pipeline_recommendation"][feature] = {}

    for _, row in reconciliation_df.iterrows():
        feature = row["feature"]
        residue = row["residue_type"]
        reconciled = row["reconciled_role"]
        thr_rec = row.get("threshold_recommendation", "use_overall")

        if row.get("per_residue_role") == "insufficient_data" or not row.get("sufficient_data"):
            config["per_residue_thresholds"][feature][residue] = {
                "role": "insufficient_data",
                "n_positives": int(row.get("n_positives", 0)),
                "threshold_recommendation": thr_rec,
                "reconciled_role": reconciled,
                "caveat": row.get("caveat", "insufficient data for residue-specific calibration"),
            }
            config["pipeline_recommendation"][feature][residue] = _pipeline_entry_for_row(row, summary_df)
            continue

        config["per_residue_thresholds"][feature][residue] = {
            "role": row.get("per_residue_role"),
            "threshold": _safe_float(row.get("per_residue_threshold_mean")),
            "filter_direction": row.get("filter_direction_majority_residue") or None,
            "degenerate_filter": bool(row.get("degenerate_filter_residue", False)),
            "threshold_recommendation": thr_rec,
            "threshold_difference_pct": _safe_float(row.get("threshold_difference_pct")),
            "auc_mean_residue": _safe_float(row.get("auc_mean_residue")),
            "auc_note": row.get("auc_note"),
            "recall_at_threshold": _safe_float(row.get("recall_at_operating_threshold_mean_residue")),
            "specificity_at_threshold": _safe_float(row.get("specificity_at_operating_threshold_mean_residue")),
            "recall1_threshold": _safe_float(row.get("recall1_threshold_mean_residue")),
            "specificity_at_recall1": _safe_float(row.get("specificity_at_recall1_mean_residue")),
            "reconciled_role": reconciled,
            "pass_rate_residue": _safe_float(row.get("pass_rate_residue")),
            "rescued_by_residue_analysis": bool(row.get("rescued_by_residue_analysis")),
        }
        config["pipeline_recommendation"][feature][residue] = _pipeline_entry_for_row(row, summary_df)

    return config


def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _pipeline_entry_for_row(row: pd.Series | dict, summary_df: pd.DataFrame) -> dict:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    feature = row["feature"]
    reconciled = row["reconciled_role"]
    thr_rec = row.get("threshold_recommendation", "use_overall")
    overall_thr = row.get("overall_threshold_mean", float("nan"))
    residue_thr = row.get("per_residue_threshold_mean", float("nan"))
    caveat = row.get("caveat", "")
    filter_dir = (
        row.get("filter_direction_majority_residue")
        or row.get("filter_direction_majority")
        or ""
    )
    is_degenerate = bool(
        row.get("degenerate_filter_residue") or row.get("degenerate_filter")
    )

    if reconciled in ("hard_filter", "permissive_filter") and not is_degenerate:
        if thr_rec == "use_per_residue" and not np.isnan(residue_thr):
            threshold = _safe_float(residue_thr)
            note = ""
        elif not np.isnan(overall_thr):
            threshold = _safe_float(overall_thr)
            note = "using overall threshold"
        else:
            threshold = None
            note = caveat or "no threshold available"
        entry = {
            "apply_as": reconciled,
            "threshold": threshold,
            "filter_direction": filter_dir or None,
        }
        if note:
            entry["note"] = note
        if row.get("rescued_by_residue_analysis"):
            entry["note"] = (entry.get("note", "") + " rescued by per-residue analysis").strip()
        return entry

    if is_degenerate:
        return {
            "apply_as": "rank_input",
            "threshold": None,
            "filter_direction": filter_dir or None,
            "note": caveat or "degenerate filter — specificity too low",
        }

    entry = {"apply_as": "rank_input", "threshold": None}
    if row.get("per_residue_role") == "insufficient_data":
        entry["note"] = caveat or "insufficient data"
    elif row.get("rescued_by_residue_analysis"):
        entry["note"] = "rescued only in rank stage"
    elif caveat:
        entry["note"] = caveat
    else:
        entry["note"] = "use in ranker stage only"
    return entry


def build_pipeline_readable_df(filter_config: dict, reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in reconciliation_df.iterrows():
        feature = row["feature"]
        residue = row["residue_type"]
        pipeline = filter_config.get("pipeline_recommendation", {}).get(feature, {}).get(residue, {})
        rows.append({
            "feature": feature,
            "residue_type": residue,
            "apply_as": pipeline.get("apply_as"),
            "threshold": pipeline.get("threshold"),
            "threshold_source": row.get("threshold_recommendation"),
            "reconciled_role": row.get("reconciled_role"),
            "overall_role": row.get("overall_role"),
            "per_residue_role": row.get("per_residue_role"),
            "note": pipeline.get("note", ""),
            "rescued_by_residue_analysis": row.get("rescued_by_residue_analysis", False),
            "sufficient_data": row.get("sufficient_data", False),
        })
    return pd.DataFrame(rows)


def print_per_residue_summary_table(
    summary_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    residue_types: list[str],
) -> None:
    """Print extended summary with per-residue reconciled roles."""
    if reconciliation_df.empty:
        return

    features = summary_df.loc[
        summary_df["role_final"] != "drop", "feature"
    ].tolist()

    print("\n" + "=" * 120)
    print("  PER-RESIDUE RECONCILED ROLES")
    print("=" * 120)
    print(
        f"{'Feature':<22}| {'Overall Role':<16}| "
        + "| ".join(f"{r:<12}" for r in residue_types[:8])
        + "| Notes"
    )
    print("-" * 120)

    for feature in features:
        overall_row = summary_df.loc[summary_df["feature"] == feature]
        if overall_row.empty:
            continue
        overall_role = overall_row.iloc[0]["role_final"]
        notes_parts = []

        res_cells = []
        for residue in residue_types[:8]:
            sub = reconciliation_df[
                (reconciliation_df["feature"] == feature)
                & (reconciliation_df["residue_type"] == residue)
            ]
            if sub.empty:
                res_cells.append("-")
                continue
            r = sub.iloc[0]
            reconciled = r["reconciled_role"]
            cell = reconciled[:12] if len(reconciled) <= 12 else reconciled[:10] + ".."

            if r.get("per_residue_role") == "insufficient_data":
                cell = "insufficient"[:12]
            elif r.get("threshold_recommendation") == "use_overall" and reconciled in (
                "hard_filter", "permissive_filter"
            ):
                cell = f"{reconciled[:4]}(OVR)"[:12]
            elif r.get("threshold_recommendation") == "use_per_residue" and reconciled in (
                "hard_filter", "permissive_filter"
            ):
                cell = f"{reconciled}*"[:12]
                notes_parts.append(f"{residue}: per-residue threshold differs >20%")

            if r.get("rescued_by_residue_analysis"):
                cell = f"{cell}↑"[:12]
                notes_parts.append(f"{residue}: rescued")
            elif (
                overall_role in ("hard_filter", "permissive_filter")
                and reconciled == "rank_input"
                and r.get("sufficient_data")
            ):
                cell = f"{cell}↓"[:12]
                notes_parts.append(f"{residue}: demoted")

            res_cells.append(cell)

        notes = "; ".join(notes_parts) if notes_parts else ""
        print(
            f"{feature:<22}| {overall_role:<16}| "
            + "| ".join(f"{c:<12}" for c in res_cells)
            + f"| {notes}"
        )

    print("-" * 120)
    print("Legend: OVR=using overall threshold, *=per-residue threshold differs >20%, "
          "↑=upgraded/rescued, ↓=demoted")
    print("=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_fold_splits(
    groups: np.ndarray,
    n_folds: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_folds)
    indices = np.arange(len(groups))
    splits = []
    for train_idx, val_idx in gkf.split(indices, groups=groups):
        splits.append((train_idx, val_idx))
    return splits


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("  FEATURE FILTER ANALYSIS — SUMMARY")
    print("=" * 120)
    display_cols = [
        "feature", "role_final", "pass_rate", "auc_mean",
        "threshold_mean", "filter_direction_majority",
        "degenerate_filter", "vif", "redundant_flag",
        "pairwise_partner", "recommendation",
    ]
    sub = summary_df.reindex(columns=display_cols).copy()
    sub["pass_rate"] = sub["pass_rate"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
    sub["auc_mean"] = sub["auc_mean"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    sub["threshold_mean"] = sub["threshold_mean"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "NA"
    )
    sub["degenerate_filter"] = sub["degenerate_filter"].map(
        lambda x: str(bool(x)) if pd.notna(x) else "NA"
    )
    sub["vif"] = sub["vif"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
    sub["redundant_flag"] = sub["redundant_flag"].map(
        lambda x: str(bool(x)) if pd.notna(x) else "NA"
    )
    sub["pairwise_partner"] = sub["pairwise_partner"].fillna("").astype(str)
    print(sub.to_string(index=False))
    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate feature columns as hard filters via clustered cross-validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--training", required=False, help="Path to training CSV.")
    parser.add_argument("--labels", required=False, help="Path to label CSV.")
    parser.add_argument("--clusters", default=None, help="Path to MMseqs2 cluster CSV.")
    parser.add_argument("--run-mmseqs", default=None, metavar="FASTA",
                        help="Run MMseqs2 easy-cluster on FASTA and use output clusters.")
    parser.add_argument("--features", nargs="*", default=None,
                        help="Space-separated feature column names (auto-detect if omitted).")
    parser.add_argument("--output-dir", default="./feature_analysis_output/",
                        help="Output directory (default: ./feature_analysis_output/).")
    parser.add_argument("--min-recall", type=float, default=0.95,
                        help="Minimum recall for threshold setting (default: 0.95).")
    parser.add_argument("--min-specificity", type=float, default=0.05,
                        help="Minimum specificity for a non-degenerate filter (default: 0.05).")
    parser.add_argument("--auc-threshold", type=float, default=0.70,
                        help="AUC cutoff for Scenario 1 pass (default: 0.70).")
    parser.add_argument("--effect-size", type=float, default=0.10,
                        help="Minimum rank-biserial effect size (default: 0.10).")
    parser.add_argument("--pass-rate", type=float, default=0.90,
                        help="Minimum fold pass rate for filter role (default: 0.90).")
    parser.add_argument("--threshold-cv", type=float, default=0.20,
                        help="Reference threshold CV for reporting only (default: 0.20). "
                        "Does not affect filter role assignment.")
    parser.add_argument("--n-folds", type=int, default=10, help="CV folds (default: 10).")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--min-residue-positives", type=int, default=50,
                        help="Minimum positive examples for per-residue analysis (default: 50).")
    parser.add_argument("--threshold-diff-pct", type=float, default=20.0,
                        help="Threshold %% difference to prefer per-residue threshold (default: 20).")
    parser.add_argument("--skip-per-residue", action="store_true",
                        help="Skip per-residue analysis; use overall thresholds only.")
    parser.add_argument("--residue-types", nargs="*", default=None,
                        help="Residue types for per-residue analysis (default: all in data).")
    parser.add_argument(
        "--features-residue-type", nargs="*", default=None,
        dest="features_residue_type",
        help=(
            "Features analyzed at (Name, Warhead, Residue) combo level with site "
            "deduplication (ResNum/Chain ignored). Use for residue-type descriptors "
            "constant across sites on the same protein-warhead combo."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    logger.info("Starting feature filter analysis")

    # MMseqs-only mode still needs output dir
    if args.run_mmseqs and not args.training:
        cluster_path = run_mmseqs_clustering(args.run_mmseqs, output_dir, logger)
        print(f"[INFO] Cluster CSV ready: {cluster_path}")
        print("[INFO] Re-run with --training, --labels, and --clusters to analyze features.")
        return

    if not args.training or not args.labels:
        parser.error("--training and --labels are required unless using --run-mmseqs alone.")

    # Resolve cluster file
    cluster_path = args.clusters
    if args.run_mmseqs:
        cluster_path = run_mmseqs_clustering(args.run_mmseqs, output_dir, logger)

    # Load data
    train = load_csv(args.training, "training")
    labels = load_csv(args.labels, "labels")
    train = assign_targets(train, labels, logger)

    protein_id_map = build_protein_id_map(labels, logger)
    cluster_map = None
    if cluster_path:
        cluster_map = load_cluster_map(cluster_path, logger)
    else:
        msg = (
            "No --clusters MMseqs2 file provided. Cross-validation will group by "
            "protein ID (from Proteins/UniProt column), but will NOT merge "
            "sequence-similar proteins — provide --clusters or --run-mmseqs for "
            "30% sequence-identity clustering."
        )
        logger.warning(msg)
        print(f"[WARN] {msg}")

    train = assign_cluster_groups(train, protein_id_map, cluster_map, logger)

    # Features
    residue_type_features: list[str] = []
    if args.features_residue_type:
        residue_type_features = list(args.features_residue_type)
        missing_rt = [f for f in residue_type_features if f not in train.columns]
        if missing_rt:
            sys.exit(
                f"[ERROR] --features-residue-type columns not found in training CSV: {missing_rt}"
            )

    if args.features:
        features = list(args.features)
        missing = [f for f in features if f not in train.columns]
        if missing:
            sys.exit(f"[ERROR] Feature columns not found in training CSV: {missing}")
    elif args.features_residue_type:
        features = []
    else:
        features = auto_detect_features(train)
        logger.info("Auto-detected %d feature columns", len(features))

    if not features and not residue_type_features:
        sys.exit("[ERROR] Provide --features and/or --features-residue-type.")

    overlap = set(residue_type_features) & set(features)
    if overlap:
        logger.info(
            "Features in both --features and --features-residue-type "
            "(site-level + combo-level): %s",
            ", ".join(sorted(overlap)),
        )

    # Output subdirs
    per_feature_dir = output_dir / "per_feature"
    summary_dir = output_dir / "summary"
    per_feature_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    y = train["is_target"].values.astype(int)
    groups = train["_cluster_group"].values

    n_unique_groups = len(np.unique(groups))
    n_folds = min(args.n_folds, n_unique_groups)
    if n_folds < args.n_folds:
        logger.warning(
            "Only %d unique cluster groups; reducing n_folds from %d to %d",
            n_unique_groups, args.n_folds, n_folds,
        )
        print(f"[WARN] Reducing n_folds to {n_folds} (only {n_unique_groups} cluster groups)")

    fold_splits = build_fold_splits(groups, n_folds, args.random_seed)

    # Assign fold labels to each row (validation fold)
    fold_assignment = np.full(len(train), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(fold_splits):
        fold_assignment[val_idx] = fold
    train["_fold_assignment"] = fold_assignment

    summary_rows = []
    feature_fold_failures: dict[str, list[bool]] = {}
    all_fold_results: dict[str, pd.DataFrame] = {}

    if features:
        for feature in tqdm(features, desc="Analyzing features", unit="feature"):
            scores = pd.to_numeric(train[feature], errors="coerce").values
            valid_mask = ~np.isnan(scores)

            if valid_mask.sum() < 10:
                logger.warning("Feature '%s': insufficient non-NaN values; marking as drop", feature)
                summary_rows.append({
                    "feature": feature,
                    "pass_rate": 0.0,
                    "auc_mean": float("nan"),
                    "auc_std": float("nan"),
                    "auc_cv": float("nan"),
                    "threshold_mean": float("nan"),
                    "threshold_std": float("nan"),
                    "threshold_cv": float("nan"),
                    "recall1_threshold_mean": float("nan"),
                    "specificity_at_recall1_mean": float("nan"),
                    "scenario_majority": float("nan"),
                    "role_final": "drop",
                    "n_folds_passed": 0,
                    "recommendation": "drop",
                    "recommended_operating_threshold": float("nan"),
                })
                continue

            # Zero variance check
            if np.nanstd(scores[valid_mask]) == 0:
                logger.warning("Feature '%s': zero variance; marking as drop", feature)
                summary_rows.append({
                    "feature": feature,
                    "pass_rate": 0.0,
                    "auc_mean": float("nan"),
                    "auc_std": float("nan"),
                    "auc_cv": float("nan"),
                    "threshold_mean": float("nan"),
                    "threshold_std": float("nan"),
                    "threshold_cv": float("nan"),
                    "recall1_threshold_mean": float("nan"),
                    "specificity_at_recall1_mean": float("nan"),
                    "scenario_majority": float("nan"),
                    "role_final": "drop",
                    "n_folds_passed": 0,
                    "recommendation": "drop",
                    "recommended_operating_threshold": float("nan"),
                })
                continue

            fold_rows = []
            fold_both_failed = []

            for fold, (train_idx, val_idx) in enumerate(fold_splits):
                # Restrict to non-NaN for this feature
                tr_idx = train_idx[valid_mask[train_idx]]
                va_idx = val_idx[valid_mask[val_idx]]
                if len(tr_idx) == 0:
                    continue

                row = analyze_feature_in_fold(
                    tr_idx, va_idx, y, scores, fold, args, logger, feature
                )
                fold_rows.append(row)

                fold_both_failed.append(row["role"] == "rank_input")

            feature_fold_failures[feature] = fold_both_failed
            fold_df = pd.DataFrame(fold_rows)
            all_fold_results[feature] = fold_df

            fold_path = per_feature_dir / f"{feature}_fold_results.csv"
            fold_df.to_csv(fold_path, index=False)

            # Overall metrics on full dataset (for residue export)
            y_valid = y[valid_mask]
            s_valid = scores[valid_mask]
            auc_all, mw_all, es_all, _ = compute_discriminative_metrics(y_valid, s_valid)

            summary = summarize_feature_across_folds(fold_df, feature, args, logger)
            summary_rows.append(summary)

            # Residue-level export
            residue_df = train.loc[valid_mask].copy()
            residue_df["fold_assignment"] = fold_assignment[valid_mask]
            residue_df["auc_overall"] = auc_all
            residue_df["mw_pval_overall"] = mw_all
            residue_df["effect_size_overall"] = es_all
            residue_df["operating_threshold_mean"] = summary["threshold_mean"]
            residue_df["recall1_threshold_mean"] = summary["recall1_threshold_mean"]
            residue_df["specificity_at_recall1_mean"] = summary["specificity_at_recall1_mean"]
            residue_df["scenario_overall"] = summary["scenario_majority"]
            residue_df["role_final"] = summary["role_final"]
            residue_df["filter_direction"] = summary.get("filter_direction_majority", "")
            residue_df["degenerate_filter"] = summary.get("degenerate_filter", False)
            residue_path = per_feature_dir / f"{feature}_residue_data.csv"
            residue_df.to_csv(residue_path, index=False)

        pairwise_df, rescued_partners = run_pairwise_analysis(
            feature_fold_failures, fold_splits, train, features, y, args,
            output_dir, logger, args.random_seed,
        )

        summary_df = pd.DataFrame(summary_rows)
        filter_feats = summary_df.loc[
            summary_df["role_final"].isin(["hard_filter", "permissive_filter"]), "feature"
        ].tolist()

        vif_df = run_orthogonality_analysis(train, filter_feats, y, output_dir, logger)
        vif_map = dict(zip(vif_df["feature"], vif_df["vif"])) if len(vif_df) else {}
        redundant_map = dict(zip(vif_df["feature"], vif_df["redundant"])) if len(vif_df) else {}

        summary_df["vif"] = summary_df["feature"].map(lambda f: vif_map.get(f, float("nan")))
        summary_df["redundant_flag"] = summary_df["feature"].map(
            lambda f: redundant_map.get(f, False)
        )
        summary_df["pairwise_partner"] = summary_df["feature"].map(
            lambda f: rescued_partners.get(f, "")
        )

        summary_path = summary_dir / "feature_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved feature summary to %s", summary_path)

        print_summary_table(summary_df)

        if not args.skip_per_residue:
            logger.info("Starting per-residue-type threshold analysis")
            print("\n[INFO] Running per-residue-type threshold analysis ...")
            reconciliation_df, _, _, _ = run_per_residue_analysis(
                train=train,
                features=features,
                summary_df=summary_df,
                fold_splits=fold_splits,
                fold_assignment=fold_assignment,
                y=y,
                args=args,
                output_dir=output_dir,
                logger=logger,
                exclude_features=set(residue_type_features),
            )
            residue_types_for_table = (
                [r.upper() for r in args.residue_types]
                if args.residue_types
                else sorted(train["Residue"].astype(str).str.strip().str.upper().unique())
            )
            print_per_residue_summary_table(summary_df, reconciliation_df, residue_types_for_table)
        else:
            logger.info("Per-residue analysis skipped (--skip-per-residue)")
            print("[INFO] Per-residue analysis skipped (--skip-per-residue)")
    else:
        logger.info("No --features specified; skipping site-level feature analysis")
        print("[INFO] No site-level features to analyze (--features-residue-type only run)")

    if residue_type_features:
        logger.info(
            "Starting residue-type-level combo-dedup analysis for %d feature(s)",
            len(residue_type_features),
        )
        print(
            f"\n[INFO] Running residue-type-level combo analysis for: "
            f"{', '.join(residue_type_features)} ..."
        )
        rtl_results_df = run_residue_type_level_analysis(
            train=train,
            features=residue_type_features,
            args=args,
            output_dir=output_dir,
            logger=logger,
        )
        print_residue_type_level_summary_table(rtl_results_df)

    print(f"\n[INFO] All outputs written to: {output_dir.resolve()}")
    print(f"[INFO] Log file: {(output_dir / 'analysis.log').resolve()}")


if __name__ == "__main__":
    main()
