"""
LGBMRanker for Nucleophilic Residue Ranking
============================================
Ranks residues within a protein by predicted covalent reactivity
using LightGBM's LambdaRank objective with DCG-style scoring.

Usage:
    python lgbm_ranker.py --training training.csv --labels labels.csv \
        --features Abs_Side_SASA deprotonation_prob Reactivity_Score \
        --topk 10 --pdb_folder ./pdbs

    # Train without residue type one-hot inputs (res_CYS, res_SER, ...):
    python Training_Cov_Screen.py --training training.csv --labels labels.csv \
        --no-residue-type

CV and test-set CSVs written to --output_dir:
    cv_fold_results.csv  (per-fold train-split CV metrics + mean row)
    test_label_results.csv, test_query_groups.csv,
    test_warhead_accuracy.csv, test_residue_accuracy.csv,
    test_shap_per_candidate.csv, test_shap_global.csv

Pinned clustering / splits (for ablation runs on an identical held-out set):
    # Full model — cluster once, export splits
    python Training_Cov_Screen.py ... --deep-cluster \\
        --export-split query_group_splits.csv

    # Ablations — reuse clusters.tsv + split CSV (skip MMseqs2 reruns)
    python Training_Cov_Screen.py ... --deep-cluster \\
        --clusters-tsv mmseqs_output/clusters.tsv \\
        --split-csv query_group_splits.csv
"""

import os
import re
import sys
import json
import pickle
import argparse
import warnings
import shutil
import requests
import subprocess
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

VALID_RESIDUES = {"CYS", "HIS", "SER", "TYR", "THR", "LYS"}
NUCLEOPHILE_RESIDUES = ("CYS", "HIS", "SER", "THR", "TYR", "LYS")

# WHy this? to boost the imablance residue where?
RARITY_BOOST = {
    "CYS": 1.0,
    "LYS": 2.0,
    "HIS": 2.5,
    "TYR": 4.0,
    "THR": 3.0,
    "SER": 1.5,
}

PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

DEFAULT_RANK_BONUS_EPSILON = 0.05
REWARD_MODES = ("hit_at_k", "hit_at_top_pct")

FILTER_OPS = {
    "gt":  lambda a, b: a >  b,
    "lt":  lambda a, b: a <  b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "eq":  lambda a, b: a == b,
}


# ─────────────────────────────────────────────
# PDB UTILITIES
# ─────────────────────────────────────────────

def download_pdb(pdb_id: str, save_folder: str) -> str | None:
    """Download a PDB file from RCSB if not found locally."""
    pdb_id = pdb_id.upper()
    save_path = os.path.join(save_folder, f"{pdb_id}.pdb")
    if os.path.exists(save_path):
        return save_path

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    print(f"  Downloading {pdb_id} from RCSB...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(save_path, "w") as f:
            f.write(r.text)
        print(f"  Saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"  WARNING: Could not download {pdb_id}: {e}")
        return None


def extract_fasta_from_pdbs(pdb_folder: str, pdb_ids: list[str], output_fasta: str) -> str:
    """
    Extract sequences from PDB files (downloading missing ones)
    and write a FASTA file for MMseqs2 clustering.
    """
    try:
        from Bio.PDB import PDBParser, PPBuilder
    except ImportError:
        print("ERROR: BioPython required. Run: pip install biopython")
        sys.exit(1)

    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    written = 0

    with open(output_fasta, "w") as fout:
        for pdb_id in set(pdb_ids):
            pdb_id_upper = pdb_id.upper()
            # Search for file case-insensitively
            pdb_path = None
            for fname in os.listdir(pdb_folder):
                if fname.upper().replace(".PDB", "") == pdb_id_upper:
                    pdb_path = os.path.join(pdb_folder, fname)
                    break

            if pdb_path is None:
                pdb_path = download_pdb(pdb_id, pdb_folder)
            
            #Doesn't work even after download
            if pdb_path is None or not os.path.exists(pdb_path):
                print(f"  WARNING: No PDB found for {pdb_id}, skipping.")
                continue

            try:
                structure = parser.get_structure(pdb_id, pdb_path)
                peptides = ppb.build_peptides(structure)
                sequence = "".join(str(pp.get_sequence()) for pp in peptides)
                if sequence:
                    fout.write(f">{pdb_id_upper}\n{sequence}\n")
                    written += 1
            except Exception as e:
                print(f"  WARNING: Could not parse {pdb_path}: {e}")

    print(f"  Wrote {written} sequences to {output_fasta}")
    return output_fasta


def extract_chain_fasta_from_pdbs(
    pdb_folder: str,
    pdb_ids: list[str],
    output_fasta: str,
    min_chain_length: int = 30,
) -> int:
    """
    Extract one FASTA entry per polymer chain (PDBID__CHAIN headers)
    for per-chain MMseqs2 clustering.
    """
    try:
        from Bio.PDB import PDBParser, PPBuilder
    except ImportError:
        print("ERROR: BioPython required. Run: pip install biopython")
        sys.exit(1)

    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    written = 0

    with open(output_fasta, "w") as fout:
        for pdb_id in set(pdb_ids):
            pdb_id_upper = pdb_id.upper()
            pdb_path = None
            for fname in os.listdir(pdb_folder):
                if fname.upper().replace(".PDB", "") == pdb_id_upper:
                    pdb_path = os.path.join(pdb_folder, fname)
                    break

            if pdb_path is None:
                pdb_path = download_pdb(pdb_id, pdb_folder)

            if pdb_path is None or not os.path.exists(pdb_path):
                print(f"  WARNING: No PDB found for {pdb_id}, skipping.")
                continue

            try:
                structure = parser.get_structure(pdb_id, pdb_path)
                for model in structure:
                    for chain in model:
                        peptides = ppb.build_peptides(chain)
                        sequence = "".join(str(pp.get_sequence()) for pp in peptides)
                        if len(sequence) < min_chain_length:
                            continue
                        chain_id = str(chain.id).strip() or " "
                        header = f"{pdb_id_upper}__{chain_id}"
                        fout.write(f">{header}\n{sequence}\n")
                        written += 1
            except Exception as e:
                print(f"  WARNING: Could not parse {pdb_path}: {e}")

    print(f"  Wrote {written} chain sequences to {output_fasta}")
    return written


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node

    def find(self, node: str) -> str:
        self.add(node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _parse_chain_fasta_header(header: str) -> tuple[str, str] | None:
    if "__" not in header:
        return None
    pdb_id, chain = header.rsplit("__", 1)
    pdb_id = pdb_id.strip().upper()
    if not pdb_id:
        return None
    return pdb_id, chain


def merge_chain_clusters_to_pdb_map(
    pdb_ids: list[str],
    chain_cluster_map: dict[str, str],
) -> dict[str, str]:
    """
    Lift per-chain MMseqs2 clusters to PDB-level clusters.

    Two PDBs share a cluster when any of their chains belong to the same
    sequence cluster (conservative leakage control for whole-structure screening).
    """
    pdb_ids_upper = [str(pid).strip().upper() for pid in pdb_ids]
    rep_to_pdbs: dict[str, list[str]] = defaultdict(list)

    for member, rep in chain_cluster_map.items():
        parsed = _parse_chain_fasta_header(member)
        if parsed is None:
            continue
        pdb_id, _chain = parsed
        rep_to_pdbs[rep.upper()].append(pdb_id)

    uf = _UnionFind()
    for pdb_id in pdb_ids_upper:
        uf.add(pdb_id)

    for pdbs in rep_to_pdbs.values():
        unique_pdbs = list(dict.fromkeys(pdbs))
        if len(unique_pdbs) < 2:
            continue
        anchor = unique_pdbs[0]
        for other in unique_pdbs[1:]:
            uf.union(anchor, other)

    components: dict[str, list[str]] = defaultdict(list)
    for pdb_id in pdb_ids_upper:
        components[uf.find(pdb_id)].append(pdb_id)

    cluster_map: dict[str, str] = {}
    for members in components.values():
        cluster_rep = min(members)
        for pdb_id in members:
            cluster_map[pdb_id] = cluster_rep
    return cluster_map


def _clean_mmseqs_artifacts(output_dir: str, db: str, clustered: str, tmp: str, tsv_path: str) -> None:
    """Remove stale MMseqs2 DB/cluster files so reruns do not fail."""
    db_name = os.path.basename(db)
    cluster_name = os.path.basename(clustered)
    removed: list[str] = []

    for fname in os.listdir(output_dir):
        path = os.path.join(output_dir, fname)
        if fname == os.path.basename(tsv_path):
            os.remove(path)
            removed.append(fname)
            continue
        if fname.startswith(f"{db_name}.") or fname == db_name:
            os.remove(path)
            removed.append(fname)
            continue
        if fname.startswith(f"{cluster_name}.") or fname == cluster_name:
            os.remove(path)
            removed.append(fname)

    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
        removed.append(f"{os.path.basename(tmp)}/")

    if removed:
        print(f"  Cleared {len(removed)} stale MMseqs2 artifact(s) from {output_dir}")


def _print_subprocess_failure(step_name: str, cmd: list[str], result: subprocess.CompletedProcess) -> None:
    """Print detailed diagnostics when an MMseqs2 subprocess fails."""
    print(f"  WARNING: MMseqs2 step failed ({step_name})")
    print(f"    Command   : {' '.join(cmd)}")
    print(f"    Exit code : {result.returncode}")
    for stream_name, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        cleaned = (text or "").strip()
        if not cleaned:
            print(f"    {stream_name}: (empty)")
            continue
        lines = cleaned.splitlines()
        print(f"    {stream_name} ({len(lines)} line(s)):")
        preview = lines if len(lines) <= 40 else lines[:20] + ["...", f"({len(lines) - 40} lines omitted)", "..."] + lines[-20:]
        for line in preview:
            print(f"      {line}")


def run_mmseqs2(fasta_path: str, output_dir: str,
                seq_identity: float = 0.5, coverage: float = 0.8) -> dict[str, str]:
    """
    Run MMseqs2 clustering and return {protein_id: cluster_representative}.
    Falls back to treating each protein as its own cluster if MMseqs2 unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)
    db = os.path.join(output_dir, "seqdb")
    clustered = os.path.join(output_dir, "clusters")
    tmp = os.path.join(output_dir, "tmp")
    tsv_path = os.path.join(output_dir, "clusters.tsv")

    # Check mmseqs available
    if subprocess.run(["which", "mmseqs"],
                      capture_output=True).returncode != 0:
        print("  WARNING: mmseqs2 not found. Using each protein as its own cluster.")
        cluster_map = {}
        with open(fasta_path) as f:
            for line in f:
                if line.startswith(">"):
                    pid = line.strip().lstrip(">")
                    cluster_map[pid] = pid
        return cluster_map

    _clean_mmseqs_artifacts(output_dir, db, clustered, tmp, tsv_path)

    cmds: list[tuple[str, list[str]]] = [
        ("createdb", ["mmseqs", "createdb", fasta_path, db]),
        ("cluster", [
            "mmseqs", "cluster", db, clustered, tmp,
            "--min-seq-id", str(seq_identity),
            "-c", str(coverage), "--cov-mode", "0",
        ]),
        ("createtsv", ["mmseqs", "createtsv", db, db, clustered, tsv_path]),
    ]
    failed_step: str | None = None
    for step_name, cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _print_subprocess_failure(step_name, cmd, result)
            failed_step = step_name
            break

    cluster_map = {}
    if os.path.exists(tsv_path) and failed_step is None:
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    rep, member = parts
                    cluster_map[member.upper()] = rep.upper()
        print(f"  MMseqs2 found {len(set(cluster_map.values()))} clusters "
              f"from {len(cluster_map)} proteins.")
    else:
        print("  WARNING: MMseqs2 TSV not produced. Using each protein as its own cluster.")
        with open(fasta_path) as f:
            for line in f:
                if line.startswith(">"):
                    pid = line.strip().lstrip(">")
                    cluster_map[pid] = pid

    return cluster_map


# ─────────────────────────────────────────────
# WARHEAD MATCHING
# ─────────────────────────────────────────────

def sanitize_name_part(value: object, max_length: int = 72) -> str:
    """Match CovSite Name sanitization for warhead suffixes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        text = ""
    else:
        text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return (text or "unnamed")[:max_length]


def ligand_base_name(name: object, warhead: object = None) -> str:
    """
    Ligand identity with the CovSite warhead suffix removed.

    CovSite Names are typically PDB-SMILES-Warhead or
    PDB-SMILES-Warhead-SiteTag. Ranking stays per warhead query group, but hit
    counting / perfect-match collapse to this base name so
    ...-Aldehyde and ...-Carbamate_amide_like count as one ligand.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""

    parts = text.split("-")
    if warhead is not None and not (isinstance(warhead, float) and pd.isna(warhead)):
        warhead_text = str(warhead).strip()
        if warhead_text:
            san = sanitize_name_part(warhead_text)
            for idx in range(len(parts) - 1, 1, -1):
                if parts[idx].upper() == san.upper():
                    return "-".join(parts[:idx] + parts[idx + 1 :])
            suffix = "-" + san
            if text.upper().endswith(suffix.upper()):
                return text[: -len(suffix)]

    # Fallback for labels that only carry Name: drop the warhead segment
    # (index 2) under the PDB-SMILES-Warhead[-Site] convention.
    if len(parts) >= 3:
        return "-".join(parts[:2] + parts[3:])
    return text


def warhead_matches(training_warhead: str, frankenstein_warhead: str) -> bool:
    """
    Match training warhead against (possibly comma-separated)
    Frankenstein_Warhead field from labels CSV.
    """
    if pd.isna(frankenstein_warhead) or pd.isna(training_warhead):
        return False
    tw = training_warhead.strip().lower()
    fw_list = [w.strip().lower() for w in str(frankenstein_warhead).split(",")]
    return tw in fw_list


def parse_frankenstein_warheads(raw) -> set[str]:
    if pd.isna(raw):
        return set()
    return {w.strip().lower() for w in str(raw).split(",") if w.strip()}


def _normalize_resnum(val) -> str:
    """Coerce ResNum so 45 and 45.0 compare equal."""
    if pd.isna(val):
        return ""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _matching_training_warheads(
    name: str,
    frankenstein_raw,
    warheads_by_name: dict[str, list[str]],
) -> list[str]:
    """Return lowercased training warheads for *name* that match the label."""
    matched = []
    for wh in warheads_by_name.get(name, []):
        if warhead_matches(wh, frankenstein_raw):
            matched.append(wh.strip().lower())
    return matched


# ─────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────

def load_and_merge(training_csv: str, labels_csv: str) -> pd.DataFrame:
    """
    Load training and labels CSVs, merge on ligand base Name + Warhead,
    assign binary relevance label (1 = target residue, 0 = other).

    CovSite-style Names that differ only by warhead suffix
    (e.g. ...-Aldehyde vs ...-Carbamate_amide_like) share one ligand base
    name. Each (base Name, Warhead) pair is one query group. When a label's
    Frankenstein_Warhead matches multiple training warheads under that base
    name, each warhead becomes its own query group with the same target site.
    """
    print("\n[1/7] Loading data...")
    train_df = pd.read_csv(training_csv, sep=",", engine="c", low_memory=False)
    label_df = pd.read_csv(labels_csv, sep=",", engine="c", low_memory=False)

    print(f"  Training rows: {len(train_df)}")
    print(f"  Label rows:    {len(label_df)}")

    train_df["_name_upper"]    = train_df["Name"].str.strip().str.upper()
    train_df["_warhead_lower"] = train_df["Warhead"].str.strip().str.lower()
    train_df["_base_name"] = [
        ligand_base_name(name, warhead).upper()
        for name, warhead in zip(train_df["Name"], train_df["Warhead"])
    ]
    train_df["Residue"]        = train_df["Residue"].str.strip().str.upper()
    train_df["_resnum_str"]    = train_df["ResNum"].map(_normalize_resnum)
    train_df["_chain_upper"]   = train_df["Chain"].astype(str).str.strip().str.upper()
    train_df["pdb_id"]         = train_df["_name_upper"].str.split("-").str[0]

    label_df["_name_upper"] = label_df["Name"].str.strip().str.upper()
    label_df["_base_name"] = label_df["Name"].map(
        lambda value: ligand_base_name(value).upper()
    )

    train_df = train_df[train_df["Residue"].isin(VALID_RESIDUES)].copy()
    train_df = train_df.drop_duplicates(
        subset=["_base_name", "Residue", "_resnum_str", "_chain_upper", "_warhead_lower"],
        keep="first",
    )

    warheads_by_name: dict[str, list[str]] = (
        train_df.groupby("_base_name")["Warhead"]
        .apply(lambda s: list(s.unique()))
        .to_dict()
    )

    # One target row per (base Name, Warhead) that matches the label's warhead set
    target_rows: list[dict] = []
    unmatched_labels = 0

    for _, lrow in label_df.iterrows():
        base_name = lrow["_base_name"]
        target_r = str(lrow["Residue"]).strip().upper()
        target_n = _normalize_resnum(lrow["ResNum"])
        target_c = str(lrow["Chain"]).strip().upper()
        frank_wh = lrow.get("Frankenstein_Warhead", lrow.get("Warhead", ""))

        matched_whs = _matching_training_warheads(
            base_name, frank_wh, warheads_by_name
        )
        if not matched_whs:
            unmatched_labels += 1
            continue

        for wh_lower in matched_whs:
            target_rows.append({
                "_base_name":          base_name,
                "_warhead_lower":      wh_lower,
                "tgt_residue":         target_r,
                "tgt_resnum":          target_n,
                "tgt_chain":           target_c,
                "target_residue_type": target_r,
                "label_pdb_id":        base_name,
            })

    if not target_rows:
        print("ERROR: No training rows matched any label. Check name/warhead formats.")
        sys.exit(1)

    targets_df = pd.DataFrame(target_rows)

    # Drop (base Name, Warhead) groups where labels disagree on the target site
    site_cols = ["tgt_residue", "tgt_resnum", "tgt_chain"]
    n_sites = (
        targets_df.groupby(["_base_name", "_warhead_lower"])[site_cols]
        .nunique(dropna=False)
        .max(axis=1)
    )
    conflict_keys = set(n_sites[n_sites > 1].index)
    if conflict_keys:
        print(f"  WARNING: {len(conflict_keys)} query group(s) have conflicting "
              f"labels — dropping them.")
        keep_mask = ~targets_df.set_index(["_base_name", "_warhead_lower"]).index.isin(
            conflict_keys
        )
        targets_df = targets_df.loc[keep_mask].copy()

    targets_df = targets_df.drop_duplicates(
        subset=["_base_name", "_warhead_lower", *site_cols],
        keep="first",
    )

    merged = train_df.merge(
        targets_df,
        on=["_base_name", "_warhead_lower"],
        how="inner",
    )

    merged["relevance"] = (
        (merged["Residue"] == merged["tgt_residue"])
        & (merged["_resnum_str"] == merged["tgt_resnum"])
        & (merged["_chain_upper"] == merged["tgt_chain"])
    ).astype(int)

    # Query group = ligand base name × warhead (warhead suffixes collapsed)
    merged["query_group"] = (
        merged["_base_name"] + "__" + merged["_warhead_lower"]
    )

    # Drop query groups with no positive label (target site absent in training)
    has_target = merged.groupby("query_group")["relevance"].max()
    valid_groups = has_target[has_target == 1].index
    merged = merged[merged["query_group"].isin(valid_groups)].copy()

    print(f"  Matched rows:       {len(merged)}")
    print(f"  Ligand base names:  {merged['_base_name'].nunique()}")
    print(f"  Valid query groups: {merged['query_group'].nunique()}")
    print(f"  Unmatched labels:   {unmatched_labels}")
    print(f"  Target residue distribution:")
    print(merged[merged["relevance"] == 1]["target_residue_type"]
          .value_counts().to_string(header=False))

    return merged


# ─────────────────────────────────────────────
# LARGE-POOL PREFILTER (optional, train only)
# ─────────────────────────────────────────────

@dataclass
class SimpleFilter:
    col: str
    op: str
    value: float
    label: str = ""

    def __post_init__(self):
        self.label = f"{self.col} {self.op} {self.value}"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.col not in df.columns:
            sys.exit(f"[ERROR] Prefilter column '{self.col}' not in training CSV.")
        numeric = pd.to_numeric(df[self.col], errors="coerce")
        return FILTER_OPS[self.op](numeric, self.value)


def parse_simple_filters(tokens: list[str]) -> list[SimpleFilter]:
    if len(tokens) % 3 != 0:
        sys.exit(
            f"[ERROR] --large-pool-filter requires groups of 3: COL OP VALUE. "
            f"Got {len(tokens)} tokens."
        )
    filters = []
    for i in range(0, len(tokens), 3):
        col, op, val_str = tokens[i], tokens[i + 1], tokens[i + 2]
        if op not in FILTER_OPS:
            sys.exit(
                f"[ERROR] Unknown prefilter operator '{op}'. "
                f"Valid: {', '.join(FILTER_OPS)}"
            )
        try:
            val = float(val_str)
        except ValueError:
            sys.exit(f"[ERROR] Prefilter value must be numeric, got '{val_str}'")
        filters.append(SimpleFilter(col=col, op=op, value=val))
    return filters


def apply_large_pool_prefilter(
    df: pd.DataFrame,
    filters: list[SimpleFilter],
    min_pool_size: int,
) -> tuple[pd.DataFrame, dict]:
    """
    For query groups with more than min_pool_size candidates, drop rows that
    fail all simple feature filters. Small groups are unchanged. Labeled target
    rows (relevance=1) are always kept.
    """
    pool_sizes = df.groupby("query_group", sort=False).size()
    large_groups = set(pool_sizes[pool_sizes > min_pool_size].index)

    filter_labels = " AND ".join(f.label for f in filters)
    meta = {
        "enabled": True,
        "min_pool_size": min_pool_size,
        "filters": [f.label for f in filters],
        "n_large_groups": len(large_groups),
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_removed": 0,
        "groups_dropped": 0,
    }

    if not large_groups:
        print(f"  Large-pool prefilter: no query groups above {min_pool_size}")
        return df, meta

    pass_mask = pd.Series(True, index=df.index)
    for filt in filters:
        pass_mask &= filt.apply(df)

    keep_mask = (
        ~df["query_group"].isin(large_groups)
        | pass_mask
        | (df["relevance"] == 1)
    )
    filtered = df[keep_mask].copy()

    has_target = filtered.groupby("query_group")["relevance"].max()
    valid_groups = has_target[has_target == 1].index
    groups_dropped = filtered["query_group"].nunique() - len(valid_groups)
    filtered = filtered[filtered["query_group"].isin(valid_groups)].copy()

    rows_in_large = int(df["query_group"].isin(large_groups).sum())
    rows_removed = rows_in_large - int(
        filtered["query_group"].isin(large_groups).sum()
    )

    meta.update({
        "rows_after": len(filtered),
        "rows_removed": rows_removed,
        "groups_dropped": int(groups_dropped),
    })

    print(f"\n  Large-pool prefilter (train only):")
    print(f"    Threshold: pool size > {min_pool_size}")
    print(f"    Filters:   {filter_labels}")
    print(f"    Large query groups: {len(large_groups)} / {df['query_group'].nunique()}")
    print(f"    Rows removed:       {rows_removed} "
          f"({rows_in_large} candidates in large groups)")
    if groups_dropped:
        print(f"    WARNING: {groups_dropped} query group(s) dropped "
              f"(target lost after prefilter)")

    return filtered, meta


# ─────────────────────────────────────────────
# CLUSTERING & SPLITTING
# ─────────────────────────────────────────────

def load_clusters_tsv(tsv_path: str) -> dict[str, str]:
    """
    Load MMseqs2 createtsv output: rep\\tmember → {member: rep} (uppercase keys).
    """
    cluster_map: dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            cluster_map[member.upper()] = rep.upper()
    return cluster_map


def build_cluster_map(merged: pd.DataFrame,
                      pdb_folder: str | None,
                      mmseqs_dir: str,
                      seq_identity: float,
                      coverage: float,
                      deep_cluster: bool = False,
                      min_chain_length: int = 30,
                      clusters_tsv: str | None = None) -> dict[str, str]:
    """Build protein→cluster mapping via MMseqs2 (or fallback)."""
    print("\n[2/7] Building sequence clusters...")
    pdb_ids = merged["pdb_id"].unique().tolist()

    if clusters_tsv:
        if not os.path.isfile(clusters_tsv):
            sys.exit(f"[ERROR] --clusters-tsv not found: {clusters_tsv}")
        print(f"  Reusing precomputed clusters from {clusters_tsv}")
        if deep_cluster:
            print("  Mode: deep-cluster (PDB merge from chain-level TSV)")
        raw_map = load_clusters_tsv(clusters_tsv)
        if not raw_map:
            sys.exit(f"[ERROR] --clusters-tsv is empty or unreadable: {clusters_tsv}")
        if deep_cluster:
            cluster_map = merge_chain_clusters_to_pdb_map(pdb_ids, raw_map)
            n_chain_clusters = len(set(raw_map.values()))
            n_pdb_clusters = len(set(cluster_map.values()))
            print(f"  Deep-cluster: {n_chain_clusters} chain clusters → "
                  f"{n_pdb_clusters} PDB clusters from {len(pdb_ids)} proteins.")
        else:
            cluster_map = raw_map
            print(f"  Loaded {len(set(cluster_map.values()))} clusters "
                  f"for {len(cluster_map)} members.")
        for pid in pdb_ids:
            if pid.upper() not in cluster_map:
                cluster_map[pid.upper()] = pid.upper()
        return cluster_map

    if pdb_folder is None:
        print("  No PDB folder specified — using each protein as its own cluster.")
        return {pid.upper(): pid.upper() for pid in pdb_ids}

    os.makedirs(mmseqs_dir, exist_ok=True)

    if deep_cluster:
        print("  Mode: deep-cluster (per-chain MMseqs2, PDB merge on shared chain clusters)")
        fasta_path = os.path.join(mmseqs_dir, "chains.fasta")
        n_chains = extract_chain_fasta_from_pdbs(
            pdb_folder, pdb_ids, fasta_path, min_chain_length=min_chain_length,
        )
        if n_chains == 0:
            print("  WARNING: No chain sequences extracted; using each protein as its own cluster.")
            return {pid.upper(): pid.upper() for pid in pdb_ids}

        chain_cluster_map = run_mmseqs2(fasta_path, mmseqs_dir, seq_identity, coverage)
        cluster_map = merge_chain_clusters_to_pdb_map(pdb_ids, chain_cluster_map)
        n_chain_clusters = len(set(chain_cluster_map.values()))
        n_pdb_clusters = len(set(cluster_map.values()))
        print(f"  Deep-cluster: {n_chain_clusters} chain clusters → {n_pdb_clusters} PDB clusters "
              f"from {len(pdb_ids)} proteins.")
    else:
        fasta_path = os.path.join(mmseqs_dir, "proteins.fasta")
        extract_fasta_from_pdbs(pdb_folder, pdb_ids, fasta_path)
        cluster_map = run_mmseqs2(fasta_path, mmseqs_dir, seq_identity, coverage)

    # Ensure all pdb_ids have a cluster (fallback to self)
    for pid in pdb_ids:
        if pid.upper() not in cluster_map:
            cluster_map[pid.upper()] = pid.upper()

    return cluster_map


def export_query_group_splits(merged: pd.DataFrame, path: str) -> None:
    """Write one row per query_group with train/test split."""
    out = (
        merged.groupby("query_group", as_index=False)["split"]
        .first()
        .sort_values("query_group")
    )
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)
    n_test = int((out["split"] == "test").sum())
    print(f"  Exported query-group splits: {path} "
          f"({len(out)} groups, {n_test} test)")


def _attach_cluster_ids(merged: pd.DataFrame, cluster_map: dict[str, str]) -> pd.DataFrame:
    merged = merged.copy()
    merged["cluster_id"] = merged["pdb_id"].str.upper().map(cluster_map)
    merged["cluster_id"] = merged["cluster_id"].fillna(merged["pdb_id"].str.upper())
    return merged


def _apply_split_csv(merged: pd.DataFrame, split_csv: str) -> pd.DataFrame:
    splits = pd.read_csv(split_csv)
    required = {"query_group", "split"}
    if not required.issubset(splits.columns):
        sys.exit(
            f"[ERROR] --split-csv must contain columns {sorted(required)}, "
            f"got {list(splits.columns)}"
        )

    split_map = (
        splits.drop_duplicates("query_group", keep="first")
        .set_index("query_group")["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_dict()
    )
    invalid = sorted({v for v in split_map.values() if v not in {"train", "test"}})
    if invalid:
        sys.exit(
            f"[ERROR] --split-csv split values must be 'train' or 'test', got: {invalid}"
        )

    qgs_in_data = set(merged["query_group"].unique())
    missing = sorted(qgs_in_data - set(split_map.keys()))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        sys.exit(
            f"[ERROR] --split-csv missing {len(missing)} query group(s) present in "
            f"training data, e.g. {preview}{suffix}"
        )

    merged = merged.copy()
    merged["split"] = merged["query_group"].map(split_map)
    if merged["split"].isna().any():
        sys.exit("[ERROR] --split-csv failed to assign split for some rows.")
    return merged


def assign_splits(merged: pd.DataFrame,
                  cluster_map: dict[str, str],
                  test_size: float = 0.2,
                  n_folds: int = 5,
                  random_state: int = 42,
                  split_csv: str | None = None,
                  export_split: str | None = None) -> pd.DataFrame:
    """
    Assign train/test split at the protein-cluster level,
    stratified by target residue type, or load a fixed split CSV.

    Returns merged df with 'split' and 'cluster_id' columns.
    """
    print("\n[3/7] Splitting data...")
    merged = _attach_cluster_ids(merged, cluster_map)

    if split_csv:
        if not os.path.isfile(split_csv):
            sys.exit(f"[ERROR] --split-csv not found: {split_csv}")
        print(f"  Using fixed query-group splits from {split_csv}")
        merged = _apply_split_csv(merged, split_csv)
    else:
        print(f"  Stratified cluster holdout: {test_size * 100:.0f}% test "
              f"(random_state={random_state})")

        # One row per query_group with its cluster + target residue type
        qg_meta = (merged.groupby("query_group")
                   .agg(cluster_id=("cluster_id", "first"),
                        target_res=("target_residue_type", "first"))
                   .reset_index())

        # Unique clusters with their dominant residue type
        cluster_meta = (qg_meta.groupby("cluster_id")
                        .agg(target_res=("target_res", lambda x: x.mode()[0]))
                        .reset_index())

        rng = np.random.RandomState(random_state)
        test_clusters = set()

        # Stratified holdout by residue type
        for res_type, grp in cluster_meta.groupby("target_res"):
            clusters = grp["cluster_id"].values
            n_test = max(1, int(len(clusters) * test_size))
            chosen = rng.choice(clusters, size=n_test, replace=False)
            test_clusters.update(chosen)

        merged["split"] = merged["cluster_id"].apply(
            lambda c: "test" if c in test_clusters else "train"
        )

    train_qg = merged[merged["split"] == "train"]["query_group"].nunique()
    test_qg = merged[merged["split"] == "test"]["query_group"].nunique()
    print(f"  Train query groups: {train_qg}")
    print(f"  Test  query groups: {test_qg}")

    if export_split:
        export_query_group_splits(merged, export_split)

    return merged


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(merged: pd.DataFrame,
                   feature_cols: list[str],
                   residue_specific_cols: list[str],
                   normalize_within_protein: bool = False,
                   resnum_equals: list[int] | None = None,
                   include_residue_type: bool = True) -> pd.DataFrame:
    """
    - Optionally one-hot encode Residue type (res_CYS, res_SER, ...)
    - Optionally add binary resnum_eq_N features (--res-num)
    - Optionally normalize specified features within protein
    - Residue-specific features: same value for all residues of same type
      within a protein×warhead group — left as raw QM values per user request
    """
    print("\n[4/7] Building features...")

    df = merged.copy()

    ohe_cols: list[str] = []
    if include_residue_type:
        for res in VALID_RESIDUES:
            df[f"res_{res}"] = (df["Residue"] == res).astype(int)
        ohe_cols = [f"res_{r}" for r in VALID_RESIDUES]
        print(f"  Residue type one-hot: {ohe_cols}")
    else:
        print("  Residue type one-hot: disabled")

    resnum_cols: list[str] = []
    if resnum_equals:
        if "_resnum_str" not in df.columns:
            df["_resnum_str"] = df["ResNum"].map(_normalize_resnum)
        for n in resnum_equals:
            key = _normalize_resnum(n)
            col = f"resnum_eq_{key}"
            df[col] = (df["_resnum_str"] == key).astype(int)
            resnum_cols.append(col)
        print(f"  ResNum binary features: {resnum_cols}")

    # Optional within-protein normalization
    if normalize_within_protein and feature_cols:
        print(f"  Normalizing {len(feature_cols)} features within protein...")
        for col in feature_cols:
            if col in df.columns:
                df[f"{col}_norm"] = df.groupby("query_group")[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
                )

    # Confirm residue-specific cols exist
    for col in residue_specific_cols:
        if col not in df.columns:
            print(f"  WARNING: residue-specific feature '{col}' not in data, skipping.")

    # Final feature list
    norm_cols = ([f"{c}_norm" for c in feature_cols
                  if f"{c}_norm" in df.columns]
                 if normalize_within_protein else [])
    raw_cols  = [c for c in feature_cols + residue_specific_cols
                 if c in df.columns]

    all_feat_cols = list(dict.fromkeys(ohe_cols + resnum_cols + raw_cols + norm_cols))
    print(f"  Total features: {len(all_feat_cols)}")
    print(f"  Features: {all_feat_cols}")

    return df, all_feat_cols


# ─────────────────────────────────────────────
# SAMPLE WEIGHTS
# ─────────────────────────────────────────────

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Assign sample weights boosting rare nucleophile target residues.
    Weight is applied at the query-group level based on target residue type.
    """
    weights = df["target_residue_type"].map(RARITY_BOOST).fillna(1.0).values
    return weights


# ─────────────────────────────────────────────
# DCG SCORING
# ─────────────────────────────────────────────

def dcg_score(relevances: np.ndarray, k: int = 10) -> float:
    """
    DCG@K with log2 discounting.
    relevances: array of relevance labels in predicted rank order.
    """
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_score(relevances: np.ndarray, k: int = 10) -> float:
    """NDCG@K — DCG normalised by ideal DCG."""
    dcg  = dcg_score(relevances, k)
    ideal = dcg_score(np.sort(relevances)[::-1], k)
    return dcg / ideal if ideal > 0 else 0.0


def _rank_bonus_absolute(target_rank: int, hit_at_k: bool) -> float:
    """1/rank bonus when the target is within absolute top-K."""
    return 1.0 / target_rank if hit_at_k else 0.0


def _rank_bonus_percent(rank_frac: float, hit_at_top_pct: bool,
                        top_pct_threshold: float) -> float:
    """Graded bonus inside the top-X% zone: 1 at rank 1, 0 at the zone edge."""
    if not hit_at_top_pct:
        return 0.0
    p = top_pct_threshold / 100.0
    if p <= 0:
        return 0.0
    return max(0.0, 1.0 - rank_frac / p)


def evaluate_predictions(
    df: pd.DataFrame,
    score_col: str,
    k: int = 10,
    top_pct_threshold: float = 10.0,
    reward_mode: str = "hit_at_k",
    epsilon: float = DEFAULT_RANK_BONUS_EPSILON,
) -> pd.DataFrame:
    """
    Per query-group evaluation.

    Always reports absolute (Hit@K) and percent (Hit@top-X%) metrics.
    Composite objective uses one reward mode:
      hit_at_k        -> Hit@K + epsilon * (1/rank) when hit
      hit_at_top_pct  -> Hit@top-X% + epsilon * graded bonus inside zone
    """
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")

    p_frac = top_pct_threshold / 100.0
    results = []

    for qg, grp in df.groupby("query_group"):
        grp_sorted = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        target_rows = grp_sorted[grp_sorted["relevance"] == 1]

        if target_rows.empty:
            continue

        target_rank = int(target_rows.index[0] + 1)
        n_residues  = int(len(grp_sorted))
        hit_at_k    = int(target_rank <= k)
        relevances  = grp_sorted["relevance"].values
        ndcg        = ndcg_score(relevances, k)
        res_type    = grp_sorted["target_residue_type"].iloc[0]

        if n_residues <= 1:
            rank_frac = 0.0
        else:
            rank_frac = (target_rank - 1) / (n_residues - 1)
        top_pct = 100.0 * (1.0 - rank_frac)
        search_space_reduction = (
            1.0 - (target_rank / n_residues) if n_residues > 0 else 0.0
        )

        hit_at_top_pct = int(rank_frac <= p_frac) if p_frac > 0 else int(hit_at_k)
        bonus_absolute = _rank_bonus_absolute(target_rank, bool(hit_at_k))
        bonus_percent  = _rank_bonus_percent(
            rank_frac, bool(hit_at_top_pct), top_pct_threshold
        )

        if reward_mode == "hit_at_top_pct":
            hit_primary = hit_at_top_pct
            rank_bonus  = bonus_percent
        else:
            hit_primary = hit_at_k
            rank_bonus  = bonus_absolute

        objective = float(hit_primary) + epsilon * rank_bonus

        results.append({
            "query_group":            qg,
            "target_residue_type":    res_type,
            "target_rank":            target_rank,
            "hit_at_k":               hit_at_k,
            "hit_at_top_pct":         hit_at_top_pct,
            "ndcg_at_k":              ndcg,
            "n_residues":             n_residues,
            "rank_frac":              float(rank_frac),
            "top_pct":                float(top_pct),
            "search_space_reduction": float(search_space_reduction),
            "rank_bonus_absolute":    float(bonus_absolute),
            "rank_bonus_percent":     float(bonus_percent),
            "rank_bonus":             float(rank_bonus),
            "objective":              float(objective),
        })

    return pd.DataFrame(results)


def _metric_stats_when_hit(
    eval_df: pd.DataFrame,
    hit_col: str,
    value_col: str,
) -> dict[str, float | int]:
    """Mean/median of *value_col* among rows where *hit_col* is true."""
    if eval_df.empty or hit_col not in eval_df.columns or value_col not in eval_df.columns:
        return {"avg": float("nan"), "median": float("nan"), "n": 0}
    hits = eval_df.loc[eval_df[hit_col] == 1, value_col]
    if hits.empty:
        return {"avg": float("nan"), "median": float("nan"), "n": 0}
    return {
        "avg": float(hits.mean()),
        "median": float(hits.median()),
        "n": int(len(hits)),
    }


def _rank_stats_when_hit(eval_df: pd.DataFrame, hit_col: str) -> dict[str, float | int]:
    """Mean/median target rank among rows where *hit_col* is true."""
    return _metric_stats_when_hit(eval_df, hit_col, "target_rank")


def _ss_stats_when_hit(eval_df: pd.DataFrame, hit_col: str) -> dict[str, float | int]:
    """Mean/median search-space reduction among rows where *hit_col* is true."""
    return _metric_stats_when_hit(eval_df, hit_col, "search_space_reduction")


def _metric_stats_when_miss(
    eval_df: pd.DataFrame,
    hit_col: str,
    value_col: str,
) -> dict[str, float | int]:
    """Mean/median of *value_col* among rows where *hit_col* is false."""
    if eval_df.empty or hit_col not in eval_df.columns or value_col not in eval_df.columns:
        return {"avg": float("nan"), "median": float("nan"), "n": 0}
    misses = eval_df.loc[eval_df[hit_col] == 0, value_col]
    if misses.empty:
        return {"avg": float("nan"), "median": float("nan"), "n": 0}
    return {
        "avg": float(misses.mean()),
        "median": float(misses.median()),
        "n": int(len(misses)),
    }


def _ss_stats_when_miss(eval_df: pd.DataFrame, hit_col: str) -> dict[str, float | int]:
    """Mean/median search-space reduction among rows where *hit_col* is false."""
    return _metric_stats_when_miss(eval_df, hit_col, "search_space_reduction")


def summarize_eval_metrics(
    eval_df: pd.DataFrame,
    k: int,
    top_pct_threshold: float,
    reward_mode: str,
    epsilon: float,
) -> dict:
    """Aggregate per-query metrics into one summary dict."""
    if eval_df.empty:
        return {
            "hit_rate": 0.0,
            "hit_at_top_pct": 0.0,
            "ndcg": 0.0,
            "avg_rank": float("nan"),
            "rank_bonus": 0.0,
            "objective": 0.0,
            "avg_top_pct": float("nan"),
            "avg_search_space_reduction": float("nan"),
            "median_search_space_reduction": float("nan"),
            "avg_ss_reduction_when_hit_at_k": float("nan"),
            "median_ss_reduction_when_hit_at_k": float("nan"),
            "avg_ss_reduction_when_hit_at_top_pct": float("nan"),
            "median_ss_reduction_when_hit_at_top_pct": float("nan"),
            "avg_ss_reduction_when_hit": float("nan"),
            "median_ss_reduction_when_hit": float("nan"),
            "avg_ss_reduction_when_miss_at_k": float("nan"),
            "median_ss_reduction_when_miss_at_k": float("nan"),
            "avg_ss_reduction_when_miss_at_top_pct": float("nan"),
            "median_ss_reduction_when_miss_at_top_pct": float("nan"),
            "avg_ss_reduction_when_miss": float("nan"),
            "median_ss_reduction_when_miss": float("nan"),
            "avg_rank_when_hit_at_k": float("nan"),
            "median_rank_when_hit_at_k": float("nan"),
            "n_hit_at_k": 0,
            "avg_rank_when_hit_at_top_pct": float("nan"),
            "median_rank_when_hit_at_top_pct": float("nan"),
            "n_hit_at_top_pct": 0,
            "n_miss_at_k": 0,
            "n_miss_at_top_pct": 0,
            "avg_rank_when_hit": float("nan"),
            "median_rank_when_hit": float("nan"),
        }

    hit_k_stats = _rank_stats_when_hit(eval_df, "hit_at_k")
    hit_pct_stats = _rank_stats_when_hit(eval_df, "hit_at_top_pct")
    ss_hit_k_stats = _ss_stats_when_hit(eval_df, "hit_at_k")
    ss_hit_pct_stats = _ss_stats_when_hit(eval_df, "hit_at_top_pct")
    ss_miss_k_stats = _ss_stats_when_miss(eval_df, "hit_at_k")
    ss_miss_pct_stats = _ss_stats_when_miss(eval_df, "hit_at_top_pct")
    active_hit_col = (
        "hit_at_top_pct" if reward_mode == "hit_at_top_pct" else "hit_at_k"
    )
    active_stats = (
        hit_pct_stats if reward_mode == "hit_at_top_pct" else hit_k_stats
    )
    active_ss_stats = (
        ss_hit_pct_stats if reward_mode == "hit_at_top_pct" else ss_hit_k_stats
    )
    active_ss_miss_stats = (
        ss_miss_pct_stats if reward_mode == "hit_at_top_pct" else ss_miss_k_stats
    )

    return {
        "hit_rate": float(eval_df["hit_at_k"].mean()),
        "hit_at_top_pct": float(eval_df["hit_at_top_pct"].mean()),
        "ndcg": float(eval_df["ndcg_at_k"].mean()),
        "avg_rank": float(eval_df["target_rank"].mean()),
        "rank_bonus": float(eval_df["rank_bonus"].mean()),
        "objective": float(eval_df["objective"].mean()),
        "avg_top_pct": float(eval_df["top_pct"].mean()),
        "avg_search_space_reduction": float(eval_df["search_space_reduction"].mean()),
        "median_search_space_reduction": float(eval_df["search_space_reduction"].median()),
        "avg_ss_reduction_when_hit_at_k": ss_hit_k_stats["avg"],
        "median_ss_reduction_when_hit_at_k": ss_hit_k_stats["median"],
        "avg_ss_reduction_when_hit_at_top_pct": ss_hit_pct_stats["avg"],
        "median_ss_reduction_when_hit_at_top_pct": ss_hit_pct_stats["median"],
        "avg_ss_reduction_when_hit": active_ss_stats["avg"],
        "median_ss_reduction_when_hit": active_ss_stats["median"],
        "avg_ss_reduction_when_miss_at_k": ss_miss_k_stats["avg"],
        "median_ss_reduction_when_miss_at_k": ss_miss_k_stats["median"],
        "avg_ss_reduction_when_miss_at_top_pct": ss_miss_pct_stats["avg"],
        "median_ss_reduction_when_miss_at_top_pct": ss_miss_pct_stats["median"],
        "avg_ss_reduction_when_miss": active_ss_miss_stats["avg"],
        "median_ss_reduction_when_miss": active_ss_miss_stats["median"],
        "avg_rank_when_hit_at_k": hit_k_stats["avg"],
        "median_rank_when_hit_at_k": hit_k_stats["median"],
        "n_hit_at_k": hit_k_stats["n"],
        "avg_rank_when_hit_at_top_pct": hit_pct_stats["avg"],
        "median_rank_when_hit_at_top_pct": hit_pct_stats["median"],
        "n_hit_at_top_pct": hit_pct_stats["n"],
        "n_miss_at_k": ss_miss_k_stats["n"],
        "n_miss_at_top_pct": ss_miss_pct_stats["n"],
        "avg_rank_when_hit": active_stats["avg"],
        "median_rank_when_hit": active_stats["median"],
        "hit_criterion": active_hit_col,
        "reward_mode": reward_mode,
        "top_pct_threshold": top_pct_threshold,
        "k": k,
        "epsilon": epsilon,
    }


def summarize_screen_metrics(
    eval_label_df: pd.DataFrame,
    eval_qg_df: pd.DataFrame,
    k: int,
    top_pct_threshold: float,
    reward_mode: str,
    epsilon: float,
) -> dict:
    """
    Hit/objective at label level; rank / NDCG / search-space stats over all
    matched (Name × Warhead) query groups (may exceed label count).
    """
    label_stats = summarize_eval_metrics(
        eval_label_df, k, top_pct_threshold, reward_mode, epsilon,
    )
    qg_stats = summarize_eval_metrics(
        eval_qg_df, k, top_pct_threshold, reward_mode, epsilon,
    )
    return {
        "reward_mode": reward_mode,
        "top_pct_threshold": top_pct_threshold,
        "k": k,
        "epsilon": epsilon,
        "n_labels": int(len(eval_label_df)),
        "n_query_groups": int(len(eval_qg_df)),
        "hit_rate": label_stats["hit_rate"],
        "hit_at_top_pct": label_stats["hit_at_top_pct"],
        "objective": label_stats["objective"],
        "rank_bonus": label_stats["rank_bonus"],
        "avg_rank": qg_stats["avg_rank"],
        "median_rank": float(eval_qg_df["target_rank"].median()),
        "ndcg": qg_stats["ndcg"],
        "avg_top_pct": qg_stats["avg_top_pct"],
        "avg_search_space_reduction": qg_stats["avg_search_space_reduction"],
        "median_search_space_reduction": qg_stats["median_search_space_reduction"],
        "avg_ss_reduction_when_hit_at_k": qg_stats["avg_ss_reduction_when_hit_at_k"],
        "median_ss_reduction_when_hit_at_k": qg_stats["median_ss_reduction_when_hit_at_k"],
        "avg_ss_reduction_when_hit_at_top_pct": qg_stats["avg_ss_reduction_when_hit_at_top_pct"],
        "median_ss_reduction_when_hit_at_top_pct": qg_stats["median_ss_reduction_when_hit_at_top_pct"],
        "avg_ss_reduction_when_hit": qg_stats["avg_ss_reduction_when_hit"],
        "median_ss_reduction_when_hit": qg_stats["median_ss_reduction_when_hit"],
        "avg_ss_reduction_when_miss_at_k": qg_stats["avg_ss_reduction_when_miss_at_k"],
        "median_ss_reduction_when_miss_at_k": qg_stats["median_ss_reduction_when_miss_at_k"],
        "avg_ss_reduction_when_miss_at_top_pct": qg_stats["avg_ss_reduction_when_miss_at_top_pct"],
        "median_ss_reduction_when_miss_at_top_pct": qg_stats[
            "median_ss_reduction_when_miss_at_top_pct"
        ],
        "avg_ss_reduction_when_miss": qg_stats["avg_ss_reduction_when_miss"],
        "median_ss_reduction_when_miss": qg_stats["median_ss_reduction_when_miss"],
        "avg_rank_when_hit_at_k": qg_stats["avg_rank_when_hit_at_k"],
        "median_rank_when_hit_at_k": qg_stats["median_rank_when_hit_at_k"],
        "n_hit_at_k": qg_stats["n_hit_at_k"],
        "avg_rank_when_hit_at_top_pct": qg_stats["avg_rank_when_hit_at_top_pct"],
        "median_rank_when_hit_at_top_pct": qg_stats["median_rank_when_hit_at_top_pct"],
        "n_hit_at_top_pct": qg_stats["n_hit_at_top_pct"],
        "n_miss_at_k": qg_stats["n_miss_at_k"],
        "n_miss_at_top_pct": qg_stats["n_miss_at_top_pct"],
        "avg_rank_when_hit": qg_stats["avg_rank_when_hit"],
        "median_rank_when_hit": qg_stats["median_rank_when_hit"],
        "hit_criterion": qg_stats["hit_criterion"],
    }


def nucleophile_counts_for_series(residues: pd.Series) -> dict[str, int]:
    """Count each nucleophilic residue type in a candidate pool."""
    counts = residues.astype(str).str.strip().str.upper().value_counts()
    return {res: int(counts.get(res, 0)) for res in NUCLEOPHILE_RESIDUES}


def aggregate_nucleophile_totals(df: pd.DataFrame) -> dict[str, int]:
    """Sum nucleophile-type counts across all rows (e.g. full split before filtering)."""
    if df.empty or "Residue" not in df.columns:
        return {res: 0 for res in NUCLEOPHILE_RESIDUES}
    return nucleophile_counts_for_series(df["Residue"])


def analyze_residue_composition(
    scored_df: pd.DataFrame,
    score_col: str,
    k: int,
    top_pct_threshold: float,
) -> dict:
    """
    Diagnose rank-1 and top-pool residue-type composition per query group.

    Helps spot CYS/SER prioritization bias vs label-type matching:
      - rank1_type_match_rate: fraction where rank-1 type == labeled type
      - rank1_type_counts: what the model puts at #1 overall
      - label_type_counts: labeled target types in evaluated groups
      - rank1_given_label: per label type, distribution of rank-1 predictions
      - top_k_mean_type_fraction: mean share of each type in absolute top-K
      - top_pct_mean_type_fraction: mean share in top-X% pool
      - n_CYS, n_HIS, ...: nucleophile counts in full pool before top-K / top-% slicing
      - pool_residue_totals_before_filter: summed n_* across evaluated query groups
    """
    rows: list[dict] = []
    valid = sorted(VALID_RESIDUES)

    for qg, grp in scored_df.groupby("query_group"):
        pool_counts = nucleophile_counts_for_series(grp["Residue"])
        g = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        tgt = g[g["relevance"] == 1]
        if tgt.empty:
            continue

        label_type = str(tgt.iloc[0]["Residue"]).strip().upper()
        n = len(g)
        rank1_type = str(g.iloc[0]["Residue"]).strip().upper()

        k_eff = min(k, n)
        top_k = g.head(k_eff)["Residue"].str.strip().str.upper()
        k_counts = top_k.value_counts()

        pct_n = max(1, int(np.ceil(n * top_pct_threshold / 100.0)))
        top_pct = g.head(pct_n)["Residue"].str.strip().str.upper()
        pct_counts = top_pct.value_counts()

        row = {
            "query_group": qg,
            "label_type": label_type,
            "rank1_type": rank1_type,
            "rank1_matches_label_type": int(rank1_type == label_type),
            "n_candidates": n,
        }
        for res in NUCLEOPHILE_RESIDUES:
            row[f"n_{res}"] = pool_counts[res]
        for res in valid:
            row[f"top{k}_frac_{res}"] = float(k_counts.get(res, 0) / k_eff)
            row[f"top_pct_frac_{res}"] = float(pct_counts.get(res, 0) / pct_n)
        rows.append(row)

    if not rows:
        return {"n_query_groups": 0}

    detail = pd.DataFrame(rows)
    n_qg = len(detail)

    rank1_counts = detail["rank1_type"].value_counts(normalize=True).to_dict()
    label_counts = detail["label_type"].value_counts(normalize=True).to_dict()

    rank1_given_label = (
        detail.groupby("label_type")["rank1_type"]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
        .to_dict(orient="records")
    )

    top_k_fracs = {
        res: float(detail[f"top{k}_frac_{res}"].mean())
        for res in valid
        if f"top{k}_frac_{res}" in detail.columns
    }
    top_pct_fracs = {
        res: float(detail[f"top_pct_frac_{res}"].mean())
        for res in valid
        if f"top_pct_frac_{res}" in detail.columns
    }

    pool_totals = {
        res: int(detail[f"n_{res}"].sum())
        for res in NUCLEOPHILE_RESIDUES
        if f"n_{res}" in detail.columns
    }

    return {
        "n_query_groups": n_qg,
        "rank1_type_match_rate": float(detail["rank1_matches_label_type"].mean()),
        "rank1_type_counts": {k_: float(v) for k_, v in rank1_counts.items()},
        "label_type_counts": {k_: float(v) for k_, v in label_counts.items()},
        "rank1_given_label": rank1_given_label,
        f"top{k}_mean_type_fraction": top_k_fracs,
        f"top_pct_{top_pct_threshold:g}_mean_type_fraction": top_pct_fracs,
        "pool_residue_totals_before_filter": pool_totals,
        "detail": detail,
    }


def print_residue_composition(
    composition: dict,
    k: int,
    top_pct_threshold: float,
) -> None:
    """Print rank-1 / top-pool residue-type diagnostic tables."""
    if not composition.get("n_query_groups"):
        return

    print(f"\n  Residue-type composition (N={composition['n_query_groups']} query groups):")
    print(f"    Rank-1 type == label type : "
          f"{composition['rank1_type_match_rate']:.3f}")
    print(f"      (high hit@K with low match → right class, wrong site)")
    print(f"      (rank-1 always CYS/SER     → possible type bias)")

    print(f"\n    Labeled target type distribution:")
    for res, frac in sorted(composition["label_type_counts"].items(),
                            key=lambda x: -x[1]):
        print(f"      {res:3s}  {frac:6.1%}")

    print(f"\n    Rank-1 predicted type distribution:")
    for res, frac in sorted(composition["rank1_type_counts"].items(),
                            key=lambda x: -x[1]):
        print(f"      {res:3s}  {frac:6.1%}")

    k_key = f"top{k}_mean_type_fraction"
    pct_key = f"top_pct_{top_pct_threshold:g}_mean_type_fraction"
    if k_key in composition:
        print(f"\n    Mean type fraction in top-{k} (absolute):")
        for res, frac in sorted(composition[k_key].items(), key=lambda x: -x[1]):
            print(f"      {res:3s}  {frac:6.1%}")
    if pct_key in composition:
        print(f"\n    Mean type fraction in top-{top_pct_threshold:g}% pool:")
        for res, frac in sorted(composition[pct_key].items(), key=lambda x: -x[1]):
            print(f"      {res:3s}  {frac:6.1%}")

    pool_totals = composition.get("pool_residue_totals_before_filter")
    if pool_totals:
        print(f"\n    Nucleophile pool totals before filtering (test):")
        for res in NUCLEOPHILE_RESIDUES:
            print(f"      {res:3s}  {pool_totals.get(res, 0):>8,}")

    r1gl = composition.get("rank1_given_label", [])
    if r1gl:
        print(f"\n    Rank-1 type given labeled type (fraction):")
        by_label: dict[str, list[tuple[str, float]]] = {}
        for row in r1gl:
            by_label.setdefault(row["label_type"], []).append(
                (row["rank1_type"], row["fraction"])
            )
        for label in sorted(by_label):
            parts = ", ".join(
                f"{t} {f:.0%}" for t, f in sorted(by_label[label], key=lambda x: -x[1])
            )
            print(f"      label {label:3s} → {parts}")


# ─────────────────────────────────────────────
# LGBM RANKER
# ─────────────────────────────────────────────

def get_lgbm_params(n_folds: int = 5) -> dict:
    return {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [5, 10],
        "boosting_type":    "gbdt",
        "n_estimators":     500,
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_child_samples":5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }


def train_with_crossval(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 5,
    k: int = 10,
    top_pct_threshold: float = 10.0,
    reward_mode: str = "hit_at_k",
    epsilon: float = DEFAULT_RANK_BONUS_EPSILON,
) -> tuple[lgb.LGBMRanker, dict, pd.DataFrame]:
    """
    GroupKFold cross-validation on train set, grouped by cluster_id.
    Returns best model (trained on all train data), CV mean metrics, and
    per-fold metrics as a DataFrame.
    """
    print(f"\n[5/7] Cross-validation ({n_folds} folds)...")
    print(f"  Reward mode           : {reward_mode}")
    if reward_mode == "hit_at_k":
        print(f"  Primary reward        : Hit@{k} + {epsilon} * (1/rank)")
    else:
        print(f"  Primary reward        : Hit@top-{top_pct_threshold:g}% "
              f"+ {epsilon} * graded bonus inside zone")
    print(f"  Also reporting        : absolute Hit@{k} and "
          f"Hit@top-{top_pct_threshold:g}%")

    X = train_df[feature_cols].values
    y = train_df["relevance"].values
    groups = train_df["cluster_id"].values

    def get_group_sizes(df):
        return df.groupby("query_group", sort=False).size().values

    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = defaultdict(list)
    fold_rows: list[dict] = []
    _skip_fold_keys = {
        "reward_mode", "top_pct_threshold", "k", "epsilon", "hit_criterion",
    }

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        tr_df  = train_df.iloc[tr_idx].sort_values("query_group").reset_index(drop=True)
        val_df = train_df.iloc[val_idx].sort_values("query_group").reset_index(drop=True)

        X_tr = tr_df[feature_cols].values
        y_tr = tr_df["relevance"].values
        w_tr = compute_sample_weights(tr_df)
        g_tr = get_group_sizes(tr_df)

        X_val = val_df[feature_cols].values
        y_val = val_df["relevance"].values
        g_val = get_group_sizes(val_df)

        model = lgb.LGBMRanker(**get_lgbm_params())
        model.fit(
            X_tr, y_tr,
            group=g_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            eval_group=[g_val],
            eval_at=[k],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        val_df = val_df.copy()
        val_df["pred_score"] = model.predict(X_val)
        eval_df = evaluate_predictions(
            val_df, "pred_score", k=k,
            top_pct_threshold=top_pct_threshold,
            reward_mode=reward_mode,
            epsilon=epsilon,
        )
        fold_summary = summarize_eval_metrics(
            eval_df, k, top_pct_threshold, reward_mode, epsilon
        )

        for key, val in fold_summary.items():
            if key in _skip_fold_keys:
                continue
            fold_metrics[key].append(val)

        fold_row = {
            k: v for k, v in fold_summary.items() if k not in _skip_fold_keys
        }
        fold_row["fold"] = fold + 1
        fold_row["n_query_groups"] = len(eval_df)
        fold_rows.append(fold_row)

        print(f"  Fold {fold+1}: Obj={fold_summary['objective']:.3f}  "
              f"Hit@{k}={fold_summary['hit_rate']:.3f}  "
              f"Hit@top-{top_pct_threshold:g}%="
              f"{fold_summary['hit_at_top_pct']:.3f}  "
              f"rank|hit={fold_summary['avg_rank_when_hit']:.2f}  "
              f"bonus={fold_summary['rank_bonus']:.3f}  "
              f"NDCG@{k}={fold_summary['ndcg']:.3f}")

    cv_summary = {k_: float(np.mean(v)) for k_, v in fold_metrics.items()}
    cv_summary.update({
        "reward_mode": reward_mode,
        "top_pct_threshold": top_pct_threshold,
        "k": k,
        "epsilon": epsilon,
    })
    print(f"\n  CV Mean → Obj={cv_summary['objective']:.3f}  "
          f"Hit@{k}={cv_summary['hit_rate']:.3f}  "
          f"Hit@top-{top_pct_threshold:g}%={cv_summary['hit_at_top_pct']:.3f}  "
          f"rank|hit={cv_summary['avg_rank_when_hit']:.2f}  "
          f"bonus={cv_summary['rank_bonus']:.3f}  "
          f"NDCG@{k}={cv_summary['ndcg']:.3f}")

    print("\n  Retraining on full training set...")
    X_all = train_df[feature_cols].values
    y_all = train_df["relevance"].values
    w_all = compute_sample_weights(train_df)
    g_all = get_group_sizes(train_df)

    final_model = lgb.LGBMRanker(**get_lgbm_params())
    final_model.fit(X_all, y_all, group=g_all, sample_weight=w_all)

    cv_folds_df = pd.DataFrame(fold_rows)
    if not cv_folds_df.empty:
        lead_cols = ["fold", "n_query_groups"]
        other_cols = [c for c in cv_folds_df.columns if c not in lead_cols]
        cv_folds_df = cv_folds_df[lead_cols + other_cols]

    return final_model, cv_summary, cv_folds_df


# ─────────────────────────────────────────────
# OUTPUT & REPORTING
# ─────────────────────────────────────────────

def _cv_folds_to_export_df(
    cv_folds_df: pd.DataFrame,
    cv_summary: dict,
) -> pd.DataFrame:
    """Append a mean row to per-fold CV metrics for CSV export."""
    mean_row: dict = {"fold": "mean"}
    for col in cv_folds_df.columns:
        if col == "fold":
            continue
        if col in cv_summary:
            mean_row[col] = cv_summary[col]
        elif pd.api.types.is_numeric_dtype(cv_folds_df[col]):
            mean_row[col] = float(cv_folds_df[col].mean())
    return pd.concat(
        [cv_folds_df, pd.DataFrame([mean_row])],
        ignore_index=True,
    )

def _format_ss_when_hit(
    summary: dict,
    hit_label: str,
    avg_key: str,
    median_key: str,
    n_key: str,
    n_total: int,
) -> str:
    avg = summary.get(avg_key, float("nan"))
    median = summary.get(median_key, float("nan"))
    n_hit = int(summary.get(n_key, 0) or 0)
    if pd.isna(avg):
        return f"    SS reduction | {hit_label:<12}: n/a (0/{n_total} hits)"
    return (
        f"    SS reduction | {hit_label:<12}: "
        f"avg {100.0 * avg:.1f}%  (median {100.0 * median:.1f}%, "
        f"{n_hit}/{n_total} hits)"
    )


def _format_ss_when_miss(
    summary: dict,
    hit_label: str,
    avg_key: str,
    median_key: str,
    n_key: str,
    n_total: int,
) -> str:
    avg = summary.get(avg_key, float("nan"))
    median = summary.get(median_key, float("nan"))
    n_miss = int(summary.get(n_key, 0) or 0)
    if pd.isna(avg):
        return f"    SS reduction | {hit_label:<12}: n/a (0/{n_total} misses)"
    return (
        f"    SS reduction | {hit_label:<12}: "
        f"avg {100.0 * avg:.1f}%  (median {100.0 * median:.1f}%, "
        f"{n_miss}/{n_total} misses)"
    )


def _format_rank_when_hit(
    summary: dict,
    hit_label: str,
    avg_key: str,
    median_key: str,
    n_key: str,
    n_total: int,
) -> str:
    avg = summary.get(avg_key, float("nan"))
    median = summary.get(median_key, float("nan"))
    n_hit = int(summary.get(n_key, 0) or 0)
    if pd.isna(avg):
        return f"    Avg rank | {hit_label:<12}: n/a (0/{n_total} hits)"
    return (
        f"    Avg rank | {hit_label:<12}: {avg:.2f}  "
        f"(median {median:.1f}, {n_hit}/{n_total} hits)"
    )


def print_results(
    eval_df: pd.DataFrame,
    k: int,
    top_pct_threshold: float,
    reward_mode: str,
    epsilon: float,
    split_name: str = "Test",
):
    summary = summarize_eval_metrics(
        eval_df, k, top_pct_threshold, reward_mode, epsilon
    )
    pct_label = f"top-{top_pct_threshold:g}%"

    print(f"\n{'='*60}")
    print(f"  {split_name} Results")
    print(f"{'='*60}")
    print(f"  Reward mode (headline) : {reward_mode}")
    print(f"  Objective              : {summary['objective']:.3f}  "
          f"(ε={epsilon})")
    print(f"  Rank bonus (active)    : {summary['rank_bonus']:.3f}")
    print()
    print(f"  Absolute metrics (K={k}):")
    print(f"    Hit@{k}              : {summary['hit_rate']:.3f}")
    print(f"    NDCG@{k}             : {summary['ndcg']:.3f}")
    print(f"    Average rank         : {summary['avg_rank']:.1f}")
    print(f"    Median rank          : {eval_df['target_rank'].median():.1f}")
    print(_format_rank_when_hit(
        summary, f"Hit@{k}",
        "avg_rank_when_hit_at_k", "median_rank_when_hit_at_k", "n_hit_at_k",
        len(eval_df),
    ))
    print()
    print(f"  Percent metrics ({pct_label}):")
    print(f"    Hit@{pct_label}      : {summary['hit_at_top_pct']:.3f}")
    print(f"    Avg top-%ile         : {summary['avg_top_pct']:.1f}")
    print(f"    Median top-%ile      : {eval_df['top_pct'].median():.1f}")
    print(_format_rank_when_hit(
        summary,
        f"Hit@{pct_label}",
        "avg_rank_when_hit_at_top_pct",
        "median_rank_when_hit_at_top_pct",
        "n_hit_at_top_pct",
        len(eval_df),
    ))
    print(f"    Avg SS reduction     : "
          f"{100.0 * summary['avg_search_space_reduction']:.1f}%")
    print(f"    Median SS reduction  : "
          f"{100.0 * summary['median_search_space_reduction']:.1f}%")
    print(_format_ss_when_hit(
        summary, f"Hit@{k}",
        "avg_ss_reduction_when_hit_at_k",
        "median_ss_reduction_when_hit_at_k",
        "n_hit_at_k",
        len(eval_df),
    ))
    print(_format_ss_when_hit(
        summary,
        f"Hit@{pct_label}",
        "avg_ss_reduction_when_hit_at_top_pct",
        "median_ss_reduction_when_hit_at_top_pct",
        "n_hit_at_top_pct",
        len(eval_df),
    ))
    print(_format_ss_when_miss(
        summary, f"Hit@{k}",
        "avg_ss_reduction_when_miss_at_k",
        "median_ss_reduction_when_miss_at_k",
        "n_miss_at_k",
        len(eval_df),
    ))
    print(_format_ss_when_miss(
        summary,
        f"Hit@{pct_label}",
        "avg_ss_reduction_when_miss_at_top_pct",
        "median_ss_reduction_when_miss_at_top_pct",
        "n_miss_at_top_pct",
        len(eval_df),
    ))
    headline = (
        f"Hit@{pct_label}" if reward_mode == "hit_at_top_pct" else f"Hit@{k}"
    )
    print()
    print(f"  Headline ({reward_mode}, {headline}):")
    if pd.isna(summary["avg_rank_when_hit"]):
        print("    Avg rank when hit    : n/a")
        print("    Median rank when hit : n/a")
    else:
        print(f"    Avg rank when hit    : {summary['avg_rank_when_hit']:.2f}")
        print(f"    Median rank when hit : {summary['median_rank_when_hit']:.1f}")
    if pd.isna(summary["avg_ss_reduction_when_hit"]):
        print("    Avg SS red. when hit : n/a")
        print("    Med SS red. when hit : n/a")
    else:
        print(f"    Avg SS red. when hit : "
              f"{100.0 * summary['avg_ss_reduction_when_hit']:.1f}%")
        print(f"    Med SS red. when hit : "
              f"{100.0 * summary['median_ss_reduction_when_hit']:.1f}%")
    if pd.isna(summary.get("avg_ss_reduction_when_miss")):
        print("    Avg SS red. when miss: n/a")
        print("    Med SS red. when miss: n/a")
    else:
        print(f"    Avg SS red. when miss: "
              f"{100.0 * summary['avg_ss_reduction_when_miss']:.1f}%")
        print(f"    Med SS red. when miss: "
              f"{100.0 * summary['median_ss_reduction_when_miss']:.1f}%")

    print(f"\n  Per Residue Type:")
    per_res = (eval_df.groupby("target_residue_type")
               .agg(
                   N=("hit_at_k", "count"),
                   objective=("objective", "mean"),
                   hit_rate=("hit_at_k", "mean"),
                   hit_top_pct=("hit_at_top_pct", "mean"),
                   ndcg=("ndcg_at_k", "mean"),
                   avg_rank=("target_rank", "mean"),
                   avg_top_pct=("top_pct", "mean"),
               )
               .reset_index())
    print(per_res.to_string(index=False, float_format="%.3f"))
    print(f"{'='*60}\n")


def compute_candidate_ranks(
    df: pd.DataFrame,
    score_col: str = "pred_score",
    group_col: str = "query_group",
) -> pd.Series:
    """Rank candidates within each query group (1 = highest pred_score)."""
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' required for candidate ranks.")
    return (
        df.groupby(group_col, sort=False)[score_col]
        .rank(ascending=False, method="first")
        .astype(int)
    )


def export_shap_csvs(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    output_dir: str | None = None,
    prefix: str = "test",
    per_candidate_path: str | None = None,
    global_path: str | None = None,
    max_rows: int | None = None,
    random_state: int = 42,
    feature_matrix: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export TreeSHAP values per candidate row and global mean SHAP per feature.

    Per-candidate CSV columns: identifiers, candidate_rank, pred_score, then for
    each model input: {feature} and shap_{feature} (value + SHAP contribution).
    Global CSV: mean_shap, mean_abs_shap, std_shap, plus LGBM gain importance
    (distinct from SHAP — gain is tree split frequency, not attribution).
    """
    try:
        import shap
    except ImportError:
        print("[WARN] shap not installed; skipping SHAP export. pip install shap")
        return pd.DataFrame(), pd.DataFrame()

    if df.empty:
        print("[WARN] No rows for SHAP export.")
        return pd.DataFrame(), pd.DataFrame()

    missing = [c for c in feature_cols if c not in df.columns]
    if feature_matrix is None and missing:
        print(f"[WARN] SHAP skipped; missing feature columns: {missing[:5]}")
        return pd.DataFrame(), pd.DataFrame()
    if feature_matrix is not None and missing:
        print(f"  Note: {len(missing)} model feature(s) taken from prepared matrix "
              f"(not in df), e.g. {missing[:3]}")

    work = df.copy()
    row_positions: np.ndarray | None = None
    if max_rows is not None and len(work) > max_rows:
        print(f"[WARN] SHAP subsampled to {max_rows:,} rows (from {len(work):,})")
        sampled_idx = work.sample(n=max_rows, random_state=random_state).index
        row_positions = df.index.get_indexer(sampled_idx)
        work = df.loc[sampled_idx].copy()

    if feature_matrix is not None:
        if feature_matrix.shape[0] != len(df):
            sys.exit(
                "[ERROR] feature_matrix row count must match dataframe length "
                "before SHAP subsampling."
            )
        if row_positions is not None:
            X = feature_matrix[row_positions].astype(np.float32)
        else:
            X = feature_matrix.astype(np.float32)
    else:
        X = work[feature_cols].values.astype(np.float32)

    if "pdb_id" not in work.columns:
        if "Name" in work.columns:
            work["pdb_id"] = work["Name"].astype(str).str.split("-").str[0]
        elif "_name_upper" in work.columns:
            work["pdb_id"] = work["_name_upper"].astype(str).str.split("-").str[0]

    if "pdb_id" in work.columns:
        work["pdb_name"] = work["pdb_id"]
    if "Warhead" in work.columns:
        work["warhead_name"] = work["Warhead"]

    if "pred_score" in work.columns:
        work["candidate_rank"] = compute_candidate_ranks(work)

    print(f"  Computing SHAP for {len(work):,} candidates × {len(feature_cols)} features...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else np.mean(
            shap_values, axis=0
        )
    shap_values = np.asarray(shap_values, dtype=np.float64)

    meta_candidates = [
        "Name", "pdb_id", "pdb_name", "Warhead", "warhead_name", "query_group",
        "Residue", "ResNum", "Chain",
        "pred_score", "candidate_rank", "relevance",
    ]
    meta_cols = [c for c in meta_candidates if c in work.columns]
    meta_df = work[meta_cols].reset_index(drop=True)

    value_cols: dict[str, pd.Series] = {}
    shap_cols: dict[str, pd.Series] = {}
    for j, feat in enumerate(feature_cols):
        if feat in work.columns:
            value_cols[feat] = work[feat].reset_index(drop=True)
        else:
            value_cols[feat] = pd.Series(X[:, j], name=feat)
        shap_cols[f"shap_{feat}"] = pd.Series(shap_values[:, j], name=f"shap_{feat}")

    interleaved: list[pd.DataFrame] = [meta_df]
    for feat in feature_cols:
        interleaved.append(value_cols[feat].to_frame(name=feat))
        interleaved.append(shap_cols[f"shap_{feat}"].to_frame(name=f"shap_{feat}"))
    detail_df = pd.concat(interleaved, axis=1)

    fi_map = {}
    if hasattr(model, "feature_importances_"):
        fi_map = dict(zip(feature_cols, model.feature_importances_))

    global_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_shap": np.mean(shap_values, axis=0),
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "std_shap": np.std(shap_values, axis=0),
        "lgbm_gain_importance": [fi_map.get(f, np.nan) for f in feature_cols],
    }).sort_values("mean_abs_shap", ascending=False)

    if per_candidate_path is None:
        if output_dir is None:
            raise ValueError("Provide output_dir or per_candidate_path for SHAP export.")
        per_candidate_path = os.path.join(output_dir, f"{prefix}_shap_per_candidate.csv")
    if global_path is None:
        per_path = Path(per_candidate_path)
        if per_path.stem.endswith("_per_candidate"):
            global_path = str(
                per_path.with_name(
                    per_path.stem.replace("_per_candidate", "_global") + per_path.suffix
                )
            )
        elif output_dir is not None:
            global_path = os.path.join(output_dir, f"{prefix}_shap_global.csv")
        else:
            global_path = str(
                per_path.with_name(per_path.stem + "_global" + per_path.suffix)
            )

    detail_df.to_csv(per_candidate_path, index=False)
    global_df.to_csv(global_path, index=False)
    print(f"  SHAP per candidate: {per_candidate_path}")
    print(f"  SHAP global:        {global_path}")

    return detail_df, global_df


def export_screen_eval_csvs(
    test_eval_df: pd.DataFrame,
    scored_test_df: pd.DataFrame,
    labels_csv: str,
    output_dir: str,
    reward_mode: str,
    prefix: str = "test",
    perfect_match: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Write per-label, query-group, warhead, and residue accuracy CSVs.

    Uses the same helpers as Cov_Screen.py (lazy import to avoid circular deps).
    Metrics are computed per query group, never pooled across candidates.
    With perfect_match, multi-warhead labels require every matched warhead
    query group to hit (still one hit per ligand base Name × site).
    """
    from Cov_Screen import (
        aggregate_label_eval_rows,
        build_label_warhead_counts,
        enrich_query_group_eval,
        per_residue_accuracy_breakdown,
        per_warhead_screen_breakdown,
    )

    os.makedirs(output_dir, exist_ok=True)
    label_wh_counts = build_label_warhead_counts(labels_csv)
    eval_qg_df = enrich_query_group_eval(test_eval_df, scored_test_df)
    eval_label_df = aggregate_label_eval_rows(
        eval_qg_df, perfect_match=perfect_match, label_wh_counts=label_wh_counts,
    )
    warhead_df = per_warhead_screen_breakdown(eval_qg_df, reward_mode)
    residue_df = per_residue_accuracy_breakdown(
        eval_qg_df, eval_label_df, reward_mode,
    )

    label_path = os.path.join(output_dir, f"{prefix}_label_results.csv")
    qg_path = os.path.join(output_dir, f"{prefix}_query_groups.csv")
    warhead_path = os.path.join(output_dir, f"{prefix}_warhead_accuracy.csv")
    residue_path = os.path.join(output_dir, f"{prefix}_residue_accuracy.csv")

    eval_label_df.to_csv(label_path, index=False)
    print(f"  Test label results:     {label_path}")
    eval_qg_df.to_csv(qg_path, index=False)
    print(f"  Test query groups:      {qg_path}")
    if warhead_df.empty:
        print("  [WARN] No warhead accuracy breakdown to export.")
    else:
        warhead_df.to_csv(warhead_path, index=False)
        print(f"  Test warhead accuracy:  {warhead_path}")
    if residue_df.empty:
        print("  [WARN] No residue accuracy breakdown to export.")
    else:
        residue_df.to_csv(residue_path, index=False)
        print(f"  Test residue accuracy:  {residue_path}")

    return eval_label_df, eval_qg_df, warhead_df, residue_df


def save_outputs(
    model: lgb.LGBMRanker,
    feature_cols: list[str],
    test_eval_df: pd.DataFrame,
    scored_test_df: pd.DataFrame,
    labels_csv: str,
    cv_summary: dict,
    cv_folds_df: pd.DataFrame,
    output_dir: str,
    k: int,
    top_pct_threshold: float,
    reward_mode: str,
    epsilon: float,
    composition: dict | None = None,
    large_pool_prefilter: dict | None = None,
    include_residue_type: bool = True,
    export_shap: bool = True,
    shap_max_rows: int | None = None,
    train_pool_totals_before_filter: dict[str, int] | None = None,
    perfect_match: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "lgbm_ranker.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_cols,
            "model_type": "ranker",
            "include_residue_type": include_residue_type,
            "k": k,
            "reward_mode": reward_mode,
            "top_pct_threshold": top_pct_threshold,
            "rank_bonus_epsilon": epsilon,
            "large_pool_prefilter": large_pool_prefilter,
            "perfect_match": perfect_match,
        }, f)
    print(f"  Model saved:   {model_path}")

    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_path = os.path.join(output_dir, "feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    print(f"  Feature importance: {fi_path}")

    if not cv_folds_df.empty:
        cv_path = os.path.join(output_dir, "cv_fold_results.csv")
        _cv_folds_to_export_df(cv_folds_df, cv_summary).to_csv(cv_path, index=False)
        print(f"  CV fold results: {cv_path}")

    res_path = os.path.join(output_dir, "test_results.csv")
    test_eval_df.to_csv(res_path, index=False)
    print(f"  Test results (qg): {res_path}")

    eval_label_df, eval_qg_df, warhead_df, residue_df = export_screen_eval_csvs(
        test_eval_df=test_eval_df,
        scored_test_df=scored_test_df,
        labels_csv=labels_csv,
        output_dir=output_dir,
        reward_mode=reward_mode,
        prefix="test",
        perfect_match=perfect_match,
    )

    shap_global_df = pd.DataFrame()
    if export_shap:
        _, shap_global_df = export_shap_csvs(
            model=model,
            df=scored_test_df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix="test",
            max_rows=shap_max_rows,
        )

    from Cov_Screen import per_residue_screen_breakdown

    test_overall = summarize_screen_metrics(
        eval_label_df, eval_qg_df, k, top_pct_threshold, reward_mode, epsilon,
    )
    per_res = per_residue_screen_breakdown(eval_label_df, eval_qg_df).to_dict(
        orient="records"
    )

    summary = {
        "k": k,
        "top_pct_threshold": top_pct_threshold,
        "reward_mode": reward_mode,
        "rank_bonus_epsilon": epsilon,
        "perfect_match": perfect_match,
        "large_pool_prefilter": large_pool_prefilter,
        "cv": cv_summary,
        "cv_folds": cv_folds_df.to_dict(orient="records"),
        "test_overall": test_overall,
        "test_per_residue": per_res,
        "test_per_residue_accuracy": residue_df.to_dict(orient="records"),
        "test_per_warhead": warhead_df.to_dict(orient="records"),
    }
    if not shap_global_df.empty:
        summary["test_shap_global"] = shap_global_df.to_dict(orient="records")
    if composition and composition.get("n_query_groups"):
        comp_export = {
            k_: v for k_, v in composition.items() if k_ != "detail"
        }
        if train_pool_totals_before_filter is not None:
            comp_export["train_pool_residue_totals_before_filter"] = (
                train_pool_totals_before_filter
            )
        summary["residue_composition"] = comp_export
        comp_path = os.path.join(output_dir, "residue_composition_detail.csv")
        detail_df = composition["detail"].copy()
        totals_row = {
            "query_group": "__test_totals__",
            "label_type": "",
            "rank1_type": "",
            "rank1_matches_label_type": "",
            "n_candidates": int(detail_df["n_candidates"].sum()),
        }
        for res in NUCLEOPHILE_RESIDUES:
            col = f"n_{res}"
            if col in detail_df.columns:
                totals_row[col] = int(detail_df[col].sum())
        detail_df = pd.concat(
            [detail_df, pd.DataFrame([totals_row])],
            ignore_index=True,
        )
        detail_df.to_csv(comp_path, index=False)
        print(f"  Residue composition: {comp_path}")
        if train_pool_totals_before_filter is not None:
            print("  Train nucleophile pool totals (before large-pool prefilter):")
            for res in NUCLEOPHILE_RESIDUES:
                print(f"    {res:3s}  {train_pool_totals_before_filter.get(res, 0):>8,}")
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON:  {summary_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LGBMRanker for nucleophilic residue ranking"
    )
    p.add_argument("--training",   required=True,
                   help="Path to training CSV")
    p.add_argument("--labels",     required=True,
                   help="Path to labels CSV")
    p.add_argument("--features",   nargs="+", default=[
                       "Abs_Side_SASA", "Rel_Side_SASA",
                       "deprotonation_prob", "pKa_shift",
                       "Accessibility_Score",
                       "Reactivity_Score",
                       "Reactivity_Score_Protonated",
                       "Reactivity_Score_Deprotonated",
                   ],
                   help="Feature column names from training CSV to use")
    p.add_argument("--residue_specific_features", nargs="+", default=[
                       "Reactivity_Score_Protonated",
                       "Reactivity_Score_Deprotonated",
                       "Fukui_Protonated", "Fukui_Deprotonated",
                       "Nucleophilicity_Index_Protonated",
                       "Nucleophilicity_Index_Deprotonated",
                   ],
                   help="Features that are constant per residue type x warhead "
                        "(used raw, not within-protein normalized)")
    p.add_argument("--topk",       type=int, default=10,
                   help="K for absolute Hit@K and NDCG@K (default 10)")
    p.add_argument(
        "--perfect-match",
        action="store_true",
        help="For label-level hit rate: multi-warhead Frankenstein labels require "
             "every matched training warhead query group to hit (default: any). "
             "Still at most one hit per ligand base Name × site. Same as Cov_Screen.",
    )
    p.add_argument("--reward-mode", choices=REWARD_MODES, default="hit_at_k",
                   help="Headline reward for CV objective: hit_at_k (default) or "
                        "hit_at_top_pct. Both absolute and percent metrics are "
                        "always reported.")
    p.add_argument("--top-pct", type=float, default=10.0,
                   help="Top-X%% threshold for Hit@top-X%% and percent-mode "
                        "reward bonus (default 10)")
    p.add_argument("--rank-bonus-epsilon", type=float,
                   default=DEFAULT_RANK_BONUS_EPSILON,
                   help="Weight on rank bonus inside the active reward zone "
                        f"(default {DEFAULT_RANK_BONUS_EPSILON})")
    p.add_argument("--pdb_folder", default=None,
                   help="Folder containing PDB files (downloads missing ones here)")
    p.add_argument("--mmseqs_dir", default="mmseqs_output",
                   help="Directory for MMseqs2 intermediate files")
    p.add_argument("--seq_identity", type=float, default=0.3,
                   help="MMseqs2 sequence identity threshold (default 0.3)")
    p.add_argument("--coverage",   type=float, default=0.8,
                   help="MMseqs2 alignment coverage (default 0.8)")
    p.add_argument("--deep-cluster", action="store_true",
                   help="Cluster per polymer chain, then merge PDBs that share any "
                        "chain cluster (better for multisubunit structures). "
                        "Default: concatenate all chains per PDB.")
    p.add_argument("--deep-cluster-min-chain-length", type=int, default=30,
                   help="Minimum chain length for --deep-cluster (default 30)")
    p.add_argument(
        "--clusters-tsv",
        default=None,
        metavar="PATH",
        help="Reuse MMseqs2 clusters.tsv instead of rerunning MMseqs2. "
             "Use the same --deep-cluster setting as when the TSV was created.",
    )
    p.add_argument(
        "--split-csv",
        default=None,
        metavar="PATH",
        help="Fixed query_group→train/test assignments (columns: query_group, split). "
             "Skips random cluster holdout; use for ablation runs.",
    )
    p.add_argument(
        "--export-split",
        default=None,
        metavar="PATH",
        help="After splitting, write query_group,split CSV for reuse with --split-csv.",
    )
    p.add_argument("--test_size",  type=float, default=0.2,
                   help="Fraction of clusters for held-out test set (default 0.2). "
                        "Ignored when --split-csv is set.")
    p.add_argument("--n_folds",    type=int, default=5,
                   help="Number of GroupKFold CV folds (default 5)")
    p.add_argument("--normalize_within_protein", action="store_true",
                   help="If set, also add within-protein normalized versions "
                        "of --features as extra features")
    p.add_argument("--res-num", type=int, nargs="+", default=None,
                   metavar="N",
                   help="Add binary feature resnum_eq_N for each residue number "
                        "N (e.g. --res-num 1 for position 1 / N-terminal signal). "
                        "Off by default.")
    p.add_argument("--no-residue-type", action="store_true",
                   help="Exclude residue type one-hot features (res_CYS, res_SER, "
                        "etc.) from model inputs")
    p.add_argument("--no-residue-specific-features", action="store_true",
                   help="Exclude --residue_specific_features (QM reactivity scores, "
                        "Fukui, nucleophilicity, etc.) from model inputs")
    p.add_argument("--large-pool-prefilter", action="store_true",
                   help="Enable feature prefilter on large query groups in the "
                        "train split only (off by default). Requires "
                        "--large-pool-filter.")
    p.add_argument("--large-pool-min-size", type=int, default=500,
                   help="Apply --large-pool-filter when a query group has more "
                        "than this many candidates (default 500). Only used with "
                        "--large-pool-prefilter.")
    p.add_argument("--large-pool-filter", nargs="+", default=None,
                   metavar="SPEC",
                   help="Prefilter spec as COL OP VALUE triplets, e.g. "
                        "Rel_Side_SASA gte 5. Multiple filters are AND-ed. "
                        "Required with --large-pool-prefilter.")
    p.add_argument("--output_dir", default="lgbm_output",
                   help="Directory for output files")
    p.add_argument("--no-shap", action="store_true",
                   help="Skip SHAP CSV export (test_shap_per_candidate/global)")
    p.add_argument("--shap-max-rows", type=int, default=None,
                   help="Optional cap on rows for SHAP (random subsample if exceeded)")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_pct <= 0 or args.top_pct > 100:
        sys.exit("[ERROR] --top-pct must be in (0, 100].")
    if args.topk < 1:
        sys.exit("[ERROR] --topk must be >= 1.")
    if args.split_csv and args.export_split:
        sys.exit("[ERROR] Use either --split-csv or --export-split, not both.")

    large_pool_prefilter_meta = None
    pool_prefilter_filters = None
    if args.large_pool_prefilter:
        if not args.large_pool_filter:
            sys.exit(
                "[ERROR] --large-pool-filter is required when "
                "--large-pool-prefilter is set."
            )
        if args.large_pool_min_size < 1:
            sys.exit("[ERROR] --large-pool-min-size must be >= 1.")
        pool_prefilter_filters = parse_simple_filters(args.large_pool_filter)

    # 1. Load & merge
    merged = load_and_merge(args.training, args.labels)

    # 2. Cluster
    cluster_map = build_cluster_map(
        merged, args.pdb_folder, args.mmseqs_dir,
        args.seq_identity, args.coverage,
        deep_cluster=args.deep_cluster,
        min_chain_length=args.deep_cluster_min_chain_length,
        clusters_tsv=args.clusters_tsv,
    )

    # 3. Split
    merged = assign_splits(
        merged, cluster_map,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_state=args.random_state,
        split_csv=args.split_csv,
        export_split=args.export_split,
    )

    # 4. Features
    residue_specific_cols = (
        [] if args.no_residue_specific_features else args.residue_specific_features
    )
    # Deduplicate feature lists
    all_req_features = list(dict.fromkeys(args.features + residue_specific_cols))
    merged, feature_cols = build_features(
        merged,
        feature_cols=all_req_features,
        residue_specific_cols=residue_specific_cols,
        normalize_within_protein=args.normalize_within_protein,
        resnum_equals=args.res_num,
        include_residue_type=not args.no_residue_type,
    )

    # 5. Train
    train_df = merged[merged["split"] == "train"].copy()
    test_df  = merged[merged["split"] == "test"].copy()

    # Sort within each query group (required by LGBMRanker)
    train_df = train_df.sort_values("query_group").reset_index(drop=True)
    test_df  = test_df.sort_values("query_group").reset_index(drop=True)

    train_pool_totals_before_filter = aggregate_nucleophile_totals(train_df)

    if args.large_pool_prefilter:
        train_df, large_pool_prefilter_meta = apply_large_pool_prefilter(
            train_df,
            pool_prefilter_filters,
            args.large_pool_min_size,
        )
        train_df = train_df.sort_values("query_group").reset_index(drop=True)

    model, cv_summary, cv_folds_df = train_with_crossval(
        train_df, feature_cols,
        n_folds=args.n_folds,
        k=args.topk,
        top_pct_threshold=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
    )

    # 6. Evaluate on test set
    print(f"\n[6/7] Evaluating on held-out test set...")
    test_df["pred_score"] = model.predict(test_df[feature_cols].values)
    test_eval_df = evaluate_predictions(
        test_df, "pred_score",
        k=args.topk,
        top_pct_threshold=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
    )
    composition = analyze_residue_composition(
        test_df, "pred_score", k=args.topk, top_pct_threshold=args.top_pct,
    )
    print_results(
        test_eval_df,
        k=args.topk,
        top_pct_threshold=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
    )
    print_residue_composition(composition, k=args.topk, top_pct_threshold=args.top_pct)

    # 7. Save outputs
    print(f"[7/7] Saving outputs to '{args.output_dir}'...")
    save_outputs(
        model, feature_cols,
        test_eval_df, test_df,
        args.labels,
        cv_summary,
        cv_folds_df,
        args.output_dir,
        k=args.topk,
        top_pct_threshold=args.top_pct,
        reward_mode=args.reward_mode,
        epsilon=args.rank_bonus_epsilon,
        composition=composition,
        large_pool_prefilter=large_pool_prefilter_meta,
        include_residue_type=not args.no_residue_type,
        export_shap=not args.no_shap,
        shap_max_rows=args.shap_max_rows,
        train_pool_totals_before_filter=train_pool_totals_before_filter,
        perfect_match=args.perfect_match,
    )

    print("Done.")


if __name__ == "__main__":
    main()