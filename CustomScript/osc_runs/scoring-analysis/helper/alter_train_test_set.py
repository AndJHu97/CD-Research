#!/usr/bin/env python3
"""Alter a pinned query_group_splits.csv (force PDBs into test, retarget size).

For Training_Cov_Screen.py reuse, only the split CSV needs changing:
    --deep-cluster --clusters-tsv ... --split-csv altered_splits.csv

This script rebuilds train/test at the *PDB-cluster* level (same deep-cluster
merge as Training_Cov_Screen), forces named PDB clusters into test, then
randomly samples additional clusters to reach --test-size (default 10%).

Usage:
    python alter_train_test_set.py \\
        --split-csv query_group_splits.csv \\
        --clusters-tsv mmseqs_output/clusters.tsv \\
        --force-pdb 5FMG \\
        --test-size 0.1 \\
        --output query_group_splits_5fmg_test.csv

Forcing PDBs from an external test set that are not in clusters.tsv:
    python alter_train_test_set.py \\
        --split-csv query_group_splits.csv \\
        --clusters-tsv mmseqs_output/clusters.tsv \\
        --chains-fasta mmseqs_output/chains.fasta \\
        --recluster --recluster-dir mmseqs_recluster \\
        --pdb-folder ../Existing_Structures \\
        --seq-identity 0.5 --coverage 0.8 \\
        --force-pdb-file external_sites.txt \\
        --output query_group_splits_external_test.csv

--recluster extracts chains for the unfamiliar PDBs, appends them to the
training chains.fasta, and reruns MMseqs2 on the combined set in
--recluster-dir. Train with the clusters.tsv written there.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description=(
			"Rewrite query_group_splits.csv: force PDB cluster(s) into test "
			"and retarget the cluster-level test fraction."
		),
	)
	p.add_argument(
		"--split-csv",
		required=True,
		type=Path,
		help="Existing query_group,split CSV from --export-split",
	)
	p.add_argument(
		"--clusters-tsv",
		required=True,
		type=Path,
		help="MMseqs2 clusters.tsv used with --deep-cluster",
	)
	p.add_argument(
		"--force-pdb",
		action="append",
		default=[],
		metavar="PDB",
		help="PDB ID whose entire deep-cluster must be in test (repeatable)",
	)
	p.add_argument(
		"--force-pdb-file",
		type=Path,
		default=None,
		metavar="PATH",
		help=(
			"Text file of PDBs to force into test, one per line. Lines like "
			"'4HCU|CYS|442|A' use the PDB field; '#' starts a comment."
		),
	)
	p.add_argument(
		"--test-size",
		type=float,
		default=0.1,
		help="Target fraction of PDB clusters in test (default 0.1)",
	)
	p.add_argument(
		"--keep-existing-test",
		action="store_true",
		help=(
			"Keep every cluster already marked test in --split-csv, add the "
			"forced clusters, and sample only what is still missing"
		),
	)
	p.add_argument(
		"--output",
		required=True,
		type=Path,
		help="Path for rewritten query_group_splits CSV",
	)
	p.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="RNG seed for sampling non-forced test clusters (default 42)",
	)
	p.add_argument(
		"--no-deep-cluster",
		action="store_true",
		help="Treat clusters.tsv as PDB-level (not chain-level deep merge)",
	)
	p.add_argument(
		"--recluster",
		action="store_true",
		help=(
			"Cluster forced PDBs missing from --clusters-tsv together with "
			"the training chains (requires mmseqs and --pdb-folder)"
		),
	)
	p.add_argument(
		"--chains-fasta",
		type=Path,
		default=None,
		help="Chain FASTA behind --clusters-tsv (default: chains.fasta beside it)",
	)
	p.add_argument(
		"--recluster-dir",
		type=Path,
		default=Path("mmseqs_recluster"),
		help="Output directory for --recluster (default mmseqs_recluster)",
	)
	p.add_argument(
		"--pdb-folder",
		type=Path,
		default=None,
		help="PDB folder for chain extraction; missing PDBs are downloaded here",
	)
	p.add_argument(
		"--seq-identity",
		type=float,
		default=0.5,
		help="MMseqs2 --min-seq-id for --recluster (default 0.5)",
	)
	p.add_argument(
		"--coverage",
		type=float,
		default=0.8,
		help="MMseqs2 -c (cov-mode 0) for --recluster (default 0.8)",
	)
	p.add_argument(
		"--min-chain-length",
		type=int,
		default=30,
		help="Shortest chain written to FASTA (default 30, as in training)",
	)
	return p.parse_args()


def load_clusters_tsv(path: Path) -> dict[str, str]:
	cluster_map: dict[str, str] = {}
	with path.open(encoding="utf-8-sig") as fh:
		for line in fh:
			parts = line.strip().split("\t")
			if len(parts) != 2:
				continue
			rep, member = parts
			cluster_map[member.upper()] = rep.upper()
	return cluster_map


def _parse_chain_header(header: str) -> str | None:
	if "__" not in header:
		return None
	pdb_id = header.rsplit("__", 1)[0].strip().upper()
	return pdb_id or None


def merge_chain_clusters_to_pdb_map(
	pdb_ids: list[str],
	chain_cluster_map: dict[str, str],
) -> dict[str, str]:
	"""Same deep-cluster lift as Training_Cov_Screen.merge_chain_clusters_to_pdb_map."""
	pdb_ids_upper = [str(pid).strip().upper() for pid in pdb_ids]
	rep_to_pdbs: dict[str, list[str]] = defaultdict(list)

	for member, rep in chain_cluster_map.items():
		pdb_id = _parse_chain_header(member)
		if pdb_id is None:
			continue
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
		rep = sorted(members)[0]
		for pdb_id in members:
			cluster_map[pdb_id] = rep
	return cluster_map


def pdb_from_query_group(query_group: str) -> str:
	return str(query_group).split("-", 1)[0].strip().upper()


def pdbs_in_clusters(raw: dict[str, str], deep_cluster: bool) -> set[str]:
	"""PDB IDs that have at least one member in a clusters TSV."""
	if not deep_cluster:
		return {member.upper() for member in raw}
	return {
		pdb_id
		for pdb_id in (_parse_chain_header(member) for member in raw)
		if pdb_id
	}


def read_force_pdbs(args: argparse.Namespace) -> list[str]:
	pdbs = list(args.force_pdb)
	if args.force_pdb_file is not None:
		if not args.force_pdb_file.is_file():
			sys.exit(f"[ERROR] --force-pdb-file not found: {args.force_pdb_file}")
		for line in args.force_pdb_file.read_text(encoding="utf-8-sig").splitlines():
			token = line.split("#", 1)[0].strip()
			if not token:
				continue
			token = token.split("|", 1)[0].split(",", 1)[0].strip()
			if token.upper() in {"PDB", "PDB_ID", "PDBID"}:
				continue
			pdbs.append(token)
	return list(dict.fromkeys(p.strip().upper() for p in pdbs if p.strip()))


def build_pdb_cluster_map(
	pdb_ids: list[str],
	raw: dict[str, str],
	deep_cluster: bool,
) -> dict[str, str]:
	if deep_cluster:
		cluster_map = merge_chain_clusters_to_pdb_map(pdb_ids, raw)
	else:
		cluster_map = {
			pid: raw.get(pid, pid) for pid in (p.upper() for p in pdb_ids)
		}

	for pid in pdb_ids:
		pid_u = pid.upper()
		if pid_u not in cluster_map:
			cluster_map[pid_u] = pid_u
	return cluster_map


# ─────────────────────────────────────────────
# RECLUSTERING (external PDBs missing from clusters.tsv)
# ─────────────────────────────────────────────

def load_screen_module() -> ModuleType:
	"""Training_Cov_Screen from helper/.. or from a flat copy beside this file."""
	here = Path(__file__).resolve().parent
	for root in (here, here.parent):
		path = root / "Training_Cov_Screen.py"
		if not path.is_file():
			continue
		spec = importlib.util.spec_from_file_location("training_cov_screen", path)
		if spec is None or spec.loader is None:
			break
		module = importlib.util.module_from_spec(spec)
		sys.modules["training_cov_screen"] = module
		spec.loader.exec_module(module)
		return module
	sys.exit(
		"[ERROR] Training_Cov_Screen.py not found beside this script or one "
		"directory up; --recluster needs its chain extraction and MMseqs2 code."
	)


def read_fasta(path: Path) -> list[tuple[str, str]]:
	records: list[tuple[str, str]] = []
	header: str | None = None
	chunks: list[str] = []
	with path.open(encoding="utf-8-sig") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			if line.startswith(">"):
				if header is not None:
					records.append((header, "".join(chunks)))
				header = line[1:].strip()
				chunks = []
			else:
				chunks.append(line)
	if header is not None:
		records.append((header, "".join(chunks)))
	return records


def mmseqs_version() -> str:
	result = subprocess.run(["mmseqs", "version"], capture_output=True, text=True)
	return (result.stdout or result.stderr or "").strip() or "unknown"


def recluster_with_external(
	args: argparse.Namespace,
	force_pdbs: list[str],
	raw: dict[str, str],
) -> tuple[Path, dict[str, str], dict]:
	"""Rerun MMseqs2 on training chains plus unfamiliar forced PDB chains."""
	known = pdbs_in_clusters(raw, deep_cluster=True)
	unfamiliar = [pdb for pdb in force_pdbs if pdb not in known]
	meta: dict = {"unfamiliar_pdbs": unfamiliar, "reclustered": False}
	if not unfamiliar:
		print(
			"[INFO] --recluster: every forced PDB is already in "
			f"{args.clusters_tsv}; reusing it"
		)
		return args.clusters_tsv, raw, meta

	print(
		f"[INFO] --recluster: {len(unfamiliar)} forced PDB(s) not in "
		f"{args.clusters_tsv}: {', '.join(unfamiliar)}"
	)
	if shutil.which("mmseqs") is None:
		sys.exit(
			"[ERROR] mmseqs not found on PATH. Without it, Training_Cov_Screen "
			"puts every chain in its own cluster, so similar proteins would not "
			"be caught. Load MMseqs2 (e.g. module load) and rerun."
		)
	if args.pdb_folder is None:
		sys.exit("[ERROR] --recluster needs --pdb-folder for chain extraction")
	if not args.pdb_folder.is_dir():
		sys.exit(f"[ERROR] --pdb-folder not found: {args.pdb_folder}")

	chains_fasta = args.chains_fasta or args.clusters_tsv.parent / "chains.fasta"
	if not chains_fasta.is_file():
		sys.exit(
			f"[ERROR] Training chain FASTA not found: {chains_fasta}. "
			"Pass --chains-fasta."
		)

	out_dir = args.recluster_dir.resolve()
	protected = {
		args.clusters_tsv.resolve().parent,
		chains_fasta.resolve().parent,
	}
	if out_dir in protected:
		sys.exit(
			f"[ERROR] --recluster-dir {out_dir} holds the original clustering. "
			"MMseqs2 clears clusters.tsv and its databases there; choose a "
			"new directory."
		)
	out_dir.mkdir(parents=True, exist_ok=True)

	train_records = read_fasta(chains_fasta)
	train_headers = {header.upper() for header, _ in train_records}
	missing_chains = sorted(set(raw) - train_headers)
	if missing_chains:
		preview = ", ".join(missing_chains[:5])
		sys.exit(
			f"[ERROR] {len(missing_chains)} chain(s) in {args.clusters_tsv} are "
			f"missing from {chains_fasta} (e.g. {preview}). The FASTA does not "
			"match the clustering; pass the chains.fasta from the same run."
		)
	extra_chains = len(train_headers - set(raw))
	if extra_chains:
		print(
			f"[WARN] {chains_fasta} has {extra_chains} chain(s) not in "
			f"{args.clusters_tsv}; they are clustered too."
		)

	screen = load_screen_module()
	new_fasta = out_dir / "new_chains.fasta"
	n_new = screen.extract_chain_fasta_from_pdbs(
		str(args.pdb_folder),
		unfamiliar,
		str(new_fasta),
		min_chain_length=args.min_chain_length,
	)
	new_records = read_fasta(new_fasta) if n_new else []
	extracted = {_parse_chain_header(header) for header, _ in new_records}
	failed = [pdb for pdb in unfamiliar if pdb not in extracted]
	if failed:
		sys.exit(
			f"[ERROR] No chains >= {args.min_chain_length} residues extracted "
			f"for: {', '.join(failed)}. They cannot be compared to the "
			"training set; fix the PDB files or drop these IDs."
		)

	combined = out_dir / "combined_chains.fasta"
	seen: set[str] = set()
	n_written = 0
	with combined.open("w", encoding="utf-8") as fh:
		for header, sequence in [*train_records, *new_records]:
			key = header.upper()
			if key in seen:
				continue
			seen.add(key)
			fh.write(f">{header}\n{sequence}\n")
			n_written += 1
	print(
		f"[INFO] Combined FASTA: {len(train_records):,} training + "
		f"{len(new_records):,} new chain(s) -> {combined}"
	)

	screen.run_mmseqs2(
		str(combined), str(out_dir), args.seq_identity, args.coverage
	)
	new_tsv = out_dir / "clusters.tsv"
	if not new_tsv.is_file():
		sys.exit(f"[ERROR] MMseqs2 did not write {new_tsv}; see messages above")
	new_raw = load_clusters_tsv(new_tsv)
	if not new_raw:
		sys.exit(f"[ERROR] Empty or unreadable clusters TSV: {new_tsv}")

	meta.update({
		"reclustered": True,
		"chains_fasta": str(chains_fasta.resolve()),
		"combined_fasta": str(combined),
		"clusters_tsv": str(new_tsv),
		"n_training_chains": len(train_records),
		"n_new_chains": len(new_records),
		"n_combined_chains": n_written,
		"seq_identity": args.seq_identity,
		"coverage": args.coverage,
		"cov_mode": 0,
		"min_chain_length": args.min_chain_length,
		"mmseqs_version": mmseqs_version(),
	})
	return new_tsv, new_raw, meta


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────

def alter_splits(
	split_df: pd.DataFrame,
	cluster_map: dict[str, str],
	force_pdbs: list[str],
	test_size: float,
	random_state: int,
	known_pdbs: set[str],
	keep_existing_test: bool = False,
) -> tuple[pd.DataFrame, dict]:
	df = split_df.copy()
	if "query_group" not in df.columns or "split" not in df.columns:
		sys.exit("[ERROR] --split-csv must have columns: query_group, split")

	original_split = df["split"].astype(str).str.strip().str.lower()
	df["pdb_id"] = df["query_group"].map(pdb_from_query_group)
	df["cluster_id"] = df["pdb_id"].map(cluster_map).fillna(df["pdb_id"])

	all_clusters = sorted(df["cluster_id"].unique())
	n_clusters = len(all_clusters)
	n_test_target = max(1, int(round(n_clusters * test_size)))

	split_pdbs = set(df["pdb_id"])
	forced_clusters: set[str] = set()
	forced_detail: list[dict] = []
	for pdb_u in force_pdbs:
		if pdb_u not in split_pdbs and pdb_u not in known_pdbs:
			sys.exit(
				f"[ERROR] Forced PDB '{pdb_u}' is in neither the split CSV nor "
				"the clusters TSV. Pass --recluster with --pdb-folder to cluster "
				"its chains against the training set."
			)
		cid = cluster_map.get(pdb_u, pdb_u)
		in_cluster = df["cluster_id"] == cid
		members = sorted(df.loc[in_cluster, "pdb_id"].unique())
		n_qg = int(in_cluster.sum())
		forced_detail.append({
			"pdb_id": pdb_u,
			"in_split_csv": pdb_u in split_pdbs,
			"cluster_id": cid,
			"split_pdbs_in_cluster": members,
			"n_query_groups": n_qg,
		})
		if not members:
			print(
				f"[INFO] Force PDB {pdb_u}: no split-CSV PDB shares its "
				"cluster; nothing to move"
			)
			continue
		forced_clusters.add(cid)
		shown = ", ".join(members[:15]) + (" ..." if len(members) > 15 else "")
		print(
			f"[INFO] Force PDB {pdb_u} -> cluster {cid} "
			f"({len(members)} PDB(s), {n_qg} query group(s)): {shown}"
		)

	kept_clusters: set[str] = set()
	if keep_existing_test:
		kept_clusters = set(df.loc[original_split == "test", "cluster_id"])
		print(
			f"[INFO] --keep-existing-test: {len(kept_clusters)} cluster(s) "
			"already in test stay in test"
		)

	fixed_clusters = forced_clusters | kept_clusters
	if len(fixed_clusters) > n_test_target:
		print(
			f"[WARN] Forced/kept clusters ({len(fixed_clusters)}) exceed target "
			f"test size ({n_test_target} = {test_size:.0%} of {n_clusters}). "
			"Keeping all of them in test.",
			file=sys.stderr,
		)
		n_test_target = len(fixed_clusters)

	remaining = [c for c in all_clusters if c not in fixed_clusters]
	n_extra = n_test_target - len(fixed_clusters)
	rng = np.random.RandomState(random_state)
	extra: list[str] = []
	if n_extra > 0:
		if n_extra > len(remaining):
			sys.exit(
				f"[ERROR] Need {n_extra} extra test clusters but only "
				f"{len(remaining)} non-forced clusters remain"
			)
		extra = list(rng.choice(remaining, size=n_extra, replace=False))

	test_clusters = fixed_clusters | set(extra)
	df["split"] = np.where(df["cluster_id"].isin(test_clusters), "test", "train")

	out = df[["query_group", "split"]].copy()
	return out, {
		"n_clusters": n_clusters,
		"n_test_clusters": len(test_clusters),
		"n_forced_clusters": len(forced_clusters),
		"n_kept_clusters": len(kept_clusters),
		"n_extra_clusters": len(extra),
		"n_train_qg": int((out["split"] == "train").sum()),
		"n_test_qg": int((out["split"] == "test").sum()),
		"n_qg_train_to_test": int(
			((original_split == "train") & (out["split"] == "test")).sum()
		),
		"n_qg_test_to_train": int(
			((original_split == "test") & (out["split"] == "train")).sum()
		),
		"forced_clusters": sorted(forced_clusters),
		"forced_detail": forced_detail,
	}


def main() -> None:
	args = parse_args()
	if not (0 < args.test_size < 1):
		sys.exit("[ERROR] --test-size must be in (0, 1)")
	if not (0 < args.seq_identity <= 1):
		sys.exit("[ERROR] --seq-identity must be in (0, 1]")
	if not (0 < args.coverage <= 1):
		sys.exit("[ERROR] --coverage must be in (0, 1]")
	if not args.split_csv.is_file():
		sys.exit(f"[ERROR] --split-csv not found: {args.split_csv}")
	if not args.clusters_tsv.is_file():
		sys.exit(f"[ERROR] --clusters-tsv not found: {args.clusters_tsv}")
	deep = not args.no_deep_cluster
	if args.recluster and not deep:
		sys.exit("[ERROR] --recluster works on chain-level clusters; drop --no-deep-cluster")

	force_pdbs = read_force_pdbs(args)
	if not force_pdbs:
		print(
			"[WARN] No --force-pdb given; only retargeting test-size",
			file=sys.stderr,
		)

	raw = load_clusters_tsv(args.clusters_tsv)
	if not raw:
		sys.exit(f"[ERROR] Empty or unreadable clusters TSV: {args.clusters_tsv}")
	clusters_tsv = args.clusters_tsv
	recluster_meta: dict = {}
	if args.recluster:
		clusters_tsv, raw, recluster_meta = recluster_with_external(
			args, force_pdbs, raw
		)

	split_df = pd.read_csv(args.split_csv)
	split_pdbs = sorted(
		{pdb_from_query_group(qg) for qg in split_df["query_group"].astype(str)}
	)
	pdb_ids = list(dict.fromkeys(split_pdbs + force_pdbs))
	print(
		f"[INFO] Building {'deep-cluster PDB' if deep else 'PDB-level'} map "
		f"for {len(pdb_ids)} PDB(s) from {clusters_tsv}"
	)
	cluster_map = build_pdb_cluster_map(pdb_ids, raw, deep)

	out, stats = alter_splits(
		split_df,
		cluster_map,
		force_pdbs=force_pdbs,
		test_size=args.test_size,
		random_state=args.random_state,
		known_pdbs=pdbs_in_clusters(raw, deep),
		keep_existing_test=args.keep_existing_test,
	)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(args.output, index=False)

	meta_path = args.output.with_name(f"{args.output.stem}_meta.json")
	meta = {
		"split_csv": str(args.split_csv.resolve()),
		"input_clusters_tsv": str(args.clusters_tsv.resolve()),
		"clusters_tsv_for_training": str(Path(clusters_tsv).resolve()),
		"deep_cluster": deep,
		"force_pdbs": force_pdbs,
		"test_size": args.test_size,
		"keep_existing_test": args.keep_existing_test,
		"random_state": args.random_state,
		"recluster": recluster_meta,
		"stats": stats,
	}
	with meta_path.open("w", encoding="utf-8") as fh:
		json.dump(meta, fh, indent=2)

	print("\n[INFO] SPLIT SUMMARY")
	print(f"  PDB clusters total : {stats['n_clusters']}")
	print(
		f"  Test clusters      : {stats['n_test_clusters']} "
		f"({stats['n_test_clusters'] / stats['n_clusters']:.1%}) "
		f"[forced={stats['n_forced_clusters']}, "
		f"kept={stats['n_kept_clusters']}, "
		f"sampled={stats['n_extra_clusters']}]"
	)
	print(f"  Train query groups : {stats['n_train_qg']:,}")
	print(f"  Test  query groups : {stats['n_test_qg']:,}")
	print(
		f"  Moved vs input     : {stats['n_qg_train_to_test']:,} train->test, "
		f"{stats['n_qg_test_to_train']:,} test->train"
	)
	if stats["forced_clusters"]:
		print(f"  Forced cluster IDs : {', '.join(stats['forced_clusters'])}")
	print(f"\n[SUCCESS] Wrote {args.output}")
	print(f"  Run record: {meta_path}")
	print(
		"Reuse with Training_Cov_Screen.py / Training_Cov_Classifier.py:\n"
		f"  --deep-cluster --clusters-tsv {clusters_tsv} "
		f"--split-csv {args.output}"
	)


if __name__ == "__main__":
	main()
