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
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

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
		"--test-size",
		type=float,
		default=0.1,
		help="Target fraction of PDB clusters in test (default 0.1)",
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
	return p.parse_args()


def load_clusters_tsv(path: Path) -> dict[str, str]:
	cluster_map: dict[str, str] = {}
	with path.open(encoding="utf-8") as fh:
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


def build_pdb_cluster_map(
	pdb_ids: list[str],
	clusters_tsv: Path,
	deep_cluster: bool,
) -> dict[str, str]:
	raw = load_clusters_tsv(clusters_tsv)
	if not raw:
		sys.exit(f"[ERROR] Empty or unreadable clusters TSV: {clusters_tsv}")

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


def alter_splits(
	split_df: pd.DataFrame,
	cluster_map: dict[str, str],
	force_pdbs: list[str],
	test_size: float,
	random_state: int,
) -> tuple[pd.DataFrame, dict]:
	df = split_df.copy()
	if "query_group" not in df.columns or "split" not in df.columns:
		sys.exit("[ERROR] --split-csv must have columns: query_group, split")

	df["pdb_id"] = df["query_group"].map(pdb_from_query_group)
	df["cluster_id"] = df["pdb_id"].map(cluster_map).fillna(df["pdb_id"])

	all_clusters = sorted(df["cluster_id"].unique())
	n_clusters = len(all_clusters)
	n_test_target = max(1, int(round(n_clusters * test_size)))

	forced_clusters: set[str] = set()
	for pdb in force_pdbs:
		pdb_u = pdb.strip().upper()
		if pdb_u not in set(df["pdb_id"]):
			sys.exit(
				f"[ERROR] Forced PDB '{pdb_u}' not found in any query_group "
				f"of the split CSV"
			)
		cid = cluster_map.get(pdb_u, pdb_u)
		forced_clusters.add(cid)
		members = sorted(df.loc[df["cluster_id"] == cid, "pdb_id"].unique())
		print(
			f"[INFO] Force PDB {pdb_u} -> cluster {cid} "
			f"({len(members)} PDB(s), "
			f"{int((df['cluster_id'] == cid).sum())} query group(s))"
		)

	n_forced = len(forced_clusters)
	if n_forced > n_test_target:
		print(
			f"[WARN] Forced clusters ({n_forced}) exceed target test size "
			f"({n_test_target} = {test_size:.0%} of {n_clusters}). "
			f"Keeping all forced clusters in test.",
			file=sys.stderr,
		)
		n_test_target = n_forced

	remaining = [c for c in all_clusters if c not in forced_clusters]
	n_extra = n_test_target - n_forced
	rng = np.random.RandomState(random_state)
	extra: list[str] = []
	if n_extra > 0:
		if n_extra > len(remaining):
			sys.exit(
				f"[ERROR] Need {n_extra} extra test clusters but only "
				f"{len(remaining)} non-forced clusters remain"
			)
		extra = list(rng.choice(remaining, size=n_extra, replace=False))

	test_clusters = forced_clusters | set(extra)
	df["split"] = np.where(df["cluster_id"].isin(test_clusters), "test", "train")

	out = df[["query_group", "split"]].copy()
	return out, {
		"n_clusters": n_clusters,
		"n_test_clusters": len(test_clusters),
		"n_forced_clusters": n_forced,
		"n_extra_clusters": len(extra),
		"n_train_qg": int((out["split"] == "train").sum()),
		"n_test_qg": int((out["split"] == "test").sum()),
		"forced_clusters": sorted(forced_clusters),
	}


def main() -> None:
	args = parse_args()
	if not (0 < args.test_size < 1):
		sys.exit("[ERROR] --test-size must be in (0, 1)")
	if not args.split_csv.is_file():
		sys.exit(f"[ERROR] --split-csv not found: {args.split_csv}")
	if not args.clusters_tsv.is_file():
		sys.exit(f"[ERROR] --clusters-tsv not found: {args.clusters_tsv}")
	if not args.force_pdb:
		print(
			"[WARN] No --force-pdb given; only retargeting test-size",
			file=sys.stderr,
		)

	split_df = pd.read_csv(args.split_csv)
	pdb_ids = sorted(
		{pdb_from_query_group(qg) for qg in split_df["query_group"].astype(str)}
	)
	deep = not args.no_deep_cluster
	print(
		f"[INFO] Building {'deep-cluster PDB' if deep else 'PDB-level'} map "
		f"for {len(pdb_ids)} PDB(s) from {args.clusters_tsv}"
	)
	cluster_map = build_pdb_cluster_map(pdb_ids, args.clusters_tsv, deep)

	out, stats = alter_splits(
		split_df,
		cluster_map,
		force_pdbs=args.force_pdb,
		test_size=args.test_size,
		random_state=args.random_state,
	)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(args.output, index=False)

	print("\n[INFO] SPLIT SUMMARY")
	print(f"  PDB clusters total : {stats['n_clusters']}")
	print(
		f"  Test clusters      : {stats['n_test_clusters']} "
		f"({stats['n_test_clusters'] / stats['n_clusters']:.1%}) "
		f"[forced={stats['n_forced_clusters']}, "
		f"sampled={stats['n_extra_clusters']}]"
	)
	print(f"  Train query groups : {stats['n_train_qg']:,}")
	print(f"  Test  query groups : {stats['n_test_qg']:,}")
	if stats["forced_clusters"]:
		print(f"  Forced cluster IDs : {', '.join(stats['forced_clusters'])}")
	print(f"\n[SUCCESS] Wrote {args.output}")
	print(
		"Reuse with Training_Cov_Screen.py:\n"
		f"  --deep-cluster --clusters-tsv {args.clusters_tsv} "
		f"--split-csv {args.output}"
	)


if __name__ == "__main__":
	main()
