#!/usr/bin/env python3
"""
Reduce_Disulfides.py

Break disulfide bonds in PDB files so the cysteines are treated as free thiols, and
optionally rerun PROPKA on the edited structures. PROPKA reports a 99.99 placeholder for
disulfide-bonded Cys because it has no free thiol to titrate; this gives those residues a
real pKa.

What it does, per disulfide (SG-SG distance < --ss-cutoff):
  * rotates each Cys side chain about its CA-CB axis (chi1 only), so bond lengths and the
    CA-CB-SG angle are unchanged, until the two SG atoms are >= --min-sep apart;
  * picks the chi1 that is closest to a staggered rotamer (-60, 60, 180 deg) while keeping
    SG >= --min-clearance from other heavy atoms (waters and hydrogens ignored);
  * deletes the matching SSBOND records and the CONECT lines for the two SG atoms;
  * renames CYX to CYS and adds a REMARK 999 line saying what was changed.

Limits: no hydrogens are added, nothing else is relaxed, and loops that move when a real
disulfide is reduced (e.g. in DsbA-like proteins) are NOT modelled. If a crystal structure
of the reduced protein exists, prefer it.

Inputs (choose one)
-------------------
--residues-csv  excluded_propka_placeholder_rows.csv written by Make_Figures.py
                (columns pdb, chain, resseq). Only disulfides involving a listed residue
                are reduced, and only those PDBs are processed.
--stems         PDB stems to process; every disulfide in each is reduced.
--all           every .pdb under --pdb-dir; every disulfide in each is reduced.

Outputs
-------
<out-dir>/<stem>.pdb                    reduced structure (same file stem as the input)
<out-dir>/reduced_disulfides_report.csv one row per reduced bond (distances, chi1, clearance)
<pka-out-dir>/<stem>.pka                with --run-propka

Usage
-----
    python Reduce_Disulfides.py --pdb-dir ../Existing_Structures --out-dir reduced_pdbs \\
        --residues-csv figures/excluded_propka_placeholder_rows.csv \\
        --download-missing --run-propka --pka-out-dir reduced_pka\
        

    python Reduce_Disulfides.py --pdb-dir ../Existing_Structures --out-dir reduced_pdbs --min-clearance 2.7 --residues-csv 10FoldCV/NoAPBS/excluded_propka_placeholder_rows.csv --download-missing --run-propka --pka-out-dir 10FoldCV/NoAPBS/reduced_pka
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
STAGGERED = np.radians([-60.0, 60.0, 180.0])
CYS_NAMES = {"CYS", "CYX"}
WATER_NAMES = {"HOH", "WAT", "H2O", "DOD"}
ResKey = Tuple[str, int, str]  # chain, resseq, icode


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def rotate_about_axis(p: np.ndarray, a: np.ndarray, b: np.ndarray, angle: float) -> np.ndarray:
    """Rotate point p by `angle` (radians) about the axis through a -> b."""
    k = b - a
    k = k / np.linalg.norm(k)
    v = p - a
    return a + v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))


def dihedral(p0, p1, p2, p3) -> float:
    b0 = p0 - p1
    b1 = p2 - p1
    b1 = b1 / np.linalg.norm(b1)
    b2 = p3 - p2
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def angular_gap(a: float, b: float) -> float:
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


# --------------------------------------------------------------------------- #
# PDB parsing
# --------------------------------------------------------------------------- #
def parse_atoms(lines: List[str]) -> List[dict]:
    atoms = []
    model = 0  # NMR-style files: every MODEL block is treated as its own structure
    for i, ln in enumerate(lines):
        if ln.startswith("MODEL"):
            model += 1
            continue
        if not ln.startswith(("ATOM", "HETATM")) or len(ln) < 54:
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            resseq = int(ln[22:26])
        except ValueError:
            continue
        name = ln[12:16].strip()
        elem = ln[76:78].strip().upper() if len(ln) >= 78 else ""
        is_h = elem in ("H", "D") if elem else name.lstrip("0123456789")[:1] == "H"
        atoms.append({
            "i": i, "name": name, "alt": ln[16], "resname": ln[17:20].strip().upper(),
            "key": (ln[21].strip(), resseq, ln[26].strip()), "xyz": xyz, "is_h": is_h, "model": model,
        })
    return atoms


def primary_atoms(atoms: List[dict]) -> Dict[ResKey, Dict[str, dict]]:
    """First atom of each name per residue, preferring the blank/'A' altloc."""
    out: Dict[ResKey, Dict[str, dict]] = {}
    for a in atoms:
        if a["name"] not in ("N", "CA", "CB", "SG"):
            continue
        slot = out.setdefault(a["key"], {})
        cur = slot.get(a["name"])
        if cur is None or (cur["alt"] not in (" ", "A") and a["alt"] in (" ", "A")):
            slot[a["name"]] = a
    return out


def res_label(key: ResKey) -> str:
    chain, num, icode = key
    return f"{chain or '_'}{num}{icode}"


# --------------------------------------------------------------------------- #
# Reduction
# --------------------------------------------------------------------------- #
def choose_rotation(prim: Dict[str, dict], partner_sg: np.ndarray, coords: np.ndarray,
                    other_mask: np.ndarray, min_sep: float, min_clear: float):
    """Pick a chi1 rotation (radians) that separates SG from its partner without clashes."""
    N, CA, CB, SG = (prim[n]["xyz"] for n in ("N", "CA", "CB", "SG"))
    candidates = []
    for deg in range(0, 360, 5):
        ang = np.radians(deg)
        new = rotate_about_axis(SG, CA, CB, ang)
        sep = float(np.linalg.norm(new - partner_sg))
        clear = float(np.linalg.norm(coords[other_mask] - new, axis=1).min()) if other_mask.any() else 99.0
        chi = dihedral(N, CA, CB, new)
        stag = min(angular_gap(chi, s) for s in STAGGERED)
        rot = min(deg, 360 - deg)
        candidates.append((deg, ang, sep, clear, stag, rot, chi))
    feasible = [c for c in candidates if c[2] >= min_sep and c[3] >= min_clear]
    if feasible:
        best = min(feasible, key=lambda c: (round(c[4], 2), c[5]))
        return best, True
    best = max(candidates, key=lambda c: min(c[2] - min_sep, c[3] - min_clear))
    return best, False


def reduce_structure(path_in: str, stem: str, wanted: Optional[Set[Tuple[str, int]]],
                     ss_cutoff: float, min_sep: float, min_clear: float):
    """Return (new_lines, report_rows). `wanted` limits reduction to bonds touching those residues.

    Multi-model files (NMR ensembles) are handled model by model; the report lists model 1.
    Alternate locations of an SG are rotated about the CA-CB axis of the same altloc when present.
    """
    with open(path_in, "r") as handle:
        lines = handle.read().splitlines()
    all_atoms = parse_atoms(lines)
    models = sorted({a["model"] for a in all_atoms})

    new_xyz: Dict[int, np.ndarray] = {}   # line index -> new coordinates
    reduced_keys: Set[ResKey] = set()
    reduced_pairs: Set[frozenset] = set()
    sg_serials: Set[int] = set()
    report = []
    labels_first: List[Tuple[ResKey, ResKey]] = []

    for mi, model in enumerate(models):
        atoms = [a for a in all_atoms if a["model"] == model]
        prim = primary_atoms(atoms)
        by_alt = {(a["key"], a["name"], a["alt"]): a for a in atoms if a["name"] in ("CA", "CB")}

        cys = [k for k in prim if all(n in prim[k] for n in ("N", "CA", "CB", "SG"))
               and prim[k]["SG"]["resname"] in CYS_NAMES]

        # Heavy, non-water atoms of THIS model, with an owner id for clash checks.
        heavy = [a for a in atoms if not a["is_h"] and a["resname"] not in WATER_NAMES]
        coords = np.array([a["xyz"] for a in heavy]) if heavy else np.zeros((0, 3))
        key_ids = {k: n for n, k in enumerate(sorted({a["key"] for a in heavy}))}
        owner = np.array([key_ids[a["key"]] for a in heavy])
        row_of_line = {a["i"]: r for r, a in enumerate(heavy)}

        pairs = []
        for x in range(len(cys)):
            for y in range(x + 1, len(cys)):
                d = float(np.linalg.norm(prim[cys[x]]["SG"]["xyz"] - prim[cys[y]]["SG"]["xyz"]))
                if d < ss_cutoff:
                    pairs.append((cys[x], cys[y], d))

        def listed(k: ResKey) -> bool:
            return wanted is None or (k[0], k[1]) in wanted

        pairs = [p for p in pairs if listed(p[0]) or listed(p[1])]
        if mi == 0:
            labels_first = [(a, b) for a, b, _ in pairs]

        for ka, kb, d0 in pairs:
            chis_before, chis_after, clears, ok_flags = [], [], [], []
            for key, partner in ((ka, kb), (kb, ka)):
                p = prim[key]
                partner_sg = prim[partner]["SG"]["xyz"]
                chis_before.append(np.degrees(dihedral(*(p[n]["xyz"] for n in ("N", "CA", "CB", "SG")))))
                other_mask = owner != key_ids.get(key, -1)
                (deg, ang, sep, clear, stag, rot, chi), ok = choose_rotation(
                    p, partner_sg, coords, other_mask, min_sep, min_clear)
                for a in atoms:  # every altloc copy of this SG rotates by the same angle
                    if a["key"] == key and a["name"] == "SG":
                        ca = by_alt.get((key, "CA", a["alt"]), p["CA"])["xyz"]
                        cb = by_alt.get((key, "CB", a["alt"]), p["CB"])["xyz"]
                        moved = rotate_about_axis(a["xyz"], ca, cb, ang)
                        new_xyz[a["i"]] = moved
                        if a["i"] in row_of_line:
                            coords[row_of_line[a["i"]]] = moved
                p["SG"]["xyz"] = new_xyz.get(p["SG"]["i"], p["SG"]["xyz"])
                chis_after.append(np.degrees(chi))
                clears.append(clear)
                ok_flags.append(ok)
                reduced_keys.add(key)
                sg_serial = lines[p["SG"]["i"]][6:11].strip()
                if sg_serial.isdigit():
                    sg_serials.add(int(sg_serial))
            reduced_pairs.add(frozenset((ka, kb)))
            if mi == 0:
                d1 = float(np.linalg.norm(prim[ka]["SG"]["xyz"] - prim[kb]["SG"]["xyz"]))
                report.append({
                    "pdb": stem, "res_a": res_label(ka), "res_b": res_label(kb), "n_models": len(models),
                    "sg_sg_before": round(d0, 2), "sg_sg_after": round(d1, 2),
                    "chi1_a_before": round(chis_before[0], 1), "chi1_a_after": round(chis_after[0], 1),
                    "chi1_b_before": round(chis_before[1], 1), "chi1_b_after": round(chis_after[1], 1),
                    "min_clearance_a": round(clears[0], 2), "min_clearance_b": round(clears[1], 2),
                    "constraints_met": bool(all(ok_flags) and d1 >= min_sep),
                })

    if not report:
        return None, report

    # Rewrite coordinates, drop disulfide bookkeeping records, rename CYX -> CYS.
    out = []
    for i, ln in enumerate(lines):
        if i in new_xyz:
            x, y, z = new_xyz[i]
            ln = ln[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + ln[54:]
        if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 27:
            try:
                key = (ln[21].strip(), int(ln[22:26]), ln[26].strip())
            except ValueError:
                key = None
            if key in reduced_keys and ln[17:20] == "CYX":
                ln = ln[:17] + "CYS" + ln[20:]
        if ln.startswith("SSBOND") and len(ln) >= 35:
            try:
                k1 = (ln[15].strip(), int(ln[17:21]), ln[21].strip())
                k2 = (ln[29].strip(), int(ln[31:35]), ln[35].strip() if len(ln) > 35 else "")
            except ValueError:
                k1 = k2 = None
            if k1 is not None and frozenset((k1, k2)) in reduced_pairs:
                continue
        if ln.startswith("CONECT"):
            first = ln[6:11].strip()
            if first.isdigit() and int(first) in sg_serials:
                continue
        out.append(ln)

    remarks = ["REMARK 999 DISULFIDE(S) REDUCED BY Reduce_Disulfides.py (CHI1 ROTATION ONLY):"]
    cur = "REMARK 999"
    for a, b in labels_first:
        item = f" {res_label(a)}-{res_label(b)}"
        if len(cur) + len(item) > 78:
            remarks.append(cur)
            cur = "REMARK 999"
        cur += item
    remarks.append(cur)
    first_atom = next((n for n, ln in enumerate(out) if ln.startswith(("ATOM", "HETATM", "MODEL"))), len(out))
    out[first_atom:first_atom] = remarks
    return out, report


# --------------------------------------------------------------------------- #
# File handling
# --------------------------------------------------------------------------- #
def index_pdbs(pdb_dir: str) -> Dict[str, str]:
    by_base: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(os.path.abspath(pdb_dir), "**", "*.pdb"), recursive=True)):
        by_base[os.path.splitext(os.path.basename(path))[0].lower()] = path
    return by_base


def find_pdb(index: Dict[str, str], stem: str) -> Optional[str]:
    s = stem.lower()
    for cand in (s, f"{s}_model1", f"{s}_clean"):
        if cand in index:
            return index[cand]
    return None


def download_pdb(stem: str, dest_dir: str, timeout: int = 60) -> Optional[str]:
    pdb_id = stem.upper()
    if not re.fullmatch(r"[0-9][A-Z0-9]{3}", pdb_id):
        print(f"[reduce] WARNING: '{stem}' is not a 4-character PDB ID; cannot download")
        return None
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"{stem}.pdb")
    try:
        with urllib.request.urlopen(RCSB_PDB_URL.format(pdb_id=pdb_id), timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[reduce] WARNING: could not download {pdb_id}: {exc}")
        return None
    if not data.strip():
        return None
    with open(out_path + ".part", "wb") as handle:
        handle.write(data)
    os.replace(out_path + ".part", out_path)
    return out_path


def run_propka(pdb_path: str, stem: str, out_dir: str) -> str:
    try:
        from propka.run import single as propka_single
    except ImportError as exc:
        raise RuntimeError("PROPKA Python package is not installed (pip install propka)") from exc
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.pka")
    molecule = propka_single(pdb_path, write_pka=False)
    molecule.write_pka(filename=out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Reduce disulfide bonds in PDB files (restore free thiols).")
    ap.add_argument("--pdb-dir", required=True, help="Directory searched recursively for input .pdb files")
    ap.add_argument("--out-dir", required=True, help="Where reduced PDBs and the report are written")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--residues-csv", help="excluded_propka_placeholder_rows.csv (pdb, chain, resseq)")
    sel.add_argument("--stems", nargs="+", help="PDB stems to process (all disulfides reduced)")
    sel.add_argument("--all", action="store_true", help="Process every .pdb under --pdb-dir")
    ap.add_argument("--ss-cutoff", type=float, default=2.5, help="SG-SG distance (A) counted as a disulfide")
    ap.add_argument("--min-sep", type=float, default=3.4, help="Minimum SG-SG distance (A) after reduction")
    ap.add_argument("--min-clearance", type=float, default=3.0,
                    help="Minimum distance (A) from the new SG to other heavy atoms")
    ap.add_argument("--download-missing", action="store_true", help="Fetch absent PDBs from RCSB into --pdb-dir")
    ap.add_argument("--run-propka", action="store_true", help="Run PROPKA on each reduced PDB")
    ap.add_argument("--pka-out-dir", default="reduced_pka", help="Where --run-propka writes <stem>.pka")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index = index_pdbs(args.pdb_dir)

    targets: Dict[str, Optional[Set[Tuple[str, int]]]] = {}
    if args.residues_csv:
        df = pd.read_csv(args.residues_csv)
        for col in ("pdb", "chain", "resseq"):
            if col not in df.columns:
                raise SystemExit(f"{args.residues_csv} needs columns pdb, chain, resseq (missing {col})")
        for _, r in df.iterrows():
            stem = os.path.splitext(os.path.basename(str(r["pdb"])))[0]
            chain = "" if pd.isna(r["chain"]) or str(r["chain"]).strip() in ("_", "nan") else str(r["chain"]).strip()
            targets.setdefault(stem, set()).add((chain, int(r["resseq"])))
    elif args.stems:
        targets = {s: None for s in args.stems}
    else:
        targets = {os.path.basename(p)[:-4]: None for p in index.values()}

    all_rows = []
    for stem, wanted in sorted(targets.items()):
        path = find_pdb(index, stem)
        if path is None and args.download_missing:
            path = download_pdb(stem, args.pdb_dir)
        if path is None:
            print(f"[reduce] WARNING: no PDB found for {stem}; skipped (use --download-missing)")
            continue
        new_lines, rows = reduce_structure(path, stem, wanted, args.ss_cutoff, args.min_sep, args.min_clearance)
        if new_lines is None:
            print(f"[reduce] {stem}: no disulfide found (SG-SG < {args.ss_cutoff} A) for the requested residues")
            continue
        out_path = os.path.join(args.out_dir, f"{stem}.pdb")
        with open(out_path, "w") as handle:
            handle.write("\n".join(new_lines) + "\n")
        for r in rows:
            flag = "" if r["constraints_met"] else "  <-- constraints NOT fully met, check"
            print(f"[reduce] {stem} {r['res_a']}-{r['res_b']}: SG-SG {r['sg_sg_before']} -> {r['sg_sg_after']} A{flag}")
        all_rows.extend(rows)
        if args.run_propka:
            try:
                print(f"[reduce] PROPKA -> {run_propka(out_path, stem, args.pka_out_dir)}")
            except Exception as exc:  # keep going for the remaining structures
                print(f"[reduce] WARNING: PROPKA failed for {stem}: {exc}")

    if all_rows:
        report_path = os.path.join(args.out_dir, "reduced_disulfides_report.csv")
        pd.DataFrame(all_rows).to_csv(report_path, index=False)
        print(f"[reduce] reduced {len(all_rows)} disulfide bond(s); report: {report_path}")
    else:
        print("[reduce] nothing was reduced")


if __name__ == "__main__":
    main()