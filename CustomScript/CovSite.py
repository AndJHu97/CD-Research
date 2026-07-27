#!/usr/bin/env python3
"""Build Cov_Screen candidates from PDB + ligand-SMILES input.

The reduced Frankenstein stage calculates only the deprotonated descriptors
needed by the saved screening model. It deliberately skips protonated
calculations, Frankenstein ranking, orbital filters, HSAB filters, and PROPKA.

Input CSV columns are detected case-insensitively:
  required: pdb/pdb_id/protein pdb and either smiles or LigID
  optional: name, residue, resnum, chain, warhead / Frankenstein_Warhead

When Warhead (or Frankenstein_Warhead) is provided, that exact text is written
to labels.csv as Frankenstein_Warhead for Cov_Screen matching / --perfect-match.
Warheads are still auto-detected from SMILES for candidate feature rows.

When a row has no SMILES, LigID is resolved from a local SDF/PDB in --pdb-dir
or downloaded from the RCSB ligand service.

Rows with a complete target site are written to labels.csv and evaluated.
Rows without a target are still scored and ranked within Name x Warhead.
Multiple labeled target sites for the same PDB + SMILES are allowed; Names are
disambiguated with a site suffix (e.g. -ACYS25) so each Name x Warhead group has
one target.
If a provided target is missing or not a supported nucleophile (CYS/SER/THR/TYR/LYS/HIS),
it is skipped with a warning and that row is treated as unlabeled (score-only).

Invalid SMILES, ligands with no detected warhead, and other hard per-row failures are
skipped; details are written to covsite_skipped_inputs.csv.

Feature progress is checkpointed under <output-dir>/feature_cache/shards/ as
unique per-Name/warhead CSVs. After enrichment finishes, uniquely named
covsite_candidates_<runid>.csv / covsite_labels_<runid>.csv are written for
Cov_Screen, previous final feature caches are deleted, and the shards are
cleared.

Screen-only mode (--screen-only) skips all feature calculation and enrichment.
Provide an existing candidates CSV with --candidates and optionally --labels to
run Cov_Screen directly.

Virtual-screening mode (--VS) requires Residue/ResNum/Chain on every input row
and calculates features only for that target site × ligand warhead pair
(no whole-protein nucleophile enumeration). It writes the usual candidates /
labels CSVs and does not run Cov_Screen ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem


HERE = Path(__file__).resolve().parent
DEPENDENCIES_DIR = HERE / "dependencies"
VALID_RESIDUES = ("CYS", "SER", "THR", "TYR", "LYS", "HIS")
DESCRIPTOR_COLUMNS = (
    "Fukui_Deprotonated",
    "Nucleophile_HOMO_Deprotonated",
    "Electrophile_LUMO_Deprotonated",
    "HOMO_LUMO_Gap_Deprotonated",
    "Partial_Charge_Deprotonated",
    "Nucleophilicity_Index_Deprotonated",
)
FINAL_CANDIDATES_GLOB = "covsite_candidates_*.csv"
FINAL_LABELS_GLOB = "covsite_labels_*.csv"


def log1p_signed(values: Any) -> np.ndarray:
    """Compatibility function referenced by the deprotonation model pickle."""
    array = np.asarray(values)
    return np.where(array < 0, array, np.log1p(array))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deprotonated-only covalent-site features and run "
            "Cov_Screen in score-only and, when targets exist, evaluation mode. "
            "Use --screen-only to skip feature calculation and run Cov_Screen "
            "on existing candidates (+ optional labels). Use --VS to featurize "
            "only each row's target site and skip Cov_Screen ranking."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help=(
            "CSV containing PDB IDs and ligand SMILES "
            "(required unless --screen-only)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Cov_Screen model bundle (.pkl); required unless --VS "
            "(VS writes features only and skips Cov_Screen ranking)"
        ),
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Local PDB directory/cache (required unless --screen-only)",
    )
    parser.add_argument(
        "--output-dir", default="covsite_output", help="Output directory"
    )
    parser.add_argument(
        "--scripts-dir",
        default=None,
        help=(
            "Dependency directory override "
            "(default: dependencies beside CovSite.py)"
        ),
    )
    parser.add_argument(
        "--deprotonation-model",
        default=None,
        help="No-APBS deprotonation model (default: auto-detect deprot_xgb_noapbs.pkl)",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help=(
            "Skip feature calculation/enrichment and run Cov_Screen on "
            "--candidates (and optional --labels)"
        ),
    )
    parser.add_argument(
        "--VS",
        dest="vs",
        action="store_true",
        help=(
            "Virtual-screening mode: require Residue/ResNum/Chain on every "
            "input row and calculate features only for that target site "
            "(skip other nucleophilic residues). Writes candidates/labels "
            "CSVs and does not run Cov_Screen ranking."
        ),
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help=(
            "Existing candidates feature CSV "
            "(required with --screen-only; ignored otherwise)"
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Existing labels CSV for Cov_Screen evaluation "
            "(optional with --screen-only; omitted -> score-only ranking)"
        ),
    )
    parser.add_argument("--pdb-column", default=None)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--ligid-column", default=None)
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--residue-column", default=None)
    parser.add_argument("--resnum-column", default=None)
    parser.add_argument("--chain-column", default=None)
    parser.add_argument(
        "--warhead-column",
        default=None,
        help=(
            "Optional input column copied as-is to labels Frankenstein_Warhead "
            "(comma-separated allowed). If omitted, auto-detects "
            "Frankenstein_Warhead / Warhead columns."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for future parallel xTB execution; currently must be 1",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Recalculate cached xTB descriptors"
    )
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument(
        "--reward-mode", choices=("hit_at_k", "hit_at_top_pct"), default=None
    )
    parser.add_argument("--top-pct", type=float, default=None)
    parser.add_argument("--rank-bonus-epsilon", type=float, default=None)
    parser.add_argument("--normalize-within-protein", action="store_true")
    parser.add_argument("--perfect-match", action="store_true")
    parser.add_argument("--strict-warhead-coverage", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--shap-max-rows", type=int, default=None)
    args = parser.parse_args()
    if args.vs and args.screen_only:
        parser.error("--VS and --screen-only cannot be used together")
    if args.screen_only:
        if not args.candidates:
            parser.error("--screen-only requires --candidates")
        if not args.model:
            parser.error("--screen-only requires --model")
        if args.input_csv:
            parser.error(
                "Do not pass input_csv with --screen-only; "
                "use --candidates and optional --labels instead"
            )
    elif args.vs:
        if not args.input_csv:
            parser.error("input_csv is required with --VS")
        if not args.pdb_dir:
            parser.error("--pdb-dir is required with --VS")
        if args.candidates or args.labels:
            parser.error(
                "--candidates/--labels are only valid with --screen-only"
            )
    else:
        if not args.input_csv:
            parser.error("input_csv is required unless --screen-only is set")
        if not args.pdb_dir:
            parser.error("--pdb-dir is required unless --screen-only is set")
        if not args.model:
            parser.error("--model is required unless --VS is set")
        if args.candidates or args.labels:
            parser.error(
                "--candidates/--labels are only valid with --screen-only"
            )
    return args


def script_search_roots(scripts_dir: str | None) -> list[Path]:
    root = Path(scripts_dir).resolve() if scripts_dir else DEPENDENCIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"CovSite dependency directory not found: {root}"
        )
    return [root]


def locate_file(filename: str, roots: list[Path], required: bool = True) -> Path | None:
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate.resolve()
    if required:
        searched = "\n  ".join(str(root / filename) for root in roots)
        raise FileNotFoundError(f"Could not locate {filename}. Searched:\n  {searched}")
    return None


def load_module(name: str, path: Path) -> ModuleType:
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: tuple[str, ...],
    required: bool,
) -> str | None:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    if explicit:
        matched = lookup.get(str(explicit).strip().lower())
        if matched is None:
            raise ValueError(f"Column {explicit!r} is not in the input CSV")
        return str(matched)
    for candidate in candidates:
        if candidate in lookup:
            return str(lookup[candidate])
    if required:
        raise ValueError(
            f"Could not detect a column matching {candidates}; "
            f"available columns: {list(df.columns)}"
        )
    return None


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def pdb_id_from_value(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return Path(text).stem.upper()


def sanitize_name_part(value: Any, max_length: int = 72) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", clean_text(value)).strip("_")
    return (text or "unnamed")[:max_length]


def normalized_resnum(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def target_is_complete(row: pd.Series) -> bool:
    return bool(
        clean_text(row["_target_residue"])
        and normalized_resnum(row["_target_resnum"])
        and clean_text(row["_target_chain"])
    )


def target_field_count(row: pd.Series) -> int:
    return sum(
        (
            bool(clean_text(row["_target_residue"])),
            bool(normalized_resnum(row["_target_resnum"])),
            bool(clean_text(row["_target_chain"])),
        )
    )


def resolve_pdb(
    pdb_id: str, pdb_dir: Path, residue_module: ModuleType
) -> Path:
    path = residue_module.find_or_download_pdb_file(
        pdb_id,
        str(pdb_dir),
        recursive=True,
        download_missing=True,
    )
    if path is None:
        raise FileNotFoundError(
            f"PDB {pdb_id} was not found in {pdb_dir} and could not be downloaded"
        )
    return Path(path).resolve()


def resolve_smiles_from_ligid(
    pdb_path: Path,
    ligand_id: str,
    frankenstein: ModuleType,
) -> str:
    smiles = frankenstein.extract_ligand_from_pdb(str(pdb_path), ligand_id)
    if not smiles:
        raise ValueError(
            f"Could not resolve LigID {ligand_id!r} from {pdb_path.parent} "
            "or the RCSB ligand service"
        )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(
            f"LigID {ligand_id!r} resolved to invalid SMILES: {smiles!r}"
        )
    return Chem.MolToSmiles(molecule, canonical=True)


def sasa_rows(pdb_path: Path, frankenstein: ModuleType) -> pd.DataFrame:
    rsa_path = frankenstein.hn_adv.run_freesasa(str(pdb_path))
    exposure = frankenstein.hn_adv.parse_rsa_file(rsa_path)
    rows: list[dict[str, Any]] = []
    for (residue, chain, resnum), (abs_sasa, rel_sasa) in exposure.items():
        residue = str(residue).strip().upper()
        if residue not in VALID_RESIDUES:
            continue
        rows.append(
            {
                "Residue": residue,
                "Chain": "" if str(chain).strip() == "_" else str(chain).strip(),
                "ResNum": normalized_resnum(resnum),
                "Abs_Side_SASA": abs_sasa,
                "Rel_Side_SASA": rel_sasa,
            }
        )
    if not rows:
        raise RuntimeError(f"No supported nucleophilic residues found in {pdb_path}")
    return pd.DataFrame(rows).drop_duplicates(
        subset=["Residue", "Chain", "ResNum"], keep="first"
    )


def filter_sites_to_target(
    sites: pd.DataFrame,
    residue: str,
    resnum: str,
    chain: str,
) -> pd.DataFrame:
    """Keep only the requested Residue/ResNum/Chain site."""
    residue = clean_text(residue).upper()
    resnum = normalized_resnum(resnum)
    chain = clean_text(chain).upper()
    mask = (
        (sites["Residue"].astype(str).str.upper() == residue)
        & (sites["ResNum"].map(normalized_resnum) == resnum)
        & (sites["Chain"].astype(str).str.strip().str.upper() == chain)
    )
    return sites.loc[mask].copy()


def predict_deprotonation(
    sites: pd.DataFrame,
    pdb_path: Path,
    model_path: Path,
    pipeline: Any,
    deprot_module: ModuleType,
) -> tuple[pd.DataFrame, list[str]]:
    cache = deprot_module.load_prediction_cache()
    errors: list[str] = []
    probabilities: list[float] = []
    for _, row in sites.iterrows():
        residue = str(row["Residue"]).upper()
        chain = str(row["Chain"]).strip().upper()
        resnum = int(normalized_resnum(row["ResNum"]))
        cache_key = deprot_module.make_prediction_cache_key(
            str(pdb_path), chain, residue, resnum, str(model_path),
            12.0, 6.0, deprot_module.PHYSIOLOGIC_PH, False,
        )
        cached = cache.get(cache_key, {})
        if "deprotonation_prob" in cached:
            probabilities.append(float(cached["deprotonation_prob"]))
            continue
        try:
            features = deprot_module.build_features(
                str(pdb_path),
                chain,
                residue,
                resnum,
                12.0,
                6.0,
                deprot_module.PHYSIOLOGIC_PH,
                use_apbs=False,
            )
            probability = float(np.clip(pipeline.predict(features)[0], 0.0, 1.0))
            probabilities.append(probability)
            cache[cache_key] = {"deprotonation_prob": probability}
        except Exception as exc:
            probabilities.append(float("nan"))
            errors.append(f"{pdb_path.name} {chain}:{residue}:{resnum}: {exc}")
    deprot_module.save_prediction_cache(cache)
    out = sites.copy()
    out["deprotonation_prob"] = probabilities
    return out, errors


def unique_warheads(
    smiles: str, frankenstein: ModuleType
) -> list[dict[str, Any]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid ligand SMILES: {smiles}")
    detected = frankenstein.detect_electrophile_warheads(molecule)
    if not detected:
        return []
    name_counts = Counter(item[0] for item in detected)
    ordinals: defaultdict[str, int] = defaultdict(int)
    warheads: list[dict[str, Any]] = []
    for name, smarts, reactive_idx, hsab in detected:
        ordinals[name] += 1
        label = name
        if name_counts[name] > 1:
            label = f"{name} [atom {reactive_idx}; instance {ordinals[name]}]"
        warheads.append(
            {
                "Warhead": label,
                "Warhead_Base": name,
                "Warhead_SMARTS": smarts,
                "Reactive_Atom_Index": int(reactive_idx),
                "Warhead_Instance": ordinals[name],
                "HSAB_Class": hsab,
            }
        )
    return warheads


def skipped_row_record(
    input_row: pd.Series,
    reason: str,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "Source_Row": int(input_row["_source_row"]),
        "PDB_ID": clean_text(input_row.get("_pdb_id", "")),
        "Name": clean_text(input_row.get("_input_name", "")),
        "LigID": clean_text(input_row.get("_ligid", "")),
        "electrophile_smiles": clean_text(input_row.get("_smiles", "")),
        "Reason": reason,
        "Detail": detail,
    }


def write_skipped_inputs_csv(
    output_dir: Path, skipped_rows: list[dict[str, Any]]
) -> Path | None:
    path = output_dir / "covsite_skipped_inputs.csv"
    if not skipped_rows:
        if path.is_file():
            path.unlink()
        return None
    pd.DataFrame(skipped_rows).to_csv(path, index=False)
    return path


def deprotonated_descriptors(
    smiles: str,
    warhead: dict[str, Any],
    frankenstein: ModuleType,
    reactivity_cache: dict[str, Any],
    nucleophile_cache: dict[str, Any],
    use_cache: bool,
    residues: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    residue_types = residues if residues is not None else VALID_RESIDUES
    for residue in residue_types:
        residue = str(residue).strip().upper()
        if residue not in VALID_RESIDUES:
            continue
        surrogate_key = frankenstein.get_surrogate_key(residue)
        try:
            score = frankenstein.compute_reactivity_score(
                smiles,
                warhead["Reactive_Atom_Index"],
                surrogate_key,
                protonation_state="deprotonated",
                cache=reactivity_cache,
                nucleophile_cache=nucleophile_cache,
                use_cache=use_cache,
            )
        except Exception as exc:
            raise RuntimeError(
                f"xTB descriptor calculation failed for {warhead['Warhead']} / "
                f"{residue}: {exc}"
            ) from exc
        result[residue] = {
            "Fukui_Deprotonated": score.get("fukui", np.nan),
            "Nucleophile_HOMO_Deprotonated": score.get("nucleophile_homo", np.nan),
            "Electrophile_LUMO_Deprotonated": score.get("electrophile_lumo", np.nan),
            "HOMO_LUMO_Gap_Deprotonated": score.get("homo_lumo_gap", np.nan),
            "Partial_Charge_Deprotonated": score.get("partial_charge", np.nan),
            "Nucleophilicity_Index_Deprotonated": score.get(
                "nucleophilicity_index", np.nan
            ),
        }
    return result


def add_descriptor_columns(
    sites: pd.DataFrame, descriptors: dict[str, dict[str, float]]
) -> pd.DataFrame:
    out = sites.copy()
    for column in DESCRIPTOR_COLUMNS:
        out[column] = out["Residue"].map(
            lambda residue: descriptors.get(str(residue).upper(), {}).get(
                column, np.nan
            )
        )
    return out


def site_name_tag(row: pd.Series) -> str:
    """Stable short tag for a labeled target site (used to disambiguate Names)."""
    chain = sanitize_name_part(clean_text(row["_target_chain"]).upper() or "X", 8)
    residue = sanitize_name_part(
        clean_text(row["_target_residue"]).upper() or "UNK", 8
    )
    resnum = sanitize_name_part(normalized_resnum(row["_target_resnum"]) or "0", 12)
    return f"{chain}{residue}{resnum}"


def make_base_name(
    row: pd.Series,
    warhead_label: str,
    duplicate_target_counts: Counter[tuple[str, str]],
    multi_target_supplied_names: set[str],
) -> str:
    """Build query-group Name.

    Multiple labeled target sites for the same PDB + SMILES (or same supplied
    Name) are allowed: Names are disambiguated with a site suffix so each
    Name × Warhead group has one target.
    """
    supplied = clean_text(row["_input_name"])
    if supplied:
        name = supplied
    else:
        pdb_id = row["_pdb_id"]
        smiles = row["_smiles"]
        parts = [
            sanitize_name_part(pdb_id),
            sanitize_name_part(smiles),
            sanitize_name_part(warhead_label),
        ]
        name = "-".join(parts)

    if target_is_complete(row):
        key = (row["_pdb_id"], row["_smiles"])
        needs_suffix = duplicate_target_counts[key] > 1
        if supplied and supplied.casefold() in multi_target_supplied_names:
            needs_suffix = True
        if needs_suffix:
            name = f"{name}-{site_name_tag(row)}"
    return name


def disambiguate_name_warhead_site_conflicts(
    candidates: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Ensure each Name × Warhead maps to at most one labeled site.

    If conflicts remain after initial naming, append a site tag to labels and
    expand matching candidate Name×Warhead blocks once per disambiguated Name.
    """
    if labels.empty or "Name" not in labels.columns:
        return candidates, labels, []

    out_labels = labels.copy()
    messages: list[str] = []

    normalized = out_labels.assign(
        _name=out_labels["Name"].astype(str).str.strip().str.upper(),
        _warhead=out_labels["Frankenstein_Warhead"]
        .astype(str)
        .str.strip()
        .str.lower(),
        _site=(
            out_labels["Residue"].astype(str).str.strip().str.upper()
            + ":"
            + out_labels["ResNum"].map(normalized_resnum)
            + ":"
            + out_labels["Chain"].astype(str).str.strip().str.upper()
        ),
    )
    conflicts = normalized.groupby(["_name", "_warhead"])["_site"].nunique()
    conflicts = conflicts[conflicts > 1]
    if conflicts.empty:
        return candidates, out_labels, messages

    rename_map: dict[tuple[str, str, str], str] = {}
    for name_u, warhead_u in conflicts.index:
        subset = normalized[
            (normalized["_name"] == name_u) & (normalized["_warhead"] == warhead_u)
        ]
        for _, row in subset.iterrows():
            old_name = str(row["Name"])
            warhead = str(row["Frankenstein_Warhead"])
            site = str(row["_site"])
            chain, residue, resnum = site.split(":")
            tag = (
                f"{sanitize_name_part(chain or 'X', 8)}"
                f"{sanitize_name_part(residue or 'UNK', 8)}"
                f"{sanitize_name_part(resnum or '0', 12)}"
            )
            new_name = (
                old_name
                if old_name.upper().endswith(tag.upper())
                else f"{old_name}-{tag}"
            )
            rename_map[(old_name, warhead, site)] = new_name
            messages.append(
                f"Disambiguated Name/Warhead for multiple target sites: "
                f"{old_name!r} / {warhead!r} @ {site} -> {new_name!r}"
            )

    new_label_names: list[str] = []
    for _, row in out_labels.iterrows():
        site = (
            f"{str(row['Residue']).strip().upper()}:"
            f"{normalized_resnum(row['ResNum'])}:"
            f"{str(row['Chain']).strip().upper()}"
        )
        key = (str(row["Name"]), str(row["Frankenstein_Warhead"]), site)
        new_label_names.append(rename_map.get(key, str(row["Name"])))
    out_labels["Name"] = new_label_names

    if candidates.empty or "Warhead" not in candidates.columns:
        return candidates, out_labels, messages

    conflicted_old = {(n, w) for (n, w, _s) in rename_map}
    new_names_by_old: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (old_name, warhead, _site), new_name in rename_map.items():
        key = (old_name, warhead)
        if new_name not in new_names_by_old[key]:
            new_names_by_old[key].append(new_name)

    untouched_mask = [
        (str(name), str(warhead)) not in conflicted_old
        for name, warhead in zip(candidates["Name"], candidates["Warhead"])
    ]
    parts: list[pd.DataFrame] = []
    if any(untouched_mask):
        parts.append(candidates.loc[untouched_mask].copy())

    for (old_name, warhead), new_names in new_names_by_old.items():
        block = candidates[
            (candidates["Name"].astype(str) == old_name)
            & (candidates["Warhead"].astype(str) == warhead)
        ]
        if block.empty:
            continue
        for new_name in new_names:
            part = block.copy()
            part["Name"] = new_name
            parts.append(part)

    out_candidates = pd.concat(parts, ignore_index=True) if parts else candidates
    return out_candidates, out_labels, messages


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def shard_cache_key(source_row: int, name: str, warhead: str) -> tuple[int, str, str]:
    return (
        int(source_row),
        str(name).strip().upper(),
        str(warhead).strip().lower(),
    )


def shard_filename(source_row: int, name: str, warhead: str) -> str:
    digest = hashlib.sha1(
        f"{source_row}|{name}|{warhead}".encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"row{int(source_row):05d}_"
        f"{sanitize_name_part(name, 48)}_"
        f"{digest}.csv"
    )


def delete_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            print(f"[WARN] Could not delete cache file {path}: {exc}")


def write_feature_shard(
    shard_dir: Path,
    frame: pd.DataFrame,
    source_row: int,
    name: str,
    warhead: str,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / shard_filename(source_row, name, warhead)
    frame.to_csv(path, index=False)
    return path


def load_feature_shards(
    shard_dir: Path,
) -> dict[tuple[int, str, str], pd.DataFrame]:
    loaded: dict[tuple[int, str, str], pd.DataFrame] = {}
    if not shard_dir.is_dir():
        return loaded
    for path in sorted(shard_dir.glob("row*.csv")):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable shard {path.name}: {exc}")
            continue
        if frame.empty or "Source_Row" not in frame.columns or "Name" not in frame.columns:
            continue
        warhead = frame["Warhead"].iloc[0] if "Warhead" in frame.columns else ""
        key = shard_cache_key(
            int(frame["Source_Row"].iloc[0]),
            str(frame["Name"].iloc[0]),
            str(warhead),
        )
        loaded[key] = frame
    return loaded


def ligand_row_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "Name": frame["Name"].iloc[0],
        "electrophile_smiles": frame["electrophile_smiles"].iloc[0],
        "PDB_ID": frame["PDB_ID"].iloc[0],
        "LigID": frame["LigID"].iloc[0] if "LigID" in frame.columns else "",
        "Warhead": frame["Warhead"].iloc[0] if "Warhead" in frame.columns else "",
    }


def write_final_feature_cache(
    output_dir: Path,
    run_id: str,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    shard_dir: Path,
) -> tuple[Path, Path]:
    """Write uniquely named finals, retire previous finals, and clear shards."""
    previous = sorted(output_dir.glob(FINAL_CANDIDATES_GLOB)) + sorted(
        output_dir.glob(FINAL_LABELS_GLOB)
    )
    candidates_path = output_dir / f"covsite_candidates_{run_id}.csv"
    labels_path = output_dir / f"covsite_labels_{run_id}.csv"
    candidates.to_csv(candidates_path, index=False)
    labels.to_csv(labels_path, index=False)

    # Stable aliases for the latest completed feature build.
    alias_candidates = output_dir / "covsite_candidates.csv"
    alias_labels = output_dir / "covsite_labels.csv"
    candidates.to_csv(alias_candidates, index=False)
    labels.to_csv(alias_labels, index=False)

    delete_paths([path for path in previous if path.resolve() not in {
        candidates_path.resolve(), labels_path.resolve()
    }])
    if shard_dir.is_dir():
        delete_paths(sorted(shard_dir.glob("row*.csv")))
    return candidates_path, labels_path


def cov_screen_common_args(args: argparse.Namespace) -> list[str]:
    options: list[str] = []
    if args.topk is not None:
        options += ["--topk", str(args.topk)]
    if args.reward_mode:
        options += ["--reward-mode", args.reward_mode]
    if args.top_pct is not None:
        options += ["--top-pct", str(args.top_pct)]
    if args.rank_bonus_epsilon is not None:
        options += ["--rank-bonus-epsilon", str(args.rank_bonus_epsilon)]
    if args.normalize_within_protein:
        options.append("--normalize-within-protein")
    if args.perfect_match:
        options.append("--perfect-match")
    if args.strict_warhead_coverage:
        options.append("--strict-warhead-coverage")
    if args.no_shap:
        options.append("--no-shap")
    if args.shap_max_rows is not None:
        options += ["--shap-max-rows", str(args.shap_max_rows)]
    return options


def run_cov_screen(command: list[str], description: str) -> None:
    print(f"\n[CovSite] Running Cov_Screen {description}...")
    subprocess.run(command, check=True)


def launch_cov_screen(
    args: argparse.Namespace,
    cov_screen_path: Path,
    model_path: Path,
    candidates_path: Path,
    labels_path: Path | None,
    output_dir: Path,
) -> None:
    """Run score-only Cov_Screen, plus labeled evaluation when labels exist."""
    common = cov_screen_common_args(args)
    score_command = [
        sys.executable,
        str(cov_screen_path),
        "--model",
        str(model_path),
        "--training",
        str(candidates_path),
        "--export-scores",
        str(output_dir / "covsite_scores.csv"),
        "--export-query-groups",
        str(output_dir / "covsite_query_groups.csv"),
        "--export-summary",
        str(output_dir / "covsite_score_summary.json"),
        *common,
    ]
    run_cov_screen(score_command, "score-only ranking")

    if labels_path is not None:
        eval_command = [
            sys.executable,
            str(cov_screen_path),
            "--model",
            str(model_path),
            "--training",
            str(candidates_path),
            "--labels",
            str(labels_path),
            "--export-results",
            str(output_dir / "covsite_label_results.csv"),
            "--export-scores",
            str(output_dir / "covsite_labeled_scores.csv"),
            "--export-query-groups",
            str(output_dir / "covsite_labeled_query_groups.csv"),
            "--export-summary",
            str(output_dir / "covsite_evaluation_summary.json"),
            *common,
        ]
        run_cov_screen(eval_command, "labeled evaluation")


def run_screen_only(args: argparse.Namespace) -> None:
    """Skip feature generation and run Cov_Screen on provided CSVs."""
    model_path = Path(args.model).resolve()
    candidates_path = Path(args.candidates).resolve()
    labels_path = Path(args.labels).resolve() if args.labels else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_path}")
    if labels_path is not None and not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    roots = script_search_roots(args.scripts_dir)
    cov_screen_path = locate_file("Cov_Screen.py", roots)

    candidates = pd.read_csv(candidates_path, low_memory=False)
    if candidates.empty:
        raise SystemExit("[ERROR] Candidates CSV has no rows")

    n_labels = 0
    if labels_path is not None:
        labels = pd.read_csv(labels_path, low_memory=False)
        n_labels = len(labels)
        if labels.empty:
            print("[CovSite] Labels CSV is empty; running score-only only")
            labels_path = None

    print("[CovSite] Screen-only mode")
    print(f"  Candidates: {candidates_path} ({len(candidates)} row(s))")
    if labels_path is not None:
        print(f"  Labels:     {labels_path} ({n_labels} labeled site(s))")
    else:
        print("  Labels:     (none — score-only ranking)")
    print(f"  Output dir: {output_dir}")

    launch_cov_screen(
        args,
        cov_screen_path,
        model_path,
        candidates_path,
        labels_path,
        output_dir,
    )

    print("\n[CovSite] Complete (screen-only)")
    print(f"  Candidates:    {candidates_path}")
    if labels_path is not None:
        print(f"  Labels:        {labels_path} ({n_labels} labeled site(s))")
        print(f"  Eval results:  {output_dir / 'covsite_label_results.csv'}")
        print(f"  Eval summary:  {output_dir / 'covsite_evaluation_summary.json'}")
    print(f"  Score ranking: {output_dir / 'covsite_scores.csv'}")
    print(f"  Score summary: {output_dir / 'covsite_score_summary.json'}")


def main() -> None:
    args = parse_args()
    if args.workers != 1:
        raise SystemExit("[ERROR] --workers currently must be 1")

    if args.screen_only:
        run_screen_only(args)
        return

    input_path = Path(args.input_csv).resolve()
    model_path = Path(args.model).resolve() if args.model else None
    pdb_dir = Path(args.pdb_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    roots = script_search_roots(args.scripts_dir)
    frankenstein_path = locate_file("Frankenstein.py", roots)
    residue_path = locate_file("add_residue_info.py", roots)
    ligand_path = locate_file("add_ligand_info.py", roots)
    interaction_path = locate_file("add_interaction_info.py", roots)
    nterminal_path = locate_file("Adding_N_terminal.py", roots)
    cov_screen_path = (
        None if args.vs else locate_file("Cov_Screen.py", roots)
    )
    deprot_script_path = locate_file("Deprotonation_Model.py", roots)

    deprot_model_path = (
        Path(args.deprotonation_model).resolve()
        if args.deprotonation_model
        else locate_file("deprot_xgb_noapbs.pkl", roots)
    )
    if deprot_model_path is None:
        raise FileNotFoundError("No deprotonation model was found")

    frankenstein = load_module("covsite_frankenstein", frankenstein_path)
    # Share xTB caches with standalone `python Frankenstein.py` in this project
    # root. Otherwise dependencies/Frankenstein.py would write a separate
    # dependencies/reactivity_cache.json and never see the root cache.
    frankenstein.CACHE_FILE = str(HERE / "reactivity_cache.json")
    frankenstein.NUCLEOPHILE_CACHE_FILE = str(HERE / "nucleophile_cache.json")
    residue_module = load_module("covsite_add_residue_info", residue_path)
    ligand_module = load_module("covsite_add_ligand_info", ligand_path)
    interaction_module = load_module(
        "covsite_add_interaction_info", interaction_path
    )
    nterminal_module = load_module("covsite_adding_n_terminal", nterminal_path)
    deprot_module = load_module("covsite_deprotonation_model", deprot_script_path)

    raw = pd.read_csv(input_path, low_memory=False).reset_index(drop=True)
    if raw.empty:
        raise SystemExit("[ERROR] Input CSV has no rows")
    pdb_col = detect_column(
        raw, args.pdb_column, ("pdb", "pdb_id", "protein pdb", "protein_pdb"), True
    )
    smiles_col = detect_column(
        raw,
        args.smiles_column,
        ("smiles", "electrophile_smiles", "electrophile smiles"),
        False,
    )
    ligid_col = detect_column(
        raw,
        args.ligid_column,
        ("ligid", "lig_id", "ligand_id", "ligand id"),
        False,
    )
    if smiles_col is None and ligid_col is None:
        raise ValueError(
            "Input CSV must contain a SMILES column, a LigID column, or both"
        )
    name_col = detect_column(raw, args.name_column, ("name",), False)
    residue_col = detect_column(
        raw, args.residue_column, ("residue", "resname", "aa"), False
    )
    resnum_col = detect_column(
        raw,
        args.resnum_column,
        ("resnum", "res_num", "residue_number", "residue_num"),
        False,
    )
    chain_col = detect_column(raw, args.chain_column, ("chain",), False)
    frank_warhead_col = detect_column(
        raw,
        args.warhead_column,
        ("frankenstein_warhead", "frankenstein warhead"),
        False,
    )
    warhead_col = None
    if frank_warhead_col is None:
        warhead_col = detect_column(
            raw,
            args.warhead_column,
            ("warhead",),
            False,
        )
    requested_warhead_col = frank_warhead_col or warhead_col

    work = raw.copy()
    work["_source_row"] = work.index
    work["_pdb_id"] = work[pdb_col].map(pdb_id_from_value)
    work["_smiles"] = work[smiles_col].map(clean_text) if smiles_col else ""
    work["_ligid"] = (
        work[ligid_col].map(clean_text).str.upper() if ligid_col else ""
    )
    work["_input_name"] = work[name_col].map(clean_text) if name_col else ""
    work["_target_residue"] = (
        work[residue_col].map(clean_text).str.upper() if residue_col else ""
    )
    work["_target_resnum"] = work[resnum_col] if resnum_col else ""
    work["_target_chain"] = (
        work[chain_col].map(clean_text).str.upper() if chain_col else ""
    )
    work["_requested_warheads"] = (
        work[requested_warhead_col].map(clean_text)
        if requested_warhead_col
        else ""
    )
    if requested_warhead_col:
        print(
            f"[CovSite] Using input warhead column {requested_warhead_col!r} "
            "as labels Frankenstein_Warhead (exact text; no conversion)"
        )
    missing_ligand = (work["_smiles"] == "") & (work["_ligid"] == "")
    if (work["_pdb_id"] == "").any() or missing_ligand.any():
        bad = work.index[
            (work["_pdb_id"] == "") | missing_ligand
        ].tolist()
        raise ValueError(
            f"Rows missing PDB or both SMILES and LigID values: {bad}"
        )
    partial_targets = [
        int(row["_source_row"])
        for _, row in work.iterrows()
        if target_field_count(row) not in (0, 3)
    ]
    if partial_targets:
        raise ValueError(
            "Target sites must provide Residue, ResNum, and Chain together. "
            f"Partial targets found on rows: {partial_targets}"
        )
    if args.vs:
        missing_vs_targets = [
            int(row["_source_row"])
            for _, row in work.iterrows()
            if not target_is_complete(row)
        ]
        if missing_vs_targets:
            raise ValueError(
                "--VS requires Residue, ResNum, and Chain on every input row. "
                f"Missing complete targets on rows: {missing_vs_targets}"
            )
        print(
            "[CovSite] VS mode: calculating features only for each row's "
            "target site; Cov_Screen ranking will be skipped"
        )

    resolved_pdb_paths: dict[str, Path] = {}
    skipped_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row_idx, row in work.iterrows():
        if clean_text(row["_smiles"]):
            continue
        pdb_id = row["_pdb_id"]
        try:
            if pdb_id not in resolved_pdb_paths:
                resolved_pdb_paths[pdb_id] = resolve_pdb(
                    pdb_id, pdb_dir, residue_module
                )
            work.at[row_idx, "_smiles"] = resolve_smiles_from_ligid(
                resolved_pdb_paths[pdb_id],
                row["_ligid"],
                frankenstein,
            )
        except Exception as exc:
            detail = str(exc)
            reason = "ligid_resolve_failed"
            skipped_rows.append(skipped_row_record(row, reason, detail))
            warnings.append(
                f"Input row {row['_source_row']}: {reason}: {detail}"
            )
            print(
                f"[CovSite] WARNING: skipping row {row['_source_row']} "
                f"({reason}): {detail}"
            )
            work.at[row_idx, "_smiles"] = ""

    # Drop rows that never got a usable ligand SMILES.
    for _, row in work.iterrows():
        if clean_text(row["_smiles"]):
            continue
        source_row = int(row["_source_row"])
        if any(int(item["Source_Row"]) == source_row for item in skipped_rows):
            continue
        reason = "missing_smiles"
        detail = "No SMILES after LigID/SMILES resolution"
        skipped_rows.append(skipped_row_record(row, reason, detail))
        warnings.append(f"Input row {source_row}: {reason}: {detail}")
        print(f"[CovSite] WARNING: skipping row {source_row} ({reason})")

    work = work[work["_smiles"].map(clean_text) != ""].copy()
    if work.empty:
        skipped_path = write_skipped_inputs_csv(output_dir, skipped_rows)
        if skipped_path is not None:
            print(f"[CovSite] Wrote skipped inputs: {skipped_path}")
        raise SystemExit("[ERROR] No input rows with usable ligand SMILES remain")

    complete_target_keys = [
        (row["_pdb_id"], row["_smiles"])
        for _, row in work.iterrows()
        if target_is_complete(row)
    ]
    duplicate_target_counts = Counter(complete_target_keys)

    # Supplied Names that are reused across different target sites.
    supplied_name_sites: dict[str, set[str]] = defaultdict(set)
    for _, row in work.iterrows():
        supplied = clean_text(row["_input_name"])
        if not supplied or not target_is_complete(row):
            continue
        site = (
            f"{clean_text(row['_target_residue']).upper()}:"
            f"{normalized_resnum(row['_target_resnum'])}:"
            f"{clean_text(row['_target_chain']).upper()}"
        )
        supplied_name_sites[supplied.casefold()].add(site)
    multi_target_supplied_names = {
        name for name, sites in supplied_name_sites.items() if len(sites) > 1
    }

    deprot_pipeline = joblib.load(deprot_model_path)
    reactivity_cache = frankenstein.load_reactivity_cache()
    nucleophile_cache = frankenstein.load_nucleophile_cache()
    # In VS mode, cache FreeSASA rows only and deprotonate the requested site
    # per input row. Otherwise cache fully prepared nucleophile sites.
    pdb_sasa_cache: dict[str, tuple[Path, pd.DataFrame]] = {}
    pdb_cache: dict[str, tuple[Path, pd.DataFrame]] = {}
    pdb_failures: dict[str, str] = {}
    chemistry_cache: dict[
        Any, list[tuple[dict[str, Any], dict[str, dict[str, float]]]] | None
    ] = {}
    chemistry_failures: dict[Any, tuple[str, str]] = {}
    candidate_frames: list[pd.DataFrame] = []
    ligand_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    run_id = make_run_id()
    shard_dir = output_dir / "feature_cache" / "shards"
    existing_shards = load_feature_shards(shard_dir)
    if existing_shards:
        print(
            f"[CovSite] Resuming {len(existing_shards)} cached feature shard(s) "
            f"from {shard_dir}"
        )

    for _, input_row in work.iterrows():
        pdb_id = input_row["_pdb_id"]
        smiles = input_row["_smiles"]

        if pdb_id in pdb_failures:
            reason, detail = "pdb_prepare_failed", pdb_failures[pdb_id]
            skipped_rows.append(skipped_row_record(input_row, reason, detail))
            warnings.append(
                f"Input row {input_row['_source_row']}: {reason}: {detail}"
            )
            print(
                f"[CovSite] WARNING: skipping row {input_row['_source_row']} "
                f"({reason}): {detail}"
            )
            continue

        if args.vs:
            if pdb_id not in pdb_sasa_cache:
                try:
                    pdb_path = resolved_pdb_paths.get(pdb_id)
                    if pdb_path is None:
                        pdb_path = resolve_pdb(pdb_id, pdb_dir, residue_module)
                    sasa_sites = sasa_rows(pdb_path, frankenstein)
                    pdb_sasa_cache[pdb_id] = (pdb_path, sasa_sites)
                except Exception as exc:
                    detail = str(exc)
                    pdb_failures[pdb_id] = detail
                    reason = "pdb_prepare_failed"
                    skipped_rows.append(
                        skipped_row_record(input_row, reason, detail)
                    )
                    warnings.append(
                        f"Input row {input_row['_source_row']}: "
                        f"{reason}: {detail}"
                    )
                    print(
                        f"[CovSite] WARNING: skipping row "
                        f"{input_row['_source_row']} ({reason}): {detail}"
                    )
                    continue

            pdb_path, sasa_sites = pdb_sasa_cache[pdb_id]
            target_residue = clean_text(input_row["_target_residue"]).upper()
            target_resnum = normalized_resnum(input_row["_target_resnum"])
            target_chain = clean_text(input_row["_target_chain"]).upper()
            sites = filter_sites_to_target(
                sasa_sites, target_residue, target_resnum, target_chain
            )
            if sites.empty:
                reason = (
                    "not a supported nucleophilic residue"
                    if target_residue not in VALID_RESIDUES
                    else "target_site_not_found"
                )
                detail = (
                    f"Target {target_chain}:{target_residue}:{target_resnum} "
                    f"is {reason.replace('_', ' ')} in {pdb_path.name}"
                )
                skipped_rows.append(
                    skipped_row_record(input_row, reason, detail)
                )
                warnings.append(
                    f"Input row {input_row['_source_row']}: {reason}: {detail}"
                )
                print(
                    f"[CovSite] WARNING: skipping row "
                    f"{input_row['_source_row']} ({reason}): {detail}"
                )
                continue
            sites, deprot_errors = predict_deprotonation(
                sites, pdb_path, deprot_model_path, deprot_pipeline, deprot_module
            )
            warnings.extend(deprot_errors)
            row_has_valid_target = True
            chemistry_key: Any = (smiles, target_residue)
            residue_filter: tuple[str, ...] | None = (target_residue,)
        else:
            if pdb_id not in pdb_cache:
                try:
                    pdb_path = resolved_pdb_paths.get(pdb_id)
                    if pdb_path is None:
                        pdb_path = resolve_pdb(pdb_id, pdb_dir, residue_module)
                    sites = sasa_rows(pdb_path, frankenstein)
                    sites, deprot_errors = predict_deprotonation(
                        sites,
                        pdb_path,
                        deprot_model_path,
                        deprot_pipeline,
                        deprot_module,
                    )
                    warnings.extend(deprot_errors)
                    pdb_cache[pdb_id] = (pdb_path, sites)
                except Exception as exc:
                    detail = str(exc)
                    pdb_failures[pdb_id] = detail
                    reason = "pdb_prepare_failed"
                    skipped_rows.append(
                        skipped_row_record(input_row, reason, detail)
                    )
                    warnings.append(
                        f"Input row {input_row['_source_row']}: "
                        f"{reason}: {detail}"
                    )
                    print(
                        f"[CovSite] WARNING: skipping row "
                        f"{input_row['_source_row']} ({reason}): {detail}"
                    )
                    continue

            pdb_path, sites = pdb_cache[pdb_id]
            # Complete targets that are missing/unsupported are treated as unlabeled:
            # still build candidates + score ranking, but omit from labels.csv.
            row_has_valid_target = False
            if target_is_complete(input_row):
                target_residue = clean_text(input_row["_target_residue"]).upper()
                target_resnum = normalized_resnum(input_row["_target_resnum"])
                target_chain = clean_text(input_row["_target_chain"]).upper()
                target_exists = (
                    (sites["Residue"].astype(str).str.upper() == target_residue)
                    & (sites["ResNum"].map(normalized_resnum) == target_resnum)
                    & (
                        sites["Chain"].astype(str).str.strip().str.upper()
                        == target_chain
                    )
                ).any()
                if target_exists:
                    row_has_valid_target = True
                else:
                    reason = (
                        "not a supported nucleophilic residue"
                        if target_residue not in VALID_RESIDUES
                        else "not found among supported nucleophilic sites"
                    )
                    msg = (
                        f"Input row {input_row['_source_row']} target "
                        f"{target_chain}:{target_residue}:{target_resnum} "
                        f"is {reason} in {pdb_path.name}; treating as "
                        "unlabeled (score-only)"
                    )
                    warnings.append(msg)
                    print(f"[CovSite] WARNING: {msg}")
            chemistry_key = smiles
            residue_filter = None

        if chemistry_key not in chemistry_cache:
            try:
                chemistry = []
                for warhead in unique_warheads(smiles, frankenstein):
                    descriptors = deprotonated_descriptors(
                        smiles,
                        warhead,
                        frankenstein,
                        reactivity_cache,
                        nucleophile_cache,
                        use_cache=not args.no_cache,
                        residues=residue_filter,
                    )
                    chemistry.append((warhead, descriptors))
                chemistry_cache[chemistry_key] = chemistry
                frankenstein.save_reactivity_cache(reactivity_cache)
                frankenstein.save_nucleophile_cache(nucleophile_cache)
            except ValueError as exc:
                detail = str(exc)
                reason = (
                    "invalid_smiles"
                    if "Invalid ligand SMILES" in detail
                    else "warhead_detection_failed"
                )
                chemistry_cache[chemistry_key] = None
                chemistry_failures[chemistry_key] = (reason, detail)
            except Exception as exc:
                detail = str(exc)
                reason = "descriptor_calculation_failed"
                chemistry_cache[chemistry_key] = None
                chemistry_failures[chemistry_key] = (reason, detail)
                frankenstein.save_reactivity_cache(reactivity_cache)
                frankenstein.save_nucleophile_cache(nucleophile_cache)

        chemistry = chemistry_cache[chemistry_key]
        if chemistry is None:
            reason, detail = chemistry_failures.get(
                chemistry_key, ("chemistry_failed", "Unknown chemistry failure")
            )
            skipped_rows.append(skipped_row_record(input_row, reason, detail))
            warnings.append(
                f"Input row {input_row['_source_row']}: {reason}: {detail}"
            )
            print(
                f"[CovSite] WARNING: skipping row {input_row['_source_row']} "
                f"({reason}): {detail}"
            )
            continue
        if not chemistry:
            reason = "no_warhead_detected"
            detail = f"No electrophile warhead matched for SMILES: {smiles}"
            skipped_rows.append(skipped_row_record(input_row, reason, detail))
            warnings.append(
                f"Input row {input_row['_source_row']}: {reason} for {smiles}"
            )
            print(
                f"[CovSite] WARNING: skipping row {input_row['_source_row']} "
                f"({reason})"
            )
            continue

        label_frankenstein_warhead = clean_text(
            input_row.get("_requested_warheads", "")
        )

        for warhead, descriptors in chemistry:
            name = make_base_name(
                input_row,
                warhead["Warhead"],
                duplicate_target_counts,
                multi_target_supplied_names,
            )
            cache_key = shard_cache_key(
                int(input_row["_source_row"]), name, warhead["Warhead"]
            )
            if cache_key in existing_shards:
                frame = existing_shards[cache_key]
                if args.vs:
                    target_residue = clean_text(
                        input_row["_target_residue"]
                    ).upper()
                    target_resnum = normalized_resnum(
                        input_row["_target_resnum"]
                    )
                    target_chain = clean_text(
                        input_row["_target_chain"]
                    ).upper()
                    filtered = filter_sites_to_target(
                        frame,
                        target_residue,
                        target_resnum,
                        target_chain,
                    )
                    if filtered.empty:
                        print(
                            f"[CovSite] Ignoring multi-site shard for VS "
                            f"{name} / {warhead['Warhead']}; recalculating"
                        )
                        del existing_shards[cache_key]
                    else:
                        frame = filtered
                        existing_shards[cache_key] = frame
                        write_feature_shard(
                            shard_dir,
                            frame,
                            int(input_row["_source_row"]),
                            name,
                            warhead["Warhead"],
                        )
                        print(
                            f"[CovSite] Reusing shard for {name} / "
                            f"{warhead['Warhead']}"
                        )
                else:
                    print(
                        f"[CovSite] Reusing shard for {name} / "
                        f"{warhead['Warhead']}"
                    )
            if cache_key not in existing_shards:
                frame = add_descriptor_columns(sites, descriptors)
                frame.insert(0, "Name", name)
                frame.insert(1, "PDB_ID", pdb_id)
                frame.insert(2, "electrophile_smiles", smiles)
                frame.insert(3, "LigID", input_row["_ligid"])
                for key, value in warhead.items():
                    frame[key] = value
                frame["Source_Row"] = int(input_row["_source_row"])
                write_feature_shard(
                    shard_dir,
                    frame,
                    int(input_row["_source_row"]),
                    name,
                    warhead["Warhead"],
                )
                existing_shards[cache_key] = frame

            candidate_frames.append(frame)
            ligand_rows.append(ligand_row_from_frame(frame))
            if row_has_valid_target:
                label_rows.append(
                    {
                        "Name": name,
                        "Residue": input_row["_target_residue"],
                        "ResNum": normalized_resnum(input_row["_target_resnum"]),
                        "Chain": input_row["_target_chain"],
                        "Frankenstein_Warhead": (
                            label_frankenstein_warhead or warhead["Warhead"]
                        ),
                        "electrophile_smiles": smiles,
                        "PDB_ID": pdb_id,
                        "LigID": input_row["_ligid"],
                    }
                )

    frankenstein.save_reactivity_cache(reactivity_cache)
    frankenstein.save_nucleophile_cache(nucleophile_cache)
    if not candidate_frames:
        skipped_path = write_skipped_inputs_csv(output_dir, skipped_rows)
        if skipped_path is not None:
            print(f"[CovSite] Wrote skipped inputs: {skipped_path}")
        if warnings:
            warning_path = output_dir / "covsite_warnings.txt"
            warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
            print(f"[CovSite] Wrote warnings: {warning_path}")
        raise SystemExit("[ERROR] No candidate rows were generated")

    candidates = pd.concat(candidate_frames, ignore_index=True)
    ligand_labels = pd.DataFrame(ligand_rows).drop_duplicates(
        subset=["Name"], keep="first"
    )

    # Process one PDB at a time to avoid add_residue_info's site-only cache
    # being reused across different structures.
    enriched_groups: list[pd.DataFrame] = []
    for pdb_id, group in candidates.groupby("PDB_ID", sort=False):
        enriched, missing_pdbs, site_errors, _ = residue_module.add_residue_info(
            group.reset_index(drop=True),
            pdb_dir=str(pdb_dir),
            recursive=True,
            name_col="PDB_ID",
            chain_col="Chain",
            resnum_col="ResNum",
            residue_col="Residue",
            download_missing=True,
        )
        if missing_pdbs:
            warnings.append(f"{pdb_id}: missing PDB during residue enrichment")
        warnings.extend(site_errors)
        enriched_groups.append(enriched)
    candidates = pd.concat(enriched_groups, ignore_index=True)

    candidates, nterm_missing, nterm_warnings = (
        nterminal_module.add_n_terminal_column(
            candidates,
            pdb_dir=str(pdb_dir),
            name_col="PDB_ID",
            chain_col="Chain",
            resnum_col="ResNum",
            residue_col="Residue",
            recursive=True,
        )
    )
    if nterm_missing:
        warnings.append(
            "N-terminal metadata missing for: " + ", ".join(nterm_missing)
        )
    warnings.extend(nterm_warnings)

    candidates, ligand_warnings, ligand_errors = ligand_module.add_ligand_info(
        candidates,
        ligand_labels,
        smiles_col="electrophile_smiles",
        training_name_col="Name",
        labels_name_col="Name",
        cache_dir=str(output_dir / "ligand_cache"),
    )
    warnings.extend(ligand_warnings)
    warnings.extend(ligand_errors)
    candidates, _, _ = interaction_module.add_interaction_info(candidates)

    labels = pd.DataFrame(
        label_rows,
        columns=[
            "Name", "Residue", "ResNum", "Chain",
            "Frankenstein_Warhead", "electrophile_smiles", "PDB_ID", "LigID",
        ],
    ).drop_duplicates()
    candidates, labels, rename_messages = disambiguate_name_warhead_site_conflicts(
        candidates, labels
    )
    warnings.extend(rename_messages)
    for msg in rename_messages:
        print(f"[CovSite] WARNING: {msg}")

    # Unique final feature CSVs before Cov_Screen; retire prior finals + shards.
    candidates_path, labels_path = write_final_feature_cache(
        output_dir, run_id, candidates, labels, shard_dir
    )
    print(f"[CovSite] Wrote feature cache {candidates_path.name}")
    print(f"[CovSite] Wrote labels cache  {labels_path.name}")

    if args.vs:
        print("\n[CovSite] Complete (VS feature-only)")
        print(f"  Candidate features: {candidates_path}")
        print(f"  Labels:             {labels_path} ({len(labels)} labeled site(s))")
        print(f"  Latest aliases:     {output_dir / 'covsite_candidates.csv'}")
        print("  Ranking:            skipped (--VS)")
    else:
        launch_cov_screen(
            args,
            cov_screen_path,
            model_path,
            candidates_path,
            labels_path if not labels.empty else None,
            output_dir,
        )
        print("\n[CovSite] Complete")
        print(f"  Candidate features: {candidates_path}")
        print(f"  Labels:             {labels_path} ({len(labels)} labeled site(s))")
        print(f"  Latest aliases:     {output_dir / 'covsite_candidates.csv'}")
        print(f"  Score ranking:      {output_dir / 'covsite_scores.csv'}")

    skipped_path = write_skipped_inputs_csv(output_dir, skipped_rows)
    if skipped_path is not None:
        print(f"  Skipped inputs:     {skipped_path} ({len(skipped_rows)} row(s))")
    if warnings:
        warning_path = output_dir / "covsite_warnings.txt"
        warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"  Warnings:           {warning_path} ({len(warnings)})")


if __name__ == "__main__":
    main()
