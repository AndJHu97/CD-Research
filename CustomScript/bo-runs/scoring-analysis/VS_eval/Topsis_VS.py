"""
TOPSIS virtual screening over matched ligand-warhead variants.

Labels are used like Cov_Screen.py --VS:
  * select ligand Names;
  * match label Frankenstein_Warhead values to training Warhead values;
  * ignore label Residue/ResNum/Chain.

Training rows are selected with one or more --residue values and optional
--chain/--resnum. Features are never averaged across sites. Each
Residue/ResNum/Chain x Name x Warhead row is retained as a candidate.

Two analyses are produced:
  * per-site: TOPSIS is run independently for every selected residue site;
  * pooled: all selected residue-site x ligand x warhead rows are ranked
    together as one collective screen.

Outputs:
  topsis_per_site_ranking.csv         full independent rankings for every site
  topsis_pooled_pair_ranking.csv      full collective ranking of all site pairs
  full_pooled_ligand_ranking.csv      full ligand-level list (best pair per Name)
  hit_per_site_ranks.csv              hit subset of per-site ranking
  hit_pooled_pair_ranks.csv           hit subset of pooled pair ranking
  enrichment_factors.csv              EF@1/5/10% plus LogAUC / adjusted LogAUC
  feature_statistics.csv              per-site and pooled single-feature stats
  feature_enrichment_factors.csv      single-feature EF + LogAUC / adjusted LogAUC
  matched_site_warhead_rows.csv       exact source rows used
  unmatched_hit_names.csv             unmatched hits with Reason/Detail
                                      (name_level_dedup = stereo/Name collapse)

Example:
  python TOPSIS_VS.py \
      --training ../training_usp8_extended_full.csv \
      --labels ../labels_usp8.csv \
      --hits usp8_hits.csv \
      --residue CYS \
      --benefit Nucleophilicity_Index_Deprotonated Hydrophobic_Fit Geo_Fit \
      --cost HOMO_LUMO_Gap_Deprotonated \
      --weights 1 1 1 1 \
      --output-dir usp8_topsis

    python TOPSIS_VS.py --training bmx_candidates.csv --hits bmx_hits.csv --residue CYS --benefit Nucleophilicity_Index_Deprotonated Hydrophobic_Fit Hydrogen_DPAL_Fit Hydrogen_APDL_Fit --cost HOMO_LUMO_Gap_Deprotonated Geo_Fit_Deviation --weights 1 1 1 1 1 1 --output-dir bmx_topsis

Hits CSV may use Name_Hit and/or electrophile_hit (SMILES matched to
training electrophile_smiles). Optional warhead_type next to each hit row
restricts that ligand to those Warhead_Base values for hit labeling and
ranking (comma-separated allowed). Optional electrophile_miss (and/or
Name_Miss) with its own warhead_type column restricts which warhead variants
of listed decoys are ranked; anything not in the hit list remains a miss.
Use --multiple-hits to count each Name × Warhead × SMILES variant as a
separate hit (default: one per Name). Use --best-warhead to keep only the
highest TOPSIS-scoring warhead per electrophile_smiles × site before
EF/LogAUC. Use --entropy-weights to set criterion weights from Shannon
entropy of each feature in the candidate pool
(per site / pooled) instead of fixed --weights.

Omit --labels to rank every training row (after residue filters) with no
Frankenstein warhead matching.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


SCORING_DIR = Path(__file__).resolve().parent.parent
if str(SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_DIR))

from Cov_Screen import (  # noqa: E402
    VALID_RESIDUES,
    _normalize_resnum,
    filter_vs_site_eval,
    ligand_base_name,
    load_merged_for_vs,
    load_unlabeled_candidates,
)


EF_FRACTIONS = (0.01, 0.05, 0.10)

# Mysinger & Shoichet LogAUC on a semilog ROC (FPR from λ to 1).
# https://wiki.docking.org/index.php/LogAUC
LOGAUC_LAMBDA = 0.001
# logAUC of the random ROC (y = x) between λ and 1; AF3 / DISI adjusted metric.
RANDOM_LOGAUC = 0.14462

try:
    from rdkit import Chem

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    Chem = None
    _HAS_RDKIT = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TOPSIS ranking of VS-matched Name x Warhead candidates"
    )
    parser.add_argument("--training", required=True, help="Training feature CSV")
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Labels CSV for Names/warheads (optional). "
            "Omit to rank all training candidates after residue filters."
        ),
    )
    parser.add_argument(
        "--hits",
        "--input",
        dest="hits",
        required=True,
        help=(
            "CSV of positives: Name_Hit and/or electrophile_hit "
            "(SMILES matched to training electrophile_smiles). "
            "Optional per-row warhead_type restricts that hit to those "
            "Warhead_Base values. Optional electrophile_miss / Name_Miss "
            "with warhead_type restricts decoy warhead variants ranked."
        ),
    )
    parser.add_argument(
        "--name-hit-column",
        default="Name_Hit",
        help="Positive-name column in --hits (default: Name_Hit)",
    )
    parser.add_argument(
        "--smiles-hit-column",
        default="electrophile_hit",
        help=(
            "Positive-SMILES column in --hits (default: electrophile_hit). "
            "Matched against training electrophile_smiles."
        ),
    )
    parser.add_argument(
        "--name-miss-column",
        default="Name_Miss",
        help="Optional decoy-name column in --hits (default: Name_Miss)",
    )
    parser.add_argument(
        "--smiles-miss-column",
        default="electrophile_miss",
        help=(
            "Optional decoy-SMILES column in --hits "
            "(default: electrophile_miss). Used with an adjacent "
            "warhead_type column to restrict which miss warhead variants "
            "are ranked; ligands not in the hit list are still misses."
        ),
    )
    parser.add_argument(
        "--warhead-type-column",
        default="warhead_type",
        help=(
            "Per-row Warhead_Base column paired with hit/miss entity columns "
            "(default: warhead_type). Duplicate headers "
            "(hit Warhead_Type + miss Warhead_Type) are supported; each is "
            "bound to the preceding Electrophile_Hit / Electrophile_Miss "
            "column. Comma-separated values allowed."
        ),
    )
    parser.add_argument(
        "--multiple-hits",
        action="store_true",
        help=(
            "Count each Name × Warhead × electrophile_smiles variant as its "
            "own hit for EF and LogAUC (e.g. same SMILES with different "
            "warheads). Default: collapse to one hit per ligand Name."
        ),
    )
    parser.add_argument(
        "--best-warhead",
        action="store_true",
        help=(
            "For duplicate electrophile_smiles at the same site (multiple "
            "warheads), keep only the best-scoring warhead by TOPSIS score "
            "(benefit/cost) before ranking exports and EF/LogAUC. "
            "Single-feature EF uses that feature's oriented score."
        ),
    )
    parser.add_argument(
        "--residue",
        nargs="+",
        required=True,
        help="One or more training-site residue types (e.g. CYS or CYS SER)",
    )
    parser.add_argument("--chain", default=None, help="Optional chain filter")
    parser.add_argument("--resnum", default=None, help="Optional residue-number filter")
    parser.add_argument(
        "--benefit",
        nargs="*",
        default=[],
        metavar="FEATURE",
        help="Features where larger values are preferred",
    )
    parser.add_argument(
        "--cost",
        nargs="*",
        default=[],
        metavar="FEATURE",
        help="Features where smaller values are preferred",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional non-negative weights in feature order: all --benefit "
            "features followed by all --cost features. Default: equal. "
            "Ignored when --entropy-weights is set."
        ),
    )
    parser.add_argument(
        "--entropy-weights",
        action="store_true",
        help=(
            "Derive TOPSIS criterion weights from Shannon entropy of each "
            "feature across the ranked candidate pool (per site and pooled). "
            "Low-variance criteria get near-zero weight. Incompatible with "
            "explicit --weights."
        ),
    )
    parser.add_argument(
        "--missing-policy",
        choices=("median", "drop"),
        default="median",
        help=(
            "Handle missing feature values by criterion median "
            "(default) or drop the candidate"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="topsis_results",
        help="Output directory (default: topsis_results)",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def normalize_smiles(value: object) -> str:
    """
    Match key for electrophile SMILES (case-insensitive).

    Canonicalize with RDKit when available, then casefold so hit CSV values
    match training electrophile_smiles regardless of letter case.

    Do not call this on already-casefolded keys (e.g. ranking electrophile_smiles
    after build_site_candidates); use smiles_match_key instead — re-parsing
    casefolded SMILES fails (carbonyl O → o) and floods RDKit errors.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if _HAS_RDKIT:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            text = Chem.MolToSmiles(mol)
    return text.casefold()


def smiles_match_key(value: object) -> str:
    """Casefold-only SMILES key (no RDKit). Use when values are pre-normalized."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.casefold()


def normalize_warhead_base(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    # Drop instance tags if present on Warhead-like strings.
    text = re.sub(r"\s*\[atom[^\]]*\]", "", text, flags=re.IGNORECASE).strip()
    return text.casefold()


def resolve_column(columns: pd.Index, wanted: str) -> str | None:
    """Case-insensitive column lookup; returns the actual column name or None."""
    wanted_key = wanted.strip().casefold()
    for col in columns:
        base = str(col).strip()
        # Pandas mangles duplicate headers to Warhead_Type.1 — match stem.
        stem = base.split(".")[0].strip().casefold()
        if stem == wanted_key or base.casefold() == wanted_key:
            return str(col)
    return None


def resolve_paired_warhead_column(
    columns: pd.Index,
    entity_col: str | None,
    warhead_wanted: str,
) -> str | None:
    """
    Prefer the warhead_type column immediately after *entity_col* (handles
    duplicate Warhead_Type headers for hit vs miss). Else first matching name.
    """
    if entity_col is None:
        return None
    cols = [str(c) for c in columns]
    if entity_col not in cols:
        return None
    idx = cols.index(entity_col)
    wanted_key = warhead_wanted.strip().casefold()
    if idx + 1 < len(cols):
        nxt = cols[idx + 1]
        stem = nxt.split(".")[0].strip().casefold()
        if stem == wanted_key or "warhead_type" in nxt.casefold():
            return nxt
    # Fallback: first unused warhead_type-like column after entity.
    for col in cols[idx + 1 :]:
        stem = col.split(".")[0].strip().casefold()
        if stem == wanted_key:
            return col
    return None


def parse_warhead_type_cell(raw: object) -> set[str] | None:
    """
    Parse a warhead_type cell into normalized Warhead_Base values.

    Empty / missing -> None (no restriction: any warhead).
    Non-empty -> set of allowed bases (comma-separated list OK).
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    parts = {
        normalize_warhead_base(piece)
        for piece in text.split(",")
        if normalize_warhead_base(piece)
    }
    return parts or None


def _merge_warhead_allowance(
    existing: set[str] | None,
    new: set[str] | None,
) -> set[str] | None:
    """None means unrestricted; union restricted sets; unrestricted wins."""
    if existing is None or new is None:
        return None
    return existing | new


@dataclass
class HitMissSpec:
    """Per-ligand hit/miss identity with optional per-row warhead restrictions."""

    hit_names: set[str] = field(default_factory=set)
    hit_smiles: set[str] = field(default_factory=set)
    # entity -> allowed warheads; missing key = unrestricted; None value = any
    hit_name_warheads: dict[str, set[str] | None] = field(default_factory=dict)
    hit_smiles_warheads: dict[str, set[str] | None] = field(default_factory=dict)
    miss_names: set[str] = field(default_factory=set)
    miss_smiles: set[str] = field(default_factory=set)
    miss_name_warheads: dict[str, set[str] | None] = field(default_factory=dict)
    miss_smiles_warheads: dict[str, set[str] | None] = field(default_factory=dict)

    def allowed_warheads_for_row(
        self, name: str, smiles: str
    ) -> set[str] | None:
        """
        Warheads to keep for this ligand in the ranked pool.

        None => keep all warheads. Empty set should not occur.
        Hit restrictions take precedence over miss restrictions when both
        somehow apply.
        """
        if smiles and smiles in self.hit_smiles_warheads:
            return self.hit_smiles_warheads[smiles]
        if name and name in self.hit_name_warheads:
            return self.hit_name_warheads[name]
        if smiles and smiles in self.miss_smiles_warheads:
            return self.miss_smiles_warheads[smiles]
        if name and name in self.miss_name_warheads:
            return self.miss_name_warheads[name]
        return None

    def row_is_hit(self, name: str, smiles: str, warhead_base: str) -> bool:
        if smiles and smiles in self.hit_smiles:
            allowed = self.hit_smiles_warheads.get(smiles)
            if allowed is None:
                return True
            return bool(warhead_base) and warhead_base in allowed
        if name and name in self.hit_names:
            allowed = self.hit_name_warheads.get(name)
            if allowed is None:
                return True
            return bool(warhead_base) and warhead_base in allowed
        return False


def load_hits(
    path: str,
    name_column: str,
    smiles_column: str,
    warhead_type_column: str,
    name_miss_column: str = "Name_Miss",
    smiles_miss_column: str = "electrophile_miss",
) -> HitMissSpec:
    """
    Load positives (and optional decoy warhead restrictions) from the hits CSV.

    Hit warhead_type is per electrophile_hit / Name_Hit row. Miss warhead_type
    is per electrophile_miss / Name_Miss row and only restricts which warhead
    variants of those decoys are ranked. Ligands not listed as hits remain
    misses by default.
    """
    hit_df = pd.read_csv(path)
    hit_df.columns = hit_df.columns.str.strip()

    name_col = resolve_column(hit_df.columns, name_column)
    smiles_col = resolve_column(hit_df.columns, smiles_column)
    name_miss_col = resolve_column(hit_df.columns, name_miss_column)
    smiles_miss_col = resolve_column(hit_df.columns, smiles_miss_column)

    hit_wh_col = resolve_paired_warhead_column(
        hit_df.columns,
        smiles_col or name_col,
        warhead_type_column,
    )
    miss_entity = smiles_miss_col or name_miss_col
    miss_wh_col = resolve_paired_warhead_column(
        hit_df.columns, miss_entity, warhead_type_column
    )
    # If hit and miss resolve to the same column, only use it for hits when
    # there is no separate miss entity column.
    if (
        hit_wh_col is not None
        and miss_wh_col is not None
        and hit_wh_col == miss_wh_col
        and miss_entity is not None
    ):
        miss_wh_col = None
        for col in hit_df.columns:
            col_s = str(col)
            if col_s == hit_wh_col:
                continue
            stem = col_s.split(".")[0].strip().casefold()
            if stem == warhead_type_column.strip().casefold():
                miss_wh_col = col_s
                break

    if name_col is None and smiles_col is None:
        sys.exit(
            f"[ERROR] Hit CSV must contain {name_column!r} and/or "
            f"{smiles_column!r} (case-insensitive). "
            f"Available: {', '.join(map(str, hit_df.columns))}"
        )

    spec = HitMissSpec()

    for _, row in hit_df.iterrows():
        name = normalize_name(row[name_col]) if name_col else ""
        smiles = normalize_smiles(row[smiles_col]) if smiles_col else ""
        hit_wh = (
            parse_warhead_type_cell(row[hit_wh_col]) if hit_wh_col else None
        )

        if name:
            spec.hit_names.add(name)
            if hit_wh_col is not None:
                spec.hit_name_warheads[name] = _merge_warhead_allowance(
                    spec.hit_name_warheads.get(name, set()),
                    hit_wh,
                )
        if smiles:
            spec.hit_smiles.add(smiles)
            if hit_wh_col is not None:
                spec.hit_smiles_warheads[smiles] = _merge_warhead_allowance(
                    spec.hit_smiles_warheads.get(smiles, set()),
                    hit_wh,
                )

        miss_name = normalize_name(row[name_miss_col]) if name_miss_col else ""
        miss_smiles = (
            normalize_smiles(row[smiles_miss_col]) if smiles_miss_col else ""
        )
        miss_wh = (
            parse_warhead_type_cell(row[miss_wh_col]) if miss_wh_col else None
        )

        if miss_name:
            spec.miss_names.add(miss_name)
            if miss_wh_col is not None:
                spec.miss_name_warheads[miss_name] = _merge_warhead_allowance(
                    spec.miss_name_warheads.get(miss_name, set()),
                    miss_wh,
                )
        if miss_smiles:
            spec.miss_smiles.add(miss_smiles)
            if miss_wh_col is not None:
                spec.miss_smiles_warheads[miss_smiles] = _merge_warhead_allowance(
                    spec.miss_smiles_warheads.get(miss_smiles, set()),
                    miss_wh,
                )

    if not spec.hit_names and not spec.hit_smiles:
        sys.exit(
            f"[ERROR] No non-empty positives found in {name_column!r} / "
            f"{smiles_column!r}."
        )
    return spec


def validate_feature_args(args: argparse.Namespace) -> tuple[list[str], np.ndarray | None]:
    benefit = list(dict.fromkeys(args.benefit))
    cost = list(dict.fromkeys(args.cost))
    overlap = sorted(set(benefit) & set(cost))
    if overlap:
        sys.exit(
            "[ERROR] Features cannot be both benefit and cost:\n  "
            + ", ".join(overlap)
        )

    features = benefit + cost
    if not features:
        sys.exit("[ERROR] Supply at least one --benefit or --cost feature.")

    if getattr(args, "entropy_weights", False):
        if args.weights is not None:
            sys.exit(
                "[ERROR] Use either --entropy-weights or --weights, not both."
            )
        return features, None

    if args.weights is None:
        weights = np.ones(len(features), dtype=float)
    else:
        if len(args.weights) != len(features):
            sys.exit(
                f"[ERROR] --weights has {len(args.weights)} values but "
                f"{len(features)} features were specified. Weight order is "
                "all --benefit features followed by all --cost features."
            )
        weights = np.asarray(args.weights, dtype=float)

    if not np.isfinite(weights).all() or (weights < 0).any():
        sys.exit("[ERROR] --weights must be finite and non-negative.")
    if float(weights.sum()) <= 0:
        sys.exit("[ERROR] At least one TOPSIS weight must be positive.")
    weights = weights / weights.sum()
    return features, weights


def entropy_weights(
    df: pd.DataFrame,
    criteria_cols: list[str],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Shannon-entropy criterion weights from a candidate decision matrix.

    Low-spread criteria get e_j → 1, d_j → 0, w_j → 0. Columns are shifted to
    be strictly positive before forming the probability distribution.
    """
    if not criteria_cols:
        sys.exit("[ERROR] entropy_weights requires at least one criterion.")
    missing = [c for c in criteria_cols if c not in df.columns]
    if missing:
        sys.exit(
            "[ERROR] entropy_weights missing columns:\n  " + ", ".join(missing)
        )

    work = df[criteria_cols].apply(pd.to_numeric, errors="coerce")
    if work.isna().any().any():
        work = work.fillna(work.median(numeric_only=True))
    if work.isna().any().any():
        sys.exit(
            "[ERROR] Cannot compute entropy weights: criteria still contain "
            "missing values after median fill."
        )

    X = work.to_numpy(dtype=float)
    m = X.shape[0]
    if m < 2:
        print(
            "[WARN] Fewer than 2 candidates for entropy weights; "
            "falling back to equal weights."
        )
        return np.ones(len(criteria_cols), dtype=float) / len(criteria_cols)

    # Strictly positive probabilities (handles negative / zero fit scores).
    X_shifted = X - np.nanmin(X, axis=0) + eps
    col_sums = X_shifted.sum(axis=0, keepdims=True)
    zero_cols = col_sums.ravel() <= 0
    if zero_cols.any():
        col_sums = np.where(zero_cols.reshape(1, -1), 1.0, col_sums)
    P = X_shifted / col_sums

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(P > 0, P * np.log(P), 0.0)
    e_j = -entropy_terms.sum(axis=0) / np.log(m)
    d_j = 1.0 - e_j
    d_sum = float(d_j.sum())
    if not np.isfinite(d_sum) or d_sum <= 0:
        print(
            "[WARN] Entropy diversification sum is zero; "
            "falling back to equal weights."
        )
        return np.ones(len(criteria_cols), dtype=float) / len(criteria_cols)
    return (d_j / d_sum).astype(float)


def resolve_topsis_weights(
    candidates: pd.DataFrame,
    features: list[str],
    fixed_weights: np.ndarray | None,
    *,
    use_entropy: bool,
    label: str = "",
) -> np.ndarray:
    """Fixed weights, or entropy weights from this candidate pool."""
    if use_entropy:
        weights = entropy_weights(candidates, features)
        tag = f" ({label})" if label else ""
        parts = [
            f"{feature}={weight:.4f}"
            for feature, weight in zip(features, weights)
        ]
        print(f"[INFO] Entropy weights{tag}: " + ", ".join(parts))
        return weights
    assert fixed_weights is not None
    return fixed_weights


def _raw_training_for_unmatched_audit(training_csv: str) -> pd.DataFrame:
    """Load training rows before Name×site×warhead dedup (for unmatched reasons)."""
    df = pd.read_csv(training_csv, sep=",", engine="c", low_memory=False)
    required = {"Name", "Warhead", "Residue", "ResNum", "Chain"}
    missing = sorted(required - set(df.columns))
    if missing:
        return pd.DataFrame()
    df = df.copy()
    df["Residue"] = df["Residue"].astype(str).str.strip().str.upper()
    df = df[df["Residue"].isin(VALID_RESIDUES)].copy()
    df["_base_name"] = [
        ligand_base_name(name, warhead).upper()
        for name, warhead in zip(df["Name"], df["Warhead"])
    ]
    df["_warhead_lower"] = df["Warhead"].astype(str).str.strip().str.lower()
    df["_resnum_str"] = df["ResNum"].map(_normalize_resnum)
    df["_chain_upper"] = df["Chain"].astype(str).str.strip().str.upper()
    if "Warhead_Base" in df.columns:
        df["_analysis_warhead_base"] = df["Warhead_Base"].map(normalize_warhead_base)
    else:
        df["_analysis_warhead_base"] = df["Warhead"].map(normalize_warhead_base)
    if "electrophile_smiles" in df.columns:
        df["_analysis_smiles"] = df["electrophile_smiles"].map(normalize_smiles)
    else:
        df["_analysis_smiles"] = ""
    df["_analysis_name"] = df["_base_name"].map(normalize_name)
    return df


def _filter_raw_by_site(
    raw: pd.DataFrame,
    residues: list[str],
    chain: str | None,
    resnum: int | str | None,
) -> pd.DataFrame:
    if raw.empty:
        return raw
    parts = [
        filter_vs_site_eval(raw, residue=residue, chain=chain, resnum=resnum)
        for residue in residues
    ]
    if not parts:
        return raw.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def classify_unmatched_hits(
    args: argparse.Namespace,
    spec: HitMissSpec,
    selected: pd.DataFrame,
) -> list[dict[str, str]]:
    """
    Explain unmatched positives.

    name_level_dedup: SMILES/Name was in the training CSV but dropped because
    another row shares the same CovSite base Name × site × warhead (common for
    enantiomers whose Names strip stereo).
    """
    hit_names = spec.hit_names
    hit_smiles = spec.hit_smiles
    residues = list(dict.fromkeys(str(value).strip().upper() for value in args.residue))
    raw = _raw_training_for_unmatched_audit(args.training)
    raw_site = _filter_raw_by_site(raw, residues, args.chain, args.resnum)

    present_smiles = (
        set(selected["_analysis_smiles"])
        if "_analysis_smiles" in selected.columns
        else set()
    )
    present_names = set(selected["_analysis_name"])
    selected_bases = set(selected["_analysis_name"])

    records: list[dict[str, str]] = []

    for name in sorted(hit_names - present_names):
        allowed = spec.hit_name_warheads.get(name)
        raw_rows = (
            raw_site[raw_site["_analysis_name"] == name]
            if not raw_site.empty
            else raw_site
        )
        if raw_rows.empty and not raw.empty:
            raw_any = raw[raw["_analysis_name"] == name]
            if raw_any.empty:
                reason, detail = "not_in_candidates", "Name absent from training CSV"
            else:
                reason, detail = (
                    "site_filter",
                    "Name present in training but not at selected residue/site",
                )
        elif raw_rows.empty:
            reason, detail = "not_in_candidates", "Could not audit raw training CSV"
        elif allowed is not None and not raw_rows["_analysis_warhead_base"].isin(
            allowed
        ).any():
            available = ", ".join(
                sorted({str(v) for v in raw_rows["_analysis_warhead_base"] if v})
            )
            reason, detail = (
                "warhead_type_filter",
                (
                    "Name rows exist but Warhead_Base not in this hit's "
                    f"warhead_type (allowed: {sorted(allowed)}; "
                    f"available: {available or 'none'})"
                ),
            )
        else:
            reason, detail = (
                "name_level_dedup",
                "Dropped by base Name × site × warhead dedup (keep=first)",
            )
        records.append({"Hit": f"Name:{name}", "Reason": reason, "Detail": detail})

    for smiles in sorted(hit_smiles - present_smiles):
        allowed = spec.hit_smiles_warheads.get(smiles)
        if raw_site.empty and raw.empty:
            records.append(
                {
                    "Hit": f"SMILES:{smiles}",
                    "Reason": "not_in_candidates",
                    "Detail": "Could not audit raw training CSV",
                }
            )
            continue

        raw_rows = raw_site[raw_site["_analysis_smiles"] == smiles]
        if raw_rows.empty:
            raw_any = raw[raw["_analysis_smiles"] == smiles] if not raw.empty else raw
            if raw_any.empty:
                reason = "not_in_candidates"
                detail = "SMILES absent from training electrophile_smiles"
            else:
                reason = "site_filter"
                detail = "SMILES present in training but not at selected residue/site"
            records.append(
                {"Hit": f"SMILES:{smiles}", "Reason": reason, "Detail": detail}
            )
            continue

        if allowed is not None:
            wh_ok = raw_rows[raw_rows["_analysis_warhead_base"].isin(allowed)]
            if wh_ok.empty:
                available = ", ".join(
                    sorted({str(v) for v in raw_rows["_analysis_warhead_base"] if v})
                )
                records.append(
                    {
                        "Hit": f"SMILES:{smiles}",
                        "Reason": "warhead_type_filter",
                        "Detail": (
                            "SMILES rows exist but Warhead_Base not in this hit's "
                            f"warhead_type (allowed: {sorted(allowed)}; "
                            f"available: {available or 'none'})"
                        ),
                    }
                )
                continue
            raw_rows = wh_ok

        # Same base Name × site × warhead as a kept selected row → stereo/Name dedup.
        sibling_bits: list[str] = []
        for _, row in raw_rows.iterrows():
            base = normalize_name(row["_base_name"])
            site = (
                f"{str(row['Residue']).strip().upper()}:"
                f"{str(row['ResNum']).strip()}:"
                f"{str(row['Chain']).strip().upper()}"
            )
            warhead = str(row["Warhead"]).strip()
            kept = selected[
                (selected["_analysis_name"] == base)
                & (selected["_site_key"] == site)
                & (selected["_analysis_warhead"] == warhead)
            ]
            if kept.empty and "_analysis_warhead_base" in selected.columns:
                kept = selected[
                    (selected["_analysis_name"] == base)
                    & (selected["_site_key"] == site)
                    & (
                        selected["_analysis_warhead_base"]
                        == row["_analysis_warhead_base"]
                    )
                ]
            if not kept.empty:
                kept_smiles = (
                    kept["_analysis_smiles"].dropna().astype(str).unique().tolist()
                    if "_analysis_smiles" in kept.columns
                    else []
                )
                kept_show = kept_smiles[0] if kept_smiles else "(no SMILES on kept row)"
                sibling_bits.append(
                    f"base={base}; site={site}; warhead={warhead}; "
                    f"kept_smiles={kept_show}"
                )

        if sibling_bits:
            reason = "name_level_dedup"
            detail = (
                "Present in candidates but dropped by base Name × site × "
                "warhead dedup (keep=first). " + sibling_bits[0]
            )
        else:
            base = normalize_name(raw_rows["_base_name"].iloc[0])
            if base in selected_bases:
                reason = "name_level_dedup"
                detail = (
                    f"Present under base Name {base} but this SMILES was "
                    "removed by dedup"
                )
            else:
                reason = "dropped_before_ranking"
                detail = (
                    "Present in training at site/warhead but absent after "
                    "load_unlabeled / label merge dedup"
                )

        records.append(
            {"Hit": f"SMILES:{smiles}", "Reason": reason, "Detail": detail}
        )

    return records


def load_selected_rows(
    args: argparse.Namespace,
    features: list[str],
    spec: HitMissSpec,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if args.labels:
        print(f"[INFO] Loading with labels warhead matching: {args.labels}")
        merged = load_merged_for_vs(args.training, args.labels)
    else:
        print("[INFO] No --labels: ranking all training candidates")
        merged = load_unlabeled_candidates(args.training)

    residues = list(dict.fromkeys(str(value).strip().upper() for value in args.residue))
    selected_parts = [
        filter_vs_site_eval(
            merged,
            residue=residue,
            chain=args.chain,
            resnum=args.resnum,
        )
        for residue in residues
    ]
    selected = pd.concat(selected_parts, ignore_index=True).copy()
    if selected.empty:
        sys.exit(
            "[ERROR] No matched training rows remained after the requested "
            "residue/chain/resnum filter."
        )

    missing = [feature for feature in features if feature not in selected.columns]
    if missing:
        sys.exit(
            "[ERROR] Features missing from training CSV:\n  " + ", ".join(missing)
        )

    if "Warhead_Base" in selected.columns:
        selected["_analysis_warhead_base"] = selected["Warhead_Base"].map(
            normalize_warhead_base
        )
    else:
        selected["_analysis_warhead_base"] = selected["Warhead"].map(
            normalize_warhead_base
        )

    selected["_analysis_name"] = selected["_name_upper"].map(normalize_name)
    selected["_analysis_warhead"] = (
        selected["Warhead"].astype(str).str.strip()
    )
    selected["_site_key"] = (
        selected["Residue"].astype(str).str.strip().str.upper()
        + ":"
        + selected["ResNum"].astype(str).str.strip()
        + ":"
        + selected["Chain"].astype(str).str.strip().str.upper()
    )

    # Always retain a SMILES key when available so stereo variants stay distinct.
    if "electrophile_smiles" in selected.columns:
        selected["_analysis_smiles"] = selected["electrophile_smiles"].map(
            normalize_smiles
        )

    before = len(selected)
    names = selected["_analysis_name"].astype(str)
    smiles = (
        selected["_analysis_smiles"].astype(str)
        if "_analysis_smiles" in selected.columns
        else pd.Series([""] * len(selected), index=selected.index)
    )
    warheads = selected["_analysis_warhead_base"].astype(str)
    keep = []
    for name, smi, wh in zip(names, smiles, warheads):
        allowed = spec.allowed_warheads_for_row(name, smi)
        keep.append(allowed is None or wh in allowed)
    selected = selected.loc[keep].copy()
    dropped = before - len(selected)
    if dropped:
        print(
            f"[INFO] Per-ligand warhead_type filter removed {dropped:,} / "
            f"{before:,} rows (hit/miss Warhead_Type pairing)"
        )
    if selected.empty:
        sys.exit(
            "[ERROR] No training rows remained after per-ligand warhead_type "
            "filter."
        )

    if spec.hit_smiles and "_analysis_smiles" not in selected.columns:
        sys.exit(
            "[ERROR] Training CSV has no electrophile_smiles column, "
            "but hits were provided via SMILES."
        )

    smiles_col = (
        selected["_analysis_smiles"].astype(str)
        if "_analysis_smiles" in selected.columns
        else pd.Series([""] * len(selected), index=selected.index)
    )
    selected["is_hit"] = [
        int(
            spec.row_is_hit(
                str(name),
                str(smi),
                str(wh),
            )
        )
        for name, smi, wh in zip(
            selected["_analysis_name"],
            smiles_col,
            selected["_analysis_warhead_base"],
        )
    ]

    for feature in features:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")

    unmatched = classify_unmatched_hits(args, spec, selected)

    if int(selected["is_hit"].sum()) == 0:
        sys.exit(
            "[ERROR] None of the hit Names/SMILES matched selected training rows "
            "(check per-row warhead_type pairing)."
        )
    return selected, unmatched



def build_site_candidates(
    selected: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Keep every site x Name x Warhead x SMILES row separate; never average sites.
    """
    columns = [
        "_analysis_name",
        "_analysis_warhead",
        "Residue",
        "ResNum",
        "Chain",
        "_site_key",
        "is_hit",
        *features,
    ]
    has_smiles = "_analysis_smiles" in selected.columns
    if has_smiles:
        columns.insert(2, "_analysis_smiles")
    elif "electrophile_smiles" in selected.columns:
        selected = selected.copy()
        selected["_analysis_smiles"] = selected["electrophile_smiles"].map(
            normalize_smiles
        )
        columns.insert(2, "_analysis_smiles")
        has_smiles = True

    candidates = selected[columns].rename(
        columns={
            "_analysis_name": "Name",
            "_analysis_warhead": "Warhead",
            "_analysis_smiles": "electrophile_smiles",
            "_site_key": "Site",
        }
    ).copy()
    key_cols = ["Name", "Warhead", "Residue", "ResNum", "Chain"]
    if has_smiles and "electrophile_smiles" in candidates.columns:
        key_cols.append("electrophile_smiles")
    duplicate_count = int(candidates.duplicated(key_cols).sum())
    if duplicate_count:
        print(
            f"[WARN] Dropping {duplicate_count:,} duplicate site x Name x "
            "Warhead"
            + (" x SMILES" if has_smiles else "")
            + " rows; no feature values are averaged."
        )
        candidates = candidates.drop_duplicates(key_cols, keep="first").copy()

    candidates["Name_Warhead"] = (
        candidates["Name"] + "_" + candidates["Warhead"]
    )
    if has_smiles and "electrophile_smiles" in candidates.columns:
        smiles_tag = candidates["electrophile_smiles"].fillna("").astype(str)
        candidates["Name_Warhead"] = (
            candidates["Name_Warhead"] + "_" + smiles_tag
        )
    candidates["Site_Name_Warhead"] = (
        candidates["Site"] + "_" + candidates["Name_Warhead"]
    )
    return candidates.reset_index(drop=True)


def handle_missing(
    candidates: pd.DataFrame,
    features: list[str],
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_counts = pd.DataFrame(
        {
            "feature": features,
            "n_missing_before": [
                int(candidates[feature].isna().sum()) for feature in features
            ],
        }
    )

    result = candidates.copy()
    if policy == "drop":
        before = len(result)
        result = result.dropna(subset=features).copy()
        print(f"  Missing policy drop: removed {before - len(result):,} candidates")
    else:
        for feature in features:
            median = result[feature].median()
            if pd.isna(median):
                sys.exit(
                    f"[ERROR] Feature {feature!r} is entirely missing after aggregation."
                )
            result[feature] = result[feature].fillna(float(median))

    if result.empty:
        sys.exit("[ERROR] No candidates remain after missing-value handling.")
    return result, missing_counts


def topsis_rank(
    candidates: pd.DataFrame,
    benefit_features: list[str],
    cost_features: list[str],
    weights: np.ndarray,
    rank_column: str = "overall_rank",
) -> pd.DataFrame:
    features = benefit_features + cost_features
    matrix = candidates[features].to_numpy(dtype=float)

    # Standard vector normalization followed by normalized criterion weights.
    denominators = np.sqrt(np.square(matrix).sum(axis=0))
    zero_norm = denominators == 0
    if zero_norm.any():
        constant = [features[i] for i in np.flatnonzero(zero_norm)]
        print(
            "[WARN] Zero vector norm for criterion/criteria; normalized values "
            "set to zero: " + ", ".join(constant)
        )
        denominators[zero_norm] = 1.0
    weighted = (matrix / denominators) * weights

    n_benefit = len(benefit_features)
    ideal_best = np.empty(len(features), dtype=float)
    ideal_worst = np.empty(len(features), dtype=float)
    for idx in range(len(features)):
        if idx < n_benefit:
            ideal_best[idx] = weighted[:, idx].max()
            ideal_worst[idx] = weighted[:, idx].min()
        else:
            ideal_best[idx] = weighted[:, idx].min()
            ideal_worst[idx] = weighted[:, idx].max()

    distance_best = np.sqrt(np.square(weighted - ideal_best).sum(axis=1))
    distance_worst = np.sqrt(np.square(weighted - ideal_worst).sum(axis=1))
    distance_sum = distance_best + distance_worst
    score = np.divide(
        distance_worst,
        distance_sum,
        out=np.full_like(distance_worst, 0.5),
        where=distance_sum > 0,
    )

    ranked = candidates.copy()
    ranked["distance_to_ideal_best"] = distance_best
    ranked["distance_to_ideal_worst"] = distance_worst
    ranked["topsis_score"] = score
    ranked = ranked.sort_values(
        ["topsis_score", "Name", "Warhead"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranked[rank_column] = np.arange(1, len(ranked) + 1)

    leading = [
        rank_column,
        "Site",
        "Residue",
        "ResNum",
        "Chain",
        "Name",
        "Warhead",
        "Name_Warhead",
        "Site_Name_Warhead",
        "is_hit",
        "topsis_score",
        "distance_to_ideal_best",
        "distance_to_ideal_worst",
    ]
    leading = [column for column in leading if column in ranked.columns]
    remaining = [column for column in ranked.columns if column not in leading]
    return ranked[leading + remaining]


def logauc_from_fpr_tpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    lambda_min: float = LOGAUC_LAMBDA,
) -> float:
    """
    Mysinger/Shoichet LogAUC_λ: fraction of ideal area under a semilog ROC.

    Integrates TPR vs log10(FPR) from λ to 1 with the trapezoidal rule, then
    divides by log10(1/λ). See https://wiki.docking.org/index.php/LogAUC
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.size < 2 or tpr.size < 2:
        return float("nan")

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    # Deduplicate FPR (keep last TPR at each FPR, ROC step plateaus).
    uniq_fpr, uniq_idx = np.unique(fpr, return_index=True)
    # unique returns first index; for ROC we want the last TPR at each FPR
    last_idx = []
    for value in uniq_fpr:
        matches = np.where(fpr == value)[0]
        last_idx.append(int(matches[-1]))
    fpr = fpr[last_idx]
    tpr = tpr[last_idx]

    if float(fpr.max()) < lambda_min:
        return float("nan")

    tpr_at_lambda = float(np.interp(lambda_min, fpr, tpr))
    keep = fpr > lambda_min
    fpr_seg = np.concatenate(([lambda_min], fpr[keep]))
    tpr_seg = np.concatenate(([tpr_at_lambda], tpr[keep]))

    if fpr_seg[-1] < 1.0 - 1e-15:
        fpr_seg = np.concatenate((fpr_seg, [1.0]))
        tpr_seg = np.concatenate((tpr_seg, [float(tpr_seg[-1])]))

    log_x = np.log10(fpr_seg)
    area = 0.0
    for i in range(len(log_x) - 1):
        area += (log_x[i + 1] - log_x[i]) * 0.5 * (tpr_seg[i + 1] + tpr_seg[i])
    denom = np.log10(1.0 / lambda_min)
    if denom <= 0:
        return float("nan")
    return float(area / denom)


def keep_best_warhead_per_electrophile(
    ranking: pd.DataFrame,
    score_column: str = "topsis_score",
    rank_column: str | None = None,
    *,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Keep one warhead variant per electrophile SMILES (per site).

    Among rows sharing the same normalized electrophile_smiles at the same
    Site, retain the row with the best score (TOPSIS or oriented feature).
    Rows without SMILES are left unchanged. Ranks are reassigned afterward.
    """
    if ranking.empty:
        return ranking
    if "electrophile_smiles" not in ranking.columns:
        print(
            "[WARN] --best-warhead ignored: ranking has no electrophile_smiles "
            "column"
        )
        return ranking
    if score_column not in ranking.columns:
        sys.exit(
            f"[ERROR] --best-warhead needs score column {score_column!r} "
            f"in ranking."
        )

    work = ranking.copy()
    work["_smiles_key"] = work["electrophile_smiles"].map(smiles_match_key)
    empty_mask = work["_smiles_key"].eq("")
    with_smiles = work.loc[~empty_mask].copy()
    without_smiles = work.loc[empty_mask].copy()

    before = len(with_smiles)
    if with_smiles.empty:
        return ranking

    group_cols = ["_smiles_key"]
    if "Site" in with_smiles.columns:
        group_cols.append("Site")

    with_smiles = with_smiles.sort_values(
        score_column,
        ascending=not higher_is_better,
        kind="mergesort",
    )
    kept = with_smiles.drop_duplicates(subset=group_cols, keep="first")
    dropped = before - len(kept)
    if dropped:
        print(
            f"[INFO] --best-warhead: kept best-scoring warhead per "
            f"electrophile_smiles"
            + (" × site" if "Site" in group_cols else "")
            + f"; dropped {dropped:,} duplicate warhead rows "
            f"({before:,} → {len(kept):,})"
        )

    result = pd.concat([kept, without_smiles], ignore_index=True)
    result = result.drop(columns=["_smiles_key"], errors="ignore")
    result = result.sort_values(
        score_column,
        ascending=not higher_is_better,
        kind="mergesort",
    ).reset_index(drop=True)

    if rank_column is not None:
        result[rank_column] = np.arange(1, len(result) + 1)
    return result


def hit_entity_key_columns(ranking: pd.DataFrame, multiple_hits: bool) -> list[str]:
    """
    Columns that define one positive entity for EF / LogAUC.

    Default: Name only (one hit per ligand).
    --multiple-hits: Name × Warhead × electrophile_smiles so warhead (and
    stereo SMILES) variants of the same ligand each count separately.
    """
    if not multiple_hits:
        return ["Name"]
    cols = ["Name", "Warhead"]
    if "electrophile_smiles" in ranking.columns:
        cols.append("electrophile_smiles")
    return cols


def assign_hit_entity_id(ranking: pd.DataFrame, multiple_hits: bool) -> pd.Series:
    cols = hit_entity_key_columns(ranking, multiple_hits)
    parts = []
    for col in cols:
        if col not in ranking.columns:
            parts.append(pd.Series("", index=ranking.index))
        else:
            parts.append(
                ranking[col].map(
                    lambda v: "" if pd.isna(v) else str(v).strip().casefold()
                )
            )
    entity = parts[0]
    for part in parts[1:]:
        entity = entity + "||" + part
    return entity


def compute_logauc_metrics(
    ranking: pd.DataFrame,
    rank_column: str,
    *,
    multiple_hits: bool = False,
    per_row: bool = False,
) -> dict[str, float]:
    """
    Compute LogAUC_0.001 and adjusted LogAUC for a ranked screen.

    Default: collapse to unique Name (best rank, any-hit).
    --multiple-hits: collapse to Name × Warhead × SMILES.
    per_row=True: no collapse (each ranked candidate row is an entity).

    adjusted_logAUC = logAUC - 0.14462 (fractional); percent forms use ×100.
    Random adjusted = 0; perfect adjusted ≈ 0.85538 (85.538%).
    """
    empty = {
        "logAUC": float("nan"),
        "adjusted_logAUC": float("nan"),
        "logAUC_percent": float("nan"),
        "adjusted_logAUC_percent": float("nan"),
        "logAUC_lambda": LOGAUC_LAMBDA,
    }
    if ranking.empty or "is_hit" not in ranking.columns:
        return empty

    work = ranking.copy()
    work[rank_column] = pd.to_numeric(work[rank_column], errors="coerce")
    work = work.dropna(subset=[rank_column])
    if work.empty:
        return empty

    if per_row:
        work = work.rename(columns={rank_column: "rank"})
    else:
        work["_hit_entity"] = assign_hit_entity_id(work, multiple_hits)
        work = (
            work.groupby("_hit_entity", sort=False)
            .agg(rank=(rank_column, "min"), is_hit=("is_hit", "max"))
            .reset_index(drop=True)
        )

    y_true = work["is_hit"].astype(int).to_numpy()
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return empty

    y_score = -work["rank"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    logauc = logauc_from_fpr_tpr(fpr, tpr, LOGAUC_LAMBDA)
    if not np.isfinite(logauc):
        return empty
    adjusted = logauc - RANDOM_LOGAUC
    return {
        "logAUC": float(logauc),
        "adjusted_logAUC": float(adjusted),
        "logAUC_percent": float(logauc * 100.0),
        "adjusted_logAUC_percent": float(adjusted * 100.0),
        "logAUC_lambda": LOGAUC_LAMBDA,
    }


def enrichment_factors(
    ranking: pd.DataFrame,
    level: str,
    rank_column: str = "overall_rank",
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    EF_x = (hits in top x / top x size) / (all hits / all candidates).
    """
    n_total = len(ranking)
    n_hits = int(ranking["is_hit"].sum())
    if n_total == 0 or n_hits == 0:
        return pd.DataFrame()

    prevalence = n_hits / n_total
    logauc_metrics = compute_logauc_metrics(
        ranking, rank_column, per_row=True
    )
    records: list[dict] = []
    for fraction in EF_FRACTIONS:
        top_n = max(1, int(np.ceil(fraction * n_total)))
        top = ranking.nsmallest(top_n, rank_column)
        hits_top = int(top["is_hit"].sum())
        ef = (hits_top / top_n) / prevalence
        max_possible_hits = min(top_n, n_hits)
        max_ef = (max_possible_hits / top_n) / prevalence
        record = {
                "ranking_level": level,
                "cutoff_percent": 100.0 * fraction,
                "top_n": top_n,
                "n_candidates": n_total,
                "n_total_hits": n_hits,
                "n_hits_in_top": hits_top,
                "hit_prevalence": prevalence,
                "enrichment_factor": float(ef),
                "maximum_possible_ef": float(max_ef),
                "fraction_of_hits_recovered": hits_top / n_hits,
                **logauc_metrics,
        }
        if metadata:
            record = {**metadata, **record}
        records.append(record)
    return pd.DataFrame(records)


def enrichment_any_candidate_per_ligand(
    ranking: pd.DataFrame,
    level: str,
    rank_column: str,
    metadata: dict[str, object] | None = None,
    *,
    multiple_hits: bool = False,
) -> pd.DataFrame:
    """
    Rank all candidate rows; recover hit entities within each top-X% cutoff.

    Default: one hit entity per ligand Name (any warhead/SMILES row recovers it).
    --multiple-hits: one entity per Name × Warhead × electrophile_smiles so
    warhead variants of the same electrophile each count separately.

    Cutoffs are still based on all candidate rows. EF is hit-entity recovery
    divided by the fraction of candidate rows screened.

    Also attaches Mysinger/Shoichet LogAUC_0.001 and adjusted LogAUC.
    """
    n_rows = len(ranking)
    if n_rows == 0 or int(ranking["is_hit"].sum()) == 0:
        return pd.DataFrame()

    work = ranking.copy()
    work["_hit_entity"] = assign_hit_entity_id(work, multiple_hits)
    hit_entities = set(work.loc[work["is_hit"] == 1, "_hit_entity"])
    if not hit_entities:
        return pd.DataFrame()

    unique_names = ranking["Name"].nunique()
    n_hit_entities = len(hit_entities)
    hit_counting = (
        "name_warhead_smiles" if multiple_hits else "ligand_name"
    )
    logauc_metrics = compute_logauc_metrics(
        ranking, rank_column, multiple_hits=multiple_hits
    )
    records: list[dict[str, object]] = []
    for fraction in EF_FRACTIONS:
        top_n = max(1, int(np.ceil(fraction * n_rows)))
        top = work.nsmallest(top_n, rank_column)
        recovered = set(top.loc[top["is_hit"] == 1, "_hit_entity"])
        n_recovered = len(recovered)
        screened_fraction = top_n / n_rows
        recovery_fraction = n_recovered / n_hit_entities
        max_recovered = min(top_n, n_hit_entities)
        max_recovery_fraction = max_recovered / n_hit_entities
        record: dict[str, object] = {
            "ranking_level": level,
            "hit_counting": hit_counting,
            "cutoff_percent": 100.0 * fraction,
            "top_n_candidate_rows": top_n,
            "n_candidate_rows": n_rows,
            "n_unique_ligands": unique_names,
            "n_total_hit_entities": n_hit_entities,
            "n_hit_entities_recovered": n_recovered,
            # Backward-compatible aliases (ligand wording).
            "n_total_hit_ligands": n_hit_entities,
            "n_hit_ligands_recovered": n_recovered,
            "candidate_fraction_screened": screened_fraction,
            "fraction_of_hit_ligands_recovered": recovery_fraction,
            "enrichment_factor": recovery_fraction / screened_fraction,
            "maximum_possible_ef": (
                max_recovery_fraction / screened_fraction
            ),
            **logauc_metrics,
        }
        if metadata:
            record = {**metadata, **record}
        records.append(record)
    return pd.DataFrame(records)


def rank_each_site(
    candidates: pd.DataFrame,
    features: list[str],
    benefit_features: list[str],
    cost_features: list[str],
    weights: np.ndarray | None,
    missing_policy: str,
    *,
    multiple_hits: bool = False,
    best_warhead: bool = False,
    entropy_weights_flag: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    rankings: list[pd.DataFrame] = []
    ef_tables: list[pd.DataFrame] = []
    missing_tables: list[pd.DataFrame] = []
    weight_records: list[dict[str, object]] = []
    if best_warhead and multiple_hits:
        level = "topsis_site_best_warhead_multi_hit"
    elif best_warhead:
        level = "topsis_site_best_warhead"
    elif multiple_hits:
        level = "topsis_site_all_warheads_multi_hit"
    else:
        level = "topsis_site_all_warheads_any_hit"
    for site, group in candidates.groupby("Site", sort=True):
        handled, missing = handle_missing(group, features, missing_policy)
        site_weights = resolve_topsis_weights(
            handled,
            features,
            weights,
            use_entropy=entropy_weights_flag,
            label=f"site {site}",
        )
        for feature, weight in zip(features, site_weights):
            weight_records.append(
                {
                    "analysis_scope": "per_site",
                    "Site": site,
                    "Residue": group["Residue"].iloc[0],
                    "ResNum": group["ResNum"].iloc[0],
                    "Chain": group["Chain"].iloc[0],
                    "feature": feature,
                    "normalized_weight": float(weight),
                    "weight_method": (
                        "entropy" if entropy_weights_flag else "fixed"
                    ),
                }
            )
        site_ranked = topsis_rank(
            handled,
            benefit_features,
            cost_features,
            site_weights,
            rank_column="site_rank",
        )
        if best_warhead:
            site_ranked = keep_best_warhead_per_electrophile(
                site_ranked,
                score_column="topsis_score",
                rank_column="site_rank",
                higher_is_better=True,
            )
        rankings.append(site_ranked)
        site_metadata = {
            "analysis_scope": "per_site",
            "Site": site,
            "Residue": group["Residue"].iloc[0],
            "ResNum": group["ResNum"].iloc[0],
            "Chain": group["Chain"].iloc[0],
        }
        ef_tables.append(
            enrichment_any_candidate_per_ligand(
                site_ranked,
                level,
                rank_column="site_rank",
                metadata=site_metadata,
                multiple_hits=multiple_hits,
            )
        )
        for key, value in site_metadata.items():
            missing[key] = value
        missing_tables.append(missing)

    if not rankings:
        sys.exit("[ERROR] No per-site candidates could be ranked.")
    return (
        pd.concat(rankings, ignore_index=True),
        pd.concat(ef_tables, ignore_index=True),
        pd.concat(missing_tables, ignore_index=True),
        weight_records,
    )


def feature_statistics_and_enrichment(
    candidates: pd.DataFrame,
    features: list[str],
    benefit_features: list[str],
    *,
    multiple_hits: bool = False,
    best_warhead: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate row-level feature statistics and EF without averaging sites."""
    directions = {
        feature: ("benefit" if feature in benefit_features else "cost")
        for feature in features
    }
    scopes: list[tuple[str, str, pd.DataFrame]] = [
        ("pooled", "ALL", candidates)
    ]
    scopes.extend(
        ("per_site", str(site), group)
        for site, group in candidates.groupby("Site", sort=True)
    )

    statistic_records: list[dict[str, object]] = []
    ef_tables: list[pd.DataFrame] = []
    for scope, site, group in scopes:
        site_metadata: dict[str, object] = {
            "analysis_scope": scope,
            "Site": site,
            "Residue": "ALL" if scope == "pooled" else group["Residue"].iloc[0],
            "ResNum": "ALL" if scope == "pooled" else group["ResNum"].iloc[0],
            "Chain": "ALL" if scope == "pooled" else group["Chain"].iloc[0],
        }
        for feature in features:
            keep_cols = ["Name", "Warhead", feature, "is_hit"]
            if "electrophile_smiles" in group.columns:
                keep_cols.insert(2, "electrophile_smiles")
            if "Site" in group.columns:
                keep_cols.append("Site")
            valid = group.loc[
                group[feature].notna(),
                [c for c in keep_cols if c in group.columns],
            ].copy()
            hit_values = valid.loc[valid["is_hit"] == 1, feature]
            non_hit_values = valid.loc[valid["is_hit"] == 0, feature]
            oriented = (
                valid[feature]
                if directions[feature] == "benefit"
                else -valid[feature]
            )
            has_two_classes = valid["is_hit"].nunique() == 2
            auc = (
                float(roc_auc_score(valid["is_hit"], oriented))
                if has_two_classes
                else np.nan
            )
            pr_auc = (
                float(average_precision_score(valid["is_hit"], oriented))
                if has_two_classes
                else np.nan
            )
            statistic_records.append(
                {
                    **site_metadata,
                    "feature": feature,
                    "direction": directions[feature],
                    "n_rows": len(valid),
                    "n_hit_rows": int((valid["is_hit"] == 1).sum()),
                    "n_non_hit_rows": int((valid["is_hit"] == 0).sum()),
                    "hit_mean": float(hit_values.mean()) if len(hit_values) else np.nan,
                    "non_hit_mean": (
                        float(non_hit_values.mean()) if len(non_hit_values) else np.nan
                    ),
                    "hit_median": (
                        float(hit_values.median()) if len(hit_values) else np.nan
                    ),
                    "non_hit_median": (
                        float(non_hit_values.median())
                        if len(non_hit_values)
                        else np.nan
                    ),
                    "roc_auc_oriented": auc,
                    "pr_auc_oriented": pr_auc,
                }
            )
            if valid.empty:
                continue
            feature_ranking = valid.assign(_oriented_score=oriented).sort_values(
                "_oriented_score", ascending=False
            )
            if best_warhead:
                feature_ranking = keep_best_warhead_per_electrophile(
                    feature_ranking,
                    score_column="_oriented_score",
                    rank_column=None,
                    higher_is_better=True,
                )
            feature_ranking["overall_rank"] = np.arange(
                1, len(feature_ranking) + 1
            )
            if best_warhead and multiple_hits:
                feat_level = f"single_feature_best_warhead_multi_hit:{feature}"
            elif best_warhead:
                feat_level = f"single_feature_best_warhead:{feature}"
            elif multiple_hits:
                feat_level = f"single_feature_multi_hit:{feature}"
            else:
                feat_level = f"single_feature_all_candidates_any_hit:{feature}"
            ef_tables.append(
                enrichment_any_candidate_per_ligand(
                    feature_ranking,
                    feat_level,
                    rank_column="overall_rank",
                    metadata={
                        **site_metadata,
                        "feature": feature,
                        "direction": directions[feature],
                    },
                    multiple_hits=multiple_hits,
                )
            )

    statistics = pd.DataFrame(statistic_records)
    enrichment = (
        pd.concat(ef_tables, ignore_index=True) if ef_tables else pd.DataFrame()
    )
    return statistics, enrichment


def collapse_pooled_to_ligand_ranking(
    pooled_ranking: pd.DataFrame,
    *,
    multiple_hits: bool = False,
) -> pd.DataFrame:
    """
    Full ligand-level rank list from pooled pair ranking (hits and non-hits).

    Default: one row per Name (best overall_rank / topsis_score).
    --multiple-hits: one row per Name × Warhead × electrophile_smiles.
    """
    if pooled_ranking.empty:
        return pooled_ranking.copy()
    if "overall_rank" not in pooled_ranking.columns:
        return pooled_ranking.copy()

    work = pooled_ranking.copy()
    work["_ligand_entity"] = assign_hit_entity_id(work, multiple_hits)
    sort_cols = ["overall_rank"]
    if "topsis_score" in work.columns:
        sort_cols = ["overall_rank", "topsis_score"]
    ascending = [True] + ([False] if "topsis_score" in work.columns else [])
    best = (
        work.sort_values(sort_cols, ascending=ascending)
        .groupby("_ligand_entity", sort=False, as_index=False)
        .first()
    )
    best = best.sort_values(
        ["overall_rank", "Name"],
        ascending=[True, True],
    ).reset_index(drop=True)
    best.insert(0, "ligand_rank", np.arange(1, len(best) + 1))
    drop_cols = [c for c in ("_ligand_entity", "_hit_entity") if c in best.columns]
    if drop_cols:
        best = best.drop(columns=drop_cols)
    return best


def export_results(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    per_site_ranking: pd.DataFrame,
    pooled_ranking: pd.DataFrame,
    topsis_enrichment: pd.DataFrame,
    feature_statistics: pd.DataFrame,
    feature_enrichment: pd.DataFrame,
    missing_counts: pd.DataFrame,
    unmatched_hits: list[dict[str, str]],
    features: list[str],
    weight_records: list[dict[str, object]],
) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_site_ranking.to_csv(
        out_dir / "topsis_per_site_ranking.csv", index=False
    )
    per_site_ranking.loc[per_site_ranking["is_hit"] == 1].to_csv(
        out_dir / "hit_per_site_ranks.csv", index=False
    )
    pooled_ranking.to_csv(
        out_dir / "topsis_pooled_pair_ranking.csv", index=False
    )
    pooled_ranking.loc[pooled_ranking["is_hit"] == 1].to_csv(
        out_dir / "hit_pooled_pair_ranks.csv", index=False
    )
    ligand_ranking = collapse_pooled_to_ligand_ranking(
        pooled_ranking,
        multiple_hits=bool(getattr(args, "multiple_hits", False)),
    )
    ligand_ranking.to_csv(
        out_dir / "full_pooled_ligand_ranking.csv", index=False
    )
    topsis_enrichment.to_csv(out_dir / "enrichment_factors.csv", index=False)
    feature_statistics.to_csv(out_dir / "feature_statistics.csv", index=False)
    feature_enrichment.to_csv(
        out_dir / "feature_enrichment_factors.csv", index=False
    )

    audit_cols = [
        column
        for column in [
            "Name",
            "Residue",
            "ResNum",
            "Chain",
            "Warhead",
            "query_group",
            "_analysis_name",
            "_analysis_warhead",
            "is_hit",
        ]
        if column in selected.columns
    ] + features
    selected[audit_cols].to_csv(
        out_dir / "matched_site_warhead_rows.csv", index=False
    )
    candidates.to_csv(out_dir / "site_name_warhead_candidates.csv", index=False)
    missing_counts.to_csv(out_dir / "missing_feature_counts.csv", index=False)
    unmatched_df = pd.DataFrame(
        unmatched_hits, columns=["Hit", "Reason", "Detail"]
    )
    unmatched_df.to_csv(out_dir / "unmatched_hit_names.csv", index=False)

    direction = {
        feature: "benefit" for feature in args.benefit
    } | {feature: "cost" for feature in args.cost}
    criteria_df = pd.DataFrame(weight_records)
    if not criteria_df.empty:
        criteria_df["direction"] = criteria_df["feature"].map(direction)
        # Prefer pooled rows in topsis_criteria.csv when present; else all.
        pooled_mask = criteria_df["analysis_scope"] == "pooled"
        summary = (
            criteria_df.loc[pooled_mask].copy()
            if pooled_mask.any()
            else criteria_df.copy()
        )
        summary[
            ["feature", "direction", "normalized_weight", "weight_method"]
        ].to_csv(out_dir / "topsis_criteria.csv", index=False)
        criteria_df.to_csv(out_dir / "topsis_weights_by_scope.csv", index=False)

    print("\nTOPSIS virtual screening complete")
    print(f"  Selected residue sites:    {candidates['Site'].nunique():,}")
    print(f"  Site × ligand × warhead:   {len(candidates):,}")
    print(f"  Unique ligand Names:       {candidates['Name'].nunique():,}")
    print(
        "  Positive Names matched:    "
        f"{candidates.loc[candidates['is_hit'] == 1, 'Name'].nunique():,}"
    )
    print(f"  Unmatched positives:       {len(unmatched_hits):,}")
    if unmatched_hits:
        reason_counts = unmatched_df["Reason"].value_counts()
        for reason, count in reason_counts.items():
            print(f"    {reason}: {count:,}")
    print("  Full rankings (all candidates):")
    print(
        f"    Per-site pairs:          "
        f"{out_dir / 'topsis_per_site_ranking.csv'}"
    )
    print(
        f"    Pooled pairs:            "
        f"{out_dir / 'topsis_pooled_pair_ranking.csv'}"
    )
    print(
        f"    Pooled ligands:          "
        f"{out_dir / 'full_pooled_ligand_ranking.csv'}"
    )
    print("  Hit-only subsets:")
    print(
        f"    Per-site:                "
        f"{out_dir / 'hit_per_site_ranks.csv'}"
    )
    print(
        f"    Pooled pairs:            "
        f"{out_dir / 'hit_pooled_pair_ranks.csv'}"
    )
    print(f"  TOPSIS enrichment:         {out_dir / 'enrichment_factors.csv'}")
    print(
        f"  Feature enrichment:        "
        f"{out_dir / 'feature_enrichment_factors.csv'}"
    )
    if getattr(args, "entropy_weights", False):
        print(
            f"  Entropy weights by scope:  "
            f"{out_dir / 'topsis_weights_by_scope.csv'}"
        )


def main() -> None:
    args = parse_args()
    features, weights = validate_feature_args(args)
    hit_spec = load_hits(
        args.hits,
        args.name_hit_column,
        args.smiles_hit_column,
        args.warhead_type_column,
        name_miss_column=args.name_miss_column,
        smiles_miss_column=args.smiles_miss_column,
    )
    if hit_spec.hit_names:
        print(f"[INFO] Hit Names loaded:   {len(hit_spec.hit_names):,}")
    if hit_spec.hit_smiles:
        print(f"[INFO] Hit SMILES loaded:  {len(hit_spec.hit_smiles):,}")
    n_hit_wh = sum(
        1
        for allowed in (
            *hit_spec.hit_smiles_warheads.values(),
            *hit_spec.hit_name_warheads.values(),
        )
        if allowed is not None
    )
    if n_hit_wh:
        print(
            f"[INFO] Per-hit warhead_type restrictions: {n_hit_wh:,} "
            "ligand entrie(s)"
        )
    if hit_spec.miss_smiles or hit_spec.miss_names:
        print(
            "[INFO] Miss entries with optional warhead_type: "
            f"{len(hit_spec.miss_smiles) + len(hit_spec.miss_names):,}"
        )
    if args.multiple_hits:
        print(
            "[INFO] Hit counting: Name × Warhead × SMILES "
            "(--multiple-hits)"
        )
    else:
        print("[INFO] Hit counting: one hit per ligand Name")
    if args.best_warhead:
        print(
            "[INFO] --best-warhead: keep highest TOPSIS-scoring warhead "
            "per electrophile_smiles × site"
        )
    if args.entropy_weights:
        print(
            "[INFO] --entropy-weights: criterion weights from Shannon "
            "entropy of each feature in the candidate pool "
            "(computed separately per site and for pooled)"
        )
    selected, unmatched_hits = load_selected_rows(
        args, features, hit_spec
    )
    candidates = build_site_candidates(selected, features)
    benefit_features = list(dict.fromkeys(args.benefit))
    cost_features = list(dict.fromkeys(args.cost))

    per_site_ranking, per_site_ef, per_site_missing, site_weight_records = (
        rank_each_site(
            candidates,
            features,
            benefit_features,
            cost_features,
            weights,
            args.missing_policy,
            multiple_hits=args.multiple_hits,
            best_warhead=args.best_warhead,
            entropy_weights_flag=args.entropy_weights,
        )
    )
    pooled_candidates, pooled_missing = handle_missing(
        candidates, features, args.missing_policy
    )
    pooled_missing["analysis_scope"] = "pooled"
    pooled_missing["Site"] = "ALL"
    pooled_missing["Residue"] = "ALL"
    pooled_missing["ResNum"] = "ALL"
    pooled_missing["Chain"] = "ALL"
    missing_counts = pd.concat(
        [per_site_missing, pooled_missing], ignore_index=True
    )
    pooled_weights = resolve_topsis_weights(
        pooled_candidates,
        features,
        weights,
        use_entropy=args.entropy_weights,
        label="pooled",
    )
    weight_records = list(site_weight_records)
    for feature, weight in zip(features, pooled_weights):
        weight_records.append(
            {
                "analysis_scope": "pooled",
                "Site": "ALL",
                "Residue": "ALL",
                "ResNum": "ALL",
                "Chain": "ALL",
                "feature": feature,
                "normalized_weight": float(weight),
                "weight_method": (
                    "entropy" if args.entropy_weights else "fixed"
                ),
            }
        )
    pooled_ranking = topsis_rank(
        pooled_candidates,
        benefit_features,
        cost_features,
        pooled_weights,
    )
    if args.best_warhead:
        pooled_ranking = keep_best_warhead_per_electrophile(
            pooled_ranking,
            score_column="topsis_score",
            rank_column="overall_rank",
            higher_is_better=True,
        )
    if args.best_warhead and args.multiple_hits:
        pooled_level = "topsis_pooled_best_warhead_multi_hit"
    elif args.best_warhead:
        pooled_level = "topsis_pooled_best_warhead"
    elif args.multiple_hits:
        pooled_level = "topsis_pooled_all_pairs_multi_hit"
    else:
        pooled_level = "topsis_pooled_all_pairs_any_hit"
    pooled_ef = enrichment_any_candidate_per_ligand(
        pooled_ranking,
        pooled_level,
        rank_column="overall_rank",
        metadata={
            "analysis_scope": "pooled",
            "Site": "ALL",
            "Residue": "ALL",
            "ResNum": "ALL",
            "Chain": "ALL",
        },
        multiple_hits=args.multiple_hits,
    )
    topsis_enrichment = pd.concat([per_site_ef, pooled_ef], ignore_index=True)
    feature_statistics, feature_enrichment = feature_statistics_and_enrichment(
        candidates,
        features,
        benefit_features,
        multiple_hits=args.multiple_hits,
        best_warhead=args.best_warhead,
    )
    export_results(
        args,
        selected,
        candidates,
        per_site_ranking,
        pooled_ranking,
        topsis_enrichment,
        feature_statistics,
        feature_enrichment,
        missing_counts,
        unmatched_hits,
        features,
        weight_records,
    )


if __name__ == "__main__":
    main()
