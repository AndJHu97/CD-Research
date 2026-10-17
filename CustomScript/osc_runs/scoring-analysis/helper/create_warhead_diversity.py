#!/usr/bin/env python3
"""Create warhead-diversity figure from a baby_frank detail CSV.

Panel A: Grouped warhead specificity (pie) for matched nucleophilic residues.
Panel B: Nucleophilic-residue makeup (pie) for the same filtered rows.

Filters: Is_Mismatch == False and Residue in {CYS, LYS, THR, TYR, HIS, SER}.

Usage:
    python create_warhead_diversity.py detail_baby_frank_top3.csv [--output figure.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NUCLEOPHILIC_RESIDUES = frozenset({"CYS", "LYS", "THR", "TYR", "HIS", "SER"})
RESIDUE_LABELS = {
    "CYS": "Cys",
    "LYS": "Lys",
    "SER": "Ser",
    "THR": "Thr",
    "HIS": "His",
    "TYR": "Tyr",
}

# Raw Warhead labels (lowercase) → display group for Panel A.
WARHEAD_GROUPS: dict[str, list[str]] = {
    "Michael acceptors": [
        "alpha-beta unsaturated carbonyl (michael acceptor)",
        "ketoamide (michael acceptor)",
        "acrylamide warhead",
        "vinyl sulfone",
        "vinylsulfonamide",
    ],
    "Aldehyde": ["aldehyde"],
    "Ketone": ["ketone (reactive)", "activated ketone", "fluoromethyl ketone"],
    "Phosphorus": [
        "phosphonyl fluoride (phosphonofluoridate)",
        "phosphonate (general)",
    ],
    "Boronic acid": ["boronic acid"],
    "Beta-lactam/lactone": ["beta-lactam", "beta-lactone"],
    "Halides": [
        "alkyl chloride",
        "alkyl halide (good lg)",
        "heteroaryl halide (snar)",
        "sulfonyl chloride",
    ],
    "Epoxides": [
        "epoxide",
        "alpha-beta epoxyketone (epoxide)",
        "alpha-beta epoxyketone (carbonyl)",
        "aziridine",
    ],
    "Sulfonyl": [
        "sulfonyl fluoride",
        "sulfonyl-sulfide",
        "sulfonamide (activated)",
    ],
    "Esters/acyl transfer": [
        "activated ester",
        "phenyl ester",
        "thioester",
        "carbamate (amide-like)",
    ],
    "Nitrile": ["nitrile (electrophilic)", "aromatic nitrile (cathepsin-like)"],
    "Disulfide": ["disulfide"],
    "Other": ["urea carbonyl", "isothiocyanate"],
}

_WARHEAD_TO_GROUP: dict[str, str] = {
    member.lower(): group for group, members in WARHEAD_GROUPS.items() for member in members
}

# Stable pie order (largest groups first after sorting by count).
WARHEAD_GROUP_ORDER = list(WARHEAD_GROUPS.keys())

PIE_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#BCBD22", "#17BECF",
]

RESIDUE_COLORS = {
    "CYS": "#4C72B0",
    "LYS": "#937860",
    "SER": "#DD8452",
    "THR": "#55A868",
    "HIS": "#C44E52",
    "TYR": "#8172B3",
}


def load_detail(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def filter_matched_nucleophiles(df: pd.DataFrame) -> pd.DataFrame:
    is_mismatch = df["Is_Mismatch"].astype(str).str.lower().isin({"false", "0"})
    residue = df["Residue"].astype(str).str.upper().isin(NUCLEOPHILIC_RESIDUES)
    return df.loc[is_mismatch & residue].copy()


def map_warhead_group(raw_warhead: str) -> str:
    key = str(raw_warhead).strip().lower()
    return _WARHEAD_TO_GROUP.get(key, "Other")


def warhead_group_counts(df: pd.DataFrame) -> pd.Series:
    groups = df["Warhead"].map(map_warhead_group)
    counts = groups.value_counts()
    ordered = [g for g in WARHEAD_GROUP_ORDER if g in counts.index]
    extras = [g for g in counts.index if g not in ordered]
    return counts.reindex(ordered + extras)


def residue_counts(df: pd.DataFrame) -> pd.Series:
    counts = df["Residue"].astype(str).str.upper().value_counts()
    order = [r for r in ("CYS", "SER", "THR", "LYS", "HIS", "TYR") if r in counts.index]
    return counts.reindex(order)


def _pie_autopct(values: pd.Series):
    total = values.sum()

    def fmt(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        if pct < 4.0:
            return ""
        return f"{pct:.1f}%\n(n={count})"

    return fmt


def plot_warhead_pie(ax: plt.Axes, counts: pd.Series) -> None:
    colors = PIE_COLORS[: len(counts)]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct=_pie_autopct(counts),
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        labeldistance=1.08,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        textprops={"fontsize": 8},
    )
    for t in autotexts:
        t.set_fontsize(7)
        t.set_fontweight("bold")
    ax.set_title(
        f"A  Warhead specificity (n={counts.sum():,})",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    subtitle = (
        "Matched warheads only (Is_Mismatch = False); "
        "Cys, Lys, Ser, Thr, His, Tyr nucleophiles"
    )
    ax.text(0.0, -0.08, subtitle, transform=ax.transAxes, fontsize=7, color="#555555", va="top")


def plot_residue_pie(ax: plt.Axes, counts: pd.Series) -> None:
    labels = [RESIDUE_LABELS[r] for r in counts.index]
    colors = [RESIDUE_COLORS[r] for r in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct=_pie_autopct(counts),
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        labeldistance=1.08,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax.set_title(
        f"B  Nucleophilic residue makeup (n={counts.sum():,})",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    subtitle = "Same filter as Panel A: matched warheads, seven nucleophilic residues"
    ax.text(0.0, -0.08, subtitle, transform=ax.transAxes, fontsize=7, color="#555555", va="top")


def create_figure(csv_path: Path, output_path: Path | None = None) -> Path:
    df = load_detail(csv_path)
    sub = filter_matched_nucleophiles(df)

    if sub.empty:
        raise ValueError("No rows after filtering (Is_Mismatch=False, seven nucleophilic residues).")

    wh_counts = warhead_group_counts(sub)
    res_counts = residue_counts(sub)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.35}
    )

    plot_warhead_pie(ax_a, wh_counts)
    plot_residue_pie(ax_b, res_counts)

    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_warhead_diversity.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create warhead-diversity pie charts from a baby_frank detail CSV."
    )
    parser.add_argument(
        "detail_csv",
        type=Path,
        help="Path to detail CSV (e.g. detail_baby_frank_top3.csv).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output image path (default: <csv_stem>_warhead_diversity.png).",
    )
    args = parser.parse_args()

    if not args.detail_csv.exists():
        raise SystemExit(f"File not found: {args.detail_csv}")

    out = create_figure(args.detail_csv, args.output)
    print(f"[INFO] Figure written to: {out}")


if __name__ == "__main__":
    main()
