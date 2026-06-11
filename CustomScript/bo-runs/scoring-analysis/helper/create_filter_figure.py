#!/usr/bin/env python3
"""Create a four-panel filter / ranker evaluation figure.

Panel A: SASA (RSA) threshold — recall & specificity vs threshold.
Panel B: Deprotonation probability — recall & specificity vs threshold.
Panel C: LightGBM Hit@3 by residue type (rankable-only).
Panel D: LightGBM Hit@3 by warhead class (rankable-only, N_Rankable > 100).

Usage:
    python create_filter_figure.py \\
        --sasa-sweep ../sasa_sweep.csv \\
        --deprot-sweep ../deprot_sweep.csv \\
        --breakdown ../lgbm_reactivity_breakdown.csv \\
        [--output filter_figure.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Warhead tag → display group (aligned with create_warhead_diversity.py).
WARHEAD_GROUPS: dict[str, list[str]] = {
    "Michael acceptors": [
        "alpha-beta unsaturated carbonyl (michael acceptor)",
        "ketoamide (michael acceptor)",
        "acrylamide warhead",
        "propiolamide warhead",
        "vinyl sulfone",
        "vinylsulfonamide",
        "vinyl sulfonate ester",
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
        "alpha-halo carbonyl",
        "heteroaryl halide (snar)",
        "nitro-activated aryl halide (snar)",
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
    "Nitrile": ["nitrile (electrophilic)", "aromatic nitrile (cathepsin-like)", "cyanamide"],
    "Disulfide": ["disulfide"],
    "Other": ["urea carbonyl", "isothiocyanate", "unclassified"],
}

_WARHEAD_TO_GROUP: dict[str, str] = {
    member.lower(): group for group, members in WARHEAD_GROUPS.items() for member in members
}

WARHEAD_GROUP_ORDER = list(WARHEAD_GROUPS.keys())

RESIDUE_ORDER = ("CYS", "HIS", "LYS", "SER", "THR", "TYR")
RESIDUE_LABELS = {
    "CYS": "Cys",
    "HIS": "His",
    "LYS": "Lys",
    "SER": "Ser",
    "THR": "Thr",
    "TYR": "Tyr",
}

RECALL_COLOR = "#4C72B0"
SPEC_COLOR = "#DD8452"
MODEL_COLOR = "#55A868"
RANDOM_COLOR = "#C0C0C0"
CHOSEN_LINE_COLOR = "#333333"


def load_sweep(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"threshold", "recall", "specificity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}")
    return df.sort_values("threshold").reset_index(drop=True)


def interpolate_at_threshold(sweep: pd.DataFrame, chosen: float) -> tuple[float, float]:
    thr = sweep["threshold"].to_numpy()
    recall = sweep["recall"].to_numpy()
    spec = sweep["specificity"].to_numpy()
    return float(np.interp(chosen, thr, recall)), float(np.interp(chosen, thr, spec))


def plot_threshold_curve(
    ax: plt.Axes,
    sweep: pd.DataFrame,
    *,
    panel_label: str,
    title: str,
    xlabel: str,
    chosen_threshold: float,
) -> None:
    thr = sweep["threshold"]
    recall_pct = sweep["recall"] * 100.0
    spec_pct = sweep["specificity"] * 100.0

    ax.plot(thr, recall_pct, color=RECALL_COLOR, linewidth=2.0, label="Recall")
    ax.plot(thr, spec_pct, color=SPEC_COLOR, linewidth=2.0, label="Specificity")

    recall_at, _ = interpolate_at_threshold(sweep, chosen_threshold)
    y_pct = recall_at * 100.0
    ax.axvline(
        chosen_threshold,
        color=CHOSEN_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
    )

    # Place high-recall annotations down-right so they stay inside the axes.
    if y_pct >= 90:
        xytext = (58, -52)
    elif y_pct >= 70:
        xytext = (42, -28)
    else:
        xytext = (14, 16)

    ax.annotate(
        f"threshold = {chosen_threshold:g}\nrecall = {y_pct:.1f}%",
        xy=(chosen_threshold, y_pct),
        xytext=xytext,
        textcoords="offset points",
        fontsize=8,
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=CHOSEN_LINE_COLOR, lw=0.9),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.set_title(f"{panel_label}  {title}", loc="left", fontweight="bold", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def map_warhead_group(category: str) -> str:
    tags = [t.strip().lower() for t in str(category).split(",") if t.strip()]
    for tag in tags:
        if tag in _WARHEAD_TO_GROUP:
            return _WARHEAD_TO_GROUP[tag]
    return "Other"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def load_breakdown_residues(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sub = df[df["Group"] == "Residue_Type"].copy()
    sub["Category"] = sub["Category"].astype(str).str.upper()
    sub = sub[sub["Category"].isin(RESIDUE_ORDER)].copy()
    sub["Label"] = sub["Category"].map(RESIDUE_LABELS)
    sub = sub.set_index("Category").reindex(RESIDUE_ORDER).reset_index()
    sub["Label"] = sub["Category"].map(RESIDUE_LABELS)
    return sub


def load_breakdown_warheads(csv_path: Path, min_rankable: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    wh = df[df["Group"] == "Warhead"].copy()
    wh = wh[wh["N_Rankable"].fillna(0) > 0].copy()
    wh["Warhead_Group"] = wh["Category"].map(map_warhead_group)

    rows = []
    for group, grp in wh.groupby("Warhead_Group"):
        n_rankable = int(grp["N_Rankable"].sum())
        rows.append({
            "Warhead_Group": group,
            "N_Rankable": n_rankable,
            "Hit@3_Rankable_Pct": weighted_mean(grp["Hit@3_Rankable_Pct"], grp["N_Rankable"]),
            "Random_Hit@3_Rankable_Pct": weighted_mean(
                grp["Random_Hit@3_Rankable_Pct"], grp["N_Rankable"]
            ),
        })

    out = pd.DataFrame(rows)
    out = out[out["N_Rankable"] > min_rankable].copy()

    order = [g for g in WARHEAD_GROUP_ORDER if g in set(out["Warhead_Group"])]
    extras = [g for g in out["Warhead_Group"] if g not in order]
    out["Warhead_Group"] = pd.Categorical(
        out["Warhead_Group"], categories=order + extras, ordered=True
    )
    return out.sort_values("N_Rankable", ascending=False).reset_index(drop=True)


def plot_hit3_bars(
    ax: plt.Axes,
    labels: list[str],
    model_vals: list[float],
    random_vals: list[float],
    n_rankable: list[int],
    *,
    panel_label: str,
    title: str,
    subtitle: str,
    rotate_x: float = 0,
) -> None:
    x = np.arange(len(labels))
    width = 0.36

    model_bars = ax.bar(
        x - width / 2,
        model_vals,
        width,
        label="LightGBM ranker",
        color=MODEL_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    random_bars = ax.bar(
        x + width / 2,
        random_vals,
        width,
        label="Random baseline",
        color=RANDOM_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )

    ymax = max(max(model_vals, default=0), max(random_vals, default=0))
    for bar, val in zip(model_bars, model_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=MODEL_COLOR,
        )
    for bar, val in zip(random_bars, random_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666666",
        )

    ax.set_xticks(x)
    tick_labels = [f"{lab}\n(n={n:,})" for lab, n in zip(labels, n_rankable)]
    ax.set_xticklabels(
        tick_labels,
        fontsize=8,
        rotation=rotate_x,
        ha="right" if rotate_x else "center",
    )
    if rotate_x:
        ax.tick_params(axis="x", pad=8)
    if len(labels) >= 6:
        ax.margins(x=0.06)
    ax.set_ylabel("Top 3 rankable hit (%)")
    ax.set_ylim(0, min(ymax + 18, 115))
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_title(f"{panel_label}  {title}", loc="left", fontweight="bold", fontsize=11)
    ax.text(0.0, -0.22, subtitle, transform=ax.transAxes, fontsize=7, color="#555555", va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figure(
    sasa_sweep_path: Path,
    deprot_sweep_path: Path,
    breakdown_path: Path,
    output_path: Path | None = None,
    *,
    sasa_threshold: float = 0.12,
    deprot_threshold: float = 0.14,
    min_warhead_rankable: int = 100,
) -> Path:
    sasa_sweep = load_sweep(sasa_sweep_path)
    deprot_sweep = load_sweep(deprot_sweep_path)
    residues = load_breakdown_residues(breakdown_path)
    warheads = load_breakdown_warheads(breakdown_path, min_warhead_rankable)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.5, 9.5),
        gridspec_kw={"hspace": 0.42, "wspace": 0.28},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    plot_threshold_curve(
        ax_a,
        sasa_sweep,
        panel_label="A",
        title="SASA filter operating point",
        xlabel="Relative side-chain SASA threshold",
        chosen_threshold=sasa_threshold,
    )
    plot_threshold_curve(
        ax_b,
        deprot_sweep,
        panel_label="B",
        title="Deprotonation filter operating point",
        xlabel="Deprotonation probability threshold",
        chosen_threshold=deprot_threshold,
    )

    plot_hit3_bars(
        ax_c,
        labels=residues["Label"].tolist(),
        model_vals=residues["Hit@3_Rankable_Pct"].tolist(),
        random_vals=residues["Random_Hit@3_Rankable_Pct"].tolist(),
        n_rankable=residues["N_Rankable"].astype(int).tolist(),
        panel_label="C",
        title="Top 3 hit by residue type",
        subtitle="Rankable targets only (Top 3 Rankable Hit Pct vs random baseline)",
    )
    plot_hit3_bars(
        ax_d,
        labels=warheads["Warhead_Group"].tolist(),
        model_vals=warheads["Hit@3_Rankable_Pct"].tolist(),
        random_vals=warheads["Random_Hit@3_Rankable_Pct"].tolist(),
        n_rankable=warheads["N_Rankable"].astype(int).tolist(),
        panel_label="D",
        title="Top 3 hit by warhead class",
        subtitle=f"Rankable targets only > {min_warhead_rankable}",
        rotate_x=18,
    )

    fig.subplots_adjust(left=0.07, right=0.96, top=0.96, bottom=0.11)

    if output_path is None:
        output_path = breakdown_path.with_name("filter_figure.png")
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create four-panel filter / LightGBM ranker evaluation figure."
    )
    parser.add_argument(
        "--sasa-sweep",
        type=Path,
        required=True,
        help="SASA threshold sweep CSV (from feature_analysis.py --export-sweep).",
    )
    parser.add_argument(
        "--deprot-sweep",
        type=Path,
        required=True,
        help="Deprotonation threshold sweep CSV.",
    )
    parser.add_argument(
        "--breakdown",
        type=Path,
        required=True,
        help="LightGBM reactivity breakdown CSV (residue + warhead groups).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output image path (default: filter_figure.png next to breakdown CSV).",
    )
    parser.add_argument(
        "--sasa-threshold",
        type=float,
        default=0.12,
        help="Chosen RSA threshold for Panel A annotation (default: 0.12).",
    )
    parser.add_argument(
        "--deprot-threshold",
        type=float,
        default=0.14,
        help="Chosen deprotonation threshold for Panel B annotation (default: 0.14).",
    )
    parser.add_argument(
        "--min-warhead-rankable",
        type=int,
        default=100,
        help="Panel D: minimum N_Rankable per warhead group (default: 100).",
    )
    args = parser.parse_args()

    for path in (args.sasa_sweep, args.deprot_sweep, args.breakdown):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")

    out = create_figure(
        args.sasa_sweep,
        args.deprot_sweep,
        args.breakdown,
        args.output,
        sasa_threshold=args.sasa_threshold,
        deprot_threshold=args.deprot_threshold,
        min_warhead_rankable=args.min_warhead_rankable,
    )
    print(f"[INFO] Figure written to: {out}")


if __name__ == "__main__":
    main()
