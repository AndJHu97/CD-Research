#!/usr/bin/env python3
"""
Make_Figures.py

Publication-style figures comparing the deprotonation model against PROPKA and a
trivial "reference pKa" baseline. Every figure is saved as PNG and SVG, with thick
axes and editable (non-outlined) SVG text.

Inputs
------
--oof        <prefix>_oof_predictions_with_propka.csv   (from Compare_DM_With_Propka.py)
             Needs: label, prediction, ref_pka, resname, pdb, propka_deprot_probability.
             Optional: fold, expt_pka, propka_pka, row_id.
--exclude-propka-placeholder
             Optional. Drop rows where PROPKA reports its 99.99 "not applicable" pKa (mostly
             disulfide-bonded Cys) from all method comparisons; they are saved to
             excluded_propka_placeholder_rows.csv.
--ablation   Optional LABEL=PATH pairs; each PATH is a plain <prefix>_oof_predictions.csv
             from a different training run (e.g. --no-apbs), used for the ablation figure.

Figures
-------
 01 scatter_probability     predicted vs experimental P(deprot), model | PROPKA
 02 scatter_pka             predicted vs experimental pKa, model | PROPKA
                            (model probabilities clipped to [--clip-eps, 1-eps] first)
 03 roc_pr                  ROC and precision-recall: model, PROPKA, baseline
 04 error_by_residue        absolute error per residue type
 05 paired_error_difference per-residue paired error difference, PDB-clustered bootstrap CI
 06 per_fold_metrics        R2 / AUC / MCC per fold, paired across methods
 07 confusion_matrices      at the 0.5 threshold
 08 calibration             reliability curves with prediction histograms
 09 ablation                feature-set ablation (needs --ablation)
 10 class_balance           fraction deprotonated per residue type
 11 flip_roc_pr             ROC / PR for detecting residues whose true state differs from
                            the ref-pKa baseline state ("flipped" residues)
 12 flip_recall_false_flips recall on flipped residues and false-flip rate on the rest,
                            model vs PROPKA, PDB-clustered bootstrap CIs

Flip definition (figures 11-12): a residue is "flipped" when the ref-pKa baseline and the
experimental label fall on opposite sides of 0.5 (baseline is wrong by construction).
A method "flips" a residue when its own state (P >= 0.5) differs from the baseline state.

Also writes figure_metrics_summary.csv and flip_metrics_summary.csv.
--exclude-propka-placeholder is for accounting for the 99 value (these don't work)
Usage
-----
    python helper/figure/Create_Deprotonation_Figures.py --oof final_data_and_draft/deprotonation/deprot_xgb_noapbs_oof_predictions_with_propka.csv --clip-eps 0.001 --exclude-propka-placeholder --out-dir final_data_and_draft/deprotonation/figures
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)

AXIS_LW = 2.5
# PROPKA writes 99.99 as a "not applicable" placeholder (e.g. cysteines in disulfide bonds).
PROPKA_PLACEHOLDER_MIN = 99.0
METHOD_COLORS = {
    "Model": "#1f77b4",
    "PROPKA": "#d62728",
    "Ref pKa baseline": "#7f7f7f",
}


# --------------------------------------------------------------------------- #
# Style / saving
# --------------------------------------------------------------------------- #
def set_style() -> None:
    plt.rcParams.update({
        "svg.fonttype": "none",          # keep text as text in SVG
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "axes.linewidth": AXIS_LW,
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": AXIS_LW,
        "ytick.major.width": AXIS_LW,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 2.5,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "figure.dpi": 100,
    })


def save_fig(fig, out_dir: str, name: str, dpi: int) -> None:
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {name}.png / {name}.svg")


def style_boxplot(bp, color: str) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(2)
    for key in ("whiskers", "caps"):
        for ln in bp[key]:
            ln.set_linewidth(2)
    for ln in bp["medians"]:
        ln.set_color("black")
        ln.set_linewidth(3)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def pka_to_prob(pka, ph: float):
    return 1.0 / (1.0 + 10 ** (np.asarray(pka, dtype=float) - ph))


def prob_to_pka(p, ph: float):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = (p > 0) & (p < 1)
    out[ok] = ph + np.log10((1.0 - p[ok]) / p[ok])
    return out


def read_oof(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_id" in df.columns:
        df = df.drop_duplicates("row_id", keep="first")
    return df.reset_index(drop=True)


def prepare_full(path: str, ph: float) -> pd.DataFrame:
    """All OOF rows with a prediction (no PROPKA requirement)."""
    df = read_oof(path)
    need = ["label", "prediction", "ref_pka", "resname", "pdb"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    df["resname"] = df["resname"].astype(str).str.upper().str.strip()
    df["baseline"] = pka_to_prob(df["ref_pka"], ph)
    if "expt_pka" not in df.columns:
        df["expt_pka"] = prob_to_pka(df["label"], ph)
    return df.dropna(subset=["label", "prediction", "baseline"]).reset_index(drop=True)


def prepare_propka_subset(full: pd.DataFrame, ph: float, exclude_placeholder: bool = False):
    """Rows where PROPKA has a value, so all methods are compared on identical rows.

    With exclude_placeholder, rows whose PROPKA pKa is the 99.99 "not applicable" placeholder
    are dropped for every method (model, PROPKA, baseline), so comparisons stay paired.
    Returns (subset, excluded_rows).
    """
    if "propka_deprot_probability" not in full.columns:
        raise SystemExit(
            "No propka_deprot_probability column. Pass the *_oof_predictions_with_propka.csv "
            "written by Compare_DM_With_Propka.py."
        )
    sub = full.dropna(subset=["propka_deprot_probability"]).reset_index(drop=True)
    if sub.empty:
        raise SystemExit("No rows have a PROPKA value; nothing to compare.")
    if "propka_pka" not in sub.columns:
        sub["propka_pka"] = prob_to_pka(sub["propka_deprot_probability"], ph)
    excluded = sub.iloc[0:0]
    if exclude_placeholder:
        mask = (sub["propka_pka"] >= PROPKA_PLACEHOLDER_MIN).to_numpy()
        excluded = sub[mask].reset_index(drop=True)
        sub = sub[~mask].reset_index(drop=True)
        if sub.empty:
            raise SystemExit("All PROPKA rows were excluded as placeholders; nothing to compare.")
    sub.attrs["n_propka_excluded"] = int(len(excluded))
    return sub, excluded


def get_scores(df: pd.DataFrame, include_propka: bool = True) -> Dict[str, np.ndarray]:
    scores = {"Model": df["prediction"].to_numpy(float)}
    if include_propka:
        scores["PROPKA"] = df["propka_deprot_probability"].to_numpy(float)
    scores["Ref pKa baseline"] = df["baseline"].to_numpy(float)
    return scores


def safe_r(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(y, s) -> Dict[str, float]:
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ybin = (y >= 0.5).astype(int)
    sbin = (s >= 0.5).astype(int)
    two_class = len(np.unique(ybin)) == 2
    return {
        "n": int(len(y)),
        "r2": float(r2_score(y, s)) if len(y) > 1 else np.nan,
        "mae": float(np.mean(np.abs(y - s))),
        "rmse": float(np.sqrt(np.mean((y - s) ** 2))),
        "pearson": safe_r(y, s),
        "auc": float(roc_auc_score(ybin, s)) if two_class else np.nan,
        "ap": float(average_precision_score(ybin, s)) if two_class else np.nan,
        "mcc": float(matthews_corrcoef(ybin, sbin)),
    }


def residue_colors(resnames) -> Dict[str, tuple]:
    names = sorted(set(resnames))
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    return {n: cmap(i % cmap.N) for i, n in enumerate(names)}


def pdb_bootstrap_ci(diff: np.ndarray, groups: np.ndarray, n_boot: int, seed: int):
    """Percentile CI of the mean difference, resampling whole PDB structures."""
    g = pd.DataFrame({"d": diff, "g": groups}).groupby("g")["d"].agg(["sum", "count"])
    sums = g["sum"].to_numpy()
    cnts = g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return np.percentile(boots, [2.5, 97.5])


# --------------------------------------------------------------------------- #
# Flip helpers
# --------------------------------------------------------------------------- #
def flip_setup(df: pd.DataFrame):
    """Return (label_state, baseline_state, flipped) as boolean arrays.

    flipped = the ref-pKa baseline and the experimental label are on opposite sides of 0.5.
    """
    label_state = df["label"].to_numpy(float) >= 0.5
    base_state = df["baseline"].to_numpy(float) >= 0.5
    flipped = label_state != base_state
    return label_state, base_state, flipped


def flip_score(s: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    """Probability a method assigns to the state OPPOSITE the baseline's state.

    Baseline deprotonated -> flip means protonated -> score = 1 - P.
    Baseline protonated   -> flip means deprotonated -> score = P.
    """
    s = np.asarray(s, float)
    return np.where(base_state, 1.0 - s, s)


def flip_methods(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Methods evaluated on flips. The baseline never flips itself, so it is excluded."""
    return {k: v for k, v in get_scores(df).items() if k != "Ref pKa baseline"}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_scatter_probability(df, out_dir, dpi):
    colors = residue_colors(df["resname"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    x = df["label"].to_numpy(float)
    for ax, (name, y) in zip(axes, [("Model", df["prediction"]), ("PROPKA", df["propka_deprot_probability"])]):
        y = y.to_numpy(float)
        for res, c in colors.items():
            m = (df["resname"] == res).to_numpy()
            ax.scatter(x[m], y[m], s=30, color=c, alpha=0.7, edgecolors="none", label=res)
        ax.plot([0, 1], [0, 1], "k--", lw=2, zorder=0)
        mt = compute_metrics(x, y)
        ax.text(0.04, 0.96, f"n = {mt['n']}\nr = {mt['pearson']:.2f}\nR² = {mt['r2']:.2f}\nMAE = {mt['mae']:.3f}",
                transform=ax.transAxes, va="top", fontsize=12)
        ax.set_title(name)
        ax.set_xlabel("Experimental P(deprot)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted P(deprot)")
    axes[1].legend(title="Residue", loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=1.6)
    save_fig(fig, out_dir, "fig01_scatter_probability", dpi)


def fig_scatter_pka(df, ph, out_dir, dpi, clip_eps=1e-3):
    colors = residue_colors(df["resname"])
    expt = df["expt_pka"].to_numpy(float)

    # The regressor is not bounded to [0, 1]; pKa = pH + log10((1-p)/p) is undefined there.
    # Clip to [eps, 1-eps] so every prediction maps to a pKa (clip_eps <= 0 disables clipping).
    pred = df["prediction"].to_numpy(float)
    if clip_eps > 0:
        n_clipped = int(((pred < clip_eps) | (pred > 1 - clip_eps)).sum())
        model_pka = prob_to_pka(np.clip(pred, clip_eps, 1 - clip_eps), ph)
    else:
        n_clipped = 0
        model_pka = prob_to_pka(pred, ph)
    panels = [
        ("Model", model_pka),
        ("PROPKA", df["propka_pka"].to_numpy(float)),
    ]
    finite_vals = [expt[np.isfinite(expt)]] + [p[np.isfinite(p)] for _, p in panels]
    allv = np.concatenate(finite_vals)
    lo = max(np.min(allv) - 0.5, -1.0)
    hi = min(np.max(allv) + 0.5, 17.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    for ax, (name, y) in zip(axes, panels):
        mask = np.isfinite(expt) & np.isfinite(y)
        dropped = int((~mask).sum())
        for res, c in colors.items():
            m = mask & (df["resname"] == res).to_numpy()
            ax.scatter(expt[m], y[m], s=30, color=c, alpha=0.7, edgecolors="none", label=res)
        ax.plot([lo, hi], [lo, hi], "k--", lw=2, zorder=0)
        r = safe_r(expt[mask], y[mask])
        mae = float(np.mean(np.abs(expt[mask] - y[mask])))
        rmse = float(np.sqrt(np.mean((expt[mask] - y[mask]) ** 2)))
        note = f"n = {int(mask.sum())}"
        if dropped:
            note += f" ({dropped} undefined)"
        if name == "Model" and n_clipped:
            note += f"\n{n_clipped} predictions clipped"
        ax.text(0.04, 0.96, f"{note}\nr = {r:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, va="top", fontsize=12)
        ax.set_title(name)
        ax.set_xlabel("Experimental pKa")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted pKa")
    axes[1].legend(title="Residue", loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=1.6)
    if clip_eps > 0:
        pka_lo = ph + np.log10(clip_eps / (1 - clip_eps))
        pka_hi = ph + np.log10((1 - clip_eps) / clip_eps)
        foot = (f"Model pKa is back-calculated from predicted probability clipped to "
                f"[{clip_eps:g}, {1 - clip_eps:g}], limiting it to pKa {pka_lo:.1f}–{pka_hi:.1f}.")
    else:
        foot = "Model pKa is back-calculated from predicted probability; undefined when p ≤ 0 or p ≥ 1."
    n_excl = int(df.attrs.get("n_propka_excluded", 0))
    if n_excl:
        foot += (f"\n{n_excl} rows with PROPKA placeholder pKa (99.99, e.g. disulfide-bonded Cys) "
                 f"excluded from all panels.")
    fig.text(0.5, -0.01, foot, ha="center", va="top", fontsize=10)
    save_fig(fig, out_dir, "fig02_scatter_pka", dpi)


def fig_roc_pr(df, out_dir, dpi):
    ybin = (df["label"].to_numpy(float) >= 0.5).astype(int)
    if len(np.unique(ybin)) < 2:
        print("[fig] skipping ROC/PR: only one class present")
        return
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5.6))
    for name, s in get_scores(df).items():
        c = METHOD_COLORS[name]
        fpr, tpr, _ = roc_curve(ybin, s)
        ax_roc.plot(fpr, tpr, color=c, label=f"{name} (AUC = {roc_auc_score(ybin, s):.3f})")
        prec, rec, _ = precision_recall_curve(ybin, s)
        ax_pr.plot(rec, prec, color=c, label=f"{name} (AP = {average_precision_score(ybin, s):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=2, zorder=0)
    ax_pr.axhline(ybin.mean(), color="k", ls="--", lw=2, zorder=0)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(-0.01, 1.01), ylim=(-0.01, 1.03))
    ax_roc.set_title(f"ROC (n = {len(ybin)})")
    ax_pr.set_title("Precision-recall")
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="lower left")
    save_fig(fig, out_dir, "fig03_roc_pr", dpi)


def fig_error_by_residue(df, min_n, out_dir, dpi):
    counts = df["resname"].value_counts()
    resnames = [r for r in sorted(counts.index) if counts[r] >= min_n]
    if not resnames:
        print("[fig] skipping error-by-residue: no residue type meets --min-resname-n")
        return
    scores = get_scores(df)
    width = 0.26
    offsets = np.linspace(-width, width, len(scores))
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(resnames) + 2), 5.6))
    handles = []
    for k, (name, s) in enumerate(scores.items()):
        err = np.abs(s - df["label"].to_numpy(float))
        data = [err[(df["resname"] == r).to_numpy()] for r in resnames]
        bp = ax.boxplot(data, positions=np.arange(len(resnames)) + offsets[k], widths=width * 0.9,
                        patch_artist=True, showfliers=False)
        style_boxplot(bp, METHOD_COLORS[name])
        handles.append(Patch(facecolor=METHOD_COLORS[name], edgecolor="black", alpha=0.55, label=name))
    ax.set_xticks(np.arange(len(resnames)))
    ax.set_xticklabels([f"{r}\n(n={counts[r]})" for r in resnames])
    ax.set_ylabel("|Predicted − experimental P(deprot)|")
    ax.set_title("Absolute error by residue type (outliers hidden)")
    ax.legend(handles=handles, loc="upper right", ncol=len(handles))
    save_fig(fig, out_dir, "fig04_error_by_residue", dpi)


def fig_paired_difference(df, n_boot, seed, out_dir, dpi):
    label = df["label"].to_numpy(float)
    scores = get_scores(df)
    err = {k: np.abs(v - label) for k, v in scores.items()}
    groups = df["pdb"].astype(str).to_numpy()
    comparisons = [("PROPKA", "Model vs PROPKA"), ("Ref pKa baseline", "Model vs ref pKa baseline")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    summary = []
    for ax, (other, title) in zip(axes, comparisons):
        diff = err["Model"] - err[other]
        mean = float(np.mean(diff))
        lo, hi = pdb_bootstrap_ci(diff, groups, n_boot, seed)
        try:
            p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            p = np.nan
        frac_better = float(np.mean(diff < 0))
        ax.hist(diff, bins=40, color=METHOD_COLORS["Model"], alpha=0.65, edgecolor="black", linewidth=1.5)
        ax.axvspan(lo, hi, color="gold", alpha=0.45, zorder=0)
        ax.axvline(0, color="k", ls="--", lw=2)
        ax.axvline(mean, color="firebrick", lw=3)
        ax.text(0.03, 0.97,
                f"mean Δ = {mean:+.4f}\n95% CI (PDB bootstrap):\n[{lo:+.4f}, {hi:+.4f}]\n"
                f"Wilcoxon p = {p:.2g} (row-level)\nmodel better in {frac_better*100:.0f}% of rows",
                transform=ax.transAxes, va="top", fontsize=11)
        ax.set_title(f"{title} (n = {len(diff)})")
        ax.set_xlabel("|Model error| − |comparator error|\n(negative = model better)")
        summary.append((title, mean, lo, hi, p))
    axes[0].set_ylabel("Residues")
    save_fig(fig, out_dir, "fig05_paired_error_difference", dpi)
    for title, mean, lo, hi, p in summary:
        print(f"[stats] {title}: mean Δ|err| = {mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], Wilcoxon p = {p:.3g}")


def fig_per_fold(df, out_dir, dpi):
    if "fold" not in df.columns:
        print("[fig] skipping per-fold metrics: no fold column")
        return
    folds = sorted(df["fold"].dropna().unique())
    names = list(get_scores(df).keys())
    keys = [("r2", "R²"), ("auc", "AUC-ROC"), ("mcc", "MCC")]
    table = {m: {k: [] for k, _ in keys} for m in names}
    for f in folds:
        sub = df[df["fold"] == f]
        for m, s in get_scores(sub).items():
            mt = compute_metrics(sub["label"], s)
            for k, _ in keys:
                table[m][k].append(mt[k])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (k, title) in zip(axes, keys):
        mat = np.array([table[m][k] for m in names], dtype=float)  # methods x folds
        for j in range(mat.shape[1]):
            if np.all(np.isfinite(mat[:, j])):
                ax.plot(range(len(names)), mat[:, j], color="0.6", lw=1.5, alpha=0.6, zorder=1)
        for i, m in enumerate(names):
            vals = mat[i][np.isfinite(mat[i])]
            if len(vals) == 0:
                continue
            bp = ax.boxplot([vals], positions=[i], widths=0.5, patch_artist=True, showfliers=False)
            style_boxplot(bp, METHOD_COLORS[m])
            ax.scatter(np.full(len(vals), i), vals, s=55, color=METHOD_COLORS[m],
                       edgecolor="black", linewidth=1.5, zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per fold")
    fig.suptitle(f"{len(folds)} folds, rows with PROPKA values; grey lines link the same fold", y=1.02, fontsize=12)
    save_fig(fig, out_dir, "fig06_per_fold_metrics", dpi)


def fig_confusion(df, out_dir, dpi):
    ybin = (df["label"].to_numpy(float) >= 0.5).astype(int)
    scores = get_scores(df)
    fig, axes = plt.subplots(1, len(scores), figsize=(5.2 * len(scores), 5))
    for ax, (name, s) in zip(axes, scores.items()):
        sb = (s >= 0.5).astype(int)
        cm = confusion_matrix(ybin, sb, labels=[0, 1])
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]*100:.0f}%)", ha="center", va="center",
                        fontsize=15, fontweight="bold", color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Protonated", "Deprotonated"])
        ax.set_yticklabels(["Protonated", "Deprotonated"], rotation=90, va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Experimental")
        ax.set_title(f"{name}\nMCC = {matthews_corrcoef(ybin, sb):.3f}")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(AXIS_LW)
    save_fig(fig, out_dir, "fig07_confusion_matrices", dpi)


def fig_calibration(df, out_dir, dpi, n_bins=10):
    y = df["label"].to_numpy(float)
    edges = np.linspace(0, 1, n_bins + 1)
    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(6.8, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    for name, s in get_scores(df).items():
        c = METHOD_COLORS[name]
        idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if m.any():
                xs.append(s[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, "-o", color=c, markersize=8, markeredgecolor="black", markeredgewidth=1.5, label=name)
        ax_h.hist(s, bins=edges, histtype="step", color=c, linewidth=2.5)
    ax.plot([0, 1], [0, 1], "k--", lw=2, zorder=0)
    ax.set_ylabel("Mean experimental P(deprot)")
    ax.set_title(f"Calibration (n = {len(y)})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    ax_h.set_yscale("log")
    ax_h.set_xlabel("Predicted P(deprot)")
    ax_h.set_ylabel("Count")
    save_fig(fig, out_dir, "fig08_calibration", dpi)


def fig_ablation(full_df, specs: List[str], out_dir, dpi):
    if not specs:
        print("[fig] skipping ablation: pass --ablation LABEL=PATH ... to enable")
        return
    frames = {}
    for spec in specs:
        if "=" not in spec:
            print(f"[fig] ignoring malformed --ablation entry '{spec}' (expected LABEL=PATH)")
            continue
        label, path = spec.split("=", 1)
        frames[label.strip()] = read_oof(path.strip())
    if not frames:
        return
    ordered = dict(frames)
    ordered["Full model"] = full_df
    if all("row_id" in f.columns for f in ordered.values()):
        common = set.intersection(*[set(f["row_id"]) for f in ordered.values()])
        ordered = {k: f[f["row_id"].isin(common)].sort_values("row_id") for k, f in ordered.items()}
        print(f"[ablation] {len(common)} rows shared across all runs")
    else:
        print("[ablation] row_id missing in at least one file; using all rows per run")

    results = {}
    base = ordered["Full model"]
    results["Ref pKa baseline"] = compute_metrics(base["label"], base["baseline"])
    for k, f in ordered.items():
        results[k] = compute_metrics(f["label"], f["prediction"])

    labels = list(results.keys())
    colors = []
    for k in labels:
        colors.append(METHOD_COLORS["Ref pKa baseline"] if k == "Ref pKa baseline"
                      else METHOD_COLORS["Model"] if k == "Full model" else "#ff7f0e")
    keys = [("r2", "R²"), ("auc", "AUC-ROC"), ("mcc", "MCC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.7 * len(labels) + 3), sharey=True)
    for ax, (k, title) in zip(axes, keys):
        vals = [results[l][k] for l in labels]
        ypos = np.arange(len(labels))
        ax.barh(ypos, vals, color=colors, edgecolor="black", linewidth=2, height=0.65)
        for yp, v in zip(ypos, vals):
            if np.isfinite(v):
                ax.text(v + (0.01 if v >= 0 else -0.01), yp, f"{v:.3f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=11)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.axvline(0, color="k", lw=AXIS_LW)
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            ax.set_xlim(min(0, min(finite)) - 0.05, max(finite) * 1.18 + 0.02)
    n = results["Full model"]["n"]
    fig.suptitle(f"Feature ablation (out-of-fold, n = {n})", y=1.02, fontsize=15, fontweight="bold")
    save_fig(fig, out_dir, "fig09_ablation", dpi)


def fig_class_balance(full_df, out_dir, dpi):
    ybin = (full_df["label"].to_numpy(float) >= 0.5)
    groups = [("All", np.ones(len(full_df), dtype=bool))]
    for r in sorted(full_df["resname"].unique()):
        groups.append((r, (full_df["resname"] == r).to_numpy()))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(groups) + 3), 5.2))
    for i, (name, m) in enumerate(groups):
        n = int(m.sum())
        frac = float(ybin[m].mean()) if n else np.nan
        ax.bar(i, frac, color=METHOD_COLORS["Model"], edgecolor="black", linewidth=2, width=0.7,
               label="Deprotonated (P ≥ 0.5)" if i == 0 else None)
        ax.bar(i, 1 - frac, bottom=frac, color="0.85", edgecolor="black", linewidth=2, width=0.7,
               label="Protonated (P < 0.5)" if i == 0 else None)
        ax.text(i, 1.02, f"n={n}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Fraction of residues")
    ax.set_title("Class balance at pH of label")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save_fig(fig, out_dir, "fig10_class_balance", dpi)


def fig_flip_roc_pr(df, out_dir, dpi):
    """ROC / PR for detecting residues whose true state differs from the baseline state."""
    _, base_state, flipped = flip_setup(df)
    n, n_flip = len(flipped), int(flipped.sum())
    if n_flip == 0 or n_flip == n:
        print(f"[fig] skipping flip ROC/PR: {n_flip} flipped of {n} rows (need both flipped and non-flipped)")
        return

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5.6))
    for name, s in flip_methods(df).items():
        c = METHOD_COLORS[name]
        fs = flip_score(s, base_state)
        fpr, tpr, _ = roc_curve(flipped.astype(int), fs)
        ax_roc.plot(fpr, tpr, color=c, label=f"{name} (AUC = {roc_auc_score(flipped, fs):.3f})")
        prec, rec, _ = precision_recall_curve(flipped.astype(int), fs)
        ax_pr.plot(rec, prec, color=c, label=f"{name} (AP = {average_precision_score(flipped, fs):.3f})")
    prevalence = n_flip / n
    ax_roc.plot([0, 1], [0, 1], "k--", lw=2, zorder=0)
    ax_pr.axhline(prevalence, color="k", ls="--", lw=2, zorder=0)
    ax_roc.set(xlabel="False positive rate (non-flipped residues)",
               ylabel="True positive rate (flipped residues)",
               xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax_pr.set(xlabel="Recall (flipped residues)", ylabel="Precision",
              xlim=(-0.01, 1.01), ylim=(-0.01, 1.03))
    ax_roc.set_title(f"ROC: detecting baseline flips\n({n_flip} flipped of {n})")
    ax_pr.set_title(f"Precision-recall (chance = {prevalence:.2f})")
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="upper right")
    fig.text(0.5, -0.04,
             "Flipped = ref-pKa baseline state differs from experimental state at 0.5. "
             "Score = probability assigned to the state opposite the baseline's.",
             ha="center", fontsize=10)
    save_fig(fig, out_dir, "fig11_flip_roc_pr", dpi)


def fig_flip_recall_false_flips(df, n_boot, seed, out_dir, dpi):
    """Recall on flipped residues and false-flip rate on non-flipped residues (threshold 0.5)."""
    label_state, base_state, flipped = flip_setup(df)
    n_flip, n_keep = int(flipped.sum()), int((~flipped).sum())
    if n_flip == 0 or n_keep == 0:
        print(f"[fig] skipping flip recall/false flips: {n_flip} flipped, {n_keep} non-flipped rows")
        return
    groups = df["pdb"].astype(str).to_numpy()

    rows = []
    for name, s in flip_methods(df).items():
        state = np.asarray(s, float) >= 0.5
        moved = state != base_state                      # method flips this residue
        correct_flip = moved & flipped                   # flipped, and method flipped it (to the right state)
        false_flip = moved & ~flipped                    # not flipped, but method flipped it anyway

        rec_ind = (state == label_state)[flipped].astype(float)
        ffr_ind = moved[~flipped].astype(float)
        rec_lo, rec_hi = pdb_bootstrap_ci(rec_ind, groups[flipped], n_boot, seed)
        ffr_lo, ffr_hi = pdb_bootstrap_ci(ffr_ind, groups[~flipped], n_boot, seed)
        n_moved = int(moved.sum())
        rows.append({
            "method": name,
            "n_rows": len(df),
            "n_flipped": n_flip,
            "n_nonflipped": n_keep,
            "correct_flips": int(correct_flip.sum()),
            "missed_flips": n_flip - int(correct_flip.sum()),
            "false_flips": int(false_flip.sum()),
            "recall_flipped": float(rec_ind.mean()),
            "recall_ci_lo": float(rec_lo),
            "recall_ci_hi": float(rec_hi),
            "false_flip_rate": float(ffr_ind.mean()),
            "false_flip_ci_lo": float(ffr_lo),
            "false_flip_ci_hi": float(ffr_hi),
            "flip_precision": float(correct_flip.sum() / n_moved) if n_moved else np.nan,
            "net_flips": int(correct_flip.sum()) - int(false_flip.sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "flip_metrics_summary.csv"), index=False)
    print(summary.round(4).to_string(index=False))

    panels = [
        ("recall_flipped", "recall_ci_lo", "recall_ci_hi", "correct_flips", n_flip,
         "Recall on flipped residues", f"Flipped residues (n = {n_flip})"),
        ("false_flip_rate", "false_flip_ci_lo", "false_flip_ci_hi", "false_flips", n_keep,
         "False-flip rate", f"Non-flipped residues (n = {n_keep})"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, (val, lo, hi, cnt, denom, ylabel, title) in zip(axes, panels):
        for i, r in summary.iterrows():
            ax.bar(i, r[val], color=METHOD_COLORS[r["method"]], edgecolor="black", linewidth=2, width=0.6)
            ax.errorbar(i, r[val], yerr=[[max(r[val] - r[lo], 0)], [max(r[hi] - r[val], 0)]],
                        color="black", capsize=7, capthick=2.5, elinewidth=2.5, zorder=3)
            ax.text(i, r[hi] + 0.02, f"{r[val]*100:.1f}%\n({int(r[cnt])}/{denom})",
                    ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary["method"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, min(1.0, float(summary[hi].max()) + 0.12) if val == "false_flip_rate" else 1.18)
    fig.text(0.5, -0.04,
             "Threshold 0.5. Error bars: 95% CI, PDB-clustered bootstrap. A method that never flips "
             "(the ref-pKa baseline) scores 0% on both.",
             ha="center", fontsize=10)
    save_fig(fig, out_dir, "fig12_flip_recall_false_flips", dpi)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Make comparison figures (PNG + SVG).")
    ap.add_argument("--oof", required=True, help="*_oof_predictions_with_propka.csv")
    ap.add_argument("--out-dir", default="figures", help="Output directory (default: figures)")
    ap.add_argument("--ph", type=float, default=7.4, help="pH used for labels/PROPKA conversion (default 7.4)")
    ap.add_argument("--ablation", nargs="*", default=[], help="LABEL=PATH pairs of plain OOF CSVs from other runs")
    ap.add_argument("--min-resname-n", type=int, default=5, help="Min rows for a residue type in per-residue plots")
    ap.add_argument("--n-boot", type=int, default=5000, help="Bootstrap resamples (PDB-clustered)")
    ap.add_argument("--clip-eps", type=float, default=1e-3,
                    help="Clip model probabilities to [eps, 1-eps] before converting to pKa in figure 2 "
                         "(default 0.001; 0 disables, leaving out-of-range predictions undefined)")
    ap.add_argument("--exclude-propka-placeholder", action="store_true",
                    help=f"Drop rows where PROPKA reports its 99.99 'not applicable' pKa (>= {PROPKA_PLACEHOLDER_MIN:g}; "
                         "mostly disulfide-bonded Cys) from every method's comparison, and write them to "
                         "excluded_propka_placeholder_rows.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not 0 <= args.clip_eps < 0.5:
        raise SystemExit("--clip-eps must be in [0, 0.5)")

    set_style()
    os.makedirs(args.out_dir, exist_ok=True)

    full_df = prepare_full(args.oof, args.ph)
    df, excluded = prepare_propka_subset(full_df, args.ph, args.exclude_propka_placeholder)
    print(f"[data] {len(full_df)} OOF rows; {len(df)} with PROPKA values (used for method comparisons)")
    if args.exclude_propka_placeholder:
        print(f"[data] excluded {len(excluded)} row(s) with PROPKA pKa >= {PROPKA_PLACEHOLDER_MIN:g} "
              f"(placeholder): {excluded['resname'].value_counts().to_dict() if len(excluded) else {}}")
        if len(excluded):
            keep = [c for c in ("row_id", "pdb", "chain", "resseq", "resname", "expt_pka", "label",
                                "prediction", "propka_pka") if c in excluded.columns]
            excluded[keep].to_csv(os.path.join(args.out_dir, "excluded_propka_placeholder_rows.csv"), index=False)

    # Summary table
    rows = []
    for name, s in get_scores(df).items():
        rows.append({"method": name, **compute_metrics(df["label"], s)})
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "figure_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"[data] wrote {summary_path}")

    fig_scatter_probability(df, args.out_dir, args.dpi)
    fig_scatter_pka(df, args.ph, args.out_dir, args.dpi, args.clip_eps)
    fig_roc_pr(df, args.out_dir, args.dpi)
    fig_error_by_residue(df, args.min_resname_n, args.out_dir, args.dpi)
    fig_paired_difference(df, args.n_boot, args.seed, args.out_dir, args.dpi)
    fig_per_fold(df, args.out_dir, args.dpi)
    fig_confusion(df, args.out_dir, args.dpi)
    fig_calibration(df, args.out_dir, args.dpi)
    fig_ablation(full_df, args.ablation, args.out_dir, args.dpi)
    fig_class_balance(full_df, args.out_dir, args.dpi)
    fig_flip_roc_pr(df, args.out_dir, args.dpi)
    fig_flip_recall_false_flips(df, args.n_boot, args.seed, args.out_dir, args.dpi)
    print(f"[done] figures in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()