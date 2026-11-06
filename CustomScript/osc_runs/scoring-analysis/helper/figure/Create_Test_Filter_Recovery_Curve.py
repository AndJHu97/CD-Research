#!/usr/bin/env python3

"""Target-site recovery vs. percent of candidates filtered out, per query group.



Each query group (unique ``Name``) is ranked by ``pred_score`` (highest = rank 1).

For a filter level of X% filtered out, the top (100 - X)% of *that group's*

candidates are retained (rounded up, at least one residue unless X = 100). The

cutoff is therefore applied inside every group, never to the pooled residues.



Outputs

\-------

1. \<output>.png/.svg        Line graph: % of true sites (relevance == 1) retained

                            vs. % filtered out, with the random-ranking baseline.

2. \<output>_roc.png/.svg    Per-group-cutoff ROC curve on a log FPR axis (the

                            curve behind the adjusted logAUC).

3. \<output>_stats.csv       Hit rate, enrichment factor, etc. at each filter level.

4. Adjusted logAUC printed to the console and written to \<output>_summary.txt.



Metrics

\-------

Hit rate (site recovery) = true sites retained / total true sites (pooled).

Enrichment factor (EF)   = hit rate / fraction of all candidates retained.

Adjusted logAUC          = logAUC (TPR integrated over log10 FPR from 0.001 to 1,

                           divided by log10(1/0.001) = 3) minus the random

                           baseline 0.14462. 0% = random, 85.5% = perfect.

                           (Method of the AlphaFold3 covalent docking paper,

                           London lab.) The ROC is built by sweeping the

                           per-group retained fraction, so FPR/TPR come from

                           the same per-group cutoffs as the line graph.



Usage:

    python Create_Filter_Recovery_Curve.py \\\\

        --input path/to/test_predictions.csv \\\\

        --output filter_recovery.png

"""



from __future__ import annotations



import argparse

from pathlib import Path



import matplotlib



matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np

import pandas as pd

from sklearn.metrics import auc

GROUP_COL = "Name"

SCORE_COL = "pred_score"

RELEVANCE_COL = "relevance"



DEFAULT_FILTER_PCTS = [0, 25, 50, 75, 99, 99.5, 99.9, 100]

LOGAUC_MIN_FPR = 0.001

RANDOM_LOGAUC = 0.14462  # random-ranking baseline for FPR in [0.001, 1]



OBSERVED_COLOR = "#1E9E3A"

RANDOM_COLOR = "#8A8A8A"

GREY = "#8A8A8A"

TIGHT_SCALE = 0.55   # x width given to 99% -> 99.9% (0-99% spans 4.0 units)

BREAK_L_MAX = 99.0   # broken axis: left panel shows 0 -> this %

BREAK_R_MIN = 99.0   # broken axis: right panel zooms from this % to 100% (contiguous)

X_100 = 4.0 + TIGHT_SCALE + 0.275  # x position of 100% filtered (nothing retained)



_trapz = getattr(np, "trapezoid", None) or np.trapz





def _configure_figure_fonts() -> None:

    """Use bold Arial for all figure text (register Windows fonts when needed)."""

    from matplotlib import font_manager



    for path in (

        Path("/mnt/c/Windows/Fonts/arial.ttf"),

        Path("/mnt/c/Windows/Fonts/arialbd.ttf"),

        Path(r"C:\Windows\Fonts\arial.ttf"),

        Path(r"C:\Windows\Fonts\arialbd.ttf"),

    ):

        if path.is_file():

            try:

                font_manager.fontManager.addfont(str(path))

            except (OSError, RuntimeError, ValueError):

                pass



    plt.rcParams.update(

        {

            "font.family": "sans-serif",

            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],

            "font.weight": "bold",

            "axes.labelweight": "bold",

            "axes.titleweight": "bold",

            "figure.titleweight": "bold",

            "legend.fontsize": 12,

            "axes.unicode_minus": False,
            "axes.linewidth": 2,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,

        }

    )





# --------------------------------------------------------------------------- #

# Data

# --------------------------------------------------------------------------- #

def load_predictions(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in (GROUP_COL, SCORE_COL, RELEVANCE_COL) if c not in df.columns]

    if missing:

        raise SystemExit(f"Input is missing required column(s): {', '.join(missing)}")



    df = df.copy()

    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")

    df[RELEVANCE_COL] = pd.to_numeric(df[RELEVANCE_COL], errors="coerce").fillna(0).astype(int)

    n_before = len(df)

    df = df.dropna(subset=[SCORE_COL])

    if len(df) < n_before:

        print(f"Dropped {n_before - len(df)} row(s) with non-numeric {SCORE_COL}.")

    if not (df[RELEVANCE_COL] == 1).any():

        raise SystemExit("No rows with relevance == 1 found.")

    return df





def add_ranks(df: pd.DataFrame) -> pd.DataFrame:

    """Rank within each query group (1 = highest score). Ties are broken by file

    order so that exactly n residues are retained for a cutoff of n."""

    df = df.copy()

    df["rank"] = (

        df.groupby(GROUP_COL)[SCORE_COL].rank(method="first", ascending=False).astype(int)

    )

    df["group_size"] = df.groupby(GROUP_COL)[SCORE_COL].transform("size")

    return df





def n_retained(sizes: np.ndarray, filtered_pct: float) -> np.ndarray:

    """Residues kept per group at a given % filtered out (ceil, min 1 unless 100%)."""

    sizes = np.asarray(sizes)

    if filtered_pct >= 100:

        return np.zeros_like(sizes, dtype=int)

    keep_frac = (100.0 - filtered_pct) / 100.0

    n = np.ceil(sizes * keep_frac - 1e-9)  # epsilon guards float error (e.g. 1000*0.001)

    return np.clip(n, 1, sizes).astype(int)



def plot_roc_linear(fpr, tpr, output, mean_auc):
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        fpr,
        tpr,
        label="CovSite (pooled retained-fraction curve)",
        linewidth=3
    )
    ax.plot([0, 1], [0, 1], linewidth=2, linestyle="--", label="Random (AUC = 0.500)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Linear Scale)")
    ax.legend(loc="lower right")

    ax.text(
        0.05, 0.95,
        f"Mean per-query ROC-AUC = {mean_auc:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def mean_per_query_roc_auc(df: pd.DataFrame) -> tuple[float, int]:
    """Calculate conventional ROC-AUC separately for each query,
    then return the unweighted mean and number of eligible queries.
    """
    from sklearn.metrics import roc_auc_score

    query_aucs = []

    for _, group in df.groupby(GROUP_COL, sort=False):
        y_true = group[RELEVANCE_COL].to_numpy()
        y_score = group[SCORE_COL].to_numpy()

        # AUC requires both positive and negative candidates.
        if len(np.unique(y_true)) < 2:
            continue

        query_aucs.append(roc_auc_score(y_true, y_score))

    if not query_aucs:
        raise SystemExit("No queries have both positive and negative candidates.")

    return float(np.mean(query_aucs)), len(query_aucs)


# --------------------------------------------------------------------------- #

# Statistics

# --------------------------------------------------------------------------- #

def summarize(df: pd.DataFrame, filter_pcts: list[float]) -> pd.DataFrame:

    sizes = df.groupby(GROUP_COL)["group_size"].first()

    total_candidates = len(df)

    pos = df[df[RELEVANCE_COL] == 1]

    n_sites = len(pos)

    pos_groups = pos[GROUP_COL].nunique()



    rows = []

    for p in filter_pcts:

        n_g = n_retained(sizes.to_numpy(), p)

        n_map = dict(zip(sizes.index, n_g))

        keep_n = pos[GROUP_COL].map(n_map)

        kept = pos["rank"] <= keep_n



        retained_candidates = int(n_g.sum())

        sites_kept = int(kept.sum())

        hit_rate = sites_kept / n_sites

        frac_retained = retained_candidates / total_candidates

        ef = hit_rate / frac_retained if frac_retained > 0 else np.nan



        # Random ranking: nominal (k%) and rounding-aware expectation.

        nominal_random = 100.0 - p

        rounded_random = float((keep_n / pos["group_size"]).mean() * 100.0)



        per_group_kept = kept.groupby(pos[GROUP_COL]).any()

        rows.append(

            {

                "pct_filtered_out": p,

                "pct_retained_nominal": 100.0 - p,

                "candidates_retained": retained_candidates,

                "pct_candidates_retained_actual": frac_retained * 100.0,

                "sites_retained": sites_kept,

                "total_sites": n_sites,

                "hit_rate_pct": hit_rate * 100.0,

                "enrichment_factor": ef,

                "queries_with_site_retained": int(per_group_kept.sum()),

                "queries_with_sites": pos_groups,

                "queries_with_site_retained_pct": per_group_kept.mean() * 100.0,

                "random_expected_pct_nominal": nominal_random,

                "random_expected_pct_with_rounding": rounded_random,

            }

        )

    return pd.DataFrame(rows)





def roc_by_fraction(df: pd.DataFrame, n_grid: int = 20000) -> tuple[np.ndarray, np.ndarray]:

    """ROC where every point is a per-group retained fraction k (top ceil(k*size))."""

    sizes = df.groupby(GROUP_COL)["group_size"].first().to_numpy()

    pos = df[df[RELEVANCE_COL] == 1]

    r = pos["rank"].to_numpy()

    s = pos["group_size"].to_numpy()

    n_pos = len(pos)

    n_neg = len(df) - n_pos

    if n_neg == 0:

        raise SystemExit("No negatives (relevance != 1) found; cannot compute FPR.")



    ks = np.unique(

        np.concatenate(

            [[0.0], np.geomspace(1e-6, 1.0, n_grid // 2), np.linspace(0.0, 1.0, n_grid // 2)]

        )

    )

    tpr = np.empty(len(ks))

    fpr = np.empty(len(ks))

    for i, k in enumerate(ks):

        n_g = np.minimum(np.ceil(sizes * k - 1e-9), sizes)

        keep_pos = int((r <= np.minimum(np.ceil(s * k - 1e-9), s)).sum())

        tpr[i] = keep_pos / n_pos

        fpr[i] = (n_g.sum() - keep_pos) / n_neg

    return np.maximum.accumulate(fpr), np.maximum.accumulate(tpr)





def adjusted_log_auc(fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, float]:

    """Return (logAUC, adjusted logAUC) as fractions."""

    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr}).groupby("fpr", as_index=False)["tpr"].max()

    x = np.geomspace(LOGAUC_MIN_FPR, 1.0, 20000)

    y = np.interp(x, roc["fpr"].to_numpy(), roc["tpr"].to_numpy())

    log_auc = _trapz(y, np.log10(x)) / np.log10(1.0 / LOGAUC_MIN_FPR)

    linear_auc = auc(fpr, tpr)

    print(f"Conventional ROC-AUC: {linear_auc:.4f} ({linear_auc * 100:.2f}%)")

    return float(log_auc), float(log_auc - RANDOM_LOGAUC)





# --------------------------------------------------------------------------- #

# Plots

# --------------------------------------------------------------------------- #

def _fmt_pct(p: float) -> str:

    return f"{p:g}%"





def x_positions(pcts, axis: str) -> np.ndarray:

    """X positions for filter levels (a smooth, continuous mapping).



    "log" (default): 0-99% is linear (4.0 units). From 99% to 100% the axis is

    compressed: x = 4 + TIGHT_SCALE * t**0.6, with t = -log10(fraction retained) - 2

    (99.9% -> t = 1). 100% (nothing retained) has no finite position, so it sits

    at X_100. "categorical": equal spacing between the listed levels.

    """

    pcts = np.asarray(pcts, dtype=float)

    if axis == "categorical":

        return np.arange(len(pcts), dtype=float)

    x = np.empty(len(pcts))

    low = pcts <= 99

    x[low] = 4.0 * pcts[low] / 99.0

    mid = (pcts > 99) & (pcts < 100)

    t = -np.log10((100.0 - pcts[mid]) / 100.0) - 2.0

    x[mid] = 4.0 + TIGHT_SCALE * np.power(np.clip(t, 0, None), 0.6)

    x[pcts >= 100] = X_100

    return x





def dense_filter_levels(requested) -> list[float]:

    """Fine grid of filter levels for drawing a smooth curve (plus requested ones)."""

    lin = np.linspace(0, 99, 200)

    u = np.linspace(0, 1, 60)  # 99% -> 99.9%, log-spaced

    tail = 100.0 - 100.0 * 10.0 ** (-(2.0 + u))

    return sorted(set(np.round(np.concatenate([lin, tail, [100.0], requested]), 6)))





def dense_levels_broken(requested) -> list[float]:

    lin = np.linspace(0, BREAK_L_MAX, 250)

    zoom = np.linspace(99.0, 99.99, 150)

    return sorted(set(np.round(np.concatenate([lin, zoom, [100.0], requested]), 6)))





def _label(ax, x, y, hr, ef, mode):

    ef_txt = "EF n/a" if np.isnan(ef) else f"EF {ef:.1f}"

    offs, ha, va = {

        "above": ((0, 12), "center", "bottom"),

        "upright": ((8, 12), "left", "bottom"),

        "belowleft": ((-10, -10), "right", "top"),

        "floor": ((-10, 10), "right", "bottom"),

    }[mode]

    ax.annotate(

        f"{hr:.1f}%\n{ef_txt}", xy=(x, y), xytext=offs, textcoords="offset points",

        ha=ha, va=va, fontsize=10, color=OBSERVED_COLOR,

    )





def plot_recovery_linear(

    stats: pd.DataFrame, dense: pd.DataFrame, adj_logauc: float, n_groups: int, output: Path

) -> None:

    """Single panel, plain linear x-axis (0-100%). The crowded 99-100% points are

    labeled in a stacked column with leader lines instead of stretching the axis."""

    _configure_figure_fonts()

    fig, ax = plt.subplots(figsize=(11.0, 7.0))

    pc = dense["pct_filtered_out"]

    below100 = dense[pc < 100]



    ax.plot(pc, dense["random_expected_pct_nominal"], ls="--", color=RANDOM_COLOR, lw=2, zorder=2)

    ax.plot(below100["pct_filtered_out"], below100["hit_rate_pct"], color=OBSERVED_COLOR,

            lw=3, zorder=3)

    last = below100.iloc[-1]

    ax.plot([last["pct_filtered_out"], 100.0], [last["hit_rate_pct"], 0.0],

            color=OBSERVED_COLOR, lw=2, ls=(0, (3, 3)), zorder=3)



    crowded = []

    for _, r in stats.iterrows():

        pp = r["pct_filtered_out"]

        ax.plot([pp], [r["random_expected_pct_nominal"]], ls="", color=RANDOM_COLOR,

                marker="s", markersize=6, zorder=2)

        ax.plot([pp], [r["hit_rate_pct"]], ls="", color=OBSERVED_COLOR, marker="o",

                markersize=8, zorder=4)

        if pp < 99:

            _label(ax, pp, r["hit_rate_pct"], r["hit_rate_pct"], r["enrichment_factor"], "above")

        else:

            crowded.append(r)



    # Stacked labels (top box <-> highest point) so leader lines do not cross.

    crowded.sort(key=lambda r: -r["hit_rate_pct"])

    for i, r in enumerate(crowded):

        ef = r["enrichment_factor"]

        ef_txt = "n/a" if np.isnan(ef) else f"{ef:.1f}"

        ax.annotate(

            f"{_fmt_pct(r['pct_filtered_out'])}:  {r['hit_rate_pct']:.1f}%  (EF {ef_txt})",

            xy=(r["pct_filtered_out"], r["hit_rate_pct"]), xytext=(96.0, 58 - 9 * i),

            textcoords="data", ha="right", va="center", fontsize=10, color=OBSERVED_COLOR,

            arrowprops={"arrowstyle": "-", "color": OBSERVED_COLOR, "lw": 0.8,

                        "relpos": (1.0, 0.5), "shrinkA": 2, "shrinkB": 5},

        )



    ticks = [pp for pp in stats["pct_filtered_out"] if pp < 99 or pp >= 100]

    ax.set_xticks(ticks)

    ax.set_xticklabels([_fmt_pct(pp) for pp in ticks])

    ax.set_xlim(-2, 102)

    ax.set_ylim(-3, 115)

    ax.set_yticks(range(0, 101, 20))

    ax.set_xlabel("Candidates filtered out within each query group", fontsize=14)

    ax.set_ylabel("True sites retained (%)", fontsize=14)

    ax.tick_params(labelsize=12)

    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8, zorder=0)

    for spine in ("top", "right"):

        ax.spines[spine].set_visible(False)



    n_sites = int(stats["total_sites"].iloc[0])

    ax.text(

        0.02, 0.50,

        f"Adjusted logAUC: {adj_logauc * 100:.1f}%",

        transform=ax.transAxes, ha="left", va="top", fontsize=12, linespacing=1.5,

        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#999999"},

    )

    handles = [

        Line2D([0], [0], color=OBSERVED_COLOR, lw=3, marker="o", markersize=8, label="Model"),

        Line2D([0], [0], color=RANDOM_COLOR, lw=2, ls="--", marker="s", markersize=6,

               label="Random ranking (k%)"),

    ]

    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.02), frameon=False)

    _save(fig, output)





def plot_recovery_broken(

    stats: pd.DataFrame, dense: pd.DataFrame, adj_logauc: float, n_groups: int, output: Path

) -> None:

    """Two-panel broken-axis figure: 0-98.6% (left) and a 99-100% zoom (right)."""

    _configure_figure_fonts()

    fig, (axl, axr) = plt.subplots(

        1, 2, sharey=True, figsize=(11.0, 7.0),

        gridspec_kw={"width_ratios": [4.0, 1.9], "wspace": 0.0},

    )

    pc = dense["pct_filtered_out"]

    d_left = dense[pc <= BREAK_L_MAX]

    d_right = dense[(pc >= BREAK_R_MIN) & (pc < 100)]

    d_right_all = dense[pc >= BREAK_R_MIN]



    for ax, dd in ((axl, d_left), (axr, d_right_all)):

        ax.plot(dd["pct_filtered_out"], dd["random_expected_pct_nominal"], ls="--",

                color=RANDOM_COLOR, lw=2, zorder=2)

    axl.plot(d_left["pct_filtered_out"], d_left["hit_rate_pct"], color=OBSERVED_COLOR, lw=3, zorder=3)

    axr.plot(d_right["pct_filtered_out"], d_right["hit_rate_pct"], color=OBSERVED_COLOR, lw=3, zorder=3)

    # 100% filtered keeps nothing, so that last drop is trivial: dashed connector.

    last = d_right.iloc[-1]

    top = dense[pc >= 100].iloc[0]

    axr.plot([last["pct_filtered_out"], 100.0], [last["hit_rate_pct"], top["hit_rate_pct"]],

             color=OBSERVED_COLOR, lw=2, ls=(0, (3, 3)), zorder=3)



    right_seen = 0

    for _, r in stats.iterrows():

        p = r["pct_filtered_out"]

        on_left = p < BREAK_R_MIN

        ax = axl if on_left else axr

        for col, color, mk, ms in (

            ("random_expected_pct_nominal", RANDOM_COLOR, "s", 6),

            ("hit_rate_pct", OBSERVED_COLOR, "o", 9),

        ):

            ax.plot([p], [r[col]], ls="", color=color, marker=mk, markersize=ms,

                    zorder=4 if col == "hit_rate_pct" else 2, clip_on=False)

        if on_left:

            mode = "above"

        elif r["hit_rate_pct"] < 5:

            mode = "floor"

        else:

            mode = "upright" if right_seen == 0 else "belowleft"

            right_seen += 1

        _label(ax, p, r["hit_rate_pct"], r["hit_rate_pct"], r["enrichment_factor"], mode)



    axl.set_xlim(-2, BREAK_L_MAX)

    axr.set_xlim(BREAK_R_MIN, 100.06)

    axl.set_xticks([p for p in stats["pct_filtered_out"] if p < BREAK_R_MIN])

    axl.set_xticklabels([_fmt_pct(p) for p in stats["pct_filtered_out"] if p < BREAK_R_MIN])

    rt = [p for p in stats["pct_filtered_out"] if p >= BREAK_R_MIN]

    axr.set_xticks(rt)

    axr.set_xticklabels([_fmt_pct(p) for p in rt])

    for lab in axr.get_xticklabels():  # 99.9% and 100% are close: tilt to avoid overlap

        lab.set_rotation(60)

        lab.set_ha("right")

        lab.set_rotation_mode("anchor")

    axl.set_ylim(-3, 115)

    axl.set_yticks(range(0, 101, 20))

    for ax in (axl, axr):

        ax.tick_params(labelsize=12)

        ax.grid(axis="y", color="#E5E5E5", linewidth=0.8, zorder=0)

        ax.spines["top"].set_visible(False)

    axl.spines["right"].set_visible(False)

    axr.spines["left"].set_visible(False)

    axr.spines["right"].set_visible(False)

    axr.tick_params(axis="y", left=False, labelleft=False)



    # Break marks

    kw = {"transform": None, "color": "k", "clip_on": False, "lw": 1.5}

    d = 0.012

    for ax, x0 in ((axl, 1.0), (axr, 0.0)):

        dx = d * (4.0 / 1.9) if ax is axr else d

        ax.plot([x0 - dx, x0 + dx], [-d * 1.6, d * 1.6], transform=ax.transAxes,

                color="k", clip_on=False, lw=1.5)



    axl.set_ylabel("True sites retained (%)", fontsize=14)

    axl.set_title("True-site recovery vs. per-query filtering", fontsize=15, loc="left")

    fig.supxlabel("Candidates filtered out within each query group", fontsize=14, fontweight="bold", y=-0.01)



    n_sites = int(stats["total_sites"].iloc[0])

    axl.text(

        0.98, 0.52,

        f"{n_groups:,} queries\n{n_sites:,} true sites\n"

        f"Adjusted logAUC: {adj_logauc * 100:.1f}%",

        transform=axl.transAxes, ha="right", va="top", fontsize=12, linespacing=1.5,

        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#999999"},

    )

    handles = [

        Line2D([0], [0], color=OBSERVED_COLOR, lw=3, marker="o", markersize=9, label="Model"),

        Line2D([0], [0], color=RANDOM_COLOR, lw=2, ls="--", marker="s", markersize=6,

               label="Random ranking (k%)"),

    ]

    axl.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.02), frameon=False)

    _save(fig, output)





def plot_recovery(
    stats: pd.DataFrame, adj_logauc: float, n_groups: int, output: Path,
    axis: str = "log", dense: pd.DataFrame | None = None,
) -> None:
    """Plot recovery against fraction retained on a reversed log x-axis."""
    _configure_figure_fonts()
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    curve = (dense if dense is not None else stats).copy()
    curve = curve[curve["pct_filtered_out"] < 100]
    curve["fraction_retained"] = (100.0 - curve["pct_filtered_out"]) / 100.0
    curve = curve.sort_values("fraction_retained", ascending=False)

    observed = stats[stats["pct_filtered_out"] < 100].copy()
    observed["fraction_retained"] = (100.0 - observed["pct_filtered_out"]) / 100.0
    observed = observed.sort_values("fraction_retained", ascending=False)

    ax.plot(curve["fraction_retained"], curve["random_expected_pct_nominal"],
            ls="--", color=RANDOM_COLOR, lw=2, zorder=2, label="Random ranking")
    ax.plot(curve["fraction_retained"], curve["hit_rate_pct"],
            color=OBSERVED_COLOR, lw=3, zorder=3, label="Model")
    ax.plot(observed["fraction_retained"], observed["hit_rate_pct"],
            ls="", color=OBSERVED_COLOR, marker="o", ms=8, zorder=4)
    ax.plot(observed["fraction_retained"], observed["random_expected_pct_nominal"],
            ls="", color=RANDOM_COLOR, marker="s", ms=5, zorder=3)

    # Avoid stacked side labels; label each requested operating point in place.
    for _, r in observed.iterrows():
        ef_txt = "n/a" if np.isnan(r["enrichment_factor"]) else f'{r["enrichment_factor"]:.1f}'
        ax.annotate(f'{r["hit_rate_pct"]:.1f}% (EF {ef_txt})',
                    xy=(r["fraction_retained"], r["hit_rate_pct"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=OBSERVED_COLOR)

    ax.set_xscale("log")
    ax.set_xlim(1.0, 0.001)
    ax.set_xticks([1.0, 0.1, 0.01, 0.001])
    ax.set_xticklabels(["100%", "10%", "1%", "0.1%"])
    ax.set_ylim(-3, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Fraction of candidates retained within each query group", fontsize=14)
    ax.set_ylabel("True sites retained (%)", fontsize=14)
    ax.set_title("True-site recovery vs. fraction of candidates retained", fontsize=15, loc="left")
    ax.tick_params(labelsize=12)
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.8, zorder=0, which="major")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], color=OBSERVED_COLOR, lw=3, marker="o", markersize=8, label="Model"),
        Line2D([0], [0], color=RANDOM_COLOR, lw=2, ls="--", marker="s", markersize=5, label="Random ranking"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
    _save(fig, output)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, log_auc: float, adj: float, output: Path) -> None:

    _configure_figure_fonts()

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    f = np.clip(fpr, 1e-7, 1.0)

    ax.plot(f, tpr, color=OBSERVED_COLOR, linewidth=3, label="Model", zorder=3)

    xs = np.geomspace(1e-5, 1, 200)

    ax.plot(xs, xs, color=RANDOM_COLOR, linestyle="--", linewidth=2, label="Random ranking", zorder=2)

    ax.axvspan(LOGAUC_MIN_FPR, 1.0, color="#A8DDB5", alpha=0.25, linewidth=0, zorder=1)



    ax.set_xscale("log")

    ax.set_xlim(LOGAUC_MIN_FPR, 1.0)

    ax.set_ylim(0, 1.02)

    ax.set_xlabel("False-positive rate (log scale)", fontsize=14)

    ax.set_ylabel("True-positive rate (true sites retained)", fontsize=14)

    ax.set_title("Per-query-cutoff ROC (log FPR)", fontsize=15, loc="left")

    ax.tick_params(labelsize=12)

    ax.grid(color="#E5E5E5", linewidth=0.8, zorder=0)

    for spine in ("top", "right"):

        ax.spines[spine].set_visible(False)

    ax.text(

        0.04, 0.96,

        f"LogAUC: {log_auc * 100:.1f}%\nAdjusted logAUC: {adj * 100:.1f}%\n(integrated over FPR 0.001–1)",

        transform=ax.transAxes, ha="right", va="top", fontsize=12, linespacing=1.5,

        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#999999"},

    )

    ax.legend(loc="lower right", frameon=False)

    _save(fig, output)





def _save(fig, output: Path) -> None:

    output.parent.mkdir(parents=True, exist_ok=True)

    png, svg = output.with_suffix(".png"), output.with_suffix(".svg")

    fig.savefig(png, dpi=300, bbox_inches="tight")

    fig.savefig(svg, format="svg", bbox_inches="tight")

    plt.close(fig)

    print(f"Saved {png}")

    print(f"Saved {svg}")





# --------------------------------------------------------------------------- #

# CLI

# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--input", "-i", type=Path, required=True, help="Per-residue prediction CSV/TSV.")

    p.add_argument(

        "--output", "-o", type=Path, default=None,

        help="Output path stem (PNG + SVG written). Default: \<input stem>_filter_recovery.png",

    )

    p.add_argument(

        "--axis", choices=["linear", "broken", "log", "categorical"], default="log",

        help="Recovery plot x-axis: log = reversed logarithmic fraction retained (default); legacy layouts remain available.",

    )

    p.add_argument(

        "--no-smooth", action="store_true",

        help="Draw straight segments between the requested levels only (no dense curve).",

    )

    p.add_argument(

        "--filter-pcts", type=float, nargs="+", default=DEFAULT_FILTER_PCTS,

        help=f"Percent filtered out per query group (default: {DEFAULT_FILTER_PCTS}).",

    )

    return p.parse_args()





def main() -> None:

    args = parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_filter_recovery.png")

    stem = output.with_suffix("")

    pcts = sorted(set(args.filter_pcts))



    df = add_ranks(load_predictions(args.input))

    n_groups = df[GROUP_COL].nunique()

    n_sites = int((df[RELEVANCE_COL] == 1).sum())

    print(f"{n_groups} query groups, {len(df)} residues, {n_sites} true sites")



    stats = summarize(df, pcts)

    if args.axis in ("linear", "broken"):

        dense = summarize(df, dense_levels_broken(pcts))

    else:

        dense = None if args.no_smooth else summarize(df, dense_filter_levels(pcts))

    fpr, tpr = roc_by_fraction(df)
    log_auc, adj = adjusted_log_auc(fpr, tpr)

    mean_auc, n_auc_queries = mean_per_query_roc_auc(df)
    pooled_auc = auc(fpr, tpr)

    print(
        f"Mean per-query ROC-AUC: {mean_auc:.4f} "
        f"({mean_auc * 100:.2f}%), "
        f"across {n_auc_queries} queries"
    )



    stats_path = stem.with_name(stem.name + "_stats.csv")

    stats.round(4).to_csv(stats_path, index=False)



    
    summary = (
        f"Mean per-query ROC-AUC: {mean_auc:.4f} "
        f"({mean_auc * 100:.2f}%)\n"
        f"Queries included in mean ROC-AUC: {n_auc_queries}\n"
        f"LogAUC (FPR {LOGAUC_MIN_FPR}-1): {log_auc * 100:.2f}%\n"
        f"Random baseline: {RANDOM_LOGAUC * 100:.3f}%\n"
        f"Adjusted logAUC: {adj * 100:.2f}% "
        f"(0% = random, 85.5% = perfect)\n"
        f"Pooled retained-fraction AUC: {pooled_auc:.4f}\n"
    )


    summary_path = stem.with_name(stem.name + "_summary.txt")

    summary_path.write_text(summary)



    show = stats[

        ["pct_filtered_out", "candidates_retained", "sites_retained", "hit_rate_pct",

         "enrichment_factor", "queries_with_site_retained_pct",

         "random_expected_pct_nominal", "random_expected_pct_with_rounding"]

    ].round(2)

    print(show.to_string(index=False))

    print(summary)

    print(f"Saved {stats_path}")



    if args.axis == "linear":

        plot_recovery_linear(stats, dense, adj, n_groups, output)

    elif args.axis == "broken":

        plot_recovery_broken(stats, dense, adj, n_groups, output)

    else:

        plot_recovery(stats, adj, n_groups, output, args.axis, dense)

    
    # Existing logarithmic ROC
    plot_roc(
        fpr, tpr, log_auc, adj,
        stem.with_name(stem.name + "_roc.png")
    )

    # Additional standard linear ROC
    plot_roc_linear(
        fpr,
        tpr,
        stem.with_name(stem.name + "_roc_linear.png"),
        mean_auc,
    )






if __name__ == "__main__":

    main()
