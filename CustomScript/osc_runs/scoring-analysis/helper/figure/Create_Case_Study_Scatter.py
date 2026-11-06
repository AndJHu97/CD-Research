#!/usr/bin/env python3
"""
Plot predicted score per residue (sorted high -> low), positives in green, others grey.
Writes both a PNG and an editable SVG.

Usage:
    python plot_hits.py predictions.csv hits.csv
    python plot_hits.py predictions.csv hits.csv -o ranked.png --per-name
    python plot_hits.py predictions.csv hits.csv --style bar   # default is scatter


python helper/figure/Create_Case_Study_Scatter.py final_data_and_draft/case-study/1QXA_covsite_scores.csv final_data_and_draft/case-study/1QXA_covsite_label_results.csv --point-size 15 --hit-size 45

A row in the predictions file is a positive if:
  - its (Residue, ResNum, Chain) matches a row in the hits file, AND
  - its Warhead is one of that hits row's matched_warheads
    (matched_warheads may be comma separated; comparison is case-insensitive).
Name / other columns are ignored for matching.
Works with comma- or tab-separated files (auto-detected).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

# keep text as real text in the SVG so it stays editable in Illustrator
plt.rcParams["svg.fonttype"] = "none"

GREEN = "#2e9e4f"
GREY = "#b5b5b5"
RES_KEY = ["Residue", "ResNum", "Chain"]
KEY = RES_KEY + ["Warhead"]

X_LEFT_LOG = 0.7   # left edge of the log axis; larger = tighter (1.0 puts rank 1 on the axis)
RED = "#ff7f0e"
TOP_PCTS = [ 3, 5, 10, 25, 50, 75, 100]   # x-axis tick positions (top % of ranked residues)
BLUE = "#1f77b4"
# shaded bands on the left: (start %, end %, color, alpha, legend label). Non-overlapping.
SHADE_BANDS = [
    (0, 3, BLUE, 0.18, "Top 3%"),
    (3, 5, RED, 0.15, "Top 5%"),
]

AXIS_LW = 1.5   # axis line thickness


def load(path):
    # sep=None lets pandas sniff comma vs tab
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize(df):
    df = df.copy()
    df["Residue"] = df["Residue"].astype(str).str.strip().str.upper()
    df["ResNum"] = df["ResNum"].astype(str).str.strip()
    df["Chain"] = df["Chain"].astype(str).str.strip()
    if "Warhead" in df.columns:
        df["Warhead"] = df["Warhead"].astype(str).str.strip().str.lower()
    return df


def explode_hits(hits):
    """One row per (Residue, ResNum, Chain, Warhead) from the comma-separated matched_warheads."""
    if "matched_warheads" not in hits.columns:
        raise SystemExit("hits file has no 'matched_warheads' column")
    h = hits[RES_KEY + ["matched_warheads"]].copy()
    h["Warhead"] = h["matched_warheads"].fillna("").astype(str).str.split(",")
    h = h.explode("Warhead")
    h["Warhead"] = h["Warhead"].str.strip().str.lower()
    h = h[h["Warhead"] != ""]
    return h[KEY].drop_duplicates()


def make_plot(preds, hit_keys, hit_res, title, out, style="scatter", warn_unmatched=True,
              point_size=4, hit_size=14, x_scale="log"):
    preds = preds.sort_values("pred_score", ascending=False).reset_index(drop=True)
    merged = preds.merge(hit_keys, on=KEY, how="left", indicator=True)
    is_hit = (merged["_merge"] == "both").values
    scores = merged["pred_score"].values
    n = len(merged)
    is_bar = style == "bar"
    # scatter: x = rank (1..n); bar: 0-based position on a linear axis
    xs = np.arange(n) if is_bar else np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    if style == "bar":
        colors = [GREEN if h else GREY for h in is_hit]
        ax.bar(xs, scores, width=1.0, color=colors, linewidth=0)
        # make the (usually few) hits easier to see
        for i in np.where(is_hit)[0]:
            ax.bar(i, scores[i], width=max(3, len(merged) * 0.004), color=GREEN, linewidth=0, zorder=3)
    else:
        ax.scatter(xs[~is_hit], scores[~is_hit], s=point_size, color=GREY, linewidths=1,
                   zorder=2, clip_on=False)
        ax.scatter(xs[is_hit], scores[is_hit], s=hit_size, color=GREEN, edgecolors="white",
                   linewidths=0.5, zorder=3, clip_on=False)

    # x-axis: top-percentile ticks (position = that fraction of the ranked list)
    tick_pos = [n * f / 100 for f in TOP_PCTS]
    if is_bar:
        pad = max(5, 0.02 * n)
        x_left, x_right = -pad, n - 1 + pad
        tick_pos = [p - 0.5 for p in tick_pos]
        x_scale = "linear"
    elif x_scale == "log":
        ax.set_xscale("log")
        x_left, x_right = X_LEFT_LOG, n * 1.15      # padding so end points aren't clipped
    elif x_scale == "sqrt":
        ax.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)),
                                             lambda v: np.square(v)))
        x_left, x_right = 0, n * 1.05
    else:
        x_left, x_right = -0.02 * n, n * 1.02
    shade_left = x_left
    ax.set_xlim(x_left, x_right)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{f:g}%" for f in TOP_PCTS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", width=AXIS_LW, length=5)

    # translucent red band over the top 1% (left edge of the plot)
    shift = 0.5 if is_bar else 0
    band_patches = []
    for start, end, color, alpha, label in SHADE_BANDS:
        lo = shade_left if start == 0 else n * start / 100 - shift
        hi = n * end / 100 - shift
        ax.axvspan(lo, hi, color=color, alpha=alpha, linewidth=0, zorder=0)
        band_patches.append(Patch(facecolor=color, alpha=alpha * 2, label=label))

    scale_note = {"log": ", log scale", "sqrt": ", sqrt scale"}.get(x_scale, "")
    ax.set_xlabel(f"Top % of residues by predicted score{scale_note} (n = {n})")
    ax.set_ylabel("Predicted score")
    ax.set_title(title)
    ax.legend(
        handles=[Patch(color=GREEN, label="Positive"), Patch(color=GREY, label="Negative")]
                + band_patches,
        frameon=False,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(AXIS_LW)
    ax.tick_params(axis="y", width=AXIS_LW, length=5)
    fig.tight_layout()

    base, ext = os.path.splitext(out)
    if ext.lower() not in (".png", ".svg"):
        base = out
    fig.savefig(base + ".png")
    fig.savefig(base + ".svg")
    plt.close(fig)

    # quick sanity check printed to the console
    hit_ranks = [i + 1 for i, h in enumerate(is_hit) if h]
    print(f"{base}.png / .svg: {int(is_hit.sum())} positives among {len(merged)} residues; ranks = {hit_ranks}")
    if warn_unmatched:
        found = merged.loc[is_hit, RES_KEY].drop_duplicates()
        chk = hit_res.merge(found, on=RES_KEY, how="left", indicator=True)
        missing = chk[chk["_merge"] == "left_only"]
        if len(missing):
            print(f"  note: {len(missing)} hit residue(s) in hits file had no matching prediction row "
                  f"(residue/chain/warhead): {missing[RES_KEY].astype(str).agg(' '.join, axis=1).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("hits")
    ap.add_argument("-o", "--out", default="ranked_predictions.png",
                    help="output name; both .png and .svg are written")
    ap.add_argument("--style", choices=["scatter", "bar"], default="scatter",
                    help="plot style (default: scatter)")
    ap.add_argument("--point-size", type=float, default=4,
                    help="scatter size of negative points (default: 4)")
    ap.add_argument("--hit-size", type=float, default=14,
                    help="scatter size of positive points (default: 14)")
    ap.add_argument("--x-scale", choices=["log", "sqrt", "linear"], default="log",
                    help="x-axis spacing for scatter (default: log). Log gives the top 1%% "
                         "the most width, sqrt a compromise, linear the least")
    ap.add_argument("--all-warheads", action="store_true",
                    help="plot every prediction row instead of only rows whose Warhead "
                         "appears in the labels' matched_warheads")
    ap.add_argument("--per-name", action="store_true",
                    help="make one plot per Name in the predictions file")
    args = ap.parse_args()

    preds = normalize(load(args.predictions))
    hits = normalize(load(args.hits))
    hit_keys = explode_hits(hits)
    hit_res = hits[RES_KEY].drop_duplicates()

    # keep only prediction rows whose warhead appears in the labels' matched_warheads
    if not args.all_warheads:
        keep = set(hit_keys["Warhead"])
        before = len(preds)
        preds = preds[preds["Warhead"].isin(keep)].reset_index(drop=True)
        print(f"warhead filter: kept {len(preds)} of {before} prediction rows "
              f"(warheads: {sorted(keep)})")
        if preds.empty:
            raise SystemExit("no prediction rows match the labels' warheads")

    if args.per_name and "Name" in preds.columns:
        for name, sub in preds.groupby("Name"):
            safe = "".join(c if c.isalnum() else "_" for c in name)[:60]
            make_plot(sub, hit_keys, hit_res, str(name)[:80], f"ranked_{safe}",
                      args.style, warn_unmatched=False,
                      point_size=args.point_size, hit_size=args.hit_size,
                      x_scale=args.x_scale)
    else:
        make_plot(preds, hit_keys, hit_res, "Predicted score by residue", args.out, args.style,
                  point_size=args.point_size, hit_size=args.hit_size, x_scale=args.x_scale)


if __name__ == "__main__":
    main()