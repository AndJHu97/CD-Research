"""Paired comparison: full model (_all) vs noncovalent-only ablation (_nc).

Question (user): is _all better at predicting (target_rank, hit_at_top_pct)
  (a) for complexes WITHOUT strong noncovalent pre-organization, and
  (b) for complexes WITH strong noncovalent pre-organization,
compared with the noncovalent-only ablation (_nc)?

The eval unit is a row (one target residue in one structure). Rows are paired
across the two files on the full Name key.
"""
import os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_CSV = r"c:\Users\Andre\Downloads\test_label_results_all.csv"
NC_CSV = r"c:\Users\Andre\Downloads\test_label_results_nc.csv"
ANN = os.path.join(HERE, "TestSet_Noncovalent_Annotation_v2.csv")
_PENDING = os.path.join(HERE, "TestSet_Noncovalent_Annotation_v2_pending.csv")
if os.path.exists(_PENDING) and os.path.getmtime(_PENDING) >= os.path.getmtime(ANN):
    ANN = _PENDING
OUT = os.path.join(HERE, "AllVsNC_Comparison.csv")

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def load():
    a = pd.read_csv(ALL_CSV)
    n = pd.read_csv(NC_CSV)
    keep = ["Name", "target_rank", "hit_at_top_pct", "n_residues"]
    a = a[keep].rename(columns={"target_rank": "rank_all",
                                "hit_at_top_pct": "hit_all",
                                "n_residues": "nres_all"})
    n = n[["Name", "target_rank", "hit_at_top_pct"]].rename(
        columns={"target_rank": "rank_nc", "hit_at_top_pct": "hit_nc"})
    m = a.merge(n, on="Name", how="inner")
    m["PDB"] = m["Name"].str[:4].str.upper()
    ann = pd.read_csv(ANN)[["PDB", "Strong noncovalent?", "Target"]]
    m = m.merge(ann, on="PDB", how="left")
    m["group"] = np.where(m["Strong noncovalent?"] == "Yes",
                          "with_strong_nc", "without_strong_nc")
    return m


def summarize(df, label):
    d = df.dropna(subset=["rank_all", "rank_nc"]).copy()
    n = len(d)
    rank_all_better = int((d.rank_all < d.rank_nc).sum())
    rank_nc_better = int((d.rank_all > d.rank_nc).sum())
    rank_tie = int((d.rank_all == d.rank_nc).sum())
    hit_all_better = int((d.hit_all > d.hit_nc).sum())
    hit_nc_better = int((d.hit_all < d.hit_nc).sum())
    hit_tie = int((d.hit_all == d.hit_nc).sum())
    row = {
        "group": label,
        "n_rows": n,
        "mean_rank_all": round(d.rank_all.mean(), 2),
        "mean_rank_nc": round(d.rank_nc.mean(), 2),
        "median_rank_all": d.rank_all.median(),
        "median_rank_nc": d.rank_nc.median(),
        "mean_hit_top_pct_all": round(d.hit_all.mean(), 4),
        "mean_hit_top_pct_nc": round(d.hit_nc.mean(), 4),
        "rank_all_wins": rank_all_better,
        "rank_nc_wins": rank_nc_better,
        "rank_ties": rank_tie,
        "hit_all_wins": hit_all_better,
        "hit_nc_wins": hit_nc_better,
        "hit_ties": hit_tie,
    }
    if HAVE_SCIPY and (rank_all_better + rank_nc_better) > 0:
        try:
            stat, p = wilcoxon(d.rank_all, d.rank_nc)
            row["wilcoxon_p_rank"] = round(p, 6)
        except Exception:
            row["wilcoxon_p_rank"] = None
    return row


def main():
    m = load()
    print("merged rows:", len(m), " PDBs:", m.PDB.nunique())
    print("annotation coverage missing:", int(m["Strong noncovalent?"].isna().sum()))

    rows = []
    rows.append(summarize(m, "overall"))
    rows.append(summarize(m[m.group == "without_strong_nc"], "without_strong_nc"))
    rows.append(summarize(m[m.group == "with_strong_nc"], "with_strong_nc"))
    # finer: split without into No vs Limited
    rows.append(summarize(m[m["Strong noncovalent?"] == "Limited"], "limited_only"))
    rows.append(summarize(m[m["Strong noncovalent?"] == "No"], "no_only"))

    out = pd.DataFrame(rows)
    try:
        out.to_csv(OUT, index=False)
        print("wrote", OUT)
    except PermissionError:
        alt = OUT.replace(".csv", "_reviewed.csv")
        out.to_csv(alt, index=False)
        print("LOCKED, wrote", alt)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))

    # also save a per-PDB collapsed view for the without-strong-nc group
    return m, out


if __name__ == "__main__":
    main()
