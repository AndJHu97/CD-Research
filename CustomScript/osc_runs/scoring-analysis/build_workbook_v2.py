"""Build the final deliverable workbook:
  - Summary (hypothesis + findings)
  - Model comparison (_all vs _nc, paired)
  - Test set (486-PDB noncovalent annotation)
  - Definitions
"""
import os
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_CSV = r"c:\Users\Andre\Downloads\test_label_results_all.csv"
NC_CSV = r"c:\Users\Andre\Downloads\test_label_results_nc.csv"
ANN = os.path.join(HERE, "TestSet_Noncovalent_Annotation_v2.csv")
_PENDING = os.path.join(HERE, "TestSet_Noncovalent_Annotation_v2_pending.csv")
if os.path.exists(_PENDING) and os.path.getmtime(_PENDING) >= os.path.getmtime(ANN):
    ANN = _PENDING
CMP = os.path.join(HERE, "AllVsNC_Comparison.csv")
_CMP_ALT = os.path.join(HERE, "AllVsNC_Comparison_reviewed.csv")
if os.path.exists(_CMP_ALT) and (not os.path.exists(CMP) or os.path.getmtime(_CMP_ALT) >= os.path.getmtime(CMP)):
    CMP = _CMP_ALT
OUT = os.path.join(HERE, "TestSet_Noncovalent_Annotation_v2.xlsx")

HEAD_FILL = PatternFill("solid", fgColor="1F4E78")
HEAD_FONT = Font(bold=True, color="FFFFFF", size=11)
SUB_FILL = PatternFill("solid", fgColor="D9E1F2")
BORDER = Border(*(Side(style="thin", color="BFBFBF"),) * 4)
WRAP = Alignment(wrap_text=True, vertical="top")
TOP = Alignment(vertical="top")


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BORDER


def write_df(ws, df, start=1, widths=None, wrap_cols=()):
    cols = list(df.columns)
    for j, name in enumerate(cols, 1):
        ws.cell(row=start, column=j, value=name)
    style_header(ws, start, len(cols))
    for i, (_, r) in enumerate(df.iterrows(), start + 1):
        for j, name in enumerate(cols, 1):
            v = r[name]
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            cell = ws.cell(row=i, column=j, value=v)
            cell.border = BORDER
            cell.alignment = WRAP if name in wrap_cols else TOP
    if widths:
        for j, name in enumerate(cols, 1):
            ws.column_dimensions[get_column_letter(j)].width = widths.get(name, 16)
    ws.freeze_panes = ws.cell(row=start + 1, column=1)


def main():
    ann = pd.read_csv(ANN)
    cmp = pd.read_csv(CMP)

    order = ["PDB", "Strong noncovalent?", "Notes",
             "Before Modern Covalent Generation (No crystal structures)",
             "All Natural?", "Not formed from substrate likeness if noncovalent",
             "Ligand", "Target", "Covalent residue", "Ligand class",
             "Release year", "Structure-guided design?",
             "Ligand heavy atoms (approx)", "Confidence"]
    ann = ann[[c for c in order if c in ann.columns]]
    ann = ann.sort_values("PDB").reset_index(drop=True)

    wb = Workbook()

    # ---- Summary ----
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Covalent test set: noncovalent pre-organization & model ablation"
    ws["A1"].font = Font(bold=True, size=14)
    by = {r.group: r for _, r in cmp.iterrows()}
    strong = by["with_strong_nc"]
    weak = by["without_strong_nc"]
    none_ = by["no_only"]
    lines = [
        "",
        "Dataset: 486 covalent complexes (CovalentInDB 2.0 test set), 741 scored target-residue rows.",
        "",
        "Hypothesis being tested:",
        "  Covalent complexes in the PDB are mostly already optimized noncovalently for their site,",
        "  so a noncovalent-only model (ablation, _nc) can rank the right residue without covalent features.",
        "",
        "Annotation of noncovalent pre-organization (per complex):",
        f"  Strong noncovalent recognition (Yes): {int((ann['Strong noncovalent?']=='Yes').sum())} PDBs",
        f"  Limited noncovalent recognition:      {int((ann['Strong noncovalent?']=='Limited').sum())} PDBs",
        f"  No real noncovalent recognition:      {int((ann['Strong noncovalent?']=='No').sum())} PDBs",
        "  (Strong = peptidomimetics, substrate/transition-state mimics, natural-product inhibitors,",
        "   clinical drugs, and elaborated non-peptide scaffolds with real pocket contacts.",
        "   Limited/No = small covalent fragments and reactive small molecules that rely mainly on the warhead.)",
        "",
        "Key finding (full model _all vs noncovalent-only ablation _nc; lower target_rank is better,",
        "higher hit_at_top_pct is better):",
        "",
        f"  1. WHERE noncovalent pre-organization is STRONG ({int(strong.n_rows)} rows): the two models are ~tied.",
        f"     Medians {strong.median_rank_all:g} vs {strong.median_rank_nc:g}; paired rank wins "
        f"{int(strong.rank_all_wins)} (all) vs {int(strong.rank_nc_wins)} (nc), Wilcoxon p={strong.wilcoxon_p_rank};",
        f"     hit_at_top_pct {strong.mean_hit_top_pct_all:.3f} vs {strong.mean_hit_top_pct_nc:.3f}.",
        "",
        f"  2. WHERE noncovalent pre-organization is WEAK/ABSENT ({int(weak.n_rows)} rows): compare mean rank "
        f"{weak.mean_rank_all:.2f} (all) vs {weak.mean_rank_nc:.2f} (nc);",
        f"     paired rank wins {int(weak.rank_all_wins)} (all) vs {int(weak.rank_nc_wins)} (nc), Wilcoxon p={weak.wilcoxon_p_rank};",
        f"     hit_at_top_pct {weak.mean_hit_top_pct_all:.3f} vs {weak.mean_hit_top_pct_nc:.3f}.",
        "",
        f"  3. The purely covalent-dependent subset (No noncovalent, {int(none_.n_rows)} rows):",
        f"     hit_at_top_pct {none_.mean_hit_top_pct_all:.3f} (all) vs {none_.mean_hit_top_pct_nc:.3f} (nc);",
        f"     paired rank wins {int(none_.rank_all_wins)} (all) vs {int(none_.rank_nc_wins)} (nc).",
        "",
        "Interpretation: the noncovalent-only ablation looks strong overall precisely because the test set",
        "is dominated by complexes that are already noncovalently pre-organized. It degrades on the cases",
        "that genuinely require covalent reasoning, where the full model recovers them. This is consistent",
        "with the theory that current covalent complexes are mostly pre-optimized noncovalently for their site.",
        "",
        "Caveat: overall MEAN target_rank can favor _nc because a few catastrophic outliers (ranks near 1700)",
        "sit in both models. Medians, paired win counts, and hit_at_top_pct are the robust metrics. See",
        "'Model comparison'.",
    ]
    for i, t in enumerate(lines, 2):
        ws.cell(row=i, column=1, value=t)
    ws.column_dimensions["A"].width = 110

    # ---- Model comparison ----
    ws2 = wb.create_sheet("Model comparison")
    ws2["A1"] = "Full model (_all) vs noncovalent-only ablation (_nc) - paired by scored row"
    ws2["A1"].font = Font(bold=True, size=12)
    note = ("rank_*_wins = # rows where that model gives the lower (better) target_rank; "
            "ties mostly rank=1 identical. hit_*_wins = higher hit_at_top_pct. "
            "wilcoxon_p_rank = paired signed-rank test on target_rank.")
    ws2["A2"] = note
    ws2["A2"].alignment = WRAP
    ws2.merge_cells("A2:N2")
    ws2.row_dimensions[2].height = 45
    cmp_w = {c: 15 for c in cmp.columns}
    cmp_w["group"] = 20
    write_df(ws2, cmp, start=4, widths=cmp_w)
    # group rename for readability handled in csv already

    # ---- Test set ----
    ws3 = wb.create_sheet("Test set")
    widths = {
        "PDB": 8, "Strong noncovalent?": 13, "Notes": 70,
        "Before Modern Covalent Generation (No crystal structures)": 18,
        "All Natural?": 10, "Not formed from substrate likeness if noncovalent": 16,
        "Ligand": 30, "Target": 30, "Covalent residue": 18, "Ligand class": 22,
        "Release year": 9, "Structure-guided design?": 12,
        "Ligand heavy atoms (approx)": 11, "Confidence": 10,
    }
    write_df(ws3, ann, start=1, widths=widths,
             wrap_cols=("Notes", "Ligand", "Target"))

    # ---- Definitions ----
    ws4 = wb.create_sheet("Definitions")
    defs = [
        ("Column", "Meaning"),
        ("Strong noncovalent?",
         "Yes = ligand has strong site-specific noncovalent recognition independent of the warhead "
         "(peptidomimetic, substrate/TS mimic, natural product, clinical drug). Limited = partial, "
         "mostly fragment-scale. No = relies essentially on the covalent warhead."),
        ("Notes", "Ligand identity, purpose, and basis for the noncovalent call."),
        ("Before Modern Covalent Generation (No crystal structures)",
         "Yes if the inhibitor predates structure-based covalent drug design (natural product / "
         "classical mechanistic probe), i.e. not designed from a crystal structure."),
        ("All Natural?", "Yes if the ligand is a natural product or direct natural metabolite/substrate analog."),
        ("Not formed from substrate likeness if noncovalent",
         "Yes if the noncovalent recognition does NOT come from mimicking the native substrate "
         "(blank when Strong noncovalent? = No)."),
        ("Covalent residue", "Residue(s) forming the covalent bond, from RCSB."),
        ("Ligand class", "Warhead / chemotype grouping."),
        ("Release year", "PDB release year (proxy for design era)."),
        ("Ligand heavy atoms (approx)", "Heavy-atom proxy parsed from the ligand name (scaffold-size indicator)."),
        ("Confidence", "Annotator confidence (High/Medium) in the noncovalent call."),
    ]
    for i, (a, b) in enumerate(defs, 1):
        ws4.cell(row=i, column=1, value=a)
        ws4.cell(row=i, column=2, value=b).alignment = WRAP
    style_header(ws4, 1, 2)
    ws4.column_dimensions["A"].width = 42
    ws4.column_dimensions["B"].width = 100

    try:
        wb.save(OUT)
        print("wrote", OUT)
    except PermissionError:
        alt = OUT.replace(".xlsx", "_pending.xlsx")
        wb.save(alt)
        print("LOCKED, wrote", alt)
    print("sheets:", wb.sheetnames)


if __name__ == "__main__":
    main()
