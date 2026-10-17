"""Annotate the corrected covalent test set (test_label_results_all/_nc).

Produces the same columns as the user's started spreadsheet
(TestSetNoncovalentScaffold2(Right).xlsx), then compares the full model
(_all) against the noncovalent-only ablation (_nc) by target_rank and
hit_at_top_pct, split by whether the complex already has strong
noncovalent recognition at the reactive site.

Sources for each call: PDB title, deposited ligand, source organism, and
the covalent residue recorded in the label file. Large families
(proteasome, coronavirus 3CL/Mpro) are classified by rule; the 224 PDBs
shared with the earlier run reuse those curated calls; the 9 structures
the user already annotated are copied verbatim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALL_CSV = Path(r"c:\Users\Andre\Downloads\test_label_results_all.csv")
NC_CSV = Path(r"c:\Users\Andre\Downloads\test_label_results_nc.csv")
META = HERE / "_rcsb_meta_v2.json"
V1 = HERE / "TestSet_Noncovalent_Annotation.csv"      # reuse curated overlap calls
STARTED = HERE / "TestSetNoncovalentScaffold2(Right).xlsx"
OUT_XLSX = HERE / "TestSet_Noncovalent_Annotation_v2.xlsx"
OUT_CSV = HERE / "TestSet_Noncovalent_Annotation_v2.csv"

STRONG = "Strong noncovalent?"
BEFORE = "Before Modern Covalent Generation (No crystal structures)"
NAT = "All Natural?"
SUB = "Not formed from substrate likeness if noncovalent"

IONS = {
    "MG", "CL", "NA", "K", "CA", "ZN", "SO4", "PO4", "GOL", "EDO", "MES", "ACT",
    "PEG", "PGE", "P6G", "1PE", "MN", "FE", "CD", "BME", "DMS", "IOD", "BR",
    "TRS", "TAM", "NAG", "FMT", "CO", "NI", "CU", "ACY", "MPD", "MRD", "SCN",
    "UNX", "FLC", "EPE", "NO3", "PE4", "P4G", "LI", "RB", "CS", "B3P", "PG4",
    "1PG", "2PE", "BTB", "IPA", "IMD", "SPM", "FUC", "GAL", "BMA", "HEM", "NAP",
    "GTP", "HX2", "PGO", "PGR", "PNH", "GAI", "BCT", "K", "SGM", "SGL", "DMSO",
}


def rec(ligand, target, klass, strong, before, natural, not_substrate, guided,
        confidence, notes):
    if strong == "No":
        not_substrate = ""
    return dict(ligand=ligand, target=target, klass=klass, strong=strong,
                before=before, natural=natural, not_substrate=not_substrate,
                guided=guided, confidence=confidence, notes=notes)


# --- the 9 structures the user already filled in on the started sheet ----------
STARTED_EXACT = {
    "1B3N": rec("Cerulenin", "Beta-ketoacyl-ACP synthase", "Natural fatty-acid mimic",
                "Yes", "Yes", "Yes", "No", "No", "High",
                "Cerulenin is a natural covalent inhibitor of B-ketoacyl-ACP synthase. Uses the unsaturated chain for pocket noncovalent recognition."),
    "1EB3": rec("4,7-dioxosebacic acid", "5-aminolaevulinic acid dehydratase", "Substrate analog",
                "Yes", "Yes", "No", "No", "No", "High",
                "Synthetic and reproduces the substrate-binding sites with strong noncovalent recognition. Used to investigate the catalytic mechanism of dehydratase."),
    "1FJ8": rec("Cerulenin", "Beta-ketoacyl-ACP synthase", "Natural fatty-acid mimic",
                "Yes", "Yes", "Yes", "No", "No", "High",
                "Cerulenin is a natural-product covalent inhibitor of beta-ketoacyl-ACP synthase. Its binding mimics the condensation transition state, with the unsaturated alkyl chain occupying a hydrophobic pocket and additional active-site interactions providing noncovalent recognition before reaction with Cys163. Used to study FabF/FabB inhibition."),
    "1FVI": rec("AMP", "Chlorella virus DNA ligase", "Adenylate catalytic intermediate",
                "Yes", "Yes", "Yes", "No", "No", "High",
                "AMP is covalently attached to Lys27 as the normal adenylated intermediate of DNA ligase. AMP has strong noncovalent recognition within the nucleotide-binding site, but this is a catalytic intermediate rather than a designed covalent inhibitor."),
    "1KE3": rec("4,4'-biphenyldiboronic acid", "AmpC beta-lactamase", "Boronic acid",
                "Yes", "Yes", "No", "No", "No", "High",
                "4,4'-Biphenyldiboronic acid is a synthetic boronic-acid inhibitor with substantial noncovalent recognition. Its biphenyl group interacts with the AmpC binding site while the boronic acid occupies the oxyanion-hole region and forms a reversible covalent interaction with Ser64."),
    "1OXM": rec("TC4 triglyceride phosphonate", "Cutinase", "Phosphonate suicide inhibitor",
                "Yes", "Yes", "No", "No", "No", "High",
                "TC4 is a synthetic triglyceride analog containing a phosphonate. Its alkyl chains reproduce substrate-recognition features, while the phosphonate mimics the first tetrahedral intermediate and forms the covalent adduct with Ser120."),
    "1P9U": rec("Peptide alpha-halo carbonyl", "Coronavirus main protease (3CLpro)", "Peptide warhead",
                "Yes", "Yes", "No", "No", "No", "High",
                "Synthetic peptide/substrate analog containing an alpha-halo carbonyl/alkyl chloride warhead. The peptide portion reproduces substrate-binding interactions and localizes the reactive group near Cys144. Used as a covalent coronavirus main-protease inhibitor and for structural anti-coronavirus drug-design studies."),
    "1QFE": rec("3-dehydroshikimate", "Type I 3-dehydroquinate dehydratase", "Catalytic intermediate",
                "Yes", "Yes", "Yes", "No", "No", "High",
                "3-Dehydroshikimate is a natural substrate/product-related metabolite that becomes covalently attached to Lys170 through a Schiff base. It has strong noncovalent recognition because it occupies the normal substrate-binding site, but this is catalytic chemistry rather than a designed covalent inhibitor."),
    "1QWZ": rec("MTSET", "Sortase B", "Small reactive probe",
                "No", "Yes", "No", "", "No", "High",
                "MTSET is a small synthetic methanethiosulfonate cysteine-reactive reagent. It has little ligand-derived noncovalent recognition; localization is primarily through access/reactivity at catalytic Cys223 rather than a substantial recognition scaffold."),
}


# --- proteasome classifier (same logic as v1) ---------------------------------
def _prot_note(ligand, origin, warhead, purpose, scaffold):
    kind = {"natural": "natural-product", "semisynthetic": "semisynthetic"}.get(origin, "synthetic")
    if scaffold == "peptide":
        body = (f"Its peptide scaffold provides strong noncovalent recognition in the "
                f"proteasome substrate channel, while the {warhead} reacts with the N-terminal threonine.")
    elif scaffold == "natural-product":
        tail = ("It was discovered as a natural product rather than built as a substrate-sequence copy."
                if origin == "natural" else
                "The scaffold comes from a natural-product inhibitor rather than a substrate-sequence peptide; this analog was prepared synthetically.")
        body = (f"Its scaffold provides strong noncovalent recognition of the proteasome substrate sites, "
                f"while the {warhead} reacts with the N-terminal threonine. {tail}")
    else:
        body = (f"Its scaffold provides strong noncovalent recognition of the proteasome, "
                f"while the {warhead} reacts with the N-terminal threonine.")
    return f"{ligand} is a {kind} covalent proteasome inhibitor. {body} {purpose}"


def _purpose(kind):
    return {
        "natural": "Used to establish how this natural product recognizes and covalently inhibits the proteasome.",
        "derivative": "Used to compare a synthetic analog of a natural-product proteasome inhibitor.",
        "clinical": "Used as a covalent proteasome drug and to define its binding mode at the active site.",
        "selectivity": "Used for structure-guided comparison of peptide sequence and subunit selectivity.",
    }.get(kind, "Used to relate warhead chemistry and peptide sequence to proteasome inhibition.")


def classify_proteasome(title):
    t = title.lower()

    def make(ligand, origin, warhead, not_sub, before, guided, scaffold, klass, pk, conf="High"):
        if origin != "natural" and pk == "natural":
            pk = "derivative"
        return rec(ligand, "Proteasome", klass, "Yes", before,
                   "Yes" if origin == "natural" else "No", not_sub, guided, conf,
                   _prot_note(ligand, origin, warhead, _purpose(pk), scaffold))

    if "calpain" in t:
        return rec("Calpain inhibitor I", "Proteasome", "Peptide aldehyde",
                   "Yes", "Yes", "No", "No", "No", "High",
                   "Calpain inhibitor I is a synthetic peptide-like aldehyde inhibitor. Its peptide sequence provides strong noncovalent recognition within the proteasome substrate-binding channel, while the aldehyde forms the covalent interaction with N-terminal Thr.")
    if "mln-273" in t or "peptidyl boronate" in t:
        return make("MLN-273", "synthetic", "boronic acid", "No", "Yes", "Partial", "peptide", "Peptide boronate", "classical")
    if "substrate analogue" in t or "substrate analog" in t:
        return make("Peptide vinyl-sulfone substrate analog", "synthetic", "vinyl sulfone", "No", "Yes", "No", "peptide", "Substrate-sequence peptide", "classical")
    if "epoxomicin" in t:
        return rec("Epoxomicin", "Proteasome", "Natural epoxyketone", "Yes", "Yes", "Yes", "Yes", "No", "High",
                   "Epoxomicin is a natural-product peptide containing an alpha,beta-epoxyketone. Its peptide scaffold provides strong noncovalent recognition of the proteasome, while the epoxyketone reacts with N-terminal Thr to form a stable covalent adduct.")
    if "dihydroeponemycin" in t:
        return make("Dihydroeponemycin", "semisynthetic", "epoxyketone", "Yes", "Yes", "No", "natural-product", "Natural-product epoxyketone", "natural")
    if "homosalinosporamide" in t:
        return make("Homosalinosporamide A", "semisynthetic", "beta-lactone", "Yes", "No", "Yes", "natural-product", "Salinosporamide analog", "derivative")
    if "hydroxylated salinosporamide" in t or "fluorosalinosporamide" in t or "salinosporamide derivatives" in t:
        return make("Salinosporamide derivative", "semisynthetic", "beta-lactone", "Yes", "No", "Yes", "natural-product", "Salinosporamide analog", "derivative")
    if "salinosporamide a" in t:
        return make("Salinosporamide A", "natural", "beta-lactone", "Yes", "Yes", "No", "natural-product", "Natural beta-lactone", "natural")
    if "omuralide" in t or "spirolactacystin" in t:
        ligand = "Spirolactacystin epimer" if "epimer" in t else ("Spirolactacystin" if "spirolactacystin" in t else "Omuralide")
        return make(ligand, "semisynthetic", "beta-lactone", "Yes", "Yes", "No", "natural-product", "Lactacystin-family beta-lactone", "natural")
    if "chimera" in t:
        return make("Syringolin-glidobactin chimera", "synthetic", "Michael acceptor", "Yes", "No", "Yes", "natural-product", "Syrbactin analog", "derivative")
    if "syringolin" in t:
        return make("Syringolin B" if "syringolin b" in t else "Syringolin A", "natural", "Michael acceptor", "Yes", "Yes", "No", "natural-product", "Natural syrbactin", "natural")
    if "glidobactin-like" in t or "hb33" in t:
        return make("Glidobactin-like natural product", "natural", "Michael acceptor", "Yes", "Yes", "No", "natural-product", "Natural syrbactin", "natural")
    if "cepafungin" in t:
        return make("Cepafungin I", "natural", "Michael acceptor", "Yes", "Yes", "No", "natural-product", "Natural syrbactin", "natural")
    if "glidobactin" in t:
        return make("Glidobactin A", "natural", "Michael acceptor", "Yes", "Yes", "No", "natural-product", "Natural syrbactin", "natural")
    if "fellutamide" in t:
        return make("Fellutamide B", "natural", "aldehyde", "Yes", "Yes", "No", "peptide", "Natural peptide aldehyde", "natural")
    if "homobelactosin" in t:
        return make("Homobelactosin C", "natural", "beta-lactone", "Yes", "Yes", "No", "natural-product", "Natural belactosin", "natural")
    if "belactosin" in t:
        if "derivative" in t or " 3e" in t:
            return make("Belactosin derivative", "synthetic", "beta-lactone", "Yes", "No", "Yes", "natural-product", "Belactosin analog", "derivative")
        return make("Belactosin C", "natural", "beta-lactone", "Yes", "Yes", "No", "natural-product", "Natural belactosin", "natural")
    if "carmaphycin a" in t or "natural product carmaphycin" in t:
        return make("Carmaphycin A", "natural", "epoxyketone", "Yes", "Yes", "No", "peptide", "Natural epoxyketone", "natural")
    if "carmaphycin" in t:
        return make("Carmaphycin analogue", "synthetic", "epoxyketone or vinyl ketone", "No", "No", "Yes", "peptide", "Designed epoxyketone", "selectivity")
    if "vibralactone" in t:
        return make("Vibralactone", "natural", "beta-lactone", "Yes", "Yes", "No", "natural-product", "Natural beta-lactone", "natural")
    if "cystargolide b derivative" in t:
        return make("Cystargolide B derivative", "synthetic", "epoxyketone-like warhead", "Yes", "No", "Yes", "natural-product", "Cystargolide analog", "derivative")
    if "cystargolide" in t:
        return make("Cystargolide B", "natural", "beta-lactone / epoxyketone-like warhead", "Yes", "Yes", "No", "natural-product", "Natural beta-lactone", "natural")
    if "macyranone" in t:
        return make("Macyranone A", "synthetic", "epoxyketone-like warhead", "Yes", "No", "Yes", "natural-product", "Syrbactin analog", "derivative", "Medium")
    if "bortezomib" in t:
        return make("Bortezomib", "synthetic", "boronic acid", "No", "Yes", "Partial", "peptide", "Clinical peptide boronate", "clinical")
    if "ixazomib" in t:
        return make("Ixazomib", "synthetic", "boronic acid", "No", "No", "Yes", "peptide", "Clinical peptide boronate", "clinical")
    if "delanzomib" in t:
        return make("Delanzomib", "synthetic", "boronic acid", "No", "No", "Yes", "peptide", "Clinical peptide boronate", "clinical")
    if "carfilzomib" in t:
        return make("Carfilzomib", "synthetic", "epoxyketone", "No", "Yes", "Partial", "peptide", "Clinical peptide epoxyketone", "clinical")
    if "onx 0914" in t or "pr-957" in t:
        return make("ONX 0914 (PR-957)", "synthetic", "epoxyketone", "No", "No", "Yes", "peptide", "Immunoproteasome-selective epoxyketone", "selectivity")
    if "pr-924" in t:
        return make("PR-924", "synthetic", "epoxyketone", "No", "No", "Yes", "peptide", "Immunoproteasome-selective epoxyketone", "selectivity")
    if "mg132" in t or "mg-132" in t:
        return make("MG132", "synthetic", "aldehyde", "No", "Yes", "No", "peptide", "Peptide aldehyde", "classical")
    if "z-leu-leu-leu" in t or "boc-ala-ala-ala" in t or "bsc4999" in t or "alpha-keto" in t or "keto phenylamide" in t:
        ligand = "Boc-Ala-Ala-Ala-CHO" if "boc-ala" in t else ("Alpha-ketoamide peptide" if ("bsc4999" in t or "alpha-keto" in t or "keto phenylamide" in t) else "Z-Leu-Leu-Leu warhead variant")
        return make(ligand, "synthetic", "aldehyde, boronate, vinyl sulfone, or ketoamide", "No", "Yes", "Partial", "peptide", "Substrate-sequence peptide", "classical")
    if "chloroacetamide" in t:
        return rec("Peptide alpha-chloroacetamide", "Proteasome", "Engineered-cysteine peptide probe",
                   "Yes", "No", "No", "No", "Yes", "High",
                   "This is a synthetic carfilzomib-derived peptide bearing an alpha-chloroacetamide. The peptide provides strong noncovalent recognition in the proteasome substrate channel, and the chloroacetamide covalently modifies an engineered non-catalytic cysteine rather than the catalytic threonine.")
    if "alkyne-pi" in t or "octreotide" in t:
        return make("Peptide boronate conjugate", "synthetic", "boronic acid", "No", "No", "Yes", "peptide", "Peptide boronate", "selectivity")
    if "-ep" in t or "epoxyketone" in t:
        return make("Peptide epoxyketone", "synthetic", "epoxyketone", "No", "No", "Yes", "peptide", "Designed peptide epoxyketone", "selectivity")
    if "plasmodium" in t:
        return make("Plasmodium-selective peptide inhibitor", "synthetic", "vinyl sulfone", "No", "No", "Yes", "peptide", "Designed peptide inhibitor", "selectivity")
    if "vinyl sulfone" in t or "ada-" in t or "(leu)3-vinyl" in t:
        return make("Peptide vinyl sulfone", "synthetic", "vinyl sulfone", "No", "No", "Yes", "peptide", "Peptide vinyl sulfone", "selectivity")
    if "boronic acid" in t or "compound 1" in t:
        return make("Peptide-like boronic acid", "synthetic", "boronic acid", "No", "No", "Yes", "peptide", "Designed peptide boronate", "selectivity")
    if " complex with " in t:
        tail = t.split(" complex with ", 1)[1]
        if tail[:1].isdigit() or tail.startswith("lx"):
            return make("Subunit-selective peptide inhibitor", "synthetic", "epoxyketone or vinyl sulfone", "No", "No", "Yes", "peptide", "Designed peptide inhibitor", "selectivity")
    return None


# --- coronavirus main protease (3CLpro / Mpro) --------------------------------
MPRO_SUBSTRATE_MARKERS = ("oxopyrrolidin", "oxopiperidin", "oxopyrrolidine", "glutamin")
MPRO_DRUGS = {
    "boceprevir": ("Boceprevir", "No"), "telaprevir": ("Telaprevir", "No"),
    "narlaprevir": ("Narlaprevir", "No"), "rupintrivir": ("Rupintrivir", "No"),
    "gc376": ("GC376", "No"), "gc-376": ("GC376", "No"),
    "n3": ("N3", "No"), "carmofur": ("Carmofur", "No"),
}


def classify_mpro(title, ligtext, hv):
    t = title.lower()
    lg = ligtext.lower()
    target = "Coronavirus main protease (3CLpro/Mpro)"

    # classical / natural covalent peptides that happen to be soaked into Mpro
    if "leupeptin" in t or "leupeptin" in lg:
        return rec("Leupeptin", target, "Natural peptide aldehyde", "Yes", "Yes", "Yes", "No", "No", "High",
                   "Leupeptin is a microbial natural-product peptide aldehyde. Its peptide reproduces substrate contacts in the Mpro active site and the aldehyde forms a reversible covalent hemithioacetal with the catalytic cysteine.")
    if "mg-132" in t or "mg132" in t or re.search(r"\bald\b", lg):
        return rec("MG132", target, "Peptide aldehyde", "Yes", "Yes", "No", "No", "No", "High",
                   "MG132 is a synthetic peptide aldehyde (Z-Leu-Leu-Leu-al). Its peptide reproduces substrate contacts and the aldehyde forms a reversible covalent adduct with the catalytic cysteine of Mpro.")
    if "z-vad" in t or "z-vad" in lg or "vad-fmk" in t:
        return rec("Z-VAD-FMK", target, "Peptide fluoromethyl ketone", "Yes", "Yes", "No", "No", "No", "High",
                   "Z-VAD-FMK is a synthetic peptide fluoromethyl ketone. Its peptide reproduces substrate contacts and the fluoromethyl ketone covalently modifies the catalytic cysteine.")

    # Raw PanDDA / PCM fragments stay fragments even if a pyrrolidinone substring
    # is present (that marker false-positives on small non-peptide hits such as 5RFQ).
    is_raw_fragment = ("pcm-" in t) or ("fragment" in t and "computational design" not in t and hv < 26)
    drug = next((v[0] for k, v in MPRO_DRUGS.items() if k in t or k in lg), None)
    has_substrate_marker = any(m in lg for m in MPRO_SUBSTRATE_MARKERS)
    is_peptidomimetic = (not is_raw_fragment) and (
        drug is not None or has_substrate_marker or hv >= 26
    )
    if is_peptidomimetic:
        # Size alone is not evidence of a substrate-register peptide. Moonshot
        # elaborations and non-peptide aldehydes/acrylamides bind strongly but
        # are not P1-lactam peptidomimetics.
        substrate_like = drug is not None or has_substrate_marker or any(
            k in t for k in ("peptid", "leupeptin", "mg-132", "mg132"))
        if substrate_like:
            name = drug or "Peptidomimetic 3CL protease inhibitor"
            note = (f"{name} is a synthetic peptidomimetic coronavirus 3CL/Mpro inhibitor. Its peptide-like "
                    f"scaffold reproduces the substrate register (including the P1 glutamine-lactam surrogate) "
                    f"and provides strong noncovalent recognition of the active site, while the warhead "
                    f"reacts with the catalytic cysteine.")
            return rec(name, target, "Peptidomimetic protease inhibitor", "Yes", "No", "No", "No", "Yes", "High", note)
        name = "Non-peptide covalent Mpro inhibitor"
        note = (f"A synthetic covalent Mpro inhibitor (encoded ligand ~{hv} heavy atoms) whose scaffold "
                f"is not a substrate-sequence peptide. The scaffold still makes substantial active-site "
                f"contacts, so noncovalent recognition is strong; the electrophile then reacts with the "
                f"catalytic cysteine. This includes elaborated Moonshot / non-peptide aldehyde series.")
        return rec(name, target, "Non-peptide covalent inhibitor", "Yes", "No", "No", "Yes", "Yes", "Medium", note)

    # fragment-screen / small covalent hit
    strong = "Limited" if hv >= 13 else "No"
    note = (f"This is a small fragment-screen or fragment-sized covalent hit against SARS-CoV-2 Mpro "
            f"(encoded ligand ~{hv} heavy atoms). It makes only limited active-site contact and was not "
            f"built as a substrate copy; the electrophile localizes it near the catalytic cysteine. "
            f"These electrophile-first fragments are the clearest cases of weak pre-existing noncovalent recognition.")
    return rec("Covalent fragment", target, "Fragment-screen covalent hit", strong, "No", "No",
               "", "Yes", "Medium", note)


# --- other mechanistic / drug-like families -----------------------------------
def classify_other(title, prots, ligtext, lig_name, hv):
    t = title.lower()
    p = prots.lower()
    lg = ligtext.lower()
    s = t + " " + p

    def R(ligand, target, klass, strong, before, natural, not_sub, guided, conf, note):
        return rec(ligand, target, klass, strong, before, natural, not_sub, guided, conf, note)

    if "dehydroquinate" in s or "dehydroquinase" in s or "aroD" in s or "dhq1" in s:
        natural = "Yes" if ("dehydroshikimate" in lg or "dehydroquinate" in lg or "dhk" in lg) else "No"
        before = "Yes" if natural == "Yes" else "No"
        return R(lig_name or "dehydroquinate-site ligand", "Type I 3-dehydroquinate dehydratase (DHQ1)",
                 "Substrate/analog Schiff base", "Yes", before, natural, "No",
                 "No" if natural == "Yes" else "Yes", "Medium",
                 "A cyclohexane-carboxylate substrate or close substrate analog that binds the DHQ1 active site and forms a covalent Schiff base with the catalytic lysine. Recognition is the normal substrate pocket; the warhead chemistry is the catalytic imine.")
    if "ketoacyl" in s or "acyl carrier protein synthase" in s or "fabf" in s or "fabb" in s:
        natural = "Yes" if "cer" in lg or "cerulenin" in lg else "No"
        return R(lig_name or "cerulenin-like inhibitor", "Beta-ketoacyl-ACP synthase (FabB/FabF)",
                 "Natural fatty-acid mimic", "Yes", "Yes", natural, "No", "No", "High",
                 "A cerulenin-type fatty-acid-mimicking covalent inhibitor. The acyl chain occupies the hydrophobic condensation pocket (strong substrate-like recognition) and the epoxide/keto warhead reacts with the active-site cysteine.")
    if "dna ligase" in s:
        return R("AMP", "DNA ligase", "Adenylate catalytic intermediate", "Yes", "Yes", "Yes", "No", "No", "High",
                 "AMP is covalently attached to the catalytic lysine as the normal adenylated ligase intermediate. It has strong noncovalent recognition in the nucleotide pocket, but this is catalytic chemistry rather than a designed inhibitor.")
    if "beta-lactamase" in s:
        return R(lig_name or "boronic-acid inhibitor", "Beta-lactamase", "Boronic acid", "Yes", "Yes", "No", "No", "No", "High",
                 "A synthetic boronic-acid inhibitor. The aromatic scaffold gives real noncovalent recognition of the active site while the boron forms a reversible covalent bond with the catalytic serine.")
    if "cutinase" in s:
        if "pincer" in s or "hybrid" in s:
            return R(lig_name or "NCN-pincer phosphonate", "Cutinase", "Organometallic phosphonate-pincer",
                     "Limited", "No", "No", "Yes", "Yes", "High",
                     "An organometallic NCN-Pt/Pd phosphonate-pincer reagent selectively modifies Ser120. The phosphonate is the covalent anchor. The pincer-metal head group occupies some active-site space but largely projects into solvent; it was not a substrate-mimicking recognition scaffold. This is distinct from an alkyl phosphonate such as 1XZL, whose hydrocarbon chain occupies the substrate-binding groove.")
        return R(lig_name or "phosphonate/phosphinate", "Cutinase", "Phosphonate suicide inhibitor",
                 "Yes", "Yes", "No", "No", "No", "High",
                 "A phosphonate/phosphinate that mimics the acyl substrate and forms a covalent adduct with the catalytic serine; the alkyl chains occupy the substrate-binding groove.")
    if "sortase" in s:
        if "e64" in lg or "e-64" in lg:
            return R("E-64", "Sortase B", "Natural epoxysuccinyl peptide", "Yes", "Yes", "Yes", "No", "No", "High",
                     "E-64 is a natural-product epoxysuccinyl peptide. Its peptide provides substrate-like recognition and the epoxide covalently modifies the catalytic cysteine.")
        if "gly-gly" in s or "gly3" in t:
            return R("MTSET-like adduct", "Sortase B", "Small reactive probe", "No", "Yes", "No", "", "No", "High",
                     "A Gly3 substrate peptide is co-bound, but it is not the covalent ligand. The scored covalent species is the small trimethylammonium-ethyl adduct of an MTSET-like methanethiosulfonate on Cys223, which has no recognition scaffold of its own.")
        if "gly" in lg:
            return R("Triglycine", "Sortase B", "Substrate peptide", "Yes", "Yes", "No", "No", "No", "Medium",
                     "A triglycine substrate peptide bound in the sortase active site; recognition is the normal substrate groove.")
        return R(lig_name or "small reagent", "Sortase B", "Small reactive probe", "No", "Yes", "No", "", "No", "Medium",
                 "A small cysteine-reactive reagent with little recognition scaffold at the catalytic cysteine.")
    if "enolpyruvyl transferase" in s or "mura" in s or "acetylglucosamine enolpyruvyl" in s or "carboxyvinyltransferase" in s:
        nat = "Yes"
        lig = "Fosfomycin"
        if "terreic" in s:
            lig = "Terreic acid"
        return R(lig, "MurA (UDP-GlcNAc enolpyruvyl transferase)", "Natural epoxide at substrate site",
                 "Yes", "Yes", nat, "No", "No", "High",
                 f"{lig} is a natural-product covalent inhibitor that modifies the catalytic cysteine; the UDP-GlcNAc substrate is co-bound, so the reactive site carries strong substrate recognition.")
    if "penicillin-binding protein" in s or "pbp" in s or ("carboxypeptidase" in s and "bor" in lg):
        return R(lig_name or "boronate inhibitor", "Penicillin-binding protein", "Boronate transition-state analog",
                 "Yes", "No", "No", "No", "Yes", "Medium",
                 "A synthetic boronate that mimics the acyl-enzyme transition state and forms a reversible covalent bond with the catalytic serine; the acyl-peptide-like portion gives substrate-like recognition.")
    if "hmg-coa synthase" in s:
        return R("F-244 (hymeglusin)", "HMG-CoA synthase", "Natural beta-lactone", "Yes", "Yes", "Yes", "No", "No", "High",
                 "F-244/hymeglusin is a natural-product beta-lactone. Its fatty chain gives substrate-like recognition and the beta-lactone acylates the catalytic cysteine.")
    if "diaminopimelate epimerase" in s:
        return R(lig_name or "aziridine analog", "Diaminopimelate epimerase", "Aziridine substrate analog",
                 "Yes", "Yes", "No", "No", "No", "High",
                 "A diaminopimelate aziridine substrate analog. It occupies the substrate site (strong substrate-like recognition) and the aziridine irreversibly alkylates a catalytic cysteine.")
    if "oligopeptidase b" in s:
        if "antipain" in s or "leupeptin" in s:
            return R("Antipain", "Oligopeptidase B", "Peptide aldehyde", "Yes", "Yes", "Yes", "No", "No", "High",
                     "Antipain is a microbial peptide aldehyde, deposited here as its own polymer chain rather than a nonpolymer ligand. Its arginine/valine peptide fills the substrate site (strong noncovalent recognition) and the aldehyde forms a covalent hemiacetal with catalytic Ser577.")
        if "chloro" in lg or "tck" in lg or "tosyl" in lg or "amino" in lg:
            return R(lig_name or "chloromethyl ketone peptide", "Oligopeptidase B", "Peptide chloromethyl ketone",
                     "Yes", "Yes", "No", "No", "No", "Medium",
                     "A synthetic peptide chloromethyl ketone (TLCK-like). The peptide gives substrate-like recognition and the chloromethyl ketone covalently modifies the active site.")
        return R("none / cryoprotectant", "Oligopeptidase B", "Apo or additive", "No", "Yes", "No", "", "No", "Low",
                 "No designed covalent ligand is present (only cryoprotectant); the labeled site is the catalytic residue of the apo enzyme.")
    if "transthyretin" in s:
        return R(lig_name or "covalent kinetic stabilizer", "Transthyretin", "Covalent kinetic stabilizer",
                 "Yes", "No", "No", "Yes", "Yes", "High",
                 "A synthetic covalent kinetic stabilizer that occupies the T4 hormone-binding pocket (strong noncovalent recognition that is not a substrate copy) and reacts with Lys15 via a sulfonyl fluoride, boronic acid, or Michael acceptor.")
    if "pteridine reductase" in s or "ptr1" in s:
        return R(lig_name or "pterin-mimetic inhibitor", "Pteridine reductase 1 (PTR1)", "Antifolate pterin-mimic",
                 "Yes", "No", "No", "No", "Yes", "Medium",
                 "A synthetic pyrrolopyrimidine antifolate that mimics the pterin substrate (strong substrate-like recognition) with NADP co-bound; the aldehyde makes a reversible covalent contact.")
    if "monoacylglycerol" in s or "monoglyceride lipase" in s or "mgl" in s:
        if "phosph" in lg:
            return R(lig_name or "fatty phosphonate", "Monoglyceride lipase", "Phosphonate substrate analog",
                     "Yes", "No", "No", "No", "Yes", "Medium",
                     "A fatty phosphonate substrate analog that reproduces the lipid-binding groove and forms a covalent adduct with the catalytic serine.")
        return R(lig_name or "carbamate inhibitor", "Monoglyceride lipase", "Covalent drug-like inhibitor",
                 "Yes", "No", "No", "Yes", "Yes", "Medium",
                 "A synthetic drug-like covalent inhibitor (carbamate / activated heterocycle) that occupies the lipid pocket and carbamoylates/acylates the catalytic serine; the scaffold is not a substrate copy.")
    if "phospholipase a2" in s or ("phospholipase" in s and "phosph" in lg):
        return R(lig_name or "fluorophosphonate lipid analog", "Phospholipase A2", "Fluorophosphonate lipid analog",
                 "Yes", "No", "No", "No", "Yes", "Medium",
                 "A fluorophosphonate lipid analog that reproduces the acyl-chain recognition and covalently modifies the catalytic serine.")
    if "lecithin:cholesterol" in s or "lcat" in s:
        return R(lig_name or "phosphonate inhibitor", "LCAT", "Phosphonate substrate analog", "Yes", "No", "No", "No", "Yes", "Medium",
                 "A phosphonate acyl mimic covalently bound at the catalytic serine of LCAT, with substrate-like recognition in the lipid pocket.")
    if re.search(r"\bakt\b", s) or "akt1" in s:
        return R(lig_name or "covalent-allosteric AKT inhibitor", "AKT1", "Covalent-allosteric kinase inhibitor",
                 "Yes", "No", "No", "Yes", "Yes", "High",
                 "A synthetic covalent-allosteric AKT inhibitor. It occupies an allosteric pocket (strong noncovalent recognition that is not a substrate copy) and an electrophile covalently modifies a cysteine.")
    if "signal peptidase" in s or "lepb" in s:
        return R(lig_name or "macrocyclic covalent inhibitor", "Signal peptidase (LepB)", "Covalent drug-like inhibitor",
                 "Yes", "No", "No", "Yes", "Yes", "Medium",
                 "A synthetic macrocyclic covalent inhibitor that binds the signal-peptidase groove and reacts with the catalytic serine; the arylomycin-derived scaffold provides strong, non-substrate recognition.")
    if "iap" in s or "bir3" in s or "smac" in s:
        return R(lig_name or "covalent SMAC mimetic", "cIAP1 BIR3", "Covalent SMAC mimetic", "Yes", "No", "No", "No", "Yes", "Medium",
                 "A synthetic SMAC mimetic that reproduces the AVPI peptide recognition in the BIR groove (substrate-like) and adds an electrophile for covalent engagement.")
    if "spliceostatin" in s or "sudemycin" in s or "sf3b" in s or "splice" in s or "spliceostatin" in lg or "sudemycin" in lg or "ssa complex" in t or "u2 5'" in t:
        natural = "No"
        return R(lig_name or "SF3B splicing modulator", "SF3B spliceosome core", "Splicing modulator",
                 "Yes", "No", natural, "Yes", "Yes", "Medium",
                 "A spliceostatin/sudemycin-class SF3B modulator (from the FR901464 natural-product lineage) that binds the branch-point adenosine pocket with strong, non-substrate recognition and an epoxide/diene electrophile.")
    if "prmt5" in s:
        return R(lig_name or "covalent PBM-site inhibitor", "PRMT5", "Covalent drug-like inhibitor", "Yes", "No", "No", "Yes", "Yes", "Medium",
                 "A synthetic covalent PRMT5 inhibitor binding the PBM site (strong, non-substrate recognition) with an electrophile that modifies a cysteine.")
    if "crm1" in s or "plumbagin" in s:
        return R("Plumbagin", "CRM1 (exportin-1)", "Natural naphthoquinone", "Limited", "Yes", "Yes", "", "No", "Medium",
                 "Plumbagin is a small natural-product naphthoquinone that covalently modifies Cys528 of CRM1. It has only limited recognition scaffold beyond the reactive quinone.")
    if "isocyanide hydratase" in s:
        return R(lig_name or "substrate analog", "Isocyanide hydratase", "Catalytic thioimidate intermediate",
                 "Yes", "Yes", "Yes", "No", "No", "Medium",
                 "A substrate-derived thioimidate intermediate on the catalytic cysteine; recognition is the normal substrate site rather than a designed inhibitor.")
    if "acetoacetate decarboxylase" in s:
        return R(lig_name or "acetylacetone", "Acetoacetate decarboxylase", "Substrate-analog Schiff base",
                 "Yes", "Yes", "No", "No", "No", "Medium",
                 "Acetylacetone (deposited as pentan-2-one) is a substrate analog of acetoacetate. It occupies the normal substrate site and forms the catalytic Schiff base with Lys116. The scaffold is short because the substrate itself is small, but the recognition is the evolved substrate pocket rather than a featureless warhead.")
    if "factor b" in s or "complement" in s:
        return R(lig_name or "4-guanidinobenzoic acid", "Complement factor B", "S1-pocket recognition motif", "Yes", "Yes", "No", "No", "No", "Medium",
                 "4-Guanidinobenzoic acid bound to the serine-protease domain of complement factor B. The guanidino group provides specific S1-pocket recognition; 1RTK itself is noncovalent (no Ser acyl-enzyme bond). Related guanidinobenzoyl-trypsin structures (e.g. 1GBT/2AH4) demonstrate the covalent acyl-enzyme chemistry, but that should not be attributed to 1RTK.")
    if "psaa" in s or "ph 6 antigen" in s or "aebsf" in lg or "aminoethyl)benzenesulfonyl" in lg:
        return R("AEBSF", "Yersinia pestis PsaA (pH 6 antigen)", "Small reactive probe", "Limited", "No", "No", "No", "No", "Medium",
                 "AEBSF is a small generic sulfonyl fluoride. In this structure it modifies Tyr52 of the PsaA adhesin, not a serine-protease S1 site, so there is little site-specific recognition beyond the reagent itself.")
    if "ycac" in s:
        return R(lig_name or "acrylamide fragment", "YcaC", "Small reactive fragment", "No", "No", "No", "", "No", "Low",
                 "A small acrylamide/propionamide fragment plus a diol; little or no recognition scaffold at the reactive site.")
    if "phenanthroline" in s or "cytochrome" in s:
        return R(lig_name or "phenanthroline modification", "Engineered metalloprotein", "Engineered modification", "No", "Yes", "No", "", "No", "Low",
                 "An engineered phenanthroline modification on a designed metalloprotein; not a recognition-driven covalent ligand.")
    if "peroxidase" in s:
        return R("none (heme enzyme)", "Peroxidase", "Not a covalent inhibitor", "No", "Yes", "No", "", "No", "Low",
                 "No designed covalent inhibitor is present; the structure is a heme peroxidase and the labeled site is a catalytic residue.")
    if "rtx cysteine protease" in s or ("cysteine protease" in s and "aza" in t):
        return R(lig_name or "aza-peptide", "RTX cysteine protease domain", "Peptide inhibitor", "Yes", "Yes", "No", "No", "No", "Medium",
                 "A synthetic aza-leucine peptide inhibitor; the peptide reproduces substrate contacts (strong substrate-like recognition) and reacts with the catalytic cysteine.")
    if "spla" in s or ("serine protease" in s and "phosph" in lg):
        return R(lig_name or "peptide phosphonate", "Staphylococcal serine protease SplA", "Peptide phosphonate", "Yes", "Yes", "No", "No", "No", "Medium",
                 "A synthetic peptidyl phosphonate; the peptide gives substrate-like recognition and the phosphonate covalently modifies the catalytic serine.")
    if "aminolaevulinic" in s or "aminolevulinic" in s:
        return R(lig_name or "diacid substrate analog", "5-aminolaevulinic acid dehydratase", "Substrate analog",
                 "Yes", "Yes", "No", "No", "No", "High",
                 "A synthetic diacid substrate analog that reproduces the two substrate sub-sites (strong substrate-like recognition) and forms a Schiff base with the catalytic lysine.")
    if title.lower().startswith("moa") or "psl1a" in s or "moa-" in t:
        if "e64" in lg or "e-64" in lg:
            return R("E-64", "MOA lectin/protease", "Natural epoxysuccinyl peptide", "Yes", "Yes", "Yes", "No", "No", "Medium",
                     "E-64 is a natural epoxysuccinyl peptide; substrate-like recognition with an epoxide that modifies the catalytic cysteine.")
        return R("Z-VAD-FMK", "MOA lectin/protease", "Peptide fluoromethyl ketone", "Yes", "Yes", "No", "No", "No", "Medium",
                 "Z-VAD-FMK is a synthetic peptide fluoromethyl ketone; substrate-like recognition with a covalent warhead at the catalytic cysteine.")
    if "esterase" in s or "lipase" in s:
        return R(lig_name or "phosphonate inhibitor", "Esterase/lipase", "Phosphonate suicide inhibitor", "Yes", "No", "No", "No", "Yes", "Medium",
                 "A phosphonate/phosphinate substrate analog that reproduces the acyl-binding groove and forms a covalent adduct with the catalytic serine.")
    return None


def load_meta():
    entries = json.loads(META.read_text(encoding="utf-8"))
    out = {}
    for e in entries:
        pid = e["rcsb_id"]
        title = (e.get("struct") or {}).get("title") or ""
        year = ((e.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:4]
        prots, lig_list = [], []
        for pe in e.get("polymer_entities") or []:
            d = (pe.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
            if d:
                prots.append(d)
        primary = None
        for ne in e.get("nonpolymer_entities") or []:
            np = ne.get("pdbx_entity_nonpoly") or {}
            cid, nm = np.get("comp_id"), np.get("name")
            if cid in IONS:
                continue
            lig_list.append(f"{cid}:{nm}")
            if primary is None:
                primary = nm
        out[pid] = {
            "title": title, "year": year,
            "prots": " || ".join(prots),
            "ligtext": " || ".join(lig_list),
            "lig_name": primary,
        }
    return out


def is_proteasome(title):
    t = title.lower()
    return "proteasome" in t or t.startswith("ycp") or " ycp " in f" {t} " or "yeast cp" in t or "20s particle" in t or "immunoproteasome" in t


def is_mpro(title, prots):
    s = (title + " " + prots).lower()
    return any(k in s for k in ("main protease", "3c-like", "3cl", "mpro", "nsp5", "3c prote", "picornain", "ev71 3c", "main proteinase"))


def heavy_atoms(name):
    parts = name.split("-")
    if len(parts) < 3:
        return 0
    return len(re.findall(r"Cl|Br|[A-Z]", parts[1]))


def main():
    allv = pd.read_csv(ALL_CSV)
    ncv = pd.read_csv(NC_CSV)
    for df in (allv, ncv):
        df["PDB"] = df["Name"].astype(str).str[:4].str.upper()
    allv["hv"] = allv["Name"].map(heavy_atoms)
    hv_by_pdb = allv.groupby("PDB")["hv"].max()

    meta = load_meta()
    v1 = pd.read_csv(V1).set_index("PDB")
    v1_cols = {
        STRONG: STRONG, "Notes": "Notes", BEFORE: BEFORE, NAT: NAT, SUB: SUB,
        "Ligand": "Ligand", "Target": "Target", "Ligand class": "Ligand class",
        "Structure-guided design?": "Structure-guided design?", "Confidence": "Confidence",
    }

    grouped = allv.groupby("PDB").agg(
        residue=("Residue", lambda s: ",".join(sorted(set(s.astype(str))))),
        resnum=("ResNum", lambda s: ",".join(sorted(set(map(str, s))))),
        chain=("Chain", lambda s: ",".join(sorted(set(s.astype(str))))),
    )

    rows = []
    unmatched = []
    source_counts = {"started": 0, "reuse": 0, "proteasome": 0, "mpro": 0, "other": 0}
    for pdb, g in grouped.iterrows():
        info = meta[pdb]
        title, prots = info["title"], info["prots"]
        hv = int(hv_by_pdb.get(pdb, 0))
        if pdb in STARTED_EXACT:
            hit = STARTED_EXACT[pdb]; source_counts["started"] += 1
        elif pdb in v1.index:
            r = v1.loc[pdb]
            hit = dict(ligand=r["Ligand"], target=r["Target"], klass=r["Ligand class"],
                       strong=r[STRONG], before=r[BEFORE], natural=r[NAT],
                       not_substrate=("" if pd.isna(r[SUB]) else r[SUB]),
                       guided=r["Structure-guided design?"], confidence=r["Confidence"], notes=r["Notes"])
            source_counts["reuse"] += 1
        elif is_proteasome(title):
            hit = classify_proteasome(title)
            if hit is None:
                unmatched.append((pdb, "proteasome", title)); continue
            source_counts["proteasome"] += 1
        elif is_mpro(title, prots):
            hit = classify_mpro(title, info["ligtext"], hv); source_counts["mpro"] += 1
        else:
            hit = classify_other(title, prots, info["ligtext"], info["lig_name"], hv)
            if hit is None:
                unmatched.append((pdb, "other", title)); continue
            source_counts["other"] += 1
        if pdb == "6HJF":
            hit["notes"] = (
                "BrOxoPA (a bromo-oxopentanoate; the coordinate entry models it as levulinate) is a small "
                "proline-site alkylating analog on the racemase cysteines. It is a substrate-site binder, "
                "but the scaffold is only a short keto-acid, so noncovalent recognition is limited.")
        rows.append({
            "PDB": pdb,
            STRONG: hit["strong"],
            "Notes": hit["notes"],
            BEFORE: hit["before"],
            NAT: hit["natural"],
            SUB: hit["not_substrate"],
            "Ligand": hit["ligand"],
            "Target": hit["target"],
            "Covalent residue": f"{g['residue']} {g['resnum']} (chain {g['chain']})",
            "Release year": info["year"],
            "Ligand class": hit["klass"],
            "Structure-guided design?": hit["guided"],
            "Confidence": hit["confidence"],
            "Ligand heavy atoms (approx)": hv,
        })

    if unmatched:
        print("UNMATCHED", len(unmatched))
        for pdb, fam, title in unmatched:
            print(f"  [{fam}] {pdb}: {title}")
        raise SystemExit(1)

    ann = pd.DataFrame(rows).sort_values("PDB").reset_index(drop=True)
    print("source_counts", source_counts, "total", len(ann))
    print(ann[STRONG].value_counts().to_string())
    try:
        ann.to_csv(OUT_CSV, index=False)
        print("wrote", OUT_CSV.name)
    except PermissionError:
        alt = OUT_CSV.with_name(OUT_CSV.stem + "_pending.csv")
        ann.to_csv(alt, index=False)
        print("LOCKED, wrote", alt.name)
    return ann


if __name__ == "__main__":
    main()
