"""Annotate the covalent test-set PDBs for preorganized noncovalent recognition.

Follows the columns in TestSetNoncovalentScaffold2(Right).xlsx:
ligand purpose, natural vs synthetic, and whether recognition is a
substrate copy. Calls use the PDB title, deposited ligand, citation, and
the covalent residue recorded in the test-label file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = Path(r"c:\Users\Andre\Downloads\test_label_results (1).csv")
META = HERE / "_rcsb_meta.json"
STARTED = HERE / "TestSetNoncovalentScaffold2(Right).xlsx"
OUT_XLSX = HERE / "TestSet_Noncovalent_Annotation.xlsx"
OUT_CSV = HERE / "TestSet_Noncovalent_Annotation.csv"

COLS = [
    "PDB",
    "Strong noncovalent?",
    "Notes",
    "Before Modern Covalent Generation (No crystal structures)",
    "All Natural?",
    "Not formed from substrate likeness if noncovalent",
    "Ligand",
    "Target",
    "Covalent residue",
    "Release year",
    "Ligand class",
    "Structure-guided design?",
    "Confidence",
    "Best target rank in test file",
    "Hit in top 1% in test file",
]


def rec(
    ligand,
    target,
    klass,
    strong,
    before,
    natural,
    not_substrate,
    guided,
    confidence,
    notes,
):
    if strong == "No":
        not_substrate = ""
    return {
        "ligand": ligand,
        "target": target,
        "klass": klass,
        "strong": strong,
        "before": before,
        "natural": natural,
        "not_substrate": not_substrate,
        "guided": guided,
        "confidence": confidence,
        "notes": notes,
    }


# Exact calls for complexes that are not part of a repeated proteasome series.
# not_substrate = "Yes" means the noncovalent scaffold was NOT built as a
# substrate or transition-state copy. Epoxomicin in the started sheet is the
# template for natural products: they are marked Yes even when a peptide
# sits in the substrate channel.
SPECIAL = {
    "1A09": rec(
        "Ace-formyl-phosphotyrosyl-Glu-(N,N-dipentylamine)",
        "Src SH2 domain",
        "Phosphopeptide recognition",
        "Yes", "Yes", "No", "No", "No", "High",
        "Ace-formyl-phosphotyrosyl-Glu-(N,N-dipentylamine) is a synthetic phosphopeptide ligand of the Src SH2 domain. The phosphotyrosine and flanking residues provide strong noncovalent recognition of the pTyr pocket, with a formyl group on the ligand. Used for a thermodynamic and structural study of Src SH2 peptide recognition, not as a covalent drug.",
    ),
    "1B02": rec(
        "5-fluoro-dUMP (with tetrahydrofolate)",
        "Thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "No", "No", "No", "High",
        "5-fluoro-dUMP is a synthetic fluorinated form of the thymidylate-synthase substrate dUMP, bound with the folate cofactor. The nucleotide has strong noncovalent recognition in the substrate site and forms the normal catalytic covalent adduct with the active-site cysteine. The phosphonate and aryl-halide labels in the test file are phosphate and ring atoms, not a separate designed warhead. Used to study the thymidylate synthase mechanism.",
    ),
    "1BCR": rec(
        "Antipain",
        "Wheat serine carboxypeptidase II",
        "Natural peptide aldehyde",
        "Yes", "Yes", "Yes", "Yes", "No", "High",
        "Antipain is a microbial natural-product peptide aldehyde. Its peptide scaffold provides strong noncovalent recognition in the carboxypeptidase substrate channel, while the aldehyde forms a covalent hemiacetal with the catalytic serine. Used to study the catalytic mechanism and substrate specificity of serine carboxypeptidase II.",
    ),
    "1BCS": rec(
        "Chymostatin",
        "Wheat serine carboxypeptidase II",
        "Natural peptide aldehyde",
        "Yes", "Yes", "Yes", "Yes", "No", "High",
        "Chymostatin is a microbial natural-product peptide aldehyde. Its peptide scaffold provides strong noncovalent recognition in the carboxypeptidase substrate channel, while the aldehyde forms a covalent hemiacetal with the catalytic serine. Used to study the catalytic mechanism and substrate specificity of serine carboxypeptidase II.",
    ),
    "1CSB": rec(
        "CA030",
        "Cathepsin B",
        "Epoxysuccinyl peptide",
        "Yes", "Yes", "No", "No", "No", "High",
        "CA030 is a synthetic epoxysuccinyl-isoleucyl-proline inhibitor in the E-64 family. The dipeptide provides strong noncovalent recognition of the cathepsin B substrate sites, while the epoxysuccinyl group covalently modifies the catalytic cysteine. Used as a specific covalent cathepsin B inhibitor and as a basis for designing epoxysuccinyl inhibitors.",
    ),
    "1ESE": rec(
        "Diethyl phosphonate",
        "Streptomyces esterase",
        "Small reactive probe",
        "No", "Yes", "No", "", "No", "High",
        "Diethyl phosphonate is the small covalent adduct of a diethyl-phosphonate inhibitor on the catalytic serine. The ethyl groups give little site-specific recognition beyond the reactive phosphorus, so this is a generic serine-hydrolase probe rather than a scaffold optimized for the pocket. Used to study enantiorecognition and the catalytic triad of the esterase.",
    ),
    "1F28": rec(
        "dUMP (with BW1843U89)",
        "Thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "dUMP is the natural substrate of thymidylate synthase, bound here with the synthetic antifolate BW1843U89. dUMP has strong noncovalent recognition in the nucleotide site and forms the catalytic covalent adduct with the active-site cysteine. The phosphonate labels in the test file come from the nucleotide phosphate. Used to study ligand docking against a closed thymidylate-synthase complex.",
    ),
    "1GA4": rec(
        "Pseudoiodotyrostatin",
        "Sedolisin",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Pseudoiodotyrostatin is a synthetic iodinated analog of the natural peptide aldehyde tyrostatin. Its peptide scaffold provides strong noncovalent recognition in the sedolisin substrate channel, while the aldehyde reacts with the catalytic serine. Used to define the serine-carboxyl proteinase family and its substrate binding.",
    ),
    "1GMY": rec(
        "Dipeptidyl nitrile",
        "Cathepsin B",
        "Peptide nitrile",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "The dipeptidyl nitrile is a synthetic cathepsin B inhibitor (diphenylacetyl–methylphenylalanyl–aminoacetonitrile). The peptide groups provide strong noncovalent recognition of the S2/S3 sites, while the nitrile forms a reversible covalent thioimidate with the catalytic cysteine. Used in an early structure-based design of selective cathepsin B inhibitors, before the modern targeted-covalent drug generation.",
    ),
    "1GTJ": rec(
        "Ac-Ile-Ala-Phe-CHO",
        "Kumamolisin",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Ac-Ile-Ala-Phe-CHO is a synthetic peptide aldehyde. Its peptide sequence provides strong noncovalent recognition in the kumamolisin substrate channel, while the aldehyde reacts with the catalytic serine. Used to characterize the thermostable serine-carboxyl proteinase and its substrate specificity.",
    ),
    "1GTL": rec(
        "Ac-Ile-Pro-Phe-CHO",
        "Kumamolisin",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Ac-Ile-Pro-Phe-CHO is a synthetic peptide aldehyde. Its peptide sequence provides strong noncovalent recognition in the kumamolisin substrate channel, while the aldehyde reacts with the catalytic serine. Used to characterize the thermostable serine-carboxyl proteinase and its substrate specificity.",
    ),
    "1H2Y": rec(
        "Z-Pro-prolinal",
        "Prolyl oligopeptidase",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Z-Pro-prolinal is a synthetic dipeptide aldehyde. The proline residues provide strong noncovalent recognition in the prolyl-oligopeptidase substrate site, while the aldehyde forms a covalent hemiacetal with the catalytic serine. Used to study the catalytic mechanism and the oxyanion-binding site.",
    ),
    "1HVY": rec(
        "dUMP (with raltitrexed)",
        "Thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "dUMP is the natural substrate of human thymidylate synthase, bound with the synthetic antifolate raltitrexed. dUMP has strong noncovalent recognition in the nucleotide site and forms the catalytic covalent adduct with the active-site cysteine. The phosphonate labels in the test file come from the nucleotide phosphate. Used to describe the closed conformation of the drug complex.",
    ),
    "1KYI": rec(
        "NLVS",
        "HslV protease",
        "Peptide vinyl sulfone",
        "Yes", "Yes", "No", "No", "No", "High",
        "NLVS (4-iodo-3-nitrophenylacetyl-Leu-Leu-Leu-vinyl sulfone) is a synthetic trileucine vinyl sulfone. The leucine peptide provides strong noncovalent recognition in the HslV substrate channel, while the vinyl sulfone covalently modifies the catalytic N-terminal threonine. Used to corroborate allosteric activation of the proteasome-like HslV protease.",
    ),
    "1LXY": rec(
        "L-citrulline",
        "Arginine deiminase",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "L-citrulline is the natural reaction product, trapped covalently on arginine deiminase. It occupies the normal substrate site, so the noncovalent recognition is that of the substrate rather than a designed inhibitor. Used to determine the catalytic mechanism of arginine deiminase.",
    ),
    "1NLU": rec(
        "Pseudoiodotyrostatin",
        "Sedolisin",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Pseudoiodotyrostatin is a synthetic iodinated analog of tyrostatin, and two molecules occupy the active site. The peptide scaffold provides strong noncovalent recognition of the substrate channel, while the aldehyde reacts with the catalytic serine. Used as a model of the bi-product complex after peptide cleavage.",
    ),
    "1O41": rec(
        "RU78300 (2-formyl-6-methoxyphenyl phosphate)",
        "Src SH2 domain",
        "Fragment",
        "No", "Yes", "No", "", "Yes", "High",
        "RU78300 is a synthetic low-affinity pTyr-mimic fragment (formyl-methoxyphenyl phosphate). It contacts the phosphate pocket but does not have a substantial recognition scaffold beyond that fragment. The paper uses it to compare fragment binding with full-length SH2 inhibitors. Not a site-optimized covalent ligand.",
    ),
    "1O43": rec(
        "RU82129",
        "Src SH2 domain",
        "Elaborated pTyr mimetic",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "RU82129 is a synthetic, elaborated Src SH2 inhibitor with a biphenyl-azepanone scaffold and a formylphenyl group. The scaffold provides strong noncovalent recognition of the SH2 site and was built as a phosphotyrosine-peptide mimetic. Used to compare high-affinity inhibitor binding with low-affinity fragments.",
    ),
    "1O45": rec(
        "RU84687",
        "Src SH2 domain",
        "Elaborated pTyr mimetic",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "RU84687 is a synthetic Src SH2 inhibitor combining a biphenyl-azepanone with a formyl-phosphotyrosine. The scaffold provides strong noncovalent recognition and was built as a phosphopeptide mimetic. Used to compare high-affinity inhibitor binding with low-affinity fragments.",
    ),
    "1O4A": rec(
        "RU82197",
        "Src SH2 domain",
        "Elaborated pTyr mimetic",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "RU82197 is a synthetic Src SH2 inhibitor with a biphenyl-azepanone scaffold and a formylbenzoate. The scaffold provides strong noncovalent recognition and was built as a phosphopeptide mimetic. Used to compare high-affinity inhibitor binding with low-affinity fragments.",
    ),
    "1O4D": rec(
        "RU78262 (2-formylphenyl phosphate)",
        "Src SH2 domain",
        "Fragment",
        "No", "Yes", "No", "", "Yes", "High",
        "RU78262 is a synthetic low-affinity fragment, 2-formylphenyl dihydrogen phosphate. It is a minimal pTyr mimic with little scaffold beyond the aryl phosphate. Used in the same fragment-versus-full-inhibitor study of Src SH2. Not a site-optimized ligand.",
    ),
    "1O4E": rec(
        "RU78299 (diformyl-methylphenyl phosphate)",
        "Src SH2 domain",
        "Fragment",
        "No", "Yes", "No", "", "Yes", "High",
        "RU78299 is a synthetic low-affinity fragment, 2,6-diformyl-4-methylphenyl phosphate. It is a minimal pTyr mimic with little scaffold beyond the aryl phosphate. Used in the same fragment-versus-full-inhibitor study of Src SH2. Not a site-optimized ligand.",
    ),
    "1O4I": rec(
        "PAS219",
        "Src SH2 domain",
        "Small pTyr mimetic",
        "Limited", "Yes", "No", "No", "Yes", "Medium",
        "PAS219 is a synthetic cyclohexylmethyl formylphenyl phosphate. It is larger than the bare aryl-phosphate fragments and makes some hydrophobic contact, but it is still a small pTyr mimic rather than a full recognition scaffold. Used in the Src SH2 fragment study.",
    ),
    "1O4K": rec(
        "PASBN",
        "Src SH2 domain",
        "Small pTyr mimetic",
        "Limited", "Yes", "No", "No", "Yes", "Medium",
        "Benzyl 2-formylphenyl hydrogen phosphate is a synthetic small pTyr mimic. The benzyl group adds a little hydrophobic contact, but the ligand is still a fragment rather than a full recognition scaffold. Used in the Src SH2 fragment study.",
    ),
    "1OFI": rec(
        "NLVS",
        "HslV protease",
        "Peptide vinyl sulfone",
        "Yes", "Yes", "No", "No", "No", "High",
        "NLVS is a synthetic Leu-Leu-Leu vinyl sulfone. The peptide provides strong noncovalent recognition in the HslV substrate channel, while the vinyl sulfone covalently modifies the catalytic threonine. Used to study the asymmetric HslV–HslU complex, a prokaryotic proteasome homolog.",
    ),
    "1QD6": rec(
        "Hexadecanesulfonyl fluoride",
        "OMPLA",
        "Fatty-acyl sulfonyl fluoride",
        "Yes", "Yes", "No", "No", "No", "High",
        "Hexadecanesulfonyl fluoride is a synthetic palmitate-like reactive inhibitor. Its long hydrocarbon chain binds in the hydrophobic groove, providing meaningful noncovalent localization, while the sulfonyl fluoride covalently modifies Ser144. Used as an irreversible inhibitor to study the active site and catalytic mechanism of OMPLA.",
    ),
    "1QFM": rec(
        "Thioglycerol / glycerol",
        "Prolyl oligopeptidase",
        "Crystallization additive",
        "No", "Yes", "No", "", "No", "Low",
        "This prolyl oligopeptidase structure has no designed inhibitor. The deposited small molecules are glycerol and thioglycerol, and the test file flags a ketone near the catalytic serine. That is a crystallization additive or a detection false positive, not a recognition scaffold. Included because the test file lists it as a covalent example.",
    ),
    "1QFS": rec(
        "Z-Pro-prolinal",
        "Prolyl oligopeptidase",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Z-Pro-prolinal is a synthetic dipeptide aldehyde. The proline residues provide strong noncovalent recognition in the substrate site, while the aldehyde forms a covalent hemiacetal with the catalytic serine. Used to study how the beta-propeller domain regulates proteolysis.",
    ),
    "1QZ3": rec(
        "Hexadecanesulfonyl fluoride",
        "Carboxylesterase Est2",
        "Fatty-acyl sulfonyl fluoride",
        "Yes", "Yes", "No", "No", "No", "High",
        "Hexadecanesulfonate is the adduct of synthetic hexadecanesulfonyl fluoride, a palmitate-like inhibitor. The long hydrocarbon chain binds in the hydrophobic substrate groove, while the sulfonyl group covalently modifies the catalytic serine. Used to study a substrate-induced switch in the mechanism of thermophilic esterase Est2.",
    ),
    "1R4Z": rec(
        "(R)-IPG hexenylphosphonate",
        "Bacillus subtilis lipase A",
        "Phosphonate suicide inhibitor",
        "Yes", "Yes", "No", "No", "No", "High",
        "The (R) hex-5-enyl phosphonate is a synthetic suicide inhibitor with a lipid-like chain and a dioxolane headgroup. The chain provides noncovalent recognition in the lipase substrate site, while the phosphonate covalently modifies the catalytic serine. Used for enantioselective directed evolution and phage-display selection of lipase A.",
    ),
    "1R50": rec(
        "(S)-IPG hexenylphosphonate",
        "Bacillus subtilis lipase A",
        "Phosphonate suicide inhibitor",
        "Yes", "Yes", "No", "No", "No", "High",
        "The (S) hex-5-enyl phosphonate is a synthetic suicide inhibitor with a lipid-like chain and a dioxolane headgroup. The chain provides noncovalent recognition in the lipase substrate site, while the phosphonate covalently modifies the catalytic serine. Used for enantioselective directed evolution and phage-display selection of lipase A.",
    ),
    "1SIO": rec(
        "Ac-Ile-Pro-Phe-CHO",
        "Kumamolisin-As",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Ac-Ile-Pro-Phe-CHO is a synthetic peptide aldehyde. Its peptide sequence provides strong noncovalent recognition in the kumamolisin substrate channel, while the aldehyde reacts with the catalytic serine. Used to study this collagenase-like serine-carboxyl peptidase.",
    ),
    "1V0Y": rec(
        "Dibutyrylphosphatidylcholine",
        "Phospholipase D",
        "Substrate intermediate",
        "Yes", "Yes", "No", "No", "No", "High",
        "Dibutyrylphosphatidylcholine is a synthetic short-chain substrate analog of phosphatidylcholine. Its acyl chains provide noncovalent recognition in the phospholipase D active site, and the structure traps a covalent pentacoordinate phosphorus intermediate on the catalytic histidine. Used to take snapshots along the phospholipase D reaction pathway.",
    ),
    "1VZE": rec(
        "dUMP (with CB3717)",
        "Thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "dUMP is the natural substrate, bound with the synthetic antifolate CB3717 to an E60Q thymidylate-synthase mutant. dUMP has strong noncovalent recognition in the nucleotide site and forms the catalytic covalent adduct with the active-site cysteine. Used to separate steps in formation of the catalytic intermediates.",
    ),
    "2AAZ": rec(
        "dUMP (with CB3717)",
        "Thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "dUMP is the natural substrate of Cryptococcus neoformans thymidylate synthase, bound with the synthetic antifolate CB3717. dUMP has strong noncovalent recognition in the nucleotide site and forms the catalytic covalent adduct with the active-site cysteine. Used to compare species-specific dynamics of the enzyme.",
    ),
    "2BKL": rec(
        "Z-Ala-Pro",
        "Prolyl endopeptidase",
        "Peptide product analog",
        "Yes", "Yes", "No", "No", "No", "Medium",
        "Z-Ala-Pro is a synthetic dipeptide product analog of prolyl endopeptidase. It occupies the substrate channel and the test file flags an aldehyde-like covalent interaction at the catalytic serine. Recognition is substrate-like. Used to analyze inter-domain dynamics and specificity of prolyl endopeptidases.",
    ),
    "2FJ0": rec(
        "OTFP",
        "Juvenile hormone esterase",
        "Trifluoromethyl ketone",
        "Yes", "Yes", "No", "No", "No", "High",
        "OTFP (1,1,1-trifluoro-3-(octylthio)acetone) is a synthetic juvenile-hormone mimic. The octyl chain provides strong noncovalent recognition in the hydrophobic substrate pocket, while the trifluoromethyl ketone forms a covalent tetrahedral adduct with the catalytic serine. Used as an insect-maturation inhibitor and to map the esterase active site.",
    ),
    "2HTX": rec(
        "Polymerized glutaraldehyde",
        "Hen egg-white lysozyme",
        "Chemical crosslinker",
        "No", "Yes", "No", "", "No", "High",
        "Polymerized glutaraldehyde is a synthetic chemical crosslinker, not a recognition ligand. It modifies surface lysines of lysozyme without a scaffold optimized for a binding site. Used to identify the end products of glutaraldehyde crosslinking. It does not test whether covalent drugs are preorganized noncovalently.",
    ),
    "2HU1": rec(
        "Polymerized glutaraldehyde",
        "Hen egg-white lysozyme",
        "Chemical crosslinker",
        "No", "Yes", "No", "", "No", "High",
        "Polymerized glutaraldehyde is a synthetic chemical crosslinker, not a recognition ligand. It modifies surface lysines of lysozyme without a scaffold optimized for a binding site. Used to identify the end products of glutaraldehyde crosslinking. It does not test whether covalent drugs are preorganized noncovalently.",
    ),
    "2IPP": rec(
        "2-pyridinethiol",
        "Cathepsin B",
        "Small reactive probe",
        "No", "Yes", "No", "", "No", "High",
        "2-pyridinethiol is a small synthetic disulfide-forming reagent on the catalytic cysteine of cathepsin B. It has essentially no recognition scaffold. This is a tetragonal crystal form of the enzyme rather than a designed inhibitor complex.",
    ),
    "2J6H": rec(
        "5-oxo-L-norleucine (DON adduct)",
        "Glucosamine-6-phosphate synthase",
        "Glutamine-site inactivator",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "5-oxo-L-norleucine is the bound adduct of 6-diazo-5-oxo-L-norleucine, a natural-product glutamine analog, together with glucose-6-phosphate. It occupies the glutamine site, so recognition is substrate-like, and it covalently inactivates the N-terminal cysteine. Used to show how glutamine binding opens the ammonia channel.",
    ),
    "2XDW": rec(
        "Synthetic unnatural dipeptide",
        "Prolyl oligopeptidase",
        "Peptide aldehyde-like inhibitor",
        "Yes", "Yes", "No", "No", "Partial", "Medium",
        "The ligand is a synthetic unnatural dipeptide inhibitor of prolyl oligopeptidase. It retains strong substrate-like noncovalent recognition in the proline pocket and is covalently bound at the catalytic serine. Used to inhibit the enzyme with a non-natural dipeptide rather than as a modern covalent drug.",
    ),
    "3A73": rec(
        "15-deoxy-delta12,14-prostaglandin J2",
        "Human serum albumin",
        "Carrier-protein adduct",
        "Yes", "Yes", "Yes", "Yes", "No", "Medium",
        "15-deoxy-delta12,14-prostaglandin J2 is a natural cyclopentenone prostaglandin. It occupies a fatty-acid site on albumin and forms a covalent adduct with His146. Recognition is real but is carrier binding, not an enzyme active site optimized for a drug. Used to describe an unusual covalent albumin adduct.",
    ),
    "3AIL": rec(
        "Diethyl phosphonate (paraoxon adduct)",
        "Sulfolobus carboxylesterase",
        "Small reactive probe",
        "No", "Yes", "No", "", "No", "High",
        "The bound species is diethyl phosphonate, the small covalent adduct left by paraoxon on the catalytic serine. After the leaving group departs there is little ligand scaffold. Used to describe a thermostable archaeal carboxylesterase, not a designed recognition ligand.",
    ),
    "3CR6": rec(
        "C15 retinal analog",
        "Engineered CRABPII",
        "Retinoid analog",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "The C15 aldehyde is a synthetic truncated retinal analog. It has strong noncovalent recognition in the engineered retinoid pocket and forms a Schiff base with a lysine. Used to study wavelength regulation and binding orientation in a rhodopsin-mimic protein, not as a covalent drug.",
    ),
    "3D1L": rec(
        "2-mercaptopropionaldehyde",
        "Putative NADP oxidoreductase",
        "Small adduct",
        "Limited", "Yes", "No", "No", "No", "Low",
        "2-mercaptopropionaldehyde is a small synthetic or artifact aldehyde on a structural-genomics oxidoreductase. It can sit in the active site but it is not a substantial recognition scaffold or a designed inhibitor. Low confidence that this is a meaningful covalent-ligand complex.",
    ),
    "3F8A": rec(
        "C15 retinal analog",
        "Engineered CRABPII",
        "Retinoid analog",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "The C15 aldehyde is a synthetic truncated retinal analog. It has strong noncovalent recognition in the engineered retinoid pocket and forms a Schiff base with a lysine. Used to probe wavelength regulation in a CRABPII rhodopsin mimic.",
    ),
    "3F9D": rec(
        "C15 retinal analog",
        "Engineered CRABPII",
        "Retinoid analog",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "The C15 aldehyde is a synthetic truncated retinal analog. It has strong noncovalent recognition in the engineered retinoid pocket and forms a Schiff base with a lysine. Used to compare binding orientations caused by a single pocket mutation.",
    ),
    "3FA6": rec(
        "C15 retinal analog",
        "Engineered CRABPII",
        "Retinoid analog",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "The C15 aldehyde is a synthetic truncated retinal analog. It has strong noncovalent recognition in the engineered retinoid pocket and forms a Schiff base with a lysine. Used to compare binding orientations in CRABPII mutants.",
    ),
    "3H17": rec(
        "PMSF",
        "Esterase EstE5",
        "Small reactive probe",
        "Limited", "Yes", "No", "No", "No", "High",
        "Phenylmethanesulfonyl fluoride is a small synthetic, generic serine-modifying reagent. The benzyl group makes a limited active-site contact, but there is no large recognition scaffold of the kind seen with hexadecanesulfonyl fluoride. Used to show how PMSF blocks the catalytic serine of an HSL-family esterase.",
    ),
    "3H18": rec(
        "PMSF",
        "Esterase EstE5",
        "Small reactive probe",
        "Limited", "Yes", "No", "No", "No", "High",
        "Phenylmethanesulfonyl fluoride is a small synthetic, generic serine-modifying reagent. The benzyl group makes a limited active-site contact, but there is no large recognition scaffold. Second crystal form of the EstE5–PMSF complex, used for the same mechanistic purpose.",
    ),
    "3HHI": rec(
        "CA074",
        "Trypanosoma brucei cathepsin B",
        "Epoxysuccinyl peptide",
        "Yes", "Yes", "No", "No", "No", "High",
        "CA074 is a synthetic epoxysuccinyl-isoleucyl-proline inhibitor. The dipeptide provides strong noncovalent recognition of cathepsin B, while the epoxysuccinyl group covalently modifies the catalytic cysteine. Used as a chemical tool against a trypanosome drug target.",
    ),
    "3ICW": rec(
        "Methyl hexylphosphonate",
        "Candida antarctica lipase B",
        "Phosphonate suicide inhibitor",
        "Yes", "Yes", "No", "No", "No", "High",
        "Methyl hexylphosphonate is a synthetic suicide inhibitor. The hexyl chain provides substrate-like noncovalent recognition in the lipase acyl pocket, while the phosphonate covalently modifies the catalytic serine. Used on a circularly permuted lipase B to study the redesigned enzyme.",
    ),
    "3IVM": rec(
        "Z-Pro-prolinal",
        "Aeromonas prolyl endopeptidase",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Z-Pro-prolinal is a synthetic dipeptide aldehyde. It provides strong substrate-like noncovalent recognition in the proline pocket and forms a covalent hemiacetal with the catalytic serine. Used to describe the induced-fit mechanism of prolyl endopeptidase.",
    ),
    "3KZH": rec(
        "Glucose",
        "Putative sugar kinase",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "Medium",
        "Glucose is the natural sugar substrate, trapped as a Schiff base on a lysine of a putative Clostridium sugar kinase. Recognition is that of the substrate in its binding site, not a designed inhibitor. A structural-genomics catalytic complex.",
    ),
    "3MUO": rec(
        "Z-Pro-prolinal",
        "Aeromonas prolyl endopeptidase",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Z-Pro-prolinal is a synthetic dipeptide aldehyde bound in the closed state of prolyl endopeptidase. It provides strong substrate-like noncovalent recognition and reacts with the catalytic serine. Used to trace the route of substrate entry.",
    ),
    "3NEV": rec(
        "2-keto-3-deoxygluconate",
        "YagE aldolase",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "2-keto-3-deoxygluconate is the natural substrate, trapped covalently as a Schiff base on the aldolase lysine. Recognition is the normal sugar-phosphate/sugar-acid site. Used to assign a biochemical role to this E. coli prophage protein.",
    ),
    "3S3Q": rec(
        "K11017",
        "Schistosoma mansoni cathepsin B1",
        "Peptide vinyl sulfone",
        "Yes", "Yes", "No", "No", "No", "High",
        "K11017 is a synthetic morpholine-carbonyl leucine vinyl sulfone. The peptide provides strong noncovalent recognition of the cathepsin substrate channel, while the vinyl sulfone covalently modifies the catalytic cysteine. Used as a classical peptide-based cysteine-protease inhibitor against a schistosome target.",
    ),
    "3S3R": rec(
        "K11777",
        "Schistosoma mansoni cathepsin B1",
        "Peptide vinyl sulfone",
        "Yes", "Yes", "No", "No", "No", "High",
        "K11777 is a synthetic phenylalanine vinyl sulfone developed as a cruzain inhibitor. The peptide provides strong noncovalent recognition of the cathepsin substrate channel, while the vinyl sulfone covalently modifies the catalytic cysteine. Used here against schistosome cathepsin B1.",
    ),
    "3QSD": rec(
        "CA074",
        "Schistosoma mansoni cathepsin B1",
        "Epoxysuccinyl peptide",
        "Yes", "Yes", "No", "No", "No", "High",
        "CA074 is a synthetic epoxysuccinyl-isoleucyl-proline inhibitor. The dipeptide provides strong noncovalent recognition of the cathepsin substrate sites, while the epoxide covalently modifies the catalytic cysteine. Used for structure-based characterization of a schistosome drug target.",
    ),
    "3TK7": rec(
        "Fructose 6-phosphate",
        "Transaldolase",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "Fructose 6-phosphate is the natural substrate, covalently bound as a Schiff base on the transaldolase lysine. Recognition is the sugar-phosphate site. Used to study stereochemistry of Schiff-base formation.",
    ),
    "3TNO": rec(
        "Sedoheptulose 7-phosphate",
        "Transaldolase",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "Sedoheptulose 7-phosphate is the natural substrate, covalently bound as a Schiff base on the transaldolase lysine. Recognition is the sugar-phosphate site. Used to study stereochemistry of Schiff-base formation.",
    ),
    "3UPB": rec(
        "Arabinose 5-phosphate",
        "Transaldolase",
        "Substrate analog",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "Arabinose 5-phosphate is a sugar-phosphate substrate analog, covalently bound as a Schiff base on the transaldolase lysine. It uses the same noncovalent sugar-phosphate recognition as the natural substrates. Used as another covalent snapshot of the transaldolase mechanism.",
    ),
    "3ZFZ": rec(
        "Ceftaroline",
        "MRSA PBP2a",
        "Beta-lactam",
        "Yes", "No", "No", "No", "Yes", "High",
        "Ceftaroline is a synthetic anti-MRSA cephalosporin. The beta-lactam scaffold provides strong noncovalent recognition as a D-Ala-D-Ala transition-state mimic, and the beta-lactam acylates the catalytic serine. Developed with penicillin-binding-protein structures. Used, with muramic acid at the allosteric site, to show how ceftaroline binds PBP2a.",
    ),
    "3ZG0": rec(
        "Ceftaroline",
        "MRSA PBP2a",
        "Beta-lactam",
        "Yes", "No", "No", "No", "Yes", "High",
        "Ceftaroline is a synthetic anti-MRSA cephalosporin. The beta-lactam scaffold provides strong noncovalent recognition as a cell-wall-substrate mimic, and it acylates Ser403. A cocrystal companion to the soaked PBP2a complex, used for the same allosteric-binding question.",
    ),
    "4AMY": rec(
        "IC-1",
        "Prolyl oligopeptidase",
        "Prolyl peptidomimetic",
        "Yes", "No", "No", "No", "Yes", "High",
        "IC-1 is a synthetic P2-substituted prolylpyrrolidine inhibitor. It keeps strong substrate-like recognition in the proline pocket and is covalently bound at the catalytic serine. Used in a structure-guided series to inhibit porcine prolyl oligopeptidase.",
    ),
    "4AMZ": rec(
        "IC-2",
        "Prolyl oligopeptidase",
        "Prolyl peptidomimetic",
        "Yes", "No", "No", "No", "Yes", "High",
        "IC-2 is a synthetic cyclopentene prolylpeptidomimetic. It provides strong substrate-like noncovalent recognition and reacts at the catalytic serine. Used in a structure-guided prolyl-oligopeptidase inhibitor series.",
    ),
    "4AN0": rec(
        "IC-3",
        "Prolyl oligopeptidase",
        "Prolyl nitrile",
        "Yes", "No", "No", "No", "Yes", "High",
        "IC-3 is a synthetic phenylbutanoyl-prolylpyrrolidine nitrile. The prolyl scaffold provides strong substrate-like recognition, and the nitrile can bind covalently at the catalytic serine. Used in a structure-guided prolyl-oligopeptidase inhibitor series.",
    ),
    "4AN1": rec(
        "IC-4",
        "Prolyl oligopeptidase",
        "Prolyl peptidomimetic",
        "Yes", "No", "No", "No", "Yes", "High",
        "IC-4 is a synthetic N-benzyl prolylpyrrolidine inhibitor. It provides strong substrate-like noncovalent recognition and is covalently bound at the catalytic serine. Used in a structure-guided prolyl-oligopeptidase inhibitor series.",
    ),
    "4BCB": rec(
        "P2-substituted prolylpyrrolidine",
        "Prolyl oligopeptidase",
        "Prolyl peptidomimetic",
        "Yes", "No", "No", "No", "Yes", "High",
        "This synthetic P2-substituted N-acyl-prolylpyrrolidine was built to occupy the prolyl-oligopeptidase substrate site. Recognition is strong and substrate-like, with a covalent interaction at the catalytic serine. Used in a structure-guided inhibitor series.",
    ),
    "4BCC": rec(
        "P2-substituted prolylpyrrolidine",
        "Prolyl oligopeptidase",
        "Prolyl peptidomimetic",
        "Yes", "No", "No", "No", "Yes", "High",
        "This synthetic P2-substituted N-acyl-prolylpyrrolidine was built to occupy the prolyl-oligopeptidase substrate site. Recognition is strong and substrate-like, with a covalent interaction at the catalytic serine. Companion structure in the same inhibitor series.",
    ),
    "4DKI": rec(
        "Ceftobiprole",
        "MRSA PBP2a",
        "Beta-lactam",
        "Yes", "No", "No", "No", "Yes", "High",
        "Ceftobiprole is a synthetic anti-MRSA cephalosporin. The beta-lactam provides strong noncovalent recognition as a cell-wall-substrate mimic and acylates the catalytic serine of PBP2a. Developed against methicillin-resistant Staphylococcus aureus using penicillin-binding-protein structures.",
    ),
    "4EZ8": rec(
        "N4-hydroxy-dCMP (with dihydrofolate)",
        "Thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "No", "No", "No", "High",
        "N4-hydroxy-2'-deoxycytidine monophosphate is a synthetic nucleotide analog of the thymidylate-synthase substrate, bound with dihydrofolate. It has strong noncovalent recognition in the nucleotide site and forms a covalent adduct with the active-site cysteine. Used to describe a mouse thymidylate-synthase ternary complex.",
    ),
    "4F8B": rec(
        "PreQ0",
        "Nitrile reductase QueF",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "PreQ0 is the natural nitrile substrate of QueF, trapped as a covalent thioimide on the catalytic cysteine. Recognition is the normal substrate site. Used to characterize the catalytic intermediate of this nitrile reductase.",
    ),
    "4IW2": rec(
        "Glucose",
        "Human serum albumin",
        "Glycation",
        "No", "Yes", "Yes", "", "No", "High",
        "Glucose is the natural sugar, here as a glycation adduct on Lys195 of albumin. Albumin glycation is not strong, site-optimized ligand recognition of the kind seen for enzyme inhibitors. Used as an HSA–glucose complex rather than a designed covalent ligand.",
    ),
    "4MT4": rec(
        "Dihexanoyl glycerol",
        "CmeC outer-membrane channel",
        "Channel lipid",
        "Limited", "Yes", "No", "No", "No", "Low",
        "The deposited ligand is a synthetic short-chain diacylglycerol in the Campylobacter CmeC channel, together with detergent. The acyl chains can contact the channel, but this is not an optimized covalent inhibitor, and the activated-ester label is likely the lipid ester rather than a designed warhead. Low confidence as a covalent test-set example.",
    ),
    "4OE7": rec(
        "Pyruvate–glyoxal aldol adduct",
        "YagE aldolase",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "Medium",
        "The ligand is the aldol product of the metabolites pyruvate and glyoxal, covalently bound on the YagE lysine. Recognition is the enzyme's substrate site. Used as a catalytic complex of this aldolase, not as a designed inhibitor.",
    ),
    "4S2B": rec(
        "Tagatose 6-phosphate",
        "E. coli transaldolase TalB",
        "Substrate analog",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "Tagatose 6-phosphate is a sugar-phosphate substrate analog, covalently bound as a Schiff base on transaldolase. Recognition is the normal sugar-phosphate site. Used as a covalent mechanistic complex.",
    ),
    "4S2C": rec(
        "Fructose 6-phosphate",
        "E. coli transaldolase TalB",
        "Catalytic intermediate",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "Fructose 6-phosphate is the natural substrate, covalently bound as a Schiff base on transaldolase TalB. Recognition is the sugar-phosphate site. Used as a covalent mechanistic complex.",
    ),
    "5E5I": rec(
        "Ornithine aminotransferase inactivator",
        "Toxoplasma ornithine aminotransferase",
        "Mechanism-based inactivator",
        "Yes", "Yes", "No", "No", "Partial", "Medium",
        "The ligand is a synthetic mechanism-based inactivator covalently linked through the PLP cofactor in the ornithine site. Recognition is substrate-like. Used to inactivate the Toxoplasma enzyme and describe that adduct. Medium confidence on the exact chemotype name.",
    ),
    "5FCT": rec(
        "5-fluoro-dUMP (with methylenetetrahydrofolate)",
        "Thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "No", "No", "No", "High",
        "5-fluoro-dUMP is a synthetic fluorinated substrate analog, bound with methylenetetrahydrofolate. It has strong noncovalent recognition in the nucleotide site and forms the catalytic covalent adduct with the active-site cysteine. Used as the canonical ternary complex of mouse thymidylate synthase.",
    ),
    "5GV5": rec(
        "Phenyl phosphinate inhibitor",
        "Candida antarctica lipase B",
        "Phosphonate suicide inhibitor",
        "Yes", "Yes", "No", "No", "No", "High",
        "The ligand is a synthetic phosphinate with a phenyl substituent, covalently attached to Ser105. The aryl group provides substrate-like noncovalent recognition in the lipase pocket. Used as an active-site modifier of lipase B, in the classical suicide-inhibitor tradition.",
    ),
    "5HZQ": rec(
        "Aryl fluorosulfate probe",
        "CRABP2",
        "Chemical-biology probe",
        "Yes", "No", "No", "Yes", "Yes", "Medium",
        "The ligand is a synthetic biphenyl fluorosulfate with a polyethylene-glycol alkyne. The biaryl group provides noncovalent recognition in the retinoid-binding protein, and the fluorosulfate covalently modifies Tyr134. Used as a chemical-biology conjugate rather than a substrate analog or a classical covalent drug.",
    ),
    "5M4Z": rec(
        "N4-hydroxy-dCMP",
        "Trichinella thymidylate synthase",
        "Catalytic nucleotide adduct",
        "Yes", "Yes", "No", "No", "No", "High",
        "N4-hydroxy-dCMP is a synthetic nucleotide analog of the thymidylate-synthase substrate. It has strong noncovalent recognition in the nucleotide site and forms a covalent adduct with the active-site cysteine. Used as a mechanistic complex of the parasite enzyme.",
    ),
    "5TH7": rec(
        "MS453",
        "SETD8",
        "Drug-like covalent inhibitor",
        "Yes", "No", "No", "Yes", "Yes", "High",
        "MS453 is a synthetic quinazoline inhibitor of the lysine methyltransferase SETD8. The quinazoline and its substituents provide strong noncovalent recognition that is not a substrate copy, and an acrylamide covalently modifies a cysteine near the site. Used as a structure-guided covalent epigenetic inhibitor.",
    ),
    "5UW6": rec(
        "Z-Pro-prolinal",
        "Peptide cyclase PCY1",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Z-Pro-prolinal is a synthetic dipeptide aldehyde used to trap peptide cyclase PCY1, together with the follower peptide. It provides strong substrate-like recognition and reacts covalently in the active site. Used to visualize substrate-peptide handling by the cyclase.",
    ),
    "5UZW": rec(
        "Z-Pro-prolinal",
        "Peptide cyclase PCY1 variant",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Z-Pro-prolinal is a synthetic dipeptide aldehyde bound to a PCY1 insertion variant, together with the follower peptide. Recognition is strong and substrate-like. Used to compare peptide handling by the variant cyclase.",
    ),
    "5W1Y": rec(
        "Naphthoquinone inhibitor",
        "SETD8",
        "Drug-like covalent inhibitor",
        "Yes", "No", "No", "Yes", "Yes", "High",
        "The ligand is a synthetic 2-(methylpiperazinyl)-3-(phenylthio)naphthoquinone. The naphthoquinone scaffold provides noncovalent recognition that is not a peptide or substrate copy, and it covalently modifies a SETD8 cysteine as a Michael acceptor. Used as a covalent chemical probe of SETD8.",
    ),
    "5XPS": rec(
        "Erythrose 4-phosphate",
        "Transketolase",
        "Substrate complex",
        "Yes", "Yes", "Yes", "No", "No", "High",
        "Erythrose 4-phosphate is the natural transketolase substrate, bound with thiamine diphosphate. Recognition is the sugar-phosphate site. The test-file warhead label sits on this substrate complex. Used as a catalytic complex of Pichia stipitis transketolase.",
    ),
    "5Y41": rec(
        "Prostaglandin",
        "Nurr1 ligand-binding domain",
        "Endogenous lipid ligand",
        "Yes", "Yes", "Yes", "Yes", "No", "High",
        "The ligand is a natural prostaglandin covalently bound at Cys566 in the Nurr1 pocket. The lipid has real noncovalent recognition in the ligand-binding domain and was not built as an enzyme-substrate analog. Used to describe endogenous ligand recognition by Nurr1.",
    ),
    "5YD6": rec(
        "Prostaglandin",
        "Nurr1 ligand-binding domain",
        "Endogenous lipid ligand",
        "Yes", "Yes", "Yes", "Yes", "No", "High",
        "The ligand is a natural prostaglandin covalently bound at Cys566 in the Nurr1 pocket. The lipid has real noncovalent recognition in the ligand-binding domain and was not built as an enzyme-substrate analog. Companion Nurr1–prostaglandin complex.",
    ),
    "6AY2": rec(
        "Cyclobutane-dicarboxamide linker (compound 28)",
        "Cathepsin B",
        "Peptidomimetic ADC linker",
        "Yes", "No", "No", "No", "Yes", "High",
        "Compound 28 is a synthetic cyclobutane-1,1-dicarboxamide built as a non-peptide stand-in for the Val-Cit ADC linker. It still uses cathepsin B's substrate recognition and is covalently linked at the active-site cysteine. Used for structure-guided design of a cathepsin-B-cleavable antibody-drug-conjugate linker.",
    ),
    "6BOZ": rec(
        "MS4138",
        "SETD8",
        "Drug-like covalent inhibitor",
        "Yes", "No", "No", "Yes", "Yes", "High",
        "MS4138 is a synthetic quinazoline acrylamide. The quinazoline provides strong noncovalent recognition of SETD8 that is not a substrate copy, and the acrylamide covalently modifies a nearby cysteine. Used as a structure-guided covalent inhibitor of this methyltransferase.",
    ),
    "6C7Z": rec(
        "Julolidine retinal analog",
        "Engineered CRBPII",
        "Retinoid analog",
        "Yes", "Yes", "No", "No", "Yes", "High",
        "Julolidine is a synthetic retinal analog. It has strong noncovalent recognition in the engineered retinol-binding pocket and forms a Schiff base with a lysine. Used for protein-engineering studies of ligand color, not as a covalent drug.",
    ),
    "6G83": rec(
        "sPBTZ169",
        "M. tuberculosis DprE1",
        "Drug-like covalent inhibitor",
        "Yes", "No", "No", "Yes", "Yes", "High",
        "sPBTZ169 is a synthetic benzothiazinone anti-tubercular inhibitor. The benzothiazinone scaffold provides strong noncovalent recognition of DprE1 that is not a substrate copy, and it forms a covalent adduct with the active-site cysteine. Used as a covalent DprE1 inhibitor for tuberculosis.",
    ),
    "6H5O": rec(
        "Piperacillin",
        "MRSA PBP2a",
        "Beta-lactam",
        "Yes", "Yes", "No", "No", "No", "High",
        "Piperacillin is a synthetic penicillin from the classical beta-lactam generation. It provides strong noncovalent recognition as a cell-wall-substrate mimic and acylates the catalytic serine. This structure places that older drug on MRSA PBP2a.",
    ),
    "6HJE": rec(
        "NG-P27",
        "Trypanosoma cruzi proline racemase",
        "Small substrate analog",
        "Limited", "No", "No", "No", "Yes", "Medium",
        "NG-P27 is a small synthetic cyclopentane acetic-acid analog of proline. It occupies the substrate site but does not have a large recognition scaffold. Used as a covalent or tightly bound inhibitor to characterize proline racemase.",
    ),
    "6HJF": rec(
        "BrOxoPA (bound as levulinate)",
        "Trypanosoma cruzi proline racemase",
        "Small substrate analog",
        "Limited", "No", "No", "No", "Yes", "Medium",
        "The deposited ligand is levulinic acid, a small piece of the proline-racemase inhibitor BrOxoPA. It is a substrate-site binder without a large scaffold. Used to characterize covalent inhibition of proline racemase.",
    ),
    "6M8W": rec(
        "Ac-Ile-Ala-Phe-CHO",
        "Sedolisin",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Ac-Ile-Ala-Phe-CHO is a synthetic peptide aldehyde. Its peptide sequence provides strong noncovalent recognition in the sedolisin substrate channel, while the aldehyde reacts with the catalytic serine. A re-release of the classical serine-carboxyl proteinase inhibitor complex.",
    ),
    "6M8Y": rec(
        "Ac-Ile-Pro-Phe-CHO",
        "Sedolisin",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Ac-Ile-Pro-Phe-CHO is a synthetic peptide aldehyde. Its peptide sequence provides strong noncovalent recognition in the sedolisin substrate channel, while the aldehyde reacts with the catalytic serine. A re-release of the classical inhibitor complex.",
    ),
    "6M9C": rec(
        "Pseudotyrostatin",
        "Sedolisin",
        "Peptide aldehyde",
        "Yes", "Yes", "No", "No", "No", "High",
        "Pseudotyrostatin is a synthetic analog of the natural peptide aldehyde tyrostatin. The peptide provides strong substrate-channel recognition and the aldehyde reacts with the catalytic serine. Used to characterize sedolisin specificity.",
    ),
    "6M9D": rec(
        "Chymostatin",
        "Sedolisin",
        "Natural peptide aldehyde",
        "Yes", "Yes", "Yes", "Yes", "No", "High",
        "Chymostatin is a microbial natural-product peptide aldehyde. Its peptide scaffold provides strong noncovalent recognition in the sedolisin substrate channel, while the aldehyde reacts with the catalytic serine. Used as a natural covalent peptidase inhibitor.",
    ),
    "6M9F": rec(
        "Tyrostatin",
        "Sedolisin",
        "Natural peptide aldehyde",
        "Yes", "Yes", "Yes", "Yes", "No", "High",
        "Tyrostatin is a microbial natural-product peptide aldehyde. Its peptide scaffold provides strong noncovalent recognition in the sedolisin substrate channel, while the aldehyde reacts with the catalytic serine. Used as a natural covalent peptidase inhibitor.",
    ),
    "6MKF": rec(
        "Imipenem",
        "Enterococcus faecium PBP5",
        "Beta-lactam",
        "Yes", "Yes", "No", "No", "No", "High",
        "Imipenem is a synthetic carbapenem from the classical beta-lactam generation. It provides strong noncovalent recognition as a cell-wall-substrate mimic and acylates the catalytic serine. Used to show the covalent imipenem adduct on enterococcal PBP5.",
    ),
    "6N9J": rec(
        "Irreversible peptide inhibitor",
        "Bacteroides C11 cysteine protease",
        "Peptide inhibitor",
        "Yes", "Yes", "No", "No", "Partial", "Medium",
        "The ligand is a peptide inhibitor covalently bound in this secreted C11 cysteine protease. Peptide recognition in the substrate channel is strong. Treated as a synthetic substrate-like peptide inhibitor. Medium confidence because the deposited name does not give the exact sequence.",
    ),
    "6ND3": rec(
        "Betaine aldehyde",
        "Choline trimethylamine-lyase",
        "Metabolic product",
        "Yes", "Yes", "Yes", "No", "No", "Medium",
        "Betaine aldehyde is the natural product of choline trimethylamine-lyase. It is small, but it is the metabolite recognized by the active site and can form a covalent adduct with a cysteine. Used as the product complex of CutC, not as a designed inhibitor.",
    ),
    "6OIA": rec(
        "Perfluoro cyclopentane amino acid",
        "Ornithine aminotransferase",
        "Mechanism-based inactivator",
        "Yes", "No", "No", "No", "Yes", "High",
        "This synthetic fluorinated cyclopentane amino acid is a mechanism-based ornithine-aminotransferase inhibitor. It is recognized as a substrate analog in the ornithine/GABA site and covalently inactivates the PLP enzyme. Used as a modern, structure-informed inactivator.",
    ),
    "5OGQ": rec(
        "WRR391",
        "Schistosoma mansoni cathepsin B1",
        "Peptide vinyl sulfone",
        "Yes", "No", "No", "No", "Yes", "High",
        "WRR391 is a synthetic peptide vinyl sulfone. The peptide provides strong noncovalent recognition of the cathepsin substrate channel, while the vinyl sulfone covalently modifies the catalytic cysteine. Used in a structure-guided series against schistosome cathepsin B1.",
    ),
    "5OGR": rec(
        "WRR286",
        "Schistosoma mansoni cathepsin B1",
        "Peptide vinyl sulfone",
        "Yes", "No", "No", "No", "Yes", "High",
        "WRR286 is a synthetic phenylalanine–vinyl-sulfone inhibitor. The peptide provides strong noncovalent recognition of the cathepsin substrate channel, while the vinyl sulfone covalently modifies the catalytic cysteine. Used in a structure-guided series against schistosome cathepsin B1.",
    ),
    "6Q9N": rec(
        "Piperacillin",
        "MRSA PBP2a",
        "Beta-lactam",
        "Yes", "Yes", "No", "No", "Partial", "High",
        "Piperacillin is a synthetic classical penicillin, bound covalently at the PBP2a active site, with a quinazolinone also present at the allosteric site. The beta-lactam itself is a cell-wall-substrate mimic with strong noncovalent recognition. The penicillin predates modern covalent design; the complex was used to study allosteric quinazolinone binding.",
    ),
    "6RB0": rec(
        "Methyl hexylphosphonate",
        "Metagenomic esterase EH1",
        "Phosphonate suicide inhibitor",
        "Yes", "Yes", "No", "No", "No", "High",
        "Methyl hexylphosphonate is a synthetic suicide inhibitor. The hexyl chain provides substrate-like noncovalent recognition, and the phosphonate covalently modifies the catalytic serine. A classical lipase/esterase probe, here on a lake-metagenome esterase.",
    ),
    "6TP8": rec(
        "Diethyl phosphonate",
        "Candida antarctica lipase B",
        "Small reactive probe",
        "No", "Yes", "No", "", "No", "Medium",
        "The covalent species is diethyl phosphonate on Ser105, a small generic serine adduct. A tributyrin substrate analog is also present and supplies the lipid recognition, but that molecule is not the covalent ligand. Used to discuss substrate contacts at the rim of the lipase site.",
    ),
    "6U06": rec(
        "Isoquinolinone sulfonyl fluoride",
        "eIF4E",
        "Drug-like covalent inhibitor",
        "Yes", "No", "No", "Yes", "Yes", "High",
        "The ligand is a synthetic isoquinolinone aryl-sulfonyl fluoride found by covalent docking. The heterocycle provides noncovalent recognition of eIF4E that is not a substrate copy, and the sulfonyl fluoride covalently modifies a lysine. Used to discover lysine-targeted eIF4E inhibitors.",
    ),
    "6U09": rec(
        "Isoquinolinone sulfonyl fluoride",
        "eIF4E",
        "Drug-like covalent inhibitor",
        "Yes", "No", "No", "Yes", "Yes", "High",
        "The ligand is a synthetic isoquinolinone aryl-sulfonyl fluoride, a close analog of the 6U06 inhibitor. The scaffold provides noncovalent recognition that is not a substrate copy, and the sulfonyl fluoride covalently modifies a lysine. Used in the same covalent-docking study of eIF4E.",
    ),
    "6YI7": rec(
        "Azanitrile inhibitor",
        "Schistosoma mansoni cathepsin B1",
        "Peptide nitrile",
        "Yes", "No", "No", "No", "Yes", "High",
        "The azanitrile is a synthetic peptidomimetic cathepsin inhibitor. It provides substrate-like noncovalent recognition and the nitrile reacts covalently with the catalytic cysteine. Used as a structure-guided inhibitor of schistosome cathepsin B1.",
    ),
    "6ZU8": rec(
        "Afatinib",
        "Brachyury G177D",
        "Drug-like covalent inhibitor",
        "Yes", "No", "No", "Yes", "Yes", "High",
        "Afatinib is a synthetic quinazoline acrylamide from the modern targeted-covalent kinase-inhibitor generation. The quinazoline provides strong noncovalent recognition that is not a substrate copy, and the acrylamide can covalently modify a cysteine. Here it is bound to a Brachyury mutant rather than EGFR. Used as an off-target or repurposed covalent-drug complex.",
    ),
    "7M8R": rec(
        "1,1,1-trifluoroacetone",
        "Methane monooxygenase",
        "Small active-site probe",
        "Limited", "Yes", "No", "No", "No", "Medium",
        "1,1,1-trifluoroacetone is a small synthetic probe that binds the methane-monooxygenase active site. It has little scaffold beyond the trifluoromethyl ketone. The structure's main subject is fluorinated tryptophans in the hydroxylase. Not a site-optimized covalent ligand.",
    ),
    "7V6K": rec(
        "ML346",
        "Streptococcus pyogenes sortase A",
        "Screening-derived covalent inhibitor",
        "Yes", "No", "No", "Yes", "Partial", "Medium",
        "ML346 is a synthetic benzylidene-barbiturate inhibitor of sortase A. The aryl-barbiturate scaffold provides noncovalent recognition that is not a peptide-substrate copy, and it can react covalently at the catalytic cysteine. Used as a covalent sortase inhibitor from a screening series.",
    ),
}


def _prot_note(ligand, origin, warhead, purpose, scaffold="peptide"):
    if origin == "natural":
        kind = "natural-product"
    elif origin == "semisynthetic":
        kind = "semisynthetic"
    else:
        kind = "synthetic"
    if scaffold == "peptide":
        rec = (
            f"Its peptide scaffold provides strong noncovalent recognition in the "
            f"proteasome substrate channel, while the {warhead} reacts with the "
            f"N-terminal threonine."
        )
    elif scaffold == "natural-product":
        if origin == "natural":
            origin_clause = (
                "It was discovered as a natural product rather than built as a substrate-sequence copy."
            )
        else:
            origin_clause = (
                "The scaffold comes from a natural-product inhibitor rather than a substrate-sequence peptide; this particular analog was prepared synthetically."
            )
        rec = (
            f"Its scaffold provides strong noncovalent recognition of the proteasome "
            f"substrate sites, while the {warhead} reacts with the N-terminal threonine. "
            f"{origin_clause}"
        )
    else:
        rec = (
            f"Its scaffold provides strong noncovalent recognition of the proteasome, "
            f"while the {warhead} reacts with the N-terminal threonine."
        )
    return (
        f"{ligand} is a {kind} covalent proteasome inhibitor. {rec} {purpose}"
    )


def _purpose(title: str, kind: str) -> str:
    t = title.lower()
    if "mutant" in t and kind in {"clinical", "selectivity", "classical", "derivative"}:
        if kind == "clinical":
            return "Used to map how proteasome mutations affect binding of this clinical covalent inhibitor."
        return "Used to map how active-site or subunit mutations change inhibitor binding."
    if kind == "natural":
        return "Used to establish how this natural product recognizes and covalently inhibits the proteasome."
    if kind == "derivative":
        return "Used to compare a synthetic analog of a natural-product proteasome inhibitor."
    if kind == "clinical":
        return "Used as a covalent proteasome drug and to define its binding mode at the active site."
    if kind == "selectivity":
        return "Used for structure-guided comparison of peptide sequence and subunit selectivity."
    return "Used to relate warhead chemistry and peptide sequence to proteasome inhibition."


def classify_proteasome(title: str) -> dict:
    t = title.lower()

    def make(ligand, origin, warhead, not_sub, before, guided, scaffold, klass, purpose_kind, confidence="High"):
        if origin != "natural" and purpose_kind == "natural":
            purpose_kind = "derivative"
        purpose = _purpose(title, purpose_kind)
        return rec(
            ligand,
            "Proteasome",
            klass,
            "Yes",
            before,
            "Yes" if origin == "natural" else "No",
            not_sub,
            guided,
            confidence,
            _prot_note(ligand, origin, warhead, purpose, scaffold),
        )

    # Natural products first, including when the title also says epoxyketone.
    if "calpain" in t:
        return rec(
            "Calpain inhibitor I",
            "Proteasome",
            "Peptide aldehyde",
            "Yes", "Yes", "No", "No", "No", "High",
            "Calpain inhibitor I is a synthetic peptide-like aldehyde inhibitor. Its peptide sequence provides strong noncovalent recognition within the proteasome substrate-binding channel, while the aldehyde forms the covalent interaction with N-terminal Thr. Used as a covalent proteasome inhibitor and in studies of proteasome maturation and regulation.",
        )
    if "mln-273" in t or "peptidyl boronate" in t:
        return make(
            "MLN-273", "synthetic", "boronic acid", "No", "Yes", "Partial",
            "peptide", "Peptide boronate", "classical",
        )
    if "substrate analogue" in t or "substrate analog" in t:
        return make(
            "Peptide vinyl-sulfone substrate analog", "synthetic", "vinyl sulfone", "No", "Yes", "No",
            "peptide", "Substrate-sequence peptide", "classical",
        )
    if "epoxomicin" in t:
        return rec(
            "Epoxomicin",
            "Proteasome",
            "Natural epoxyketone",
            "Yes", "Yes", "Yes", "Yes", "No", "High",
            "Epoxomicin is a natural-product peptide containing an alpha,beta-epoxyketone. Its peptide scaffold provides strong noncovalent recognition of the proteasome, while the epoxyketone reacts with N-terminal Thr to form a stable covalent adduct. Used as a selective covalent proteasome inhibitor and to establish the structural basis of epoxyketone selectivity.",
        )
    if "dihydroeponemycin" in t:
        return make(
            "Dihydroeponemycin", "semisynthetic", "epoxyketone", "Yes", "Yes", "No",
            "natural-product", "Natural-product epoxyketone", "natural",
        )
    if "homosalinosporamide" in t:
        return make(
            "Homosalinosporamide A", "semisynthetic", "beta-lactone", "Yes", "No", "Yes",
            "natural-product", "Salinosporamide analog", "derivative",
        )
    if "hydroxylated salinosporamide" in t or "fluorosalinosporamide" in t or "salinosporamide derivatives" in t:
        return make(
            "Salinosporamide derivative", "semisynthetic", "beta-lactone", "Yes", "No", "Yes",
            "natural-product", "Salinosporamide analog", "natural",
        )
    if "salinosporamide a" in t:
        return make(
            "Salinosporamide A", "natural", "beta-lactone", "Yes", "Yes", "No",
            "natural-product", "Natural beta-lactone", "natural",
        )
    if "omuralide" in t or "spirolactacystin" in t:
        ligand = "Omuralide" if "omuralide" in t or "spirolactacystin" in t and "epimer" not in t else "Spirolactacystin / omuralide"
        if "epimer" in t:
            ligand = "Spirolactacystin epimer"
        return make(
            ligand, "semisynthetic", "beta-lactone", "Yes", "Yes", "No",
            "natural-product", "Lactacystin-family beta-lactone", "natural",
        )
    if "syringolin-glidobactin" in t or "chimera" in t:
        return make(
            "Syringolin–glidobactin chimera", "synthetic", "Michael acceptor", "Yes", "No", "Yes",
            "natural-product", "Syrbactin analog", "natural",
        )
    if "syringolin" in t:
        ligand = "Syringolin B" if "syringolin b" in t else "Syringolin A"
        return make(
            ligand, "natural", "Michael acceptor", "Yes", "Yes", "No",
            "natural-product", "Natural syrbactin", "natural",
        )
    if "glidobactin-like" in t or "hb333" in t or "hb334" in t or "hb335" in t:
        return make(
            "Glidobactin-like natural product", "natural", "Michael acceptor", "Yes", "Yes", "No",
            "natural-product", "Natural syrbactin", "natural",
        )
    if "cepafungin" in t:
        return make(
            "Cepafungin I", "natural", "Michael acceptor", "Yes", "Yes", "No",
            "natural-product", "Natural syrbactin", "natural",
        )
    if "glidobactin" in t:
        return make(
            "Glidobactin A", "natural", "Michael acceptor", "Yes", "Yes", "No",
            "natural-product", "Natural syrbactin", "natural",
        )
    if "fellutamide" in t:
        return make(
            "Fellutamide B", "natural", "aldehyde", "Yes", "Yes", "No",
            "peptide", "Natural peptide aldehyde", "natural",
        )
    if "homobelactosin" in t:
        return make(
            "Homobelactosin C", "natural", "beta-lactone", "Yes", "Yes", "No",
            "natural-product", "Natural belactosin", "natural",
        )
    if "belactosin" in t:
        deriv = "derivative" in t or " 3e" in t or "derivative 3" in t
        if deriv:
            return make(
                "Belactosin derivative", "synthetic", "beta-lactone", "Yes", "No", "Yes",
                "natural-product", "Belactosin analog", "natural",
            )
        return make(
            "Belactosin C", "natural", "beta-lactone", "Yes", "Yes", "No",
            "natural-product", "Natural belactosin", "natural",
        )
    if "carmaphycin a" in t or "natural product carmaphycin" in t:
        return make(
            "Carmaphycin A", "natural", "epoxyketone", "Yes", "Yes", "No",
            "peptide", "Natural epoxyketone", "natural",
        )
    if "carmaphycin" in t:
        return make(
            "Carmaphycin analogue", "synthetic", "epoxyketone or vinyl ketone", "No", "No", "Yes",
            "peptide", "Designed epoxyketone", "selectivity",
        )
    if "vibralactone" in t:
        return make(
            "Vibralactone", "natural", "beta-lactone", "Yes", "Yes", "No",
            "natural-product", "Natural beta-lactone", "natural",
        )
    if "cystargolide b derivative" in t or "cystargolide b deriv" in t:
        return make(
            "Cystargolide B derivative", "synthetic", "epoxyketone-like warhead", "Yes", "No", "Yes",
            "natural-product", "Cystargolide analog", "natural",
        )
    if "cystargolide" in t:
        return make(
            "Cystargolide B", "natural", "beta-lactone / epoxyketone-like warhead", "Yes", "Yes", "No",
            "natural-product", "Natural beta-lactone", "natural",
        )
    if "macyranone" in t:
        return make(
            "Macyranone A", "synthetic", "epoxyketone-like warhead", "Yes", "No", "Yes",
            "natural-product", "Syrbactin analog", "natural", "Medium",
        )
    if "bortezomib" in t:
        return make(
            "Bortezomib", "synthetic", "boronic acid", "No", "Yes", "Partial",
            "peptide", "Clinical peptide boronate", "clinical",
        )
    if "ixazomib" in t:
        return make(
            "Ixazomib", "synthetic", "boronic acid", "No", "No", "Yes",
            "peptide", "Clinical peptide boronate", "clinical",
        )
    if "delanzomib" in t:
        return make(
            "Delanzomib", "synthetic", "boronic acid", "No", "No", "Yes",
            "peptide", "Clinical peptide boronate", "clinical",
        )
    if "carfilzomib" in t:
        return make(
            "Carfilzomib", "synthetic", "epoxyketone", "No", "Yes", "Partial",
            "peptide", "Clinical peptide epoxyketone", "clinical",
        )
    if "onx 0914" in t or "pr-957" in t:
        return make(
            "ONX 0914 (PR-957)", "synthetic", "epoxyketone", "No", "No", "Yes",
            "peptide", "Immunoproteasome-selective epoxyketone", "selectivity",
        )
    if "pr-924" in t:
        return make(
            "PR-924", "synthetic", "epoxyketone", "No", "No", "Yes",
            "peptide", "Immunoproteasome-selective epoxyketone", "selectivity",
        )
    if "mg132" in t:
        return make(
            "MG132", "synthetic", "aldehyde", "No", "Yes", "No",
            "peptide", "Peptide aldehyde", "classical",
        )
    if "z-leu-leu-leu" in t or "boc-ala-ala-ala" in t or "bsc4999" in t or "alpha-keto" in t:
        ligand = "Z-Leu-Leu-Leu warhead variant"
        if "boc-ala" in t:
            ligand = "Boc-Ala-Ala-Ala-CHO"
        elif "bsc4999" in t or "keto phenylamide" in t or "alpha-keto" in t:
            ligand = "Alpha-ketoamide peptide"
        return make(
            ligand, "synthetic", "aldehyde, boronate, vinyl sulfone, or ketoamide", "No", "Yes", "Partial",
            "peptide", "Substrate-sequence peptide", "classical",
        )
    if "vinyl sulfone lu" in t or "vinyl sulfone" in t and "lu10" in t.replace(" ", ""):
        return make(
            "Peptide vinyl sulfone", "synthetic", "vinyl sulfone", "No", "No", "Yes",
            "peptide", "Peptide vinyl sulfone", "selectivity",
        )
    if "ada-" in t or "(leu)3-vinyl" in t or "vinyl sulfone" in t:
        return make(
            "Peptide vinyl sulfone", "synthetic", "vinyl sulfone", "No", "Yes", "Partial",
            "peptide", "Peptide vinyl sulfone", "classical",
        )
    if "chloroacetamide" in t:
        return rec(
            "Peptide alpha-chloroacetamide",
            "Proteasome",
            "Engineered-cysteine peptide probe",
            "Yes", "No", "No", "No", "Yes", "High",
            "This is a synthetic carfilzomib-derived peptide bearing an alpha-chloroacetamide. The peptide provides strong noncovalent recognition in the proteasome substrate channel, and the chloroacetamide covalently modifies an engineered non-catalytic cysteine (beta5 Cys48 here) rather than the catalytic threonine. Used for structure-based design of immunoproteasome-selective inhibitors.",
        )
    if "alkyne-pi" in t or "octreotide" in t:
        return make(
            "Peptide boronate conjugate", "synthetic", "boronic acid", "No", "No", "Yes",
            "peptide", "Peptide boronate", "selectivity",
        )
    if "-ep" in t or "epoxyketone" in t:
        return make(
            "Peptide epoxyketone", "synthetic", "epoxyketone", "No", "No", "Yes",
            "peptide", "Designed peptide epoxyketone", "selectivity",
        )
    if "plasmodium" in t:
        return make(
            "Plasmodium-selective peptide inhibitor", "synthetic", "vinyl sulfone", "No", "No", "Yes",
            "peptide", "Designed peptide inhibitor", "selectivity",
        )
    if "compound 1" in t or "boronic acid" in t:
        return make(
            "Peptide-like boronic acid", "synthetic", "boronic acid", "No", "No", "Yes",
            "peptide", "Designed peptide boronate", "selectivity",
        )
    # Numbered inhibitors in the beta2-selectivity series: "complex with 4", "with 13", etc.
    if " complex with " in t:
        tail = t.split(" complex with ", 1)[1]
        if tail[:1].isdigit() or tail.startswith("lx"):
            return make(
                "Subunit-selective peptide inhibitor", "synthetic", "epoxyketone or vinyl sulfone", "No", "No", "Yes",
                "peptide", "Designed peptide inhibitor", "selectivity",
            )
    return None


def is_proteasome(title: str) -> bool:
    t = title.lower()
    return (
        "proteasome" in t
        or t.startswith("ycp")
        or " ycp " in f" {t} "
        or "yeast cp" in t
        or "20s particle" in t
        or "immunoproteasome" in t
    )


def load_meta():
    entries = json.loads(META.read_text(encoding="utf-8"))
    out = {}
    for e in entries:
        pid = e["rcsb_id"]
        title = (e.get("struct") or {}).get("title") or ""
        date = ((e.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:4]
        out[pid] = {"title": title, "year": date}
    return out


def main():
    labels = pd.read_csv(CSV)
    labels["PDB"] = labels["Name"].astype(str).str[:4].str.upper()
    grouped = labels.groupby("PDB").agg(
        residue=("Residue", lambda s: ",".join(sorted(set(s.astype(str))))),
        resnum=("ResNum", lambda s: ",".join(sorted(set(map(str, s))))),
        chain=("Chain", lambda s: ",".join(sorted(set(s.astype(str))))),
        best_rank=("target_rank", "min"),
        hit_top=("hit_at_top_pct", "max"),
        n_rows=("Name", "size"),
    )
    meta = load_meta()
    rows = []
    missed = []
    for pdb, g in grouped.iterrows():
        info = meta[pdb]
        title = info["title"]
        if pdb in SPECIAL:
            hit = SPECIAL[pdb]
        elif is_proteasome(title):
            hit = classify_proteasome(title)
            if hit is None:
                missed.append((pdb, title))
                continue
        else:
            missed.append((pdb, title))
            continue
        rows.append(
            {
                "PDB": pdb,
                "Strong noncovalent?": hit["strong"],
                "Notes": hit["notes"],
                "Before Modern Covalent Generation (No crystal structures)": hit["before"],
                "All Natural?": hit["natural"],
                "Not formed from substrate likeness if noncovalent": hit["not_substrate"],
                "Ligand": hit["ligand"],
                "Target": hit["target"],
                "Covalent residue": f"{g['residue']} {g['resnum']} (chain {g['chain']})",
                "Release year": info["year"],
                "Ligand class": hit["klass"],
                "Structure-guided design?": hit["guided"],
                "Confidence": hit["confidence"],
                "Best target rank in test file": int(g["best_rank"]),
                "Hit in top 1% in test file": "Yes" if g["hit_top"] == 1 else "No",
                "_title": title,
            }
        )
    if missed:
        print("UNMATCHED", len(missed))
        for pdb, title in missed:
            print(f"  {pdb}: {title}")
        raise SystemExit(1)

    df = pd.DataFrame(rows).sort_values("PDB")
    # Summary numbers.
    n = len(df)
    strong = df["Strong noncovalent?"].value_counts()
    print("N", n)
    print(strong.to_string())
    yes = df[df["Strong noncovalent?"] == "Yes"]
    print("of Yes, not-substrate Yes", (yes["Not formed from substrate likeness if noncovalent"] == "Yes").sum())
    print("of Yes, not-substrate No", (yes["Not formed from substrate likeness if noncovalent"] == "No").sum())
    print("of Yes, natural", (yes["All Natural?"] == "Yes").sum())
    print("of Yes, before modern", (yes["Before Modern Covalent Generation (No crystal structures)"] == "Yes").sum())
    print("class counts:")
    print(df["Ligand class"].value_counts().to_string())
    print("rank by strong:")
    print(df.groupby("Strong noncovalent?")["Best target rank in test file"].describe().to_string())
    print("hit top by strong:")
    print(pd.crosstab(df["Strong noncovalent?"], df["Hit in top 1% in test file"]))

    started = pd.read_excel(STARTED)
    out = df.drop(columns=["_title"]).fillna("")
    summary = build_summary(out)
    definitions = build_definitions()
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        out.to_excel(writer, sheet_name="Test set", index=False)
        definitions.to_excel(writer, sheet_name="How the calls were made", index=False)
        started.to_excel(writer, sheet_name="Started examples", index=False)
        format_workbook(writer.book)
    out.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_XLSX, OUT_CSV)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    strong_col = "Strong noncovalent?"
    sub_col = "Not formed from substrate likeness if noncovalent"
    before_col = "Before Modern Covalent Generation (No crystal structures)"
    yes = df[df[strong_col] == "Yes"]
    n_yes = len(yes)
    n_lim = int((df[strong_col] == "Limited").sum())
    n_no = int((df[strong_col] == "No").sum())
    n_sub = int((yes[sub_col] == "No").sum())
    n_notsub = int((yes[sub_col] == "Yes").sum())
    n_nat = int((yes["All Natural?"] == "Yes").sum())
    n_before = int((yes[before_col] == "Yes").sum())
    hit_yes = int((yes["Hit in top 1% in test file"] == "Yes").sum())
    other = df[df[strong_col] != "Yes"]
    hit_other = int((other["Hit in top 1% in test file"] == "Yes").sum())
    med_yes = int(yes["Best target rank in test file"].median())
    med_other = int(other["Best target rank in test file"].median()) if len(other) else ""
    rows = [
        ("Complexes in the test file", n, "One row per PDB. The label file had 1,056 warhead rows."),
        ("Strong noncovalent recognition", f"{n_yes} ({n_yes / n:.0%})", "A real recognition scaffold is already bound at the reactive site."),
        ("Limited recognition", f"{n_lim} ({n_lim / n:.0%})", "Small active-site contact, but not a substantial scaffold."),
        ("Little or no recognition scaffold", f"{n_no} ({n_no / n:.0%})", "Generic probes, fragments, a crosslinker, glycation, or an additive."),
        ("Of the strong cases, recognition is substrate-like", f"{n_sub} ({n_sub / n_yes:.0%} of strong)", "Substrate, transition-state analog, or a peptide copied from the substrate register. Column 'Not formed from substrate likeness' = No."),
        ("Of the strong cases, recognition is not a substrate copy", f"{n_notsub} ({n_notsub / n_yes:.0%} of strong)", "Natural-product scaffolds, following the epoxomicin call in the started sheet, or drug-like scaffolds such as afatinib, SETD8, eIF4E, and DprE1."),
        ("Of the strong cases, ligand is natural", f"{n_nat} ({n_nat / n_yes:.0%} of strong)", "Natural product or unmodified metabolite. Synthetic and semisynthetic ligands are No."),
        ("Of the strong cases, from before modern covalent design", f"{n_before} ({n_before / n_yes:.0%} of strong)", "Classical probes, natural products, metabolites, or inhibitor chemistry established before the structure-guided targeted-covalent wave."),
        ("Median best rank, strong scaffold", med_yes, "Minimum target rank across warhead rows for that PDB."),
        ("Median best rank, limited or none", med_other, "The small-probe complexes are also ranked at the top of this file."),
        ("Inside the top 1%, strong scaffold", f"{hit_yes} / {n_yes}", "Poor ranks are mostly large proteasome particles, not the complexes that lack a scaffold."),
        ("Inside the top 1%, limited or none", f"{hit_other} / {n_lim + n_no}", "This file does not show the small probes ranking worse."),
    ]
    out = pd.DataFrame(rows, columns=["Finding", "Value", "What it means"])
    out.loc[len(out)] = (
        "Reading for the ablation",
        "The noncovalent-only result is expected from this set",
        "About 94% of these covalent complexes already have a recognition scaffold at the labeled site, and about 83% of those scaffolds are the substrate or a substrate copy. A model that only sees noncovalent complementarity is then scoring a pocket the ligand was already built to fit. The set is not a sample of bare warheads placed on unoptimized scaffolds.",
    )
    return out


def build_definitions() -> pd.DataFrame:
    rows = [
        ("Strong noncovalent?", "Yes", "The ligand has a recognition scaffold that contacts the site beyond the reactive atom: a peptide, substrate, nucleotide, beta-lactam, or drug-like core."),
        ("Strong noncovalent?", "Limited", "A small probe with some active-site contact and no substantial scaffold. The started sheet used only Yes/No; Limited marks this boundary instead of forcing it to Yes."),
        ("Strong noncovalent?", "No", "Generic reagent, tiny fragment, chemical crosslinker, glycation, or crystallization additive. The substrate-likeness column is left blank, as in the MTSET example."),
        ("Before Modern Covalent Generation (No crystal structures)", "Yes", "The ligand comes from classical enzymology, a natural product, a metabolite, or covalent-inhibitor chemistry established before routine structure-guided targeted-covalent design. A later crystal structure of that same ligand is still Yes."),
        ("Before Modern Covalent Generation (No crystal structures)", "No", "The ligand was designed in a structure-guided campaign after structures of the site were already being used. Most of these are 2012–2021 proteasome-selectivity series, plus later covalent probes (SETD8, eIF4E, afatinib, DprE1, anti-MRSA cephalosporins)."),
        ("All Natural?", "Yes", "Natural product or unmodified metabolite. Semisynthetic analogs and fluorinated substrate analogs are No."),
        ("Not formed from substrate likeness if noncovalent", "No", "The noncovalent fit comes from the substrate, a transition-state analog, or a peptide copied from the substrate register. This is the majority of the test set."),
        ("Not formed from substrate likeness if noncovalent", "Yes", "Strong recognition is present, and the ligand was not built as a substrate copy. Epoxomicin in the started sheet is the template for natural products. Drug-like scaffolds that are not substrate copies are also Yes."),
        ("Not formed from substrate likeness if noncovalent", "blank", "Strong noncovalent is No, so the question does not apply."),
        ("Structure-guided design?", "Yes / Partial / No", "Separate from the era column. Partial means peptide SAR or early structure-based work that was not a modern covalent-drug campaign."),
        ("Confidence", "High / Medium / Low", "High when the PDB title and ligand name identify the complex. Low for 4MT4 (channel lipid) and the prolyl-oligopeptidase additive."),
        ("Warhead labels in the test file", "Caution", "Thymidylate-synthase rows are flagged as phosphonates or aryl halides because those patterns match the nucleotide phosphate and ring. The biological covalent bond is the catalytic cysteine–nucleotide adduct. Notes say so."),
        ("What this is not", "Not an energy calculation", "Calls are from the structure title, deposited ligand, citation, and the covalent residue in the test file. They are not a rescoring of noncovalent energy."),
    ]
    return pd.DataFrame(rows, columns=["Column", "Call", "Meaning"])


def format_workbook(book):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    widths = {
        "Summary": {"A": 62, "B": 42, "C": 88},
        "Test set": {
            "A": 12, "B": 22, "C": 78, "D": 28, "E": 16, "F": 28,
            "G": 36, "H": 32, "I": 28, "J": 14, "K": 36, "L": 24, "M": 14, "N": 18, "O": 16,
        },
        "How the calls were made": {"A": 55, "B": 22, "C": 110},
        "Started examples": {"A": 12, "B": 22, "C": 78, "D": 28, "E": 16, "F": 28},
    }
    for name in book.sheetnames:
        ws = book[name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 30
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = wrap
        for col, width in widths.get(name, {}).items():
            ws.column_dimensions[col].width = width
        if name == "Test set":
            ws.auto_filter.ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
            for row in ws.iter_rows(min_row=2, max_col=1):
                ws.row_dimensions[row[0].row].height = 48
        if name == "Summary":
            ws.row_dimensions[ws.max_row].height = 72


if __name__ == "__main__":
    main()
