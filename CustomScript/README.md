# CovSite

`CovSite.py` ranks nucleophilic covalent-binding sites for protein–ligand pairs
plus a saved LightGBM ranker
(`Cov_Screen`).


## Pipeline

1. Read input CSV (PDB + SMILES and/or LigID; optional target site).
2. Detect warheads and compute deprotonated xTB descriptors per residue type
   (`--no-warhead` skips this; reactivity columns are NaN).
3. Build candidate sites (CYS, SER, THR, TYR, LYS, HIS) with SASA + deprotonation probability.
4. Enrich with residue / N-terminal / ligand / interaction features.
5. Write candidates (+ labels when valid targets exist).
6. Run `Cov_Screen`:
   - always: score-only ranking within each **Name × Warhead** query group
   - if labels exist: labeled evaluation (Hit@K / Hit@top%)

Bundled scripts live in [`dependencies/`](dependencies/README.md).

## Requirements

- Python packages: NumPy, pandas, RDKit, FreeSASA, joblib, scikit-learn,
  XGBoost, LightGBM, requests; SHAP optional
- `xtb` executable on `PATH`
- A trained Cov_Screen model bundle (`.pkl`)

## Input CSV

Headers are detected **case-insensitively**.

| Column | Required | Notes |
|---|---|---|
| `pdb` / `pdb_id` / `protein pdb` | yes | PDB ID or path stem |
| `smiles` **or** `LigID` | yes | SMILES preferred; LigID resolved from `--pdb-dir` or RCSB |
| `name` | no | Query-group name; auto-built as `PDB-SMILES-warhead` if omitted |
| `residue`, `resnum`, `chain` | no | Target site; all three required together if any provided |
| `warhead` / `Frankenstein_Warhead` | no | Optional; copied as-is to labels `Frankenstein_Warhead` for Cov_Screen matching / `--perfect-match` (comma-separated allowed). `Frankenstein_Warhead` is preferred when both columns exist. |

Supported labeled nucleophiles: **CYS, SER, THR, TYR, LYS, HIS**.

If a target is missing or unsupported (e.g. GLU), CovSite warns, treats that row
as unlabeled (score-only), and continues.

When `Warhead` / `Frankenstein_Warhead` is provided, CovSite copies that exact
text into `labels.csv` as `Frankenstein_Warhead` (no conversion). Candidate
feature rows still use auto-detected warhead names from SMILES. Cov_Screen
matches those candidate `Warhead` values against the label field
(comma-separated membership). Use `--perfect-match` to require every listed
warhead that matches a candidate to hit.

Multiple target sites for the same PDB + SMILES are supported. CovSite makes
`Name` unique per site (suffix like `-ACYS25`) so each Name × Warhead group still
has one evaluation target.

Fully skipped rows (invalid SMILES, no warhead unless `--no-warhead`, LigID/PDB
failures, descriptor failures) are written to `covsite_skipped_inputs.csv` with
a reason and detail.

## Full run

```bash
python CovSite.py input.csv \
  --model lgbm_ranker_nc.pkl \
  --pdb-dir Existing_Structures \
  --output-dir ./covsite_output \
  --top-pct 10 \
  --perfect-match
```

Useful options:

| Flag | Purpose |
|---|---|
| `--no-cache` | Recalculate xTB descriptors (ignore `reactivity_cache.json`) |
| `--ions` | Include crystallographic ions in FreeSASA / deprotonation SASA (separate `*_sasa_ions_v4.rsa` cache) |
| `--skip-sasa-deprot` | With `--VS` only: skip FreeSASA and deprotonation; confirm the target from ATOM records; `Abs_Side_SASA` / `Rel_Side_SASA` / `deprotonation_prob` are NaN. `--ions` has no effect. |
| `--no-warhead` | Skip warhead SMARTS detection and xTB; `Warhead=none`; Fukui/HOMO/LUMO/... are NaN |
| `--deprotonation-model PATH` | Override default `deprot_xgb_noapbs.pkl` |
| `--scripts-dir PATH` | Override `dependencies/` |
| `--topk`, `--reward-mode`, `--top-pct` | Cov_Screen hit metrics |
| `--perfect-match` | Pass through to Cov_Screen |
| `--no-shap` | Disable SHAP export |

Column overrides: `--pdb-column`, `--smiles-column`, `--ligid-column`,
`--name-column`, `--residue-column`, `--resnum-column`, `--chain-column`,
`--warhead-column`.

## Screen-only mode

Skip feature calculation when candidates (and optional labels) already exist:

```bash
# Score ranking + labeled evaluation
python CovSite.py --screen-only \
  --model lgbm_ranker_nc.pkl \
  --candidates covsite_candidates.csv \
  --labels covsite_labels.csv \
  --output-dir ./covsite_output_screen \
  --top-pct 10 \
  --perfect-match

# Score ranking only
python CovSite.py --screen-only \
  --model lgbm_ranker_nc.pkl \
  --candidates covsite_candidates.csv \
  --output-dir ./covsite_output_screen
```

Labels should use **`Frankenstein_Warhead`** (not bare `Warhead`) to match
candidates. Candidates use column `Warhead`.

`--pdb-dir` and `input_csv` are not required in screen-only mode.

## Virtual screening (`--VS`)

`--VS` featurizes only each row's target site (Residue / ResNum / Chain required) and skips Cov_Screen ranking. Combined with `--skip-sasa-deprot`, CovSite does not run FreeSASA or load `deprot_xgb_noapbs.pkl`; it only checks that the nucleophile exists in ATOM records. Residue geometry, ligand, and interaction features are still computed.

```bash
python CovSite.py library.csv \
  --VS --skip-sasa-deprot \
  --pdb-dir Existing_Structures \
  --output-dir ./covsite_vs_features
```

## Ligand screening ranker

`Training_CovSite_VS.py` trains a LightGBM LambdaRank model to rank **ligands at a fixed covalent site**. It calls CovSite `--VS` (passing `--skip-sasa-deprot` unless `--sasa-deprot` is set) or reuses `--candidates`. Cross-validation is **leave-one-site-out**: each fold holds out one `PDB_ID × Residue × ResNum × Chain`. Same PDB, two residues → two folds.

```bash
python Training_CovSite_VS.py library.csv hits.csv \
  --pdb-dir Existing_Structures \
  --features Geo_Fit Hydrophobic_Fit Hydrogen_DPAL_Fit \
  --output-dir ./covsite_vs_ranker \
  --best-warhead
```

Reuse an existing CovSite candidates table:

```bash
python Training_CovSite_VS.py library.csv hits.csv \
  --candidates covsite_candidates.csv \
  --features Geo_Fit Hydrophobic_Fit \
  --output-dir ./covsite_vs_ranker
```

**Library CSV** (one ligand at one site per row): `pdb` / `pdb_id`, `smiles`, `residue`, `resnum`, `chain` (optional `name`, `warhead`).

**Hits CSV** (one file): `pdb_id` plus `Electrophile_Hit` / `electrophile_hit` (RDKit canonical SMILES, matched with `pdb_id`). Optional `Name_Hit`, `warhead_type` (restricts `Warhead_Base`; comma-separated allowed). TOPSIS-style miss columns are also accepted.

`--features` are CovSite candidate headers. Columns that are constant inside every site group are dropped with a warning (site-level SASA / deprotonation never rank ligands). Univariate feature stats rank higher values as better. Sites with fewer than 2 candidates or 0 hits are skipped (EF undefined).

`--best-warhead` keeps the highest `pred_score` warhead per SMILES × site. `--multiple-hits` counts each Name × Warhead × SMILES as its own hit for EF / LogAUC.

Outputs (same metric set as TOPSIS_VS, using LGBM `pred_score`):

| File | Description |
|---|---|
| `lgbm_per_site_ranking.csv` | Held-out site rankings |
| `lgbm_pooled_pair_ranking.csv` | Concatenated OOF pairs ranked together |
| `full_pooled_ligand_ranking.csv` | Ligand-level list (best warhead per SMILES × site if `--best-warhead`) |
| `hit_per_site_ranks.csv` / `hit_pooled_pair_ranks.csv` | Hit subsets |
| `enrichment_factors.csv` | EF@1/5/10% + LogAUC / adjusted LogAUC |
| `feature_statistics.csv` / `feature_enrichment_factors.csv` | Univariate feature ranks |
| `matched_site_warhead_rows.csv` / `unmatched_hit_names.csv` | Audit |
| `models/fold_*.pkl` | Per-fold rankers (`--no-save-models` to skip) |

## Outputs

Under `--output-dir`:

| File | Description |
|---|---|
| `covsite_candidates_<runid>.csv` | Feature table for Cov_Screen |
| `covsite_labels_<runid>.csv` | Target sites (may be empty) |
| `covsite_candidates.csv` / `covsite_labels.csv` | Latest aliases |
| `covsite_scores.csv` | Score-only ranking |
| `covsite_skipped_inputs.csv` | Fully skipped input rows with reasons |
| `covsite_query_groups.csv` | Per-(Name × Warhead) score-only metrics |
| `covsite_score_summary.json` | Score-only summary |
| `covsite_label_results.csv` | Labeled evaluation results (if labels) |
| `covsite_labeled_scores.csv` | Labeled score export (if labels) |
| `covsite_evaluation_summary.json` | Evaluation summary (if labels) |
| `covsite_warnings.txt` | Skipped targets, missing warheads, etc. |
| `feature_cache/shards/` | Mid-run checkpoints (cleared after success) |

xTB caches are shared with standalone Frankenstein at the CustomScript root:

- `reactivity_cache.json`
- `nucleophile_cache.json`

## Query groups

Ranking is within each **protein–ligand × warhead** group (`Name × Warhead`),
not protein × warhead alone. A multi-warhead ligand yields one ranked list per
warhead.

## Notes

- Warhead detection uses the SMARTS list in `dependencies/Frankenstein.py`
  (~62 pattern types).
- Ligand 3D embedding uses seeded ETKDG with random-coords / UFF fallbacks in
  `dependencies/single_AA_bond.py`.
