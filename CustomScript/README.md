# CovSite

`CovSite.py` ranks nucleophilic covalent-binding sites for protein–ligand pairs
plus a saved LightGBM ranker
(`Cov_Screen`).


## Pipeline

1. Read input CSV (PDB + SMILES and/or LigID; optional target site).
2. Detect warheads and compute deprotonated xTB descriptors per residue type.
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

Supported labeled nucleophiles: **CYS, SER, THR, TYR, LYS, HIS**.

If a target is missing or unsupported (e.g. GLU), CovSite warns, treats that row
as unlabeled (score-only), and continues.

Multiple target sites for the same PDB + SMILES are supported. CovSite makes
`Name` unique per site (suffix like `-ACYS25`) so each Name × Warhead group still
has one evaluation target.

Fully skipped rows (invalid SMILES, no warhead, LigID/PDB failures, descriptor
failures) are written to `covsite_skipped_inputs.csv` with a reason and detail.

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
| `--deprotonation-model PATH` | Override default `deprot_xgb_noapbs.pkl` |
| `--scripts-dir PATH` | Override `dependencies/` |
| `--topk`, `--reward-mode`, `--top-pct` | Cov_Screen hit metrics |
| `--perfect-match` | Pass through to Cov_Screen |
| `--no-shap` | Disable SHAP export |

Column overrides: `--pdb-column`, `--smiles-column`, `--ligid-column`,
`--name-column`, `--residue-column`, `--resnum-column`, `--chain-column`.

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
