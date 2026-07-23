# CovSite dependencies

`CovSite.py` loads its bundled scripts and deprotonation model from this
directory. Use `--scripts-dir PATH` only when overriding this location.

Bundled pipeline files:

- `Frankenstein.py`
- `highlight_nucleophiles_adv_2.py`
- `single_AA_bond.py`
- `covalent_orbital_requirements.py`
- `Deprotonation_Model.py`
- `HBonds_Score.py`
- `deprot_xgb_noapbs.pkl`
- `add_residue_info.py`
- `add_ligand_info.py`
- `add_interaction_info.py`
- `Adding_N_terminal.py`
- `Cov_Screen.py`
- `Training_Cov_Screen.py`

External Python/runtime dependencies are not vendored here: NumPy, pandas,
RDKit, FreeSASA, joblib, scikit-learn, XGBoost, LightGBM, requests, SHAP
(optional), and the `xtb` executable.
