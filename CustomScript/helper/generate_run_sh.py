from pathlib import Path


template = """#!/bin/bash
#SBATCH --job-name=frankenstein_covsite_{i}
#SBATCH --output=frankenstein_covsite_{i}_%j.out
#SBATCH --error=frankenstein_covsite_{i}_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=PAS2959

set -euo pipefail

module load miniconda3/24.1.2-py310

source activate covalent_311

echo "Starting job on $(date)"
echo "Running on node $(hostname)"

export PATH="/users/PCON0009/andrewjohnhu/bin:$PATH"

echo "PATH = $PATH"
which python
which xtb
conda info --envs

python CovSite.py VS/VS_BTK_{i}.csv \
  --model lgbm_ranker_nc.pkl \
  --pdb-dir Existing_Structures \
  --output-dir VS/BTK/run{i} \
  --top-pct 10 \
  --perfect-match

"""


base = Path(__file__).parent
for i in range(1, 10):
    run_index = i
    script_path = base / f"run_BTK_{run_index}.sh"
    with script_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(template.format(i=i, run_index=run_index))

print("Generated", ", ".join(f"run_covdb_covsite_{i}.sh" for i in range(1, 4)))
