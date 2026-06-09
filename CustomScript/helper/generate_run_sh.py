from pathlib import Path


template = """#!/bin/bash
#SBATCH --job-name=frankenstein_bo_{i}
#SBATCH --output=frankenstein_bo_{i}_%j.out
#SBATCH --error=frankenstein_bo_{i}_%j.err
#SBATCH --time=12:00:00
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

python Frankenstein.py \\
    --test bo-runs/bo_leftover_input_{i}.csv \\
    --pdb-download-dir Existing_Structures \\
    --out-dir bo-runs/run{run_index} \\
    --pdb-dir Existing_Structures --reuse-existing
"""


base = Path(__file__).parent
for i in range(1, 6):
    run_index = i + 30
    script_path = base / f"run_covdb_{run_index}.sh"
    with script_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(template.format(i=i, run_index=run_index))

print("Generated", ", ".join(f"run_covdb_{i}.sh" for i in range(1, 31)))
