from pathlib import Path

template = """#!/bin/bash
#SBATCH --job-name=frankenstein_test
#SBATCH --output=frankenstein_test_%j.out
#SBATCH --error=frankenstein_test_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=PAS2959

set -euo pipefail

module load miniconda3/24.1.2-py310

source activate covalent_311

echo "Starting job on $(date)"
echo "Running on node $(hostname)"

python Frankenstein.py \\
    --test batch_pdbs_deprot_{i}.csv \\
    --pdb-download-dir Existing_Structures \\
    --out-dir Batch{i}
"""

base = Path(__file__).parent
for i in range(2, 24):
    p = base / f"run_{i}.sh"
    with p.open('w', encoding='utf-8', newline='\n') as f:
        f.write(template.format(i=i))
print('Generated', ', '.join(f'run_{i}.sh' for i in range(2,24)))
