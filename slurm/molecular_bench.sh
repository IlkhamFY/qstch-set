#!/bin/bash
#SBATCH --job-name=mol-redox
#SBATCH --account=def-ravh011
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/logs/mol_redox_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/logs/mol_redox_%j.err

# No GPU needed — pool-based GP on Morgan fingerprints is CPU-friendly

set -euo pipefail

PROJECT=/project/rrg-ravh011/ilkham/stch-botorch
RESULTS=$PROJECT/results/molecular_multi_redox

mkdir -p $RESULTS
mkdir -p $PROJECT/logs

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

# Build fresh venv in local NVMe
module load python/3.12 rdkit/2024.09.6
python -m venv --system-site-packages $SLURM_TMPDIR/mol-venv
source $SLURM_TMPDIR/mol-venv/bin/activate

# Install dependencies (rdkit is available via module on Nibi)
pip install --no-index torch botorch gpytorch 2>/dev/null || true

# Install package in editable mode
cd $PROJECT
pip install --no-index -e . 2>/dev/null || pip install -e . 2>/dev/null || true

# Also ensure rdkit available
python -c "from rdkit import Chem; print('rdkit ok')" 2>/dev/null || \
  pip install --no-index rdkit 2>/dev/null || \
  pip install rdkit 2>/dev/null || \
  echo "WARNING: rdkit not available, falling back to random features"

echo "Environment ready. Starting benchmark..."

python benchmarks/molecular_benchmark.py \
  --dataset multi_redox \
  --seeds 5 \
  --n-init 20 \
  --n-iters 30 \
  --batch-size 4 \
  --mc-samples 64 \
  --output-dir $RESULTS \
  --device cpu

echo "Done at $(date). Results in $RESULTS/"
cat $RESULTS/summary.json | python -c "
import json, sys
d = json.load(sys.stdin)
print(f\"Oracle HV: {d['oracle_hv']:.4f}\")
for m in ['random', 'qnparego', 'qstch_set']:
    r = d[m]
    pct = r['final_hv_mean'] / d['oracle_hv'] * 100
    print(f\"  {m:15s}: {r['final_hv_mean']:.4f} +/- {r['final_hv_std']:.4f} ({pct:.1f}%)\")
"
