#!/bin/bash
#SBATCH --job-name=mol-tanimoto
#SBATCH --account=def-ravh011
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/logs/mol_tanimoto_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/logs/mol_tanimoto_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# Morgan FP + Tanimoto kernel -- same pool, same dataset as RBF run.
# The only difference is the GP covariance function.

set -euo pipefail

PROJECT=/project/rrg-ravh011/ilkham/stch-botorch
RESULTS=$PROJECT/results/molecular_multi_redox_tanimoto

mkdir -p $RESULTS
mkdir -p $PROJECT/logs

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

module load python/3.12 rdkit/2024.09.6
python -m venv --system-site-packages $SLURM_TMPDIR/mol-venv
source $SLURM_TMPDIR/mol-venv/bin/activate

pip install --no-index torch botorch gpytorch 2>/dev/null || true

cd $PROJECT
pip install --no-index -e . 2>/dev/null || pip install -e . 2>/dev/null || true

python -c "from rdkit import Chem; print('rdkit ok')" 2>/dev/null || \
  echo "WARNING: rdkit not available"

echo "Environment ready. Starting Morgan+Tanimoto benchmark..."

python benchmarks/molecular_benchmark.py \
  --dataset multi_redox \
  --seeds 5 \
  --n-init 20 \
  --n-iters 30 \
  --batch-size 4 \
  --mc-samples 64 \
  --kernel tanimoto \
  --repr morgan \
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
