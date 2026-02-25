#!/bin/bash
#SBATCH --job-name=mol-selfies
#SBATCH --account=def-ravh011
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/logs/mol_selfies_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/logs/mol_selfies_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# SELFIES-TED embeddings + RBF kernel.
# Requires precomputed embeddings at data/redox_mer_selfies_ted.pt
# (SCP'd from QuantumNode after local precomputation).

set -euo pipefail

PROJECT=/project/rrg-ravh011/ilkham/stch-botorch
RESULTS=$PROJECT/results/molecular_multi_redox_selfies_ted

mkdir -p $RESULTS
mkdir -p $PROJECT/logs

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

# Check embeddings exist
if [ ! -f "$PROJECT/data/redox_mer_selfies_ted.pt" ]; then
    echo "ERROR: SELFIES-TED embeddings not found at $PROJECT/data/redox_mer_selfies_ted.pt"
    echo "Run scripts/precompute_selfies_ted.py locally and SCP to Nibi first."
    exit 1
fi
echo "Embeddings found: $(du -h $PROJECT/data/redox_mer_selfies_ted.pt)"

module load python/3.12 rdkit/2024.09.6
python -m venv --system-site-packages $SLURM_TMPDIR/mol-venv
source $SLURM_TMPDIR/mol-venv/bin/activate

pip install --no-index torch botorch gpytorch 2>/dev/null || true

cd $PROJECT
# Set PYTHONPATH explicitly — pip install -e . is unreliable on Nibi NFS
export PYTHONPATH="$PROJECT/src:$PYTHONPATH"
pip install --no-index -e . 2>/dev/null || pip install -e . 2>/dev/null || true
python -c "import stch_botorch; print('stch_botorch ok, version:', getattr(stch_botorch, '__version__', 'dev'))"

echo "Environment ready. Starting SELFIES-TED+RBF benchmark..."

python benchmarks/molecular_benchmark.py \
  --dataset multi_redox \
  --seeds 5 \
  --n-init 20 \
  --n-iters 30 \
  --batch-size 4 \
  --mc-samples 64 \
  --kernel rbf \
  --repr selfies_ted \
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
