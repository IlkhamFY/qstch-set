#!/bin/bash
#SBATCH --job-name=stch-dtlz2
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/dtlz2_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/dtlz2_%A_%a.err
#SBATCH --array=0-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# Array job: 15 tasks = 3 objectives × 5 seeds
# Index mapping:
#   0-4:   m=5,  seeds 0-4 (30 iters)
#   5-9:   m=8,  seeds 0-4 (25 iters)
#   10-14: m=10, seeds 0-4 (20 iters)

set -e

# Setup venv on local NVMe
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"

# Parse array index
IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -lt 5 ]; then
    M=5; SEED=$IDX; ITERS=30
elif [ $IDX -lt 10 ]; then
    M=8; SEED=$((IDX - 5)); ITERS=25
else
    M=10; SEED=$((IDX - 10)); ITERS=20
fi

echo "================================================================"
echo "[$(date)] DTLZ2 m=$M seed=$SEED iters=$ITERS (array idx=$IDX)"
echo "[$(date)] Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================================================"

RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/dtlz2_m${M}"
mkdir -p "$RESULTS_DIR"

cd "$STCH_ROOT"
# For m>6, skip qEHVI (exponential cost)
EXTRA_ARGS=""
if [ $M -gt 6 ]; then
    EXTRA_ARGS="--skip-qehvi"
fi

python benchmarks/dtlz_benchmark_v2.py \
    --problem DTLZ2 \
    --m "$M" \
    --seeds 1 \
    --seed-offset "$SEED" \
    --iters "$ITERS" \
    --output-dir "$RESULTS_DIR" \
    --device cuda \
    $EXTRA_ARGS

echo "[$(date)] DONE: m=$M seed=$SEED"
