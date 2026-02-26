#!/bin/bash
# DMTA Molecular Benchmark — full run across m=4,6,8 and K=5
# Array: 0=m4, 1=m6, 2=m8 (both weight modes)
# 6 tasks total (3 m-values × 2 weight modes)

#SBATCH --job-name=dmta-bench
#SBATCH --account=def-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/dmta_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/dmta_%A_%a.err
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
source /project/rrg-ravh011/ilkham/stch-botorch/slurm/setup_venv.sh

PROJECT=/project/rrg-ravh011/ilkham/stch-botorch
IDX=$SLURM_ARRAY_TASK_ID

# Task mapping: [m_values × weight_modes]
M_VALS=(4 6 8 4 6 8)
MODES=(original original original log_additive log_additive log_additive)

M=${M_VALS[$IDX]}
MODE=${MODES[$IDX]}

RESULTS_DIR="$PROJECT/results/dmta_m${M}_K5_${MODE}"
mkdir -p "$RESULTS_DIR"

echo "[$(date)] DMTA Benchmark: m=$M, K=5, weight_mode=$MODE"
echo "[$(date)] Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$STCH_ROOT"

python benchmarks/dmta_benchmark.py \
    --m "$M" \
    --seeds 5 \
    --n-init 20 \
    --n-iters 25 \
    --batch-size 5 \
    --mc-samples 128 \
    --weight-mode "$MODE" \
    --mu 0.1 \
    --output-dir "$RESULTS_DIR" \
    --device cuda

echo "[$(date)] DONE: DMTA m=$M K=5 mode=$MODE"
