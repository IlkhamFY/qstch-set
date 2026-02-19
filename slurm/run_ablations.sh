#!/bin/bash
#SBATCH --job-name=stch-ablation
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/ablation_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/ablation_%A_%a.err
#SBATCH --array=0-6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# Array job: 7 tasks
# 0-2: K ablation (K=3,5,10) on DTLZ2 m=5
# 3-6: mu ablation (mu=0.01,0.1,0.5,1.0) on DTLZ2 m=5

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"

IDX=$SLURM_ARRAY_TASK_ID
RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/ablations"
mkdir -p "$RESULTS_DIR"

cd "$STCH_ROOT"

if [ $IDX -lt 3 ]; then
    K_VALUES=(3 5 10)
    K=${K_VALUES[$IDX]}
    echo "[$(date)] K ablation: K=$K on DTLZ2 m=5"
    python benchmarks/dtlz_benchmark_v2.py \
        --problem DTLZ2 --m 5 --seeds 3 --iters 30 \
        --ablation k --k-value "$K" \
        --output-dir "$RESULTS_DIR" --device cuda
else
    MU_VALUES=(0.01 0.1 0.5 1.0)
    MU=${MU_VALUES[$((IDX - 3))]}
    echo "[$(date)] mu ablation: mu=$MU on DTLZ2 m=5"
    python benchmarks/dtlz_benchmark_v2.py \
        --problem DTLZ2 --m 5 --seeds 3 --iters 30 \
        --ablation mu --mu-value "$MU" \
        --output-dir "$RESULTS_DIR" --device cuda
fi

echo "[$(date)] DONE: ablation task $IDX"
