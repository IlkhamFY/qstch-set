#!/bin/bash
#SBATCH --job-name=stch-carside
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/results/real_world_carsideimpact/slurm_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/results/real_world_carsideimpact/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"
RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/real_world_carsideimpact"
mkdir -p "$RESULTS_DIR"
cd "$STCH_ROOT"
echo "[$(date)] Starting CarSideImpact benchmark on $(hostname)"
python benchmarks/real_world_benchmark.py --problem CarSideImpact --seeds 5 --output-dir "$RESULTS_DIR" --device cuda
echo "[$(date)] DONE"
