#!/bin/bash
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=stch_bench_full
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/logs/bench_full_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/logs/bench_full_%j.err
set -e
PROJ=/project/rrg-ravh011/ilkham/stch-botorch
cd $PROJ && mkdir -p logs
module load cuda/12.2
source $PROJ/venv/bin/activate
echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date) ==="
mkdir -p $PROJ/results/m8_K8_full $PROJ/results/m10_K10_full $PROJ/results/m8_budget_matched $PROJ/results/m10_budget_matched
python $PROJ/benchmarks/dtlz_benchmark_v2.py --problem DTLZ2 --n-obj 8 --batch-size 8 --n-iter 20 --n-seeds 5 --methods stch_set qnparego random qehvi --output-dir $PROJ/results/m8_K8_full > $PROJ/results/m8_K8_full/run.log 2>&1 &
PID_A=$!
python $PROJ/benchmarks/dtlz_benchmark_v2.py --problem DTLZ2 --n-obj 10 --batch-size 10 --n-iter 20 --n-seeds 5 --methods stch_set qnparego random qehvi --output-dir $PROJ/results/m10_K10_full > $PROJ/results/m10_K10_full/run.log 2>&1 &
PID_B=$!
python $PROJ/benchmarks/dtlz_benchmark_v2.py --problem DTLZ2 --n-obj 8 --batch-size 8 --n-iter 25 --n-seeds 5 --methods qnparego --output-dir $PROJ/results/m8_budget_matched > $PROJ/results/m8_budget_matched/run.log 2>&1 &
PID_C=$!
python $PROJ/benchmarks/dtlz_benchmark_v2.py --problem DTLZ2 --n-obj 10 --batch-size 10 --n-iter 20 --n-seeds 5 --methods qnparego --output-dir $PROJ/results/m10_budget_matched > $PROJ/results/m10_budget_matched/run.log 2>&1 &
PID_D=$!
wait $PID_A && echo "A_DONE" || echo "A_FAILED"
wait $PID_B && echo "B_DONE" || echo "B_FAILED"
wait $PID_C && echo "C_DONE" || echo "C_FAILED"
wait $PID_D && echo "D_DONE" || echo "D_FAILED"
echo "=== ALL DONE: $(date) ==="