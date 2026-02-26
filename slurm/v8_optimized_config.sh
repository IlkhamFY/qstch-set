#!/bin/bash
# v8 Optimized config: mu=0.01, K=10 (best from ablation)
# Runs DTLZ2 (m=5/8/10) + VehicleSafety + Penicillin + CarSideImpact
# with stch_set only (compare against v8 default results for baselines)

set -e

PROJECT=/project/rrg-ravh011/ilkham/stch-botorch
RESULTS_BASE="$PROJECT/results"

echo "=== v8 Optimized Config (mu=0.01, K=10) ==="
echo ""

# ─── 1. DTLZ2 (m=5/8/10 x 5 seeds, stch_set only, mu=0.01 K=10) ──
echo "=== Submitting DTLZ2 optimized ==="

JOB_DTLZ2=$(sbatch --parsable <<'DTLZ2_EOF'
#!/bin/bash
#SBATCH --job-name=v8-dtlz2-opt
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_dtlz2_opt_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_dtlz2_opt_%A_%a.err
#SBATCH --array=0-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# 0-4:   m=5,  seed=IDX
# 5-9:   m=8,  seed=IDX-5
# 10-14: m=10, seed=IDX-10

set -e
source /project/rrg-ravh011/ilkham/stch-botorch/slurm/setup_venv.sh

IDX=$SLURM_ARRAY_TASK_ID

if [ $IDX -lt 5 ]; then
    M=5; SEED=$IDX; ITERS=30
elif [ $IDX -lt 10 ]; then
    M=8; SEED=$((IDX - 5)); ITERS=25
else
    M=10; SEED=$((IDX - 10)); ITERS=20
fi

RESULTS_DIR="$PROJECT/results/dtlz2_opt_v8"
mkdir -p "$RESULTS_DIR"

echo "[$(date)] v8-opt DTLZ2 m=$M seed=$SEED iters=$ITERS"
echo "[$(date)] Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$STCH_ROOT"

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
    --methods stch_set \
    --K 10 --mu 0.01 \
    --output-dir "$RESULTS_DIR" \
    --device cuda \
    --n-workers 1 \
    $EXTRA_ARGS

echo "[$(date)] DONE: DTLZ2 m=$M seed=$SEED (opt)"
DTLZ2_EOF
)
echo "  DTLZ2 opt: $JOB_DTLZ2 (15 tasks)"

# ─── 2. Real-world (VehicleSafety + Penicillin + CarSideImpact) ───
echo ""
echo "=== Submitting real-world optimized ==="

JOB_VS=$(sbatch --parsable <<'VS_EOF'
#!/bin/bash
#SBATCH --job-name=v8-vs-opt
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_vs_opt_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_vs_opt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
source /project/rrg-ravh011/ilkham/stch-botorch/slurm/setup_venv.sh
mkdir -p "$PROJECT/results/real_world_vehiclesafety_opt_v8"
cd "$STCH_ROOT"

python benchmarks/real_world_benchmark.py \
    --problem VehicleSafety --seeds 5 \
    --K 10 --mu 0.01 \
    --output-dir "$PROJECT/results/real_world_vehiclesafety_opt_v8" \
    --device cuda

echo "[$(date)] DONE: VehicleSafety opt"
VS_EOF
)
echo "  VehicleSafety opt: $JOB_VS"

JOB_PEN=$(sbatch --parsable <<'PEN_EOF'
#!/bin/bash
#SBATCH --job-name=v8-pen-opt
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_pen_opt_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_pen_opt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
source /project/rrg-ravh011/ilkham/stch-botorch/slurm/setup_venv.sh
mkdir -p "$PROJECT/results/real_world_penicillin_opt_v8"
cd "$STCH_ROOT"

python benchmarks/real_world_benchmark.py \
    --problem Penicillin --seeds 5 \
    --K 10 --mu 0.01 \
    --output-dir "$PROJECT/results/real_world_penicillin_opt_v8" \
    --device cuda

echo "[$(date)] DONE: Penicillin opt"
PEN_EOF
)
echo "  Penicillin opt: $JOB_PEN"

JOB_CAR=$(sbatch --parsable <<'CAR_EOF'
#!/bin/bash
#SBATCH --job-name=v8-car-opt
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_car_opt_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_car_opt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
source /project/rrg-ravh011/ilkham/stch-botorch/slurm/setup_venv.sh
mkdir -p "$PROJECT/results/real_world_carsideimpact_opt_v8"
cd "$STCH_ROOT"

python benchmarks/real_world_benchmark.py \
    --problem CarSideImpact --seeds 5 \
    --K 10 --mu 0.01 \
    --output-dir "$PROJECT/results/real_world_carsideimpact_opt_v8" \
    --device cuda

echo "[$(date)] DONE: CarSideImpact opt"
CAR_EOF
)
echo "  CarSideImpact opt: $JOB_CAR"

echo ""
echo "============================================================"
echo "v8 OPTIMIZED CONFIG SUBMISSION COMPLETE"
echo "============================================================"
echo "  DTLZ2 opt (15 tasks): $JOB_DTLZ2"
echo "  VehicleSafety opt:    $JOB_VS"
echo "  Penicillin opt:       $JOB_PEN"
echo "  CarSideImpact opt:    $JOB_CAR"
echo ""
echo "Results dirs:"
echo "  $RESULTS_BASE/dtlz2_opt_v8/"
echo "  $RESULTS_BASE/real_world_vehiclesafety_opt_v8/"
echo "  $RESULTS_BASE/real_world_penicillin_opt_v8/"
echo "  $RESULTS_BASE/real_world_carsideimpact_opt_v8/"
