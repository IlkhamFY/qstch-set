#!/bin/bash
# v8 Resubmit FAILED jobs only
# Fixes applied:
#   1. CUDA fork bug: n_workers=1 forces sequential execution (no Pool fork)
#   2. Ablation: --k-value/--mu-value removed; benchmark loops internally
#   3. Budget-matched: --batch-size -> --q; n_workers 1
#   4. Ablation redesigned: 2 tasks (K-ablation, mu-ablation) not 7

set -e

PROJECT=/project/rrg-ravh011/ilkham/stch-botorch
export PYTHONPATH="$PROJECT/src:$PYTHONPATH"
RESULTS_BASE="$PROJECT/results"

echo "=== v8 Resubmit Failed Jobs ==="
echo ""

# ─── 1. DTLZ2 seeds 4-14 (array tasks 4-14) ──────────────────────
echo "=== Submitting DTLZ2 failed seeds (4-14) ==="

JOB_DTLZ2=$(sbatch --parsable <<'DTLZ2_EOF'
#!/bin/bash
#SBATCH --job-name=v8-dtlz2-retry
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_dtlz2_retry_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_dtlz2_retry_%A_%a.err
#SBATCH --array=4-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# Array index -> m, seed mapping (same as v8_resubmit_all.sh)
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

RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/dtlz2_v8"
mkdir -p "$RESULTS_DIR"

echo "[$(date)] v8 DTLZ2 retry: m=$M seed=$SEED iters=$ITERS (array idx=$IDX)"
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
    --output-dir "$RESULTS_DIR" \
    --device cuda \
    --n-workers 1 \
    $EXTRA_ARGS

echo "[$(date)] DONE: DTLZ2 m=$M seed=$SEED"
DTLZ2_EOF
)
echo "  DTLZ2 retry: $JOB_DTLZ2 (11 tasks: seeds 4-14)"

# ─── 2. ABLATIONS (fixed: 2 tasks, benchmark loops internally) ────
echo ""
echo "=== Submitting ablation studies (fixed) ==="

JOB_ABL=$(sbatch --parsable <<'ABL_EOF'
#!/bin/bash
#SBATCH --job-name=v8-ablation-retry
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_ablation_retry_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_ablation_retry_%A_%a.err
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# task 0: K ablation (K=3,5,10 looped internally)
# task 1: mu ablation (mu=0.01,0.1,0.5,1.0 looped internally)

set -e
source /project/rrg-ravh011/ilkham/stch-botorch/slurm/setup_venv.sh

IDX=$SLURM_ARRAY_TASK_ID
RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/ablations_v8"
mkdir -p "$RESULTS_DIR"
cd "$STCH_ROOT"

if [ $IDX -eq 0 ]; then
    echo "[$(date)] v8 K ablation: K=3,5,10 on DTLZ2 m=5"
    python benchmarks/dtlz_benchmark_v2.py \
        --problem DTLZ2 --m 5 --seeds 3 --iters 30 \
        --ablation k \
        --output-dir "$RESULTS_DIR" --device cuda --n-workers 1
else
    echo "[$(date)] v8 mu ablation: mu=0.01,0.1,0.5,1.0 on DTLZ2 m=5"
    python benchmarks/dtlz_benchmark_v2.py \
        --problem DTLZ2 --m 5 --seeds 3 --iters 30 \
        --ablation mu \
        --output-dir "$RESULTS_DIR" --device cuda --n-workers 1
fi

echo "[$(date)] DONE: ablation task $IDX"
ABL_EOF
)
echo "  Ablations: $JOB_ABL (2 tasks)"

# ─── 3. BUDGET-MATCHED (fixed: --q not --batch-size, n-workers 1) ─
echo ""
echo "=== Submitting budget-matched baselines (fixed) ==="

JOB_BM=$(sbatch --parsable <<'BM_EOF'
#!/bin/bash
#SBATCH --job-name=v8-budget-retry
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_budget_retry_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_budget_retry_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# 0: m=5  q=5  30 iters
# 1: m=8  q=8  25 iters
# 2: m=10 q=10 20 iters

set -e
source /project/rrg-ravh011/ilkham/stch-botorch/slurm/setup_venv.sh

IDX=$SLURM_ARRAY_TASK_ID
cd "$STCH_ROOT"

if [ $IDX -eq 0 ]; then
    M=5; ITERS=30
    RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/budget_matched_m5_v8"
elif [ $IDX -eq 1 ]; then
    M=8; ITERS=25
    RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/budget_matched_m8_v8"
else
    M=10; ITERS=20
    RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/budget_matched_m10_v8"
fi

mkdir -p "$RESULTS_DIR"

echo "[$(date)] v8 Budget-matched qNParEGO: m=$M q=$M iters=$ITERS"
python benchmarks/dtlz_benchmark_v2.py \
    --problem DTLZ2 --m "$M" --seeds 5 --iters "$ITERS" \
    --methods qnparego --q "$M" \
    --output-dir "$RESULTS_DIR" --device cuda \
    --skip-qehvi --n-workers 1

echo "[$(date)] DONE: budget-matched m=$M"
BM_EOF
)
echo "  Budget-matched: $JOB_BM (3 tasks: m=5/8/10)"

echo ""
echo "============================================================"
echo "v8 FAILED JOB RESUBMISSION COMPLETE"
echo "============================================================"
echo ""
echo "Jobs submitted:"
echo "  DTLZ2 retry (tasks 4-14): $JOB_DTLZ2"
echo "  Ablations (2 tasks):       $JOB_ABL"
echo "  Budget-matched (3 tasks):  $JOB_BM"
echo ""
echo "Results directories:"
echo "  $RESULTS_BASE/dtlz2_v8/"
echo "  $RESULTS_BASE/ablations_v8/"
echo "  $RESULTS_BASE/budget_matched_m{5,8,10}_v8/"
