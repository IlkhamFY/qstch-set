#!/bin/bash
# =============================================================================
# v8 FULL BENCHMARK RESUBMISSION
# =============================================================================
# Context: The scalarization had a critical temperature bug where weights
# multiplied inside logsumexp reduced effective mu by factor m.
# Fixed in commit 4b4050a (log-additive weights).
# ALL prior results are invalidated. This reruns everything from scratch.
#
# Usage: bash slurm/v8_resubmit_all.sh  (from project root on Nibi login node)
# =============================================================================

set -euo pipefail

PROJ="/project/rrg-ravh011/ilkham/stch-botorch"
SLURM_DIR="$PROJ/slurm"
LOG_DIR="$SLURM_DIR/logs"
RESULTS_BASE="$PROJ/results"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "v8 FULL BENCHMARK RESUBMISSION"
echo "Date: $(date)"
echo "Project: $PROJ"
echo "Git HEAD: $(cd $PROJ && git log --oneline -1)"
echo "============================================================"
echo ""

# Verify the fix is present
if ! grep -q "torch.log(weights)" "$PROJ/src/stch_botorch/scalarization.py"; then
    echo "ERROR: scalarization.py does not contain the log-additive fix!"
    echo "Pull the latest code first: cd $PROJ && git pull"
    exit 1
fi
echo "[OK] Log-additive weight fix confirmed in scalarization.py"
echo ""

# ─── 1. DTLZ2 BENCHMARKS (m=5, m=8, m=10) ────────────────────────

echo "=== Submitting DTLZ2 benchmarks ==="

JOB_DTLZ2=$(sbatch --parsable <<'DTLZ2_EOF'
#!/bin/bash
#SBATCH --job-name=v8-dtlz2
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_dtlz2_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_dtlz2_%A_%a.err
#SBATCH --array=0-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# 15 tasks = 3 obj counts × 5 seeds
# 0-4: m=5, 5-9: m=8, 10-14: m=10

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"

IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -lt 5 ]; then
    M=5; SEED=$IDX; ITERS=30
elif [ $IDX -lt 10 ]; then
    M=8; SEED=$((IDX - 5)); ITERS=25
else
    M=10; SEED=$((IDX - 10)); ITERS=20
fi

RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/dtlz2_m${M}_v8"
mkdir -p "$RESULTS_DIR"

echo "[$(date)] v8 DTLZ2 m=$M seed=$SEED iters=$ITERS (array idx=$IDX)"
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
    $EXTRA_ARGS

echo "[$(date)] DONE: DTLZ2 m=$M seed=$SEED"
DTLZ2_EOF
)
echo "  DTLZ2 array job: $JOB_DTLZ2 (15 tasks: m=5/8/10 × 5 seeds)"

# ─── 2. REAL-WORLD BENCHMARKS (VehicleSafety, Penicillin, CarSideImpact) ───

echo ""
echo "=== Submitting real-world benchmarks ==="

JOB_VS=$(sbatch --parsable <<'VS_EOF'
#!/bin/bash
#SBATCH --job-name=v8-vehiclesafety
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_vehiclesafety_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_vehiclesafety_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"
RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/real_world_vehiclesafety_v8"
mkdir -p "$RESULTS_DIR"
cd "$STCH_ROOT"
echo "[$(date)] v8 VehicleSafety on $(hostname)"
python benchmarks/real_world_benchmark.py --problem VehicleSafety --seeds 5 --output-dir "$RESULTS_DIR" --device cuda
echo "[$(date)] DONE"
VS_EOF
)
echo "  VehicleSafety: $JOB_VS"

JOB_PEN=$(sbatch --parsable <<'PEN_EOF'
#!/bin/bash
#SBATCH --job-name=v8-penicillin
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_penicillin_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_penicillin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"
RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/real_world_penicillin_v8"
mkdir -p "$RESULTS_DIR"
cd "$STCH_ROOT"
echo "[$(date)] v8 Penicillin on $(hostname)"
python benchmarks/real_world_benchmark.py --problem Penicillin --seeds 5 --output-dir "$RESULTS_DIR" --device cuda
echo "[$(date)] DONE"
PEN_EOF
)
echo "  Penicillin: $JOB_PEN"

JOB_CS=$(sbatch --parsable <<'CS_EOF'
#!/bin/bash
#SBATCH --job-name=v8-carsideimpact
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_carsideimpact_%j.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_carsideimpact_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"
RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/real_world_carsideimpact_v8"
mkdir -p "$RESULTS_DIR"
cd "$STCH_ROOT"
echo "[$(date)] v8 CarSideImpact on $(hostname)"
python benchmarks/real_world_benchmark.py --problem CarSideImpact --seeds 5 --output-dir "$RESULTS_DIR" --device cuda
echo "[$(date)] DONE"
CS_EOF
)
echo "  CarSideImpact: $JOB_CS"

# ─── 3. ABLATIONS (K and mu) ──────────────────────────────────────

echo ""
echo "=== Submitting ablation studies ==="

JOB_ABL=$(sbatch --parsable <<'ABL_EOF'
#!/bin/bash
#SBATCH --job-name=v8-ablation
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_ablation_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_ablation_%A_%a.err
#SBATCH --array=0-6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# 0-2: K ablation (K=3,5,10) on DTLZ2 m=5
# 3-6: mu ablation (mu=0.01,0.1,0.5,1.0) on DTLZ2 m=5

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"

IDX=$SLURM_ARRAY_TASK_ID
RESULTS_DIR="/project/rrg-ravh011/ilkham/stch-botorch/results/ablations_v8"
mkdir -p "$RESULTS_DIR"
cd "$STCH_ROOT"

if [ $IDX -lt 3 ]; then
    K_VALUES=(3 5 10)
    K=${K_VALUES[$IDX]}
    echo "[$(date)] v8 K ablation: K=$K on DTLZ2 m=5"
    python benchmarks/dtlz_benchmark_v2.py \
        --problem DTLZ2 --m 5 --seeds 3 --iters 30 \
        --ablation k --k-value "$K" \
        --output-dir "$RESULTS_DIR" --device cuda
else
    MU_VALUES=(0.01 0.1 0.5 1.0)
    MU=${MU_VALUES[$((IDX - 3))]}
    echo "[$(date)] v8 mu ablation: mu=$MU on DTLZ2 m=5"
    python benchmarks/dtlz_benchmark_v2.py \
        --problem DTLZ2 --m 5 --seeds 3 --iters 30 \
        --ablation mu --mu-value "$MU" \
        --output-dir "$RESULTS_DIR" --device cuda
fi

echo "[$(date)] DONE: ablation task $IDX"
ABL_EOF
)
echo "  Ablations: $JOB_ABL (7 tasks: K=3/5/10 + mu=0.01/0.1/0.5/1.0)"

# ─── 4. BUDGET-MATCHED BASELINES ──────────────────────────────────

echo ""
echo "=== Submitting budget-matched baselines ==="

JOB_BM=$(sbatch --parsable <<'BM_EOF'
#!/bin/bash
#SBATCH --job-name=v8-budget-matched
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_budget_%A_%a.out
#SBATCH --error=/project/rrg-ravh011/ilkham/stch-botorch/slurm/logs/v8_budget_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

# Budget-matched qNParEGO: q=m candidates per iteration
# 0: m=5  (q=5, 30 iters → 150 function evals)
# 1: m=8  (q=8, 25 iters → 200 function evals)
# 2: m=10 (q=10, 20 iters → 200 function evals)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/setup_venv.sh"

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
    --methods qnparego --batch-size "$M" \
    --output-dir "$RESULTS_DIR" --device cuda \
    --skip-qehvi

echo "[$(date)] DONE: budget-matched m=$M"
BM_EOF
)
echo "  Budget-matched: $JOB_BM (3 tasks: m=5/8/10)"

# ─── SUMMARY ──────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "v8 SUBMISSION COMPLETE"
echo "============================================================"
echo ""
echo "Jobs submitted:"
echo "  DTLZ2 (15 tasks):       $JOB_DTLZ2"
echo "  VehicleSafety:           $JOB_VS"
echo "  Penicillin:              $JOB_PEN"
echo "  CarSideImpact:           $JOB_CS"
echo "  Ablations (7 tasks):     $JOB_ABL"
echo "  Budget-matched (3 tasks): $JOB_BM"
echo ""
echo "Total: 28 GPU tasks"
echo ""
echo "Results directories (all _v8):"
echo "  $RESULTS_BASE/dtlz2_m{5,8,10}_v8/"
echo "  $RESULTS_BASE/real_world_{vehiclesafety,penicillin,carsideimpact}_v8/"
echo "  $RESULTS_BASE/ablations_v8/"
echo "  $RESULTS_BASE/budget_matched_m{5,8,10}_v8/"
echo ""
echo "Monitor: squeue -u ilkham"
echo "Check:   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS"
