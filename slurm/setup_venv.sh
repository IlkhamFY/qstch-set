#!/bin/bash
# Build fresh venv on local NVMe ($SLURM_TMPDIR) and activate
# Source this in job scripts: source slurm/setup_venv.sh
# NOTE: Old tarball approach was broken (hardcoded paths). Build fresh each job (~30s).

set -e

VENV_DIR="$SLURM_TMPDIR/stch-venv"

echo "[$(date)] Loading modules..."
module purge 2>/dev/null || true
module load StdEnv/2023 python/3.12 scipy-stack cuda/12.6

if [ ! -d "$VENV_DIR" ]; then
    echo "[$(date)] Building fresh venv on local NVMe ($SLURM_TMPDIR)..."
    python -m venv --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-index --upgrade pip 2>/dev/null || true
    pip install --no-index torch botorch gpytorch
    echo "[$(date)] Venv built."
else
    echo "[$(date)] Venv already exists at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

echo "[$(date)] Activated venv: $(python --version), torch=$(python -c 'import torch; print(torch.__version__)')"
echo "[$(date)] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set project root
export STCH_ROOT="/project/rrg-ravh011/ilkham/stch-botorch"
export PYTHONPATH="$STCH_ROOT/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1
