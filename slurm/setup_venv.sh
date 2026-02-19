#!/bin/bash
# Extract venv tarball to local NVMe ($SLURM_TMPDIR) and activate
# Source this in job scripts: source slurm/setup_venv.sh

set -e

VENV_TARBALL="/project/rrg-ravh011/ilkham/venvs/stch-botorch.tar.gz"
VENV_DIR="$SLURM_TMPDIR/stch-venv"

echo "[$(date)] Loading modules..."
module purge 2>/dev/null || true
module load StdEnv/2023 python/3.12 scipy-stack cuda/12.6

if [ ! -d "$VENV_DIR" ]; then
    echo "[$(date)] Extracting venv to local NVMe ($SLURM_TMPDIR)..."
    tar xzf "$VENV_TARBALL" -C "$SLURM_TMPDIR"
    echo "[$(date)] Venv extracted."
else
    echo "[$(date)] Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "[$(date)] Activated venv: $(python --version), torch=$(python -c 'import torch; print(torch.__version__)')"
echo "[$(date)] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set project root
export STCH_ROOT="/project/rrg-ravh011/ilkham/stch-botorch"
export PYTHONPATH="$STCH_ROOT/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1
