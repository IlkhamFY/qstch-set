#!/bin/bash
#SBATCH --job-name=rebuild-venv
#SBATCH --account=rrg-ravh011_gpu
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/project/rrg-ravh011/ilkham/venvs/rebuild_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zolotoymuravey@gmail.com

set -e

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.12 scipy-stack cuda/12.6

echo "[$(date)] SLURM_TMPDIR=$SLURM_TMPDIR"
echo "[$(date)] Building venv on local NVMe..."

VENV_DIR="$SLURM_TMPDIR/stch-venv"
FINAL_VENV="/project/rrg-ravh011/ilkham/venvs/stch-venv"

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[$(date)] Installing PyTorch + BoTorch from wheelhouse..."
pip install --no-index torch torchvision
pip install --no-index botorch gpytorch linear_operator

echo "[$(date)] Verifying on GPU..."
python -c "
import torch, botorch
print('torch:', torch.__version__)
print('botorch:', botorch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# Instead of cp (slow), patch paths in-place and tar to /project
echo "[$(date)] Patching activate scripts for final path $FINAL_VENV..."
OLD_PATH="$VENV_DIR"
NEW_PATH="$FINAL_VENV"
sed -i "s|$OLD_PATH|$NEW_PATH|g" "$VENV_DIR/bin/activate"
sed -i "s|$OLD_PATH|$NEW_PATH|g" "$VENV_DIR/pyvenv.cfg"
find "$VENV_DIR/bin" -maxdepth 1 -type f | xargs sed -i "s|$OLD_PATH|$NEW_PATH|g" 2>/dev/null || true

echo "[$(date)] Tarring patched venv to /project..."
rm -rf "$FINAL_VENV"
mkdir -p "$FINAL_VENV"
# Extract in place with correct path
cd "$SLURM_TMPDIR"
tar czf /project/rrg-ravh011/ilkham/venvs/stch-botorch-v2.tar.gz stch-venv/
echo "[$(date)] Tar done: $(du -sh /project/rrg-ravh011/ilkham/venvs/stch-botorch-v2.tar.gz)"

# Also extract directly to FINAL_VENV for immediate use
echo "[$(date)] Extracting to $FINAL_VENV for direct use..."
cp -a "$VENV_DIR/." "$FINAL_VENV/"

echo "[$(date)] Verifying from final path..."
source "$FINAL_VENV/bin/activate"
python -c "
import torch, botorch
print('torch:', torch.__version__)
print('botorch:', botorch.__version__)
print('cuda:', torch.cuda.is_available())
print('ALL GOOD - venv ready at: $FINAL_VENV')
"

echo "[$(date)] SUCCESS - venv ready"
