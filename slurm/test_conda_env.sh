#!/bin/bash
#SBATCH --job-name=test_conda
#SBATCH --output=logs/test_conda_%j.out
#SBATCH --error=logs/test_conda_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --mem=32GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -x  # Debug mode

echo "Testing conda environments"
echo "Node: $(hostname)"

# Show conda info
eval "$(conda shell.bash hook)" || true
conda info --envs

# Try activating each env and checking numpy/scipy
for env in vdc diffuse_vine_cop dvc dvc-env; do
    echo ""
    echo "=== Testing env: $env ==="
    conda activate "$env" 2>/dev/null && {
        echo "Activated: $CONDA_PREFIX"
        which python
        python -c "import numpy; print(f'numpy: {numpy.__version__}')" 2>&1 || echo "numpy FAILED"
        python -c "import scipy; print(f'scipy: {scipy.__version__}')" 2>&1 || echo "scipy FAILED"
        python -c "import torch; print(f'torch: {torch.__version__}')" 2>&1 || echo "torch FAILED"
    } || echo "Failed to activate: $env"
done

echo ""
echo "=== Checking user site-packages ==="
python -s -c "import sys; print('User site enabled:', hasattr(sys, 'USER_SITE')); print('sys.path:', sys.path)"
