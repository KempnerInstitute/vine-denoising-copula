#!/bin/bash
# Installation script for Vine Diffusion Copula environment
# Run this on a compute node with GPU access for proper CUDA setup

set -e  # Exit on error

echo "=================================================="
echo "Vine Diffusion Copula - Environment Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_green() { echo -e "${GREEN}✓ $1${NC}"; }
print_yellow() { echo -e "${YELLOW}➜ $1${NC}"; }
print_red() { echo -e "${RED}✗ $1${NC}"; }

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_red "conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

print_green "conda found: $(which conda)"

# Environment name
ENV_NAME="diffuse_vine_cop"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_yellow "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_yellow "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        print_yellow "Keeping existing environment. Will try to update it."
        UPDATE_ONLY=true
    fi
fi

# Create or update environment
if [ "$UPDATE_ONLY" = true ]; then
    print_yellow "Updating environment from environment.yml..."
    conda env update -n $ENV_NAME -f environment.yml --prune
else
    print_yellow "Creating new environment from environment.yml..."
    conda env create -f environment.yml
fi

print_green "Environment creation/update complete"

# Activate environment
print_yellow "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

print_green "Environment activated"

# Verify PyTorch CUDA availability
print_yellow "Verifying PyTorch and CUDA..."
python -c "
import torch
import sys

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available!')
"

# Check if CUDA is available
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    print_green "CUDA is properly configured"
else
    print_red "WARNING: CUDA not available. Make sure you're on a GPU node!"
    print_yellow "To request GPU node: salloc --partition=kempner_h100 --gres=gpu:1"
fi

# Install package in editable mode
print_yellow "Installing vine-diffusion-copula package in editable mode..."
pip install -e .

print_green "Package installed"

# Verify key dependencies
print_yellow "Verifying key dependencies..."
python -c "
import torch
import numpy
import scipy
import matplotlib
import pyvinecopulib as pv
print('✓ All core dependencies imported successfully')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - NumPy: {numpy.__version__}')
print(f'  - SciPy: {scipy.__version__}')
print(f'  - pyvinecopulib: {pv.__version__}')
"

if [ $? -eq 0 ]; then
    print_green "All dependencies verified"
else
    print_red "Some dependencies failed to import"
    exit 1
fi

# Create necessary directories
print_yellow "Creating project directories..."
mkdir -p logs checkpoints results data examples/plots

print_green "Directories created"

# Test import of local package
print_yellow "Testing local package import..."
python -c "
import vdc
from vdc.models.unet_grid import GridUNet
from vdc.data.generators import sample_bicop
print('✓ Local package imports successfully')
"

if [ $? -eq 0 ]; then
    print_green "Package imports verified"
else
    print_red "Failed to import local package"
    exit 1
fi

# Summary
echo ""
echo "=================================================="
print_green "Installation Complete!"
echo "=================================================="
echo ""
echo "Environment name: $ENV_NAME"
echo ""
echo "To activate this environment, run:"
echo "  source activate $ENV_NAME"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "Next steps:"
echo "  1. Update SLURM scripts to use: source activate $ENV_NAME"
echo "  2. Run quick test: python scripts/test_on_known_copulas.py --help"
echo "  3. Start training: sbatch scripts/slurm/train_8gpu_light.sh"
echo ""
print_green "Ready to train! 🚀"
echo ""
