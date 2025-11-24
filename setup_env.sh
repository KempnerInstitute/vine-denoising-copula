#!/bin/bash
# Script to recreate the diffuse_vine_cop conda environment
# Run this on a GPU node: ssh holygpu8a17403
# Then: bash setup_env.sh

echo "Creating conda environment with Python 3.10..."
conda create -n diffuse_vine_cop python=3.10 -y

echo "Activating environment..."
source ~/.bashrc
conda activate diffuse_vine_cop

echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install numpy scipy matplotlib seaborn pandas scikit-learn h5py pyyaml
pip install pyvinecopulib
pip install hydra-core omegaconf
pip install typer rich click tqdm
pip install tensorboard wandb
pip install diffusers

echo "Verifying installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "Environment setup complete!"
