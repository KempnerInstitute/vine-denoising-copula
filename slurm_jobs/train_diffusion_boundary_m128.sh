#!/bin/bash
#SBATCH --job-name=diffusion_boundary_m128
#SBATCH --output=logs/train_diffusion_boundary_m128_%j.out
#SBATCH --error=logs/train_diffusion_boundary_m128_%j.err
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Boundary-focused diffusion training (uniform grid with enhanced tail loss)
# Based on the successful "good run" architecture

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01

# Activate conda environment (same as other jobs)
source ~/.bashrc
conda activate diffuse_vine_cop

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to project directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Run training
python scripts/train_unified.py \
    --config configs/train_diffusion_boundary_m128.yaml \
    --model-type diffusion_unet

echo "Training job completed"

