#!/bin/bash
#SBATCH --job-name=cond_diff_smooth
#SBATCH --output=logs/slurm/cond_diff_smooth_%j.out
#SBATCH --error=logs/slurm/cond_diff_smooth_%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00

# IMPROVED Training: Conditional diffusion with smoothness losses
# 
# Key improvements over baseline:
# 1. Log-space cross-entropy loss to match density magnitudes
# 2. Tail-weighted loss for peaked copulas (Clayton, Gumbel, Joe)
# 3. TV smoothness regularization to reduce spottiness
# 4. Longer training (150k steps)
#

set -e

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Create log directory
mkdir -p logs/slurm

# Load modules
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

echo "Starting IMPROVED conditional diffusion training (150k steps)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""
echo "Key improvements:"
echo "  - Log-space CE loss for magnitude matching"
echo "  - Tail-weighted loss for peaked copulas"
echo "  - TV smoothness regularization"
echo ""
date

# Run training with improved config
python scripts/train_conditional_diffusion.py \
    --config configs/train_conditional_diffusion_smooth.yaml

echo "Training complete"
date

