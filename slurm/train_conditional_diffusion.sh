#!/bin/bash
#SBATCH --job-name=cond_diff
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/conditional_diffusion_%j.out
#SBATCH --error=logs/conditional_diffusion_%j.err

# Conditional Diffusion Training for Copula Density Estimation
#
# This trains a diffusion model that is CONDITIONED on histogram input.
# The model learns to predict density given observed samples.

set -e

# Setup
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula
mkdir -p logs

# Environment
source ~/.bashrc
module load cuda/12.4.1-fasrc01

# Activate conda environment with PyTorch
conda activate /n/netscratch/kempner_dev/hsafaai/conda_envs/diffuse_vine_cop

# Print info
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Config: configs/train_conditional_diffusion.yaml"
echo "Python: $(which python)"
echo ""

# Run training (single GPU for now)
srun python scripts/train_conditional_diffusion.py \
    --config configs/train_conditional_diffusion.yaml

echo "Training complete!"

