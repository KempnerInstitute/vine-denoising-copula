#!/bin/bash
#SBATCH --job-name=cond_diff_hybrid
#SBATCH --output=logs/slurm/cond_diff_hybrid_%j.out
#SBATCH --error=logs/slurm/cond_diff_hybrid_%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

# Approach 3: Hybrid loss - noise MSE + density ISE
# This combines diffusion training with direct density supervision

set -e

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Create log directory
mkdir -p logs/slurm

# Load modules
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

echo "Starting conditional diffusion with HYBRID loss training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
date

# Run training (uses modified script with hybrid loss)
python scripts/train_conditional_diffusion_hybrid.py \
    --config configs/train_conditional_diffusion_hybrid.yaml

echo "Training complete"
date

