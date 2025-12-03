#!/bin/bash
#SBATCH --job-name=cond_diff_long
#SBATCH --output=logs/slurm/cond_diff_long_%j.out
#SBATCH --error=logs/slurm/cond_diff_long_%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Approach 1: Train conditional diffusion for longer (100k steps)

set -e

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Create log directory
mkdir -p logs/slurm

# Load modules
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

echo "Starting conditional diffusion LONG training (100k steps)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
date

# Run training
python scripts/train_conditional_diffusion.py \
    --config configs/train_conditional_diffusion_long.yaml

echo "Training complete"
date

