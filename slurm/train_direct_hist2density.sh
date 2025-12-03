#!/bin/bash
#SBATCH --job-name=direct_h2d
#SBATCH --output=logs/slurm/direct_h2d_%j.out
#SBATCH --error=logs/slurm/direct_h2d_%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Approach 2: Direct histogram-to-density prediction (NOT diffusion)
# This is simpler and faster, directly optimizes density accuracy

set -e

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Create log directory
mkdir -p logs/slurm

# Load modules
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

echo "Starting direct histogram-to-density training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
date

# Run training
python scripts/train_direct_hist2density.py \
    --config configs/train_direct_hist2density.yaml

echo "Training complete"
date

