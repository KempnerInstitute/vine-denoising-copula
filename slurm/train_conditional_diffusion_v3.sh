#!/bin/bash
#SBATCH --job-name=cond_diff_v3
#SBATCH --output=logs/slurm/cond_diff_v3_%j.out
#SBATCH --error=logs/slurm/cond_diff_v3_%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00  # 2 days for 300k steps

# V3 Training: Improved Data Diversity
#
# Key improvements:
# - 50% mixture models for non-parametric structures
# - More mixture components (2-5)
# - All copula families including BB1, BB7
# - 50% rotation probability
# - Wider parameter ranges

set -e

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Create log directory
mkdir -p logs/slurm

# Load modules
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

echo "=========================================="
echo "Starting V3 Training (300k steps)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""
echo "Key improvements:"
echo "  - 50% mixture probability"
echo "  - 2-5 mixture components"
echo "  - BB1, BB7 families added"
echo "  - 50% rotation probability"
echo "  - Wider parameter ranges"
echo ""
date

# Run training
python scripts/train_conditional_diffusion_v2.py \
    --config configs/train_conditional_diffusion_v3.yaml

echo "Training complete"
date

