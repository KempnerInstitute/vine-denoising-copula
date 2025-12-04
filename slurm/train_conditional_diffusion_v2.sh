#!/bin/bash
#SBATCH --job-name=cond_diff_v2
#SBATCH --output=logs/slurm/cond_diff_v2_%j.out
#SBATCH --error=logs/slurm/cond_diff_v2_%j.err
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Conditional Diffusion V2: Fixed Conditioning Collapse
#
# Key fixes:
# 1. Classifier-Free Guidance (CFG) - prevents histogram copying
# 2. Anti-copying loss - explicitly penalizes copying input
# 3. Stronger smoothness regularization
#

set -e

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

mkdir -p logs/slurm

module load cuda/12.4.1-fasrc01

source ~/.bashrc
conda activate diffuse_vine_cop

echo "============================================================"
echo "Conditional Diffusion V2 - Fixed Conditioning Collapse"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""
echo "Key fixes:"
echo "  - Classifier-Free Guidance (15% dropout)"
echo "  - Anti-copying loss (penalizes histogram copying)"
echo "  - Stronger TV smoothness loss"
echo "  - CFG-guided inference at test time"
echo ""
date

python scripts/train_conditional_diffusion_v2.py \
    --config configs/train_conditional_diffusion_v2.yaml

echo "Training complete"
date

