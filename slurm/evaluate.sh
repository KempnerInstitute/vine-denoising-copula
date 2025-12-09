#!/bin/bash
#SBATCH --job-name=vdc_eval
#SBATCH --output=slurm_logs/vdc_eval_%j.out
#SBATCH --error=slurm_logs/vdc_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=kempner_dev

# =============================================================================
# Vine Diffusion Copula - Unified Evaluation Script
# =============================================================================
# Usage:
#   CHECKPOINT=path/to/model.pt sbatch slurm/evaluate.sh
#   sbatch slurm/evaluate.sh checkpoints/model.pt
# =============================================================================

set -e

# Configuration
CHECKPOINT=${1:-${CHECKPOINT:-}}
CONFIG_FILE=${CONFIG:-configs/inference/default.yaml}
MODE=${MODE:-all}  # all, bivariate, vine
DIMENSIONS=${DIMENSIONS:-"3 5 10"}

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint specified"
    echo "Usage: CHECKPOINT=path/to/model.pt sbatch slurm/evaluate.sh"
    echo "   or: sbatch slurm/evaluate.sh checkpoints/model.pt"
    exit 1
fi

# Setup
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula
mkdir -p slurm_logs

echo "=============================================="
echo "Vine Diffusion Copula - Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Dimensions: $DIMENSIONS"
echo "Start time: $(date)"
echo "=============================================="

# Load environment
source ~/.bashrc
module load python/3.10.13-fasrc01
module load cuda/12.2.0-fasrc01

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate vdc 2>/dev/null || echo "Conda env 'vdc' not found, using base"
fi

# Run evaluation
python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --mode "$MODE" \
    --dimensions $DIMENSIONS

echo ""
echo "=============================================="
echo "Evaluation completed: $(date)"
echo "=============================================="
