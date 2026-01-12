#!/bin/bash
#SBATCH --job-name=vdc_train
#SBATCH --output=slurm_logs/vdc_train_%j.out
#SBATCH --error=slurm_logs/vdc_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --account=kempner_dev

# =============================================================================
# Vine Diffusion Copula - Unified Training Script
# =============================================================================
# Usage:
#   sbatch slurm/train.sh                                    # Default config
#   sbatch slurm/train.sh configs/train/custom.yaml         # Custom config
#   CONFIG=configs/train/custom.yaml sbatch slurm/train.sh  # Via env var
# =============================================================================

set -e

# Configuration
CONFIG_FILE=${1:-${CONFIG:-configs/train/default.yaml}}
MODEL_TYPE=${MODEL_TYPE:-diffusion_unet}
RESUME_FROM=${RESUME_FROM:-}

# Setup
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula
mkdir -p slurm_logs

echo "=============================================="
echo "Vine Diffusion Copula - Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL_TYPE"
echo "Start time: $(date)"
echo "=============================================="

# Load environment
source ~/.bashrc
module load python/3.10.13-fasrc01
module load cuda/12.2.0-fasrc01

# Activate conda environment (if using conda)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate vdc 2>/dev/null || echo "Conda env 'vdc' not found, using base"
fi

# Set distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export NCCL_DEBUG=INFO
# New env var name (old one is deprecated in recent PyTorch)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Build command
CMD="torchrun --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --config $CONFIG_FILE \
    --model-type $MODEL_TYPE"

if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume $RESUME_FROM"
fi

echo "Command: $CMD"
echo ""

# Run training
$CMD

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "=============================================="
