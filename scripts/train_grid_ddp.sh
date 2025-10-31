#!/bin/bash
#SBATCH --job-name=vine-copula-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Vine Copula Training Script for SLURM
# 
# Usage:
#   sbatch scripts/train_grid_ddp.sh
#
# Multi-node usage:
#   sbatch --nodes=2 scripts/train_grid_ddp.sh

echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Load modules (adjust for your HPC system)
module purge
module load python/3.10
module load cuda/12.1
module load cudnn/8.9

# Activate environment
source activate vine-copula  # or: source /path/to/venv/bin/activate

# Create logs directory
mkdir -p logs

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"

# Training configuration
DATA_ROOT="/path/to/your/data"  # UPDATE THIS
CHECKPOINT_DIR="checkpoints/run_${SLURM_JOB_ID}"
M=64  # Grid resolution
BATCH_SIZE=32  # Per GPU
MAX_STEPS=400000
LR=3e-4

# Weights & Biases (optional)
USE_WANDB="--use_wandb"  # Remove to disable
export WANDB_PROJECT="vine-copula"
export WANDB_RUN_NAME="train_m${M}_bs${BATCH_SIZE}_${SLURM_JOB_ID}"

echo "========================================="
echo "Training Configuration"
echo "========================================="
echo "Data Root: $DATA_ROOT"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Grid Resolution (m): $M"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Max Steps: $MAX_STEPS"
echo "Learning Rate: $LR"
echo "========================================="

# Launch training with torchrun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    vdc/train/train_grid.py \
    --data_root $DATA_ROOT \
    --m $M \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_steps $MAX_STEPS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $USE_WANDB

echo "========================================="
echo "Training Complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "========================================="
