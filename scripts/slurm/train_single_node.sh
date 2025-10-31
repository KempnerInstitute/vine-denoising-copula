#!/bin/bash
#SBATCH --job-name=copula-single
#SBATCH --output=logs/train_single_%j.out
#SBATCH --error=logs/train_single_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Single-Node Multi-GPU Training
# 4 GPUs on 1 node

echo "========================================="
echo "Single-Node Copula Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "========================================="

# Setup
module purge
module load python/3.10
module load cuda/12.1
module load cudnn/8.9

source activate vine-copula

mkdir -p logs
mkdir -p checkpoints/single_node

# Environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Config
CONFIG_FILE="configs/train_large.yaml"

echo "Starting training with $SLURM_GPUS_ON_NODE GPUs..."

# Launch
torchrun \
    --nnodes=1 \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_large_scale.py \
    --config $CONFIG_FILE

echo "Training complete!"
