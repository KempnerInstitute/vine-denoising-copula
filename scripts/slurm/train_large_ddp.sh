#!/bin/bash
#SBATCH --job-name=copula-large
#SBATCH --output=logs/train_large_%j.out
#SBATCH --error=logs/train_large_%j.err
#SBATCH --nodes=4                    # 4 nodes for large-scale training
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4                 # 4 GPUs per node (total 16 GPUs)
#SBATCH --mem=256G
#SBATCH --time=72:00:00              # 72 hours
#SBATCH --partition=gpu_requeue      # Adjust for your cluster

# Large-Scale Vine Copula Training
# Total GPUs: 16 (4 nodes × 4 GPUs)
# Effective batch size: 64 × 2 (grad_accum) × 16 = 2048 samples per step
# 
# This script is designed for training on millions of copula pairs
# Expected training time: ~48-72 hours for 500k steps

set -e  # Exit on error

echo "========================================="
echo "Large-Scale Copula Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Num Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Setup environment
module purge
module load python/3.10
module load cuda/12.1
module load cudnn/8.9
module load nccl/2.18

# Activate conda environment
source activate vine-copula
# OR: source /path/to/venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p checkpoints/large_scale

# Distributed training setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0             # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=3          # GPU Direct RDMA

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"

# Config file
CONFIG_FILE="configs/train_large.yaml"

# Update data paths in config (if needed)
# sed -i 's|train_root:.*|train_root: "/path/to/actual/train/data"|' $CONFIG_FILE
# sed -i 's|val_root:.*|val_root: "/path/to/actual/val/data"|' $CONFIG_FILE

# Weights & Biases setup (optional)
export WANDB_API_KEY="your_api_key_here"  # Set your W&B API key
export WANDB_MODE="online"                # or "offline"

echo "========================================="
echo "Starting Training..."
echo "Config: $CONFIG_FILE"
echo "========================================="

# Launch training with torchrun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=3 \
    --tee=3 \
    scripts/train_large_scale.py \
    --config $CONFIG_FILE

echo "========================================="
echo "Training Complete!"
echo "Checkpoints: checkpoints/large_scale/"
echo "========================================="

# Print final checkpoint info
if [ -f "checkpoints/large_scale/best.pt" ]; then
    echo "✓ Best model saved"
fi

if [ -f "checkpoints/large_scale/final.pt" ]; then
    echo "✓ Final model saved"
fi
