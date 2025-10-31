#!/bin/bash

# Local multi-GPU training script (for workstations)
# Run this on a machine with multiple GPUs

set -e

NUM_GPUS=4  # Adjust based on available GPUs
CONFIG="configs/train_large.yaml"
CHECKPOINT_DIR="checkpoints/local"

echo "========================================="
echo "Local Multi-GPU Training"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "========================================="

# Create directories
mkdir -p logs
mkdir -p $CHECKPOINT_DIR

# Set environment
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust if needed

# Launch training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train_large_scale.py \
    --config $CONFIG \
    2>&1 | tee logs/train_local_$(date +%Y%m%d_%H%M%S).log

echo "========================================="
echo "Training complete!"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "========================================="
