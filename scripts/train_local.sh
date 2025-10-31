#!/bin/bash
# Simple training launcher for local multi-GPU training
#
# Usage:
#   bash scripts/train_local.sh

# Configuration
DATA_ROOT="${DATA_ROOT:-data/synthetic}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/local}"
M=64
BATCH_SIZE=32
MAX_STEPS=100000
LR=3e-4
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "========================================="
echo "Local Training Configuration"
echo "========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Data Root: $DATA_ROOT"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Grid Resolution: $M"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Max Steps: $MAX_STEPS"
echo "Learning Rate: $LR"
echo "========================================="

# Create directories
mkdir -p logs
mkdir -p $CHECKPOINT_DIR

# Launch with torchrun
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    vdc/train/train_grid.py \
    --data_root $DATA_ROOT \
    --m $M \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_steps $MAX_STEPS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $@

echo "========================================="
echo "Training Complete!"
echo "========================================="
