#!/bin/bash
#SBATCH --job-name=copula-light-8gpu
#SBATCH --output=logs/train_8gpu_light_%j.out
#SBATCH --error=logs/train_8gpu_light_%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_hsafaai_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=375G
#SBATCH --time=00-03:00:00

# Light training run for 8 H100 GPUs
# Expected duration: ~2.5 hours
# Purpose: Quick training test with comprehensive validation

echo "=================================================="
echo "Vine Copula Diffusion - Light Training (8 GPUs)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 375GB"
echo "=================================================="

# Environment setup
module purge
module load python/3.10
module load cuda/12.1
module load cudnn/8.9

# Activate conda environment
source activate vine-copula

# Navigate to project directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Create log directory if needed
mkdir -p logs

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# PyTorch optimizations
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=0

echo ""
echo "Environment:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPUs available: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

# Check GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Print config
echo "Training Configuration:"
echo "  Config: configs/train_8gpu_light.yaml"
echo "  Batch size: 128 per GPU (total 1024)"
echo "  Steps: 20,000 (~2.5 hours)"
echo "  Model: GridUNet, 96 channels"
echo "  Grid resolution: 64x64"
echo "  Data: On-the-fly generation"
echo ""

# Launch training with torchrun
echo "Launching training..."
echo "=================================================="

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    scripts/train_large_scale.py \
    --config configs/train_8gpu_light.yaml

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

# If training succeeded, run quick evaluation
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Running quick evaluation on latest checkpoint..."
    
    # Find latest checkpoint
    LATEST_CKPT=$(ls -t checkpoints/light_8gpu/model_*.pt 2>/dev/null | head -n 1)
    
    if [ -n "$LATEST_CKPT" ]; then
        echo "Checkpoint: $LATEST_CKPT"
        
        python scripts/test_on_known_copulas.py \
            --checkpoint "$LATEST_CKPT" \
            --families gaussian clayton gumbel frank \
            --n-samples 2000
        
        echo ""
        echo "✓ Quick evaluation completed"
    else
        echo "Warning: No checkpoint found"
    fi
fi

# Print job statistics
echo ""
echo "Job Statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State

# Archive logs
LOG_DIR="logs/run_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
cp "logs/train_8gpu_light_${SLURM_JOB_ID}.out" "$LOG_DIR/" 2>/dev/null || true
cp "logs/train_8gpu_light_${SLURM_JOB_ID}.err" "$LOG_DIR/" 2>/dev/null || true
echo "Logs archived to: $LOG_DIR"

exit $EXIT_CODE
