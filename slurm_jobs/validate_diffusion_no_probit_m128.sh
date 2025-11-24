#!/bin/bash
#SBATCH --job-name=validate_m128
#SBATCH --output=logs/validate_diffusion_no_probit_m128_%j.out
#SBATCH --error=logs/validate_diffusion_no_probit_m128_%j.err
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --mem=192GB

echo "================================================================"
echo "VALIDATION RUN: Diffusion U-Net WITHOUT Probit (m=128, LR=5e-5)"
echo "================================================================"
echo "Starting training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo ""
echo "Expected: More stable training, loss converges smoothly"
echo "Previous: m=256, LR=1e-4 - loss oscillated 5k-10k"
echo "Changes: m=256→128 (4x faster), LR=1e-4→5e-5 (more stable)"
echo "Model: Diffusion U-Net"
echo "================================================================"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

# Set CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set PyTorch DDP settings
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29503
export WORLD_SIZE=4

# Run training
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

torchrun --nproc_per_node=4 --master_port=29503 \
    scripts/train_unified.py \
    --config configs/validate_no_probit_diffusion_m128.yaml \
    --model-type diffusion_unet

echo "================================================================"
echo "Training completed at $(date)"
echo "================================================================"
