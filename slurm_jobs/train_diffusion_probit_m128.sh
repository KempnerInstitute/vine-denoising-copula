#!/bin/bash
#SBATCH --job-name=diffusion_probit_m128
#SBATCH --output=logs/train_diffusion_probit_m128_%j.out
#SBATCH --error=logs/train_diffusion_probit_m128_%j.err
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --mem=192GB

echo "================================================================"
echo "TRAINING RUN: Diffusion U-Net with PROBIT binning (m=128, LR=5e-5)"
echo "================================================================"
echo "Start time : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node(s)    : $SLURM_JOB_NODELIST"
echo "GPUs       : $SLURM_GPUS_ON_NODE per node"
echo "================================================================"

# Load modules (match validate_m128 style)
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# PyTorch DDP settings
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29513
export WORLD_SIZE=4

# Go to project root
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

echo "Launching torchrun with config: configs/train_diffusion_probit_m128.yaml"

torchrun --nproc_per_node=4 --master_port=29513 \
    scripts/train_unified.py \
    --config configs/train_diffusion_probit_m128.yaml \
    --model-type diffusion_unet

exit_code=$?

echo "================================================================"
if [ $exit_code -eq 0 ]; then
  echo "Training completed successfully at $(date)"
else
  echo "Training FAILED with exit code $exit_code at $(date)"
fi
echo "================================================================"


