#!/bin/bash
#SBATCH --job-name=validate-diffusion-no-probit
#SBATCH --output=logs/validate_diffusion_no_probit_%j.out
#SBATCH --error=logs/validate_diffusion_no_probit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --time=3:00:00
#SBATCH --mem=180GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

echo "================================================================"
echo "VALIDATION RUN: Diffusion U-Net WITHOUT Probit Transformation"
echo "================================================================"
echo "Starting training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo ""
echo "Expected: Loss drops below 7.0 within 5k steps"
echo "Baseline: Loss stuck at 8.2 with probit enabled"
echo "Config: m=256, LR=1e-4, proj_iters=15, no probit"
echo "Model: Diffusion U-Net"
echo "================================================================"

# Setup
module purge
module load cuda/12.2.0-fasrc01
source activate diffuse_vine_cop || conda activate diffuse_vine_cop || true

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run with torchrun for multi-GPU
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

torchrun \
    --standalone \
    --nproc_per_node=3 \
    scripts/train_unified.py \
    --config configs/validate_no_probit_diffusion.yaml \
    --model-type diffusion_unet

echo "================================================================"
echo "Training completed at $(date)"
echo "================================================================"
