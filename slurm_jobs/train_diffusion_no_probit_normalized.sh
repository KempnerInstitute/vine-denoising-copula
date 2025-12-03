#!/bin/bash
#SBATCH --job-name=diffusion_normalized
#SBATCH --output=logs/train_diffusion_no_probit_normalized_%j.out
#SBATCH --error=logs/train_diffusion_no_probit_normalized_%j.err
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --mem=192GB

echo "================================================================"
echo "TRAINING RUN: Diffusion NO PROBIT NORMALIZED (Nov 21 config)"
echo "================================================================"
echo "Start time : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node(s)    : $SLURM_JOB_NODELIST"
echo "GPUs       : $SLURM_GPUS_ON_NODE per node"
echo ""
echo "This is the config from Nov 21 that may have produced good fits"
echo "Key settings:"
echo "  - detach_projection: false (gradients flow through projection)"
echo "  - use_log_ise: true"
echo "  - binning: probit"
echo "  - LR: 1e-4 (higher than recent runs)"
echo "================================================================"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01

# Activate conda environment
source ~/.bashrc
conda activate diffuse_vine_cop

export CUDA_VISIBLE_DEVICES=0,1,2,3

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29533
export WORLD_SIZE=4

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

echo "Launching torchrun with config: configs/archive/train_diffusion_no_probit_normalized.yaml"

torchrun --nproc_per_node=4 --master_port=29533 \
    scripts/train_unified.py \
    --config configs/archive/train_diffusion_no_probit_normalized.yaml \
    --model-type diffusion_unet

exit_code=$?

echo "================================================================"
if [ $exit_code -eq 0 ]; then
  echo "Training completed successfully at $(date)"
else
  echo "Training FAILED with exit code $exit_code at $(date)"
fi
echo "================================================================"

