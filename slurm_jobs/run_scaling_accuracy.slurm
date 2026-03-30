#!/bin/bash
#SBATCH --job-name=vdc_scaling_accuracy
#SBATCH --output=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/logs/scaling_accuracy_%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/logs/scaling_accuracy_%j.err
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Scaling experiments with accuracy metrics (TC error, NLL error vs dimension)

set -e

# Activate environment
source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate /n/netscratch/kempner_dev/hsafaai/conda_envs/diffuse_vine_cop

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Output directory
OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula
OUTPUT_DIR="${OUTPUT_BASE}/vdc_paper_scaling_accuracy_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

# Best checkpoints
CKPT_BILINEAR="${OUTPUT_BASE}/vdc_paper_diffusion_cond_m64_tuned_bilinear_20260117_020404_55563036/checkpoints/model_step_200000.pt"
CKPT_PROBIT="${OUTPUT_BASE}/vdc_paper_diffusion_cond_m64_tuned_probit_bilinear_20260117_021859_55563037/checkpoints/model_step_200000.pt"
CKPT_DENOISER="${OUTPUT_BASE}/vdc_paper_denoiser_cond_20260113_235331_55240324/checkpoints/model_step_150000.pt"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Bilinear checkpoint: ${CKPT_BILINEAR}"
echo "Probit checkpoint: ${CKPT_PROBIT}"
echo "Denoiser checkpoint: ${CKPT_DENOISER}"
echo ""

# Run scaling experiments with bilinear model
echo "Running scaling experiments with bilinear model..."
python scripts/run_scaling_with_accuracy.py \
    --checkpoint "${CKPT_BILINEAR}" \
    --checkpoint_probit "${CKPT_PROBIT}" \
    --output_dir "${OUTPUT_DIR}/bilinear_vs_probit" \
    --dims 2 5 10 20 50 100 200 \
    --n_samples 2000 \
    --rho 0.6 \
    --repeats 3 \
    --run_baselines

# Run scaling experiments with denoiser (one-shot) model
echo ""
echo "Running scaling experiments with denoiser (one-shot) model..."
python scripts/run_scaling_with_accuracy.py \
    --checkpoint "${CKPT_DENOISER}" \
    --output_dir "${OUTPUT_DIR}/denoiser" \
    --dims 2 5 10 20 50 100 200 \
    --n_samples 2000 \
    --rho 0.6 \
    --repeats 3

echo ""
echo "Scaling accuracy experiments complete!"
echo "Results saved to: ${OUTPUT_DIR}"
