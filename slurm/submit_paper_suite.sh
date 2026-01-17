#!/bin/bash
set -euo pipefail

# Convenience helper: submits all three paper training jobs.
#
# Usage:
#   OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula ./slurm/submit_paper_suite.sh
#
# Notes:
# - This runs three independent jobs; you can comment out any you don't want.

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"
cd "${REPO_ROOT}"

# Ensure SLURM stdout/err directories exist before submission
mkdir -p logs

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "Submitting paper suite with OUTPUT_BASE=${OUTPUT_BASE}"

PARTITION="${PARTITION:-kempner_h100_priority3}"
J1=$(sbatch --partition="${PARTITION}" slurm/paper_vdc_diffusion_cond.sh | awk '{print $4}')
J2=$(sbatch --partition="${PARTITION}" slurm/paper_vdc_denoiser_cond.sh | awk '{print $4}')
J3=$(sbatch --partition="${PARTITION}" slurm/paper_vdc_enhanced_cnn_cond.sh | awk '{print $4}')

echo "Submitted:"
echo "  diffusion_cond:   ${J1}"
echo "  denoiser_cond:    ${J2}"
echo "  enhanced_cnn_cond:${J3}"

