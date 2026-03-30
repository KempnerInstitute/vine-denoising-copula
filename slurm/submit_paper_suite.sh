#!/bin/bash
set -euo pipefail

# Convenience helper: submits the single paper training job
# (enhanced denoiser model only).
#
# Usage:
#   OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula ./slurm/submit_paper_suite.sh
#
# Notes:
# - This intentionally submits one model only (denoiser_cond_enhanced).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Ensure SLURM stdout/err directories exist before submission
mkdir -p logs

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "Submitting single-model paper suite with OUTPUT_BASE=${OUTPUT_BASE}"

PARTITION="${PARTITION:-kempner_h100_priority3}"
J1=$(sbatch --partition="${PARTITION}" slurm/paper_train_enhanced.sh | awk '{print $4}')

echo "Submitted:"
echo "  denoiser_cond_enhanced: ${J1}"
