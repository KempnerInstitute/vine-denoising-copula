#!/bin/bash
#SBATCH --job-name=vdc_paper_artifacts
#SBATCH --output=logs/vdc_paper_artifacts_%j.out
#SBATCH --error=logs/vdc_paper_artifacts_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# Generate paper tables/figures from experiment outputs (no LaTeX compile).
#
# Usage:
#   OUTPUT_BASE=/path/to/output_base sbatch slurm/paper_generate_artifacts.sh
#
# Recommended usage with dependencies:
#   sbatch --dependency=afterok:<jobids...> slurm/paper_generate_artifacts.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

mkdir -p "${REPO_ROOT}/logs"

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)" || true
conda activate vdc 2>/dev/null || conda activate diffuse_vine_cop 2>/dev/null || true

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Generating paper artifacts..."
echo "OUTPUT_BASE=${OUTPUT_BASE}"

# Qualitative diffusion controls (consumed by drafts/scripts/fig_e1*_examples.py):
# - DIFFUSION_SMOOTH_SIGMA="auto" scales sigma with grid resolution (m), which is important for m=128+.
# - DIFFUSION_ENSEMBLE averages multiple DDIM samples (slower, smoother)
export DIFFUSION_SMOOTH_SIGMA="${DIFFUSION_SMOOTH_SIGMA:-auto}"
export DIFFUSION_SMOOTH_BASE="${DIFFUSION_SMOOTH_BASE:-0.35}"
export DIFFUSION_HIST_SMOOTH_SIGMA="${DIFFUSION_HIST_SMOOTH_SIGMA:-0.0}"
export DIFFUSION_X0_SMOOTH_SIGMA="${DIFFUSION_X0_SMOOTH_SIGMA:-0.0}"
export DIFFUSION_X0_SMOOTH_EVERY="${DIFFUSION_X0_SMOOTH_EVERY:-0}"
export DIFFUSION_ENSEMBLE="${DIFFUSION_ENSEMBLE:-1}"
export DIFFUSION_ENSEMBLE_MODE="${DIFFUSION_ENSEMBLE_MODE:-geometric}"

python drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force

echo "Done generating paper artifacts."
