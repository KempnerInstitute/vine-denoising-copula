#!/bin/bash
#SBATCH --job-name=vdc_e13_ablation_depth
#SBATCH --output=logs/vdc_e13_ablation_depth_%j.out
#SBATCH --error=logs/vdc_e13_ablation_depth_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e13_depth_stability_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

PYTHON_BIN="${VDC_PYTHON_BIN:-/n/home13/hsafaai/.venvs/vdc_paper/bin/python}"

OUT_JSON="${REPO_ROOT}/drafts/paper_outputs/e13_depth_stability_ablation.json"
MAX_EDGES_PER_TREE="${E13_MAX_EDGES_PER_TREE:-2}"

UNIFORM_CKPT="${UNIFORM_CKPT:-$(cat "${REPO_ROOT}/analysis/PAPER_CHECKPOINT.txt")}"
DIRECT_CKPT="${DIRECT_CKPT:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_direct_20260330_144429_2719530/checkpoints/model_step_240000.pt}"
GAUSSIAN_CKPT="${GAUSSIAN_CKPT:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_gaussian_20260330_144631_2719531/checkpoints/model_step_190000.pt}"
MULTINOMIAL_CKPT="${MULTINOMIAL_CKPT:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_multinomial_20260330_150223_2719534/checkpoints/model_step_220000.pt}"

"${PYTHON_BIN}" drafts/scripts/e13_depth_stability_benchmark.py \
  --device cuda \
  --seed 0 \
  --max-edges-per-tree "${MAX_EDGES_PER_TREE}" \
  --label-checkpoint "Uniform-mix:${UNIFORM_CKPT}" \
  --label-checkpoint "Direct:${DIRECT_CKPT}" \
  --label-checkpoint "Gaussian:${GAUSSIAN_CKPT}" \
  --label-checkpoint "Multinomial:${MULTINOMIAL_CKPT}" \
  --out "${OUT_JSON}"

echo "Wrote ${OUT_JSON}"
