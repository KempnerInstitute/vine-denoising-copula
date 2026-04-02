#!/bin/bash
#SBATCH --job-name=vdc_e16_rolldep
#SBATCH --output=logs/vdc_e16_rolldep_%j.out
#SBATCH --error=logs/vdc_e16_rolldep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${VDC_REPO_ROOT:-}" ] && [ -d "${VDC_REPO_ROOT}" ]; then
  REPO_ROOT="${VDC_REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e16_rolling_dependence_dashboard.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/drafts/paper_outputs"

module purge
module load cuda/12.2.0-fasrc01

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"

if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
else
  PYTHON_BIN="/n/netscratch/kempner_dev/hsafaai/conda_envs/vdc_paper/bin/python"
fi

export OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

cd "${REPO_ROOT}"

OUT_JSON="${E16_OUT_JSON:-${REPO_ROOT}/drafts/paper_outputs/e16_rolling_dependence.json}"

"${PYTHON_BIN}" drafts/scripts/e16_rolling_dependence_dashboard.py \
  --device "${E16_DEVICE:-cuda}" \
  --vine-type "${E16_VINE_TYPE:-dvine}" \
  --window "${E16_WINDOW:-252}" \
  --refit-every "${E16_REFIT_EVERY:-5}" \
  --start-day "${E16_START_DAY:-0}" \
  --max-days "${E16_MAX_DAYS:-0}" \
  --top-k "${E16_TOP_K:-5}" \
  --progress-every "${E16_PROGRESS_EVERY:-10}" \
  --out-json "${OUT_JSON}"

echo "Wrote ${OUT_JSON}"
