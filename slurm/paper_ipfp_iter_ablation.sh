#!/bin/bash
#SBATCH --job-name=vdc_ipfp_iter
#SBATCH --output=logs/vdc_ipfp_iter_%j.out
#SBATCH --error=logs/vdc_ipfp_iter_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --partition=kempner_priority
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/ipfp_iter_ablation.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

PYTHON_BIN="${VDC_PYTHON_BIN:-/n/home13/hsafaai/.venvs/vdc_paper/bin/python}"

"${PYTHON_BIN}" drafts/scripts/ipfp_iter_ablation.py

echo "IPFP iteration ablation completed."
