#!/bin/bash
#SBATCH --job-name=vdc_e15_structure
#SBATCH --output=logs/vdc_e15_structure_%j.out
#SBATCH --error=logs/vdc_e15_structure_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${VDC_REPO_ROOT:-}" ] && [ -d "${VDC_REPO_ROOT}" ]; then
  REPO_ROOT="${VDC_REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e15_structure_search_benchmark.py" ]; then
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

cd "${REPO_ROOT}"

OUT_JSON="${E15_OUT_JSON:-${REPO_ROOT}/drafts/paper_outputs/e15_structure_search_results.json}"

"${PYTHON_BIN}" drafts/scripts/e15_structure_search_benchmark.py \
  --datasets ${E15_DATASETS:-power gas hepmass miniboone} \
  --max-train "${E15_MAX_TRAIN:-50000}" \
  --max-test "${E15_MAX_TEST:-20000}" \
  --val-frac "${E15_VAL_FRAC:-0.2}" \
  --n-random-orders "${E15_N_RANDOM_ORDERS:-24}" \
  --exhaustive-max-d "${E15_EXHAUSTIVE_MAX_D:-5}" \
  --seed "${E15_SEED:-42}" \
  --edge-batch-size "${E15_EDGE_BATCH_SIZE:-256}" \
  --num-threads "${E15_NUM_THREADS:-16}" \
  --device "${E15_DEVICE:-cuda}" \
  --pyvine-mode "${E15_PYVINE_MODE:-nonparametric}" \
  --out-json "${OUT_JSON}"

echo "Wrote ${OUT_JSON}"
