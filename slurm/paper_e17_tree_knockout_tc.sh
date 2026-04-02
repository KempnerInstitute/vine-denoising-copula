#!/bin/bash
#SBATCH --job-name=vdc_e17_knockout_tc
#SBATCH --output=logs/vdc_e17_knockout_tc_%j.out
#SBATCH --error=logs/vdc_e17_knockout_tc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e17_tree_knockout_tc.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

PYTHON_BIN="${VDC_PYTHON_BIN:-/n/home13/hsafaai/.venvs/vdc_paper/bin/python}"
OUT_JSON="${E17_OUT_JSON:-${REPO_ROOT}/drafts/paper_outputs/e17_tree_knockout_tc_full.json}"

"${PYTHON_BIN}" drafts/scripts/e17_tree_knockout_tc.py \
  --datasets ${E17_DATASETS:-power gas credit} \
  --device cuda \
  --max-train ${E17_MAX_TRAIN:-10000} \
  --max-test ${E17_MAX_TEST:-2000} \
  --n-sim ${E17_N_SIM:-4000} \
  --n-repeats ${E17_N_REPEATS:-5} \
  --seed ${E17_SEED:-0} \
  --seed-step ${E17_SEED_STEP:-17} \
  --ksg-k ${E17_KSG_K:-5} \
  --out "${OUT_JSON}"

echo "Wrote ${OUT_JSON}"
