#!/bin/bash
#SBATCH --job-name=vdc_paper_e12_block_mi
#SBATCH --output=logs/vdc_paper_e12_block_mi_%j.out
#SBATCH --error=logs/vdc_paper_e12_block_mi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --mem=192GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT=""
if [ -n "${VDC_REPO_ROOT:-}" ] && [ -f "${VDC_REPO_ROOT}/drafts/scripts/e12_block_mi_sample_size_benchmark.py" ]; then
  REPO_ROOT="${VDC_REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e12_block_mi_sample_size_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -f "${PWD}/drafts/scripts/e12_block_mi_sample_size_benchmark.py" ]; then
  REPO_ROOT="${PWD}"
elif [ -f "${SCRIPT_DIR}/../drafts/scripts/e12_block_mi_sample_size_benchmark.py" ]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [ -z "${REPO_ROOT}" ]; then
  echo "ERROR: unable to resolve repository root"
  exit 2
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

mkdir -p "${REPO_ROOT}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e12_block_mi_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
else
  PYTHON_BIN="/n/sw/Mambaforge-23.11.0-0/bin/python"
fi

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

SEED_TAG="${E12_SEED:-42}"
RESULT_BASENAME="${E12_RESULT_BASENAME:-e12_block_mi_sample_size_results}"
OUT_JSON="${RUN_DIR}/results/${RESULT_BASENAME}_seed${SEED_TAG}.json"

"${PYTHON_BIN}" drafts/scripts/e12_block_mi_sample_size_benchmark.py \
  --device cuda \
  --scenario "${E12_SCENARIO:-gaussian_ar1}" \
  --seed "${SEED_TAG}" \
  --n-trials "${E12_TRIALS:-3}" \
  --d "${E12_D:-50}" \
  --split "${E12_SPLIT:-25}" \
  --rho "${E12_RHO:-0.7}" \
  --theta "${E12_THETA:-3.0}" \
  --m-true "${E12_M_TRUE:-2048}" \
  --sample-sizes ${E12_SAMPLE_SIZES:-100 300 1000 3000 10000 30000 100000} \
  --methods ${E12_METHODS:-dcd ksg mine infonce gaussian} \
  --mine-steps "${E12_MINE_STEPS:-800}" \
  --infonce-steps "${E12_INFONCE_STEPS:-800}" \
  --out "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e12_block_mi.log"

cp "${OUT_JSON}" "drafts/paper_outputs/${RESULT_BASENAME}_seed${SEED_TAG}.json"
cp "${OUT_JSON}" "drafts/paper_outputs/${RESULT_BASENAME}.json"

"${PYTHON_BIN}" drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
  2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_e12.log"
