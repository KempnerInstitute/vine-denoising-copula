#!/bin/bash
#SBATCH --job-name=vdc_paper_e26_ood
#SBATCH --output=logs/vdc_paper_e26_%j.out
#SBATCH --error=logs/vdc_paper_e26_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#
# ============================================================================
# Vine Denoising Copula (NeurIPS 2026) - E26 OOD BIVARIATE FAMILY BENCHMARK
# ============================================================================
# Out-of-distribution families (BB6, BB8, Tawn, Plackett) absent from the
# VDC training zoo. Compares frozen VDC vs pyvine-param vs pyvine-TLL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e26_ood_bivariate.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: E26 OOD bivariate family benchmark"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e26_ood_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

# Environment setup
module purge
module load cuda/12.2.0-fasrc01

# Use Mambaforge lib for working libffi.so.8
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)" || true
set +u
# Default to the project env python (no named conda env exists on this system).
VDC_PYTHON_BIN="${VDC_PYTHON_BIN:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2/bin/python}"
if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
elif [ -n "${VDC_CONDA_ENV_PATH:-}" ]; then
  conda activate "${VDC_CONDA_ENV_PATH}"
  PYTHON_BIN="python"
elif conda activate diffuse_vine_cop 2>/dev/null; then
  PYTHON_BIN="python"
elif conda activate vdc 2>/dev/null; then
  PYTHON_BIN="python"
else
  echo "ERROR: failed to activate conda env. Set VDC_CONDA_ENV_PATH or VDC_PYTHON_BIN."
  exit 3
fi
set -u

if [ "${PYTHON_BIN}" != "python" ] && [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: VDC_PYTHON_BIN is not executable: ${PYTHON_BIN}"
  exit 3
fi

# Add conda lib after for other libraries
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo "Testing Python..."
"${PYTHON_BIN}" -c "print('Python works')"
"${PYTHON_BIN}" -c "import numpy; print(f'numpy: {numpy.__version__}')"
"${PYTHON_BIN}" -c "import scipy; print(f'scipy: {scipy.__version__}')"
"${PYTHON_BIN}" -c "import torch; print(f'torch: {torch.__version__}')"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e26_ood_bivariate.json"
OUT_TEX="${REPO_ROOT}/drafts/tables/tab_ood_families.tex"
CKPT="${E26_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi

"${PYTHON_BIN}" drafts/scripts/e26_ood_bivariate.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  --device cuda \
  --seeds 7 17 42 \
  --n "${E26_N:-2000}" \
  --n-test "${E26_N_TEST:-2000}" \
  --m-grid "${E26_M_GRID:-64}" \
  --projection-iters "${E26_PROJECTION_ITERS:-30}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out-json "${OUT_JSON}" \
  --out-tex "${OUT_TEX}" \
  2>&1 | tee "${RUN_DIR}/logs/e26_ood.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e26_ood_bivariate.json"

echo ""
echo "============================================================================"
echo "DONE: E26 OOD bivariate benchmark completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "E26 JSON: ${OUT_JSON}"
echo "Paper cache: ${REPO_ROOT}/drafts/paper_outputs/e26_ood_bivariate.json"
echo "Table: ${OUT_TEX}"
echo ""
