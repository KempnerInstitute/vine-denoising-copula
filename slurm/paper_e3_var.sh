#!/bin/bash
#SBATCH --job-name=vdc_paper_var
#SBATCH --output=logs/vdc_paper_var_%j.out
#SBATCH --error=logs/vdc_paper_var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Diffusion Copula (ICML 2026) - E3 VaR BACKTEST JOB
# ============================================================================
# Produces:
#   RUN_DIR/results/e3_var_results.json
# and copies it to:
#   drafts/paper_outputs/e3_var_results.json
# then regenerates paper artifacts.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e3_var_backtest.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Diffusion Copula PAPER JOB: E3 VaR backtest"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e3_var_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01

# Fix libffi issue - use Mambaforge's working libffi
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)" || true
set +u
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

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "CONDA_PREFIX: ${CONDA_PREFIX:-}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
echo "Testing Python..."
"${PYTHON_BIN}" -c "print('Python works')"
"${PYTHON_BIN}" -c "import numpy; print(f'numpy: {numpy.__version__}')"
"${PYTHON_BIN}" -c "import scipy; print(f'scipy: {scipy.__version__}')"
"${PYTHON_BIN}" -c "import torch; print(f'torch: {torch.__version__}')"


cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e3_var_results.json"
CKPT="${E3_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi

"${PYTHON_BIN}" drafts/scripts/e3_var_backtest.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  --device cuda \
  --window "${E3_WINDOW:-252}" \
  --refit-every "${E3_REFIT_EVERY:-5}" \
  --n-sim "${E3_N_SIM:-5000}" \
  --alphas 0.01 0.05 \
  --max-days "${E3_MAX_DAYS:-0}" \
  --seed 42 \
  --out-json "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e3_var.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e3_var_results.json"

echo ""
echo "Regenerating paper artifacts (force refresh)..."
echo ""

export FIG_PNG_DPI="${FIG_PNG_DPI:-120}"
"${PYTHON_BIN}" drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
  2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_e3_var.log"

echo ""
echo "============================================================================"
echo "DONE: E3 VaR backtest completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "E3 JSON: ${OUT_JSON}"
echo "Paper cache: ${REPO_ROOT}/drafts/paper_outputs/e3_var_results.json"
echo ""
