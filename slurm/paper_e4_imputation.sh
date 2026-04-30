#!/bin/bash
#SBATCH --job-name=vdc_paper_impute
#SBATCH --output=logs/vdc_paper_impute_%j.out
#SBATCH --error=logs/vdc_paper_impute_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=256GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Denoising Copula (NeurIPS 2026) - E4 IMPUTATION JOB
# ============================================================================
# Produces:
#   RUN_DIR/results/e4_imputation_results.json
# and copies it to:
#   drafts/paper_outputs/e4_imputation_results.json
# then regenerates paper artifacts.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e4_imputation_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: E4 imputation"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e4_imputation_${TS}_${SLURM_JOB_ID:-nojobid}"
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
echo "E4 params: vine_type=${E4_VINE_TYPE:-dvine} missing_frac=${E4_MISSING_FRAC:-0.20} n_eval=${E4_N_EVAL:-500} candidate_pool=${E4_CANDIDATE_POOL:-20000} kernel_h=${E4_KERNEL_H:-0.05} seed=${E4_SEED:-42}"
echo "Testing Python..."
"${PYTHON_BIN}" -c "import numpy, scipy, torch; print(f'numpy={numpy.__version__}, scipy={scipy.__version__}, torch={torch.__version__}')"


cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e4_imputation_results.json"
CKPT="${E4_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi

"${PYTHON_BIN}" drafts/scripts/e4_imputation_benchmark.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  --device cuda \
  --vine-type "${E4_VINE_TYPE:-dvine}" \
  --datasets power gas credit \
  --missing-frac "${E4_MISSING_FRAC:-0.20}" \
  --n-eval "${E4_N_EVAL:-500}" \
  --candidate-pool "${E4_CANDIDATE_POOL:-20000}" \
  --kernel-h "${E4_KERNEL_H:-0.05}" \
  --max-train "${E4_MAX_TRAIN:-200000}" \
  --max-test "${E4_MAX_TEST:-50000}" \
  --seed "${E4_SEED:-42}" \
  --out-json "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e4_imputation.log"

if [ "${E4_COPY_TO_PAPER:-1}" = "1" ]; then
  cp "${OUT_JSON}" "drafts/paper_outputs/e4_imputation_results.json"
fi

if [ "${E4_REGENERATE_ARTIFACTS:-1}" = "1" ]; then
  echo ""
  echo "Regenerating paper artifacts (force refresh)..."
  echo ""

  export FIG_PNG_DPI="${FIG_PNG_DPI:-120}"
  "${PYTHON_BIN}" drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
    2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_e4_imputation.log"
fi

echo ""
echo "============================================================================"
echo "DONE: E4 imputation completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "E4 JSON: ${OUT_JSON}"
echo "Paper cache: ${REPO_ROOT}/drafts/paper_outputs/e4_imputation_results.json"
echo ""
