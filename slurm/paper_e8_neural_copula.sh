#!/bin/bash
#SBATCH --job-name=vdc_paper_e8_neural
#SBATCH --output=logs/vdc_paper_e8_neural_%j.out
#SBATCH --error=logs/vdc_paper_e8_neural_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Denoising Copula (NeurIPS 2026) - E8 Neural Copula Benchmark
# ============================================================================
# Focused rebuttal benchmark comparing VDC to ACNet on matched Archimedean
# pair-copula edge tasks. Uses the existing paper VDC artifact by default,
# so the job only needs to train/evaluate the ACNet side.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e8_neural_copula_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: E8 neural copula benchmark"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e8_neural_${TS}_${SLURM_JOB_ID:-nojobid}"
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

if [[ "${PYTHON_BIN}" != */* ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found in PATH: ${PYTHON_BIN}"
  exit 3
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"
echo "Testing Python..."
"${PYTHON_BIN}" -c "import sys, torch, scipy; print(sys.executable); print(f'torch: {torch.__version__}, cuda={torch.cuda.is_available()}'); print(f'scipy: {scipy.__version__}')"

ACNET_ROOT="${E8_ACNET_ROOT:-${REPO_ROOT}/external/ACNet}"
if [ ! -d "${ACNET_ROOT}" ]; then
  echo "Cloning ACNet into ${ACNET_ROOT}"
  mkdir -p "$(dirname "${ACNET_ROOT}")"
  git clone --depth 1 https://github.com/lingchunkai/ACNet.git "${ACNET_ROOT}"
fi

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

SEED_TAG="${E8_SEED:-0}"
OUT_JSON="${RUN_DIR}/results/e8_neural_copula_results_seed${SEED_TAG}.json"
EXTRA_ARGS=()
if [ -n "${E8_VDC_ARTIFACT:-}" ]; then
  EXTRA_ARGS+=(--vdc-artifact "${E8_VDC_ARTIFACT}")
fi
if [ -n "${E8_FAMILIES:-}" ]; then
  # Space-separated family names, e.g. "clayton frank joe"
  read -r -a E8_FAMILY_ARR <<< "${E8_FAMILIES}"
  EXTRA_ARGS+=(--families "${E8_FAMILY_ARR[@]}")
fi

"${PYTHON_BIN}" drafts/scripts/e8_neural_copula_benchmark.py \
  --acnet-root "${ACNET_ROOT}" \
  --device cuda \
  --acnet-device "${E8_ACNET_DEVICE:-cpu}" \
  --n-train "${E8_N_TRAIN:-2000}" \
  --n-test "${E8_N_TEST:-2000}" \
  --acnet-epochs "${E8_EPOCHS:-300}" \
  --acnet-batch-size "${E8_BATCH_SIZE:-200}" \
  --acnet-lr "${E8_LR:-1e-3}" \
  --acnet-depth "${E8_DEPTH:-2}" \
  --acnet-widths ${E8_WIDTHS:-10 10} \
  --acnet-eval-batch-size "${E8_EVAL_BATCH_SIZE:-128}" \
  --seed "${E8_SEED:-0}" \
  "${EXTRA_ARGS[@]}" \
  --out "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e8_neural_copula.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e8_neural_copula_results_seed${SEED_TAG}.json" || true
cp "${OUT_JSON}" "drafts/paper_outputs/e8_neural_copula_results.json" || true

echo ""
echo "============================================================================"
echo "DONE: E8 neural copula benchmark completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "E8 JSON: ${OUT_JSON}"
echo ""
