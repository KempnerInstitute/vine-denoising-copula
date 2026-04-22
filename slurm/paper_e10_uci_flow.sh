#!/bin/bash
#SBATCH --job-name=vdc_paper_e10_flow
#SBATCH --output=logs/vdc_paper_e10_flow_%j.out
#SBATCH --error=logs/vdc_paper_e10_flow_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=192GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e10_uci_flow_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: E10 UCI flow benchmark"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e10_uci_flow_${TS}_${SLURM_JOB_ID:-nojobid}"
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

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

SEED_TAG="${E10_SEED:-42}"
OUT_JSON="${RUN_DIR}/results/e10_uci_flow_results_seed${SEED_TAG}.json"

"${PYTHON_BIN}" drafts/scripts/e10_uci_flow_benchmark.py \
  --device cuda \
  --datasets ${E10_DATASETS:-power gas hepmass miniboone} \
  --max-train "${E10_MAX_TRAIN:-200000}" \
  --max-test "${E10_MAX_TEST:-50000}" \
  --seed "${SEED_TAG}" \
  --epochs "${E10_EPOCHS:-25}" \
  --batch-size "${E10_BATCH_SIZE:-2048}" \
  --eval-batch-size "${E10_EVAL_BATCH_SIZE:-4096}" \
  --lr "${E10_LR:-1e-3}" \
  --val-fraction "${E10_VAL_FRACTION:-0.1}" \
  --patience "${E10_PATIENCE:-5}" \
  --num-layers "${E10_NUM_LAYERS:-8}" \
  --hidden-dim "${E10_HIDDEN_DIM:-128}" \
  --hidden-layers "${E10_HIDDEN_LAYERS:-2}" \
  --out-json "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e10_uci_flow.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e10_uci_flow_results_seed${SEED_TAG}.json"
cp "${OUT_JSON}" "drafts/paper_outputs/e10_uci_flow_results.json"

echo ""
echo "Regenerating paper artifacts..."
echo ""
"${PYTHON_BIN}" drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
  2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_e10.log"

echo ""
echo "============================================================================"
echo "DONE: E10 UCI flow benchmark completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "E10 JSON: ${OUT_JSON}"
echo "Paper cache: ${REPO_ROOT}/drafts/paper_outputs/e10_uci_flow_results_seed${SEED_TAG}.json"
echo ""
