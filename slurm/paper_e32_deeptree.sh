#!/bin/bash
#SBATCH --job-name=vdc_paper_e32_deeptree
#SBATCH --output=logs/vdc_paper_e32_%j.out
#SBATCH --error=logs/vdc_paper_e32_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner
#SBATCH --account=kempner_bsabatini_lab
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e32_deep_tree_recovery.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: E32 deep-tree dependence recovery (LCT8)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e32_deeptree_${TS}_${SLURM_JOB_ID:-nojobid}"
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

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e32_deep_tree_recovery.json"
CKPT="${E32_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi

"${PYTHON_BIN}" drafts/scripts/e32_deep_tree_recovery.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  --device cuda \
  --d "${E32_D:-30}" \
  --theta "${E32_THETA:-2.0}" \
  --n-train "${E32_N_TRAIN:-20000}" \
  --n-test "${E32_N_TEST:-5000}" \
  --seeds ${E32_SEEDS:-7 17 42} \
  --m-grid "${E32_M_GRID:-64}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out-json "${OUT_JSON}" \
  --out-tex "drafts/tables/tab_deep_tree_recovery.tex" \
  2>&1 | tee "${RUN_DIR}/logs/e32.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e32_deep_tree_recovery.json"

echo ""
echo "============================================================================"
echo "DONE: E32 deep-tree dependence recovery completed at $(date)"
echo "============================================================================"
