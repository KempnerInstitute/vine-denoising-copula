#!/bin/bash
#SBATCH --job-name=vdc_paper_e33_cpu
#SBATCH --output=logs/vdc_paper_e33_%j.out
#SBATCH --error=logs/vdc_paper_e33_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_bsabatini_lab
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e33_cpu_timing.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: E33 matched-compute VDC CPU timing (LCT8)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e33_cpu_timing_${TS}_${SLURM_JOB_ID:-nojobid}"
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

OUT_JSON="${RUN_DIR}/results/e33_cpu_timing.json"
# Pinned to the paper checkpoint used by the E22/E25 reference numbers.
CKPT="${E33_CHECKPOINT:-${PAPER_CHECKPOINT:-${OUTPUT_BASE}/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/checkpoints/model_step_190000.pt}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi

"${PYTHON_BIN}" drafts/scripts/e33_cpu_timing.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  --dims ${E33_DIMS:-100 200} \
  --seeds ${E33_SEEDS:-7 17} \
  --n-train "${E33_N_TRAIN:-20000}" \
  --n-test "${E33_N_TEST:-5000}" \
  --theta "${E33_THETA:-2.0}" \
  --m-grid "${E33_M_GRID:-64}" \
  --cpu-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out-json "${OUT_JSON}" \
  --out-tex "drafts/tables/tab_cpu_timing.tex" \
  2>&1 | tee "${RUN_DIR}/logs/e33.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e33_cpu_timing.json"

echo ""
echo "============================================================================"
echo "DONE: E33 matched-compute VDC CPU timing completed at $(date)"
echo "============================================================================"
