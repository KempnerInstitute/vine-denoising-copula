#!/bin/bash
#SBATCH --job-name=vdc_paper_select
#SBATCH --output=logs/vdc_paper_select_%j.out
#SBATCH --error=logs/vdc_paper_select_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Denoising Copula (ICML 2026) - MODEL SELECTION / EVAL JOB
# ============================================================================
# Usage:
#   sbatch slurm/paper_vdc_model_selection.sh /path/to/ckpt1.pt /path/to/ckpt2.pt ...
#
# Creates a timestamped run directory under OUTPUT_BASE and writes:
#   results/model_selection.json
#   results/model_selection.csv
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/model_selection.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
SUITE="${SUITE:-standard}"
N_SAMPLES="${N_SAMPLES:-2000}"
WRITE_EXAMPLES="${WRITE_EXAMPLES:-1}"

if [ "$#" -lt 1 ]; then
  echo "ERROR: No checkpoints provided."
  echo "Usage: sbatch slurm/paper_vdc_model_selection.sh /path/to/ckpt1.pt [/path/to/ckpt2.pt ...]"
  exit 2
fi

echo "============================================================================"
echo "Vine Denoising Copula PAPER EVAL: model_selection"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "Output Base: ${OUTPUT_BASE}"

# Repo-local logs directory for SLURM stdout/err targets (must exist)
mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_model_selection_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,figures,checkpoints,analysis}

echo "Run Dir: ${RUN_DIR}"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)" || true
set +u
if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
elif [ -n "${VDC_CONDA_ENV_PATH:-}" ]; then
  conda activate "${VDC_CONDA_ENV_PATH}"
  PYTHON_BIN="python"
elif conda activate vdc 2>/dev/null; then
  PYTHON_BIN="python"
elif conda activate diffuse_vine_cop 2>/dev/null; then
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

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# New env var name (old one is deprecated in recent PyTorch)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
unset PYTHONHOME || true

# Some cluster images don't have system libbz2; ensure we can load it from conda.
# (Fixes: ImportError: libbz2.so.1.0: cannot open shared object file)
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

{
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  echo "Python: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi || true
} | tee "${RUN_DIR}/logs/env.txt"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"

EVAL_LOG="${RUN_DIR}/logs/model_selection.log"
EXAMPLE_ARGS=()
if [ "${WRITE_EXAMPLES}" = "1" ]; then
  EXAMPLE_ARGS=(--write-examples --examples-dir "${RUN_DIR}/figures/examples")
fi

"${PYTHON_BIN}" scripts/model_selection.py \
  --checkpoints "$@" \
  --suite "${SUITE}" \
  --n-samples "${N_SAMPLES}" \
  --device cuda \
  --diffusion-steps "${DIFFUSION_STEPS:-200}" \
  --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
  --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
  --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
  --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
  --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
  --diffusion-seed-base "${DIFFUSION_SEED_BASE:-123}" \
  "${EXAMPLE_ARGS[@]}" \
  --out-json "${RUN_DIR}/results/model_selection.json" \
  --out-csv "${RUN_DIR}/results/model_selection.csv" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "============================================================================"
echo "DONE: model_selection completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Results: ${RUN_DIR}/results/model_selection.json"
echo ""
