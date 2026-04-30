#!/bin/bash
#SBATCH --job-name=vdc_paper_rerun_complex
#SBATCH --output=logs/vdc_paper_rerun_complex_%j.out
#SBATCH --error=logs/vdc_paper_rerun_complex_%j.err
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
# Vine Denoising Copula (NeurIPS 2026) - RERUN "COMPLEX" BENCHMARK INTO AN EXISTING RUN
# ============================================================================
# Evaluates a checkpoint on the *complex synthetic* copula suite (X / ring / double-banana),
# writes results back into an existing RUN_DIR under OUTPUT_BASE.
#
# Usage:
#   sbatch slurm/paper_rerun_model_selection_complex.sh /path/to/RUN_DIR /path/to/ckpt.pt [/path/to/ckpt2.pt ...]
#
# Writes:
#   RUN_DIR/results/model_selection_complex.json
#   RUN_DIR/results/model_selection_complex.csv
#   RUN_DIR/logs/model_selection_complex.log
#   RUN_DIR/figures/examples_complex/ (example PDFs)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 2 ]; then
  echo "ERROR: Need RUN_DIR and at least one checkpoint."
  echo "Usage: sbatch slurm/paper_rerun_model_selection_complex.sh /path/to/RUN_DIR /path/to/ckpt.pt [/path/to/ckpt2.pt ...]"
  exit 2
fi

RUN_DIR="$1"
shift

echo "============================================================================"
echo "Vine Denoising Copula PAPER EVAL (rerun): model_selection (COMPLEX SUITE)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "Run Dir: ${RUN_DIR}"
echo "Checkpoints: $*"
echo ""

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${RUN_DIR}/"{results,logs,analysis,figures}
mkdir -p "${RUN_DIR}/figures/examples_complex"

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)" || true
conda activate vdc 2>/dev/null || conda activate diffuse_vine_cop 2>/dev/null || true

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

{
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  echo "Python: $(which python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi || true
} | tee "${RUN_DIR}/logs/env_rerun_model_selection_complex.txt"

cp "$0" "${RUN_DIR}/analysis/slurm_rerun_model_selection_complex.sh"

EVAL_LOG="${RUN_DIR}/logs/model_selection_complex.log"

python scripts/model_selection.py \
  --suite complex \
  --checkpoints "$@" \
  --n-samples 2000 \
  --device cuda \
  --diffusion-steps "${DIFFUSION_STEPS:-200}" \
  --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
  --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
  --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
  --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
  --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
  --write-examples \
  --examples-dir "${RUN_DIR}/figures/examples_complex" \
  --out-json "${RUN_DIR}/results/model_selection_complex.json" \
  --out-csv "${RUN_DIR}/results/model_selection_complex.csv" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "============================================================================"
echo "DONE: model_selection (COMPLEX) rerun completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Results: ${RUN_DIR}/results/model_selection_complex.json"
echo ""
