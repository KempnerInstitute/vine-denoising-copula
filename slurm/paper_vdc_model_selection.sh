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
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Diffusion Copula (ICML 2026) - MODEL SELECTION / EVAL JOB
# ============================================================================
# Usage:
#   sbatch slurm/paper_vdc_model_selection.sh /path/to/ckpt1.pt /path/to/ckpt2.pt ...
#
# Creates a timestamped run directory under OUTPUT_BASE and writes:
#   results/model_selection.json
#   results/model_selection.csv
# ============================================================================

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

if [ "$#" -lt 1 ]; then
  echo "ERROR: No checkpoints provided."
  echo "Usage: sbatch slurm/paper_vdc_model_selection.sh /path/to/ckpt1.pt [/path/to/ckpt2.pt ...]"
  exit 2
fi

echo "============================================================================"
echo "Vine Diffusion Copula PAPER EVAL: model_selection"
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
conda activate vdc 2>/dev/null || conda activate diffuse_vine_cop 2>/dev/null || true

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  echo "Python: $(which python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi || true
} | tee "${RUN_DIR}/logs/env.txt"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"

EVAL_LOG="${RUN_DIR}/logs/model_selection.log"

python scripts/model_selection.py \
  --checkpoints "$@" \
  --n-samples 2000 \
  --device cuda \
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

