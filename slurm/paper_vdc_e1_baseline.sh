#!/bin/bash
#SBATCH --job-name=vdc_paper_e1_base
#SBATCH --output=logs/vdc_paper_e1_base_%j.out
#SBATCH --error=logs/vdc_paper_e1_base_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Denoising Copula (NeurIPS 2026) - E1 BASELINE (BIVARIATE) JOB
# ============================================================================
# Usage:
#   sbatch slurm/paper_vdc_e1_baseline.sh <baseline_name>
#
# baseline_name must be one of:
#   - histogram
#   - kde_probit
#   - pyvine_param
#   - pyvine_nonpar
#
# Creates a timestamped run directory under OUTPUT_BASE and writes:
#   results/model_selection.json
#   results/model_selection.csv
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

if [ "$#" -ne 1 ]; then
  echo "ERROR: expected exactly 1 baseline argument."
  echo "Usage: sbatch slurm/paper_vdc_e1_baseline.sh <baseline_name>"
  exit 2
fi

BASELINE="$1"

echo "============================================================================"
echo "Vine Denoising Copula PAPER E1 BASELINE"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Baseline: ${BASELINE}"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_${BASELINE}_${TS}_${SLURM_JOB_ID:-nojobid}"
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
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

{
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  echo "Python: $(which python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
} | tee "${RUN_DIR}/logs/env.txt"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"

EVAL_LOG="${RUN_DIR}/logs/model_selection_baseline.log"

python scripts/model_selection.py \
  --baselines "${BASELINE}" \
  --n-samples 2000 \
  --device cpu \
  --out-json "${RUN_DIR}/results/model_selection.json" \
  --out-csv "${RUN_DIR}/results/model_selection.csv" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "Running baseline on COMPLEX suite..."
echo ""

EVAL_COMPLEX_LOG="${RUN_DIR}/logs/model_selection_baseline_complex.log"

python scripts/model_selection.py \
  --suite complex \
  --baselines "${BASELINE}" \
  --n-samples 2000 \
  --device cpu \
  --out-json "${RUN_DIR}/results/model_selection_complex.json" \
  --out-csv "${RUN_DIR}/results/model_selection_complex.csv" \
  2>&1 | tee "${EVAL_COMPLEX_LOG}"

echo ""
echo "============================================================================"
echo "DONE: E1 baseline completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Results: ${RUN_DIR}/results/model_selection.json"
echo ""
