#!/bin/bash
#SBATCH --job-name=vdc_paper_mi
#SBATCH --output=logs/vdc_paper_mi_%j.out
#SBATCH --error=logs/vdc_paper_mi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Diffusion Copula (ICML 2026) - MI ESTIMATION JOB
# ============================================================================
# Usage:
#   sbatch slurm/paper_vdc_mi_estimation.sh <estimator>
#
# estimator must be one of:
#   - ksg
#   - dcd
#   - gaussian
#   - infonce
#   - nwj
#   - mine
#   - minde
#   - mist
#
# Writes:
#   results/mi_estimation.json
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

if [ "$#" -ne 1 ]; then
  echo "ERROR: expected exactly 1 estimator argument."
  echo "Usage: sbatch slurm/paper_vdc_mi_estimation.sh <ksg|dcd|gaussian|infonce|nwj|mine|minde|mist>"
  exit 2
fi

EST="$1"
METHOD="mi_${EST}"

echo "============================================================================"
echo "Vine Diffusion Copula PAPER MI ESTIMATION"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Estimator: ${EST}"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_${METHOD}_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

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
  nvidia-smi || true
} | tee "${RUN_DIR}/logs/env.txt"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"

python scripts/mi_estimation.py \
  --estimator "${EST}" \
  --output-base "${OUTPUT_BASE}" \
  --n-samples 5000 \
  --m-true 256 \
  --seed 123 \
  --device cuda \
  --out-json "${RUN_DIR}/results/mi_estimation.json" \
  2>&1 | tee "${RUN_DIR}/logs/mi_estimation.log"

echo ""
echo "============================================================================"
echo "DONE: MI estimation completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Results: ${RUN_DIR}/results/mi_estimation.json"
echo ""
