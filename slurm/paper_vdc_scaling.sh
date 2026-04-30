#!/bin/bash
#SBATCH --job-name=vdc_paper_scaling
#SBATCH --output=logs/vdc_paper_scaling_%j.out
#SBATCH --error=logs/vdc_paper_scaling_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Denoising Copula (NeurIPS 2026) - E2 SCALING JOB
# ============================================================================
# Runs the scaling benchmark (vine build time vs dimension) and writes:
#   RUN_DIR/results/e2_scaling_results.json
# and copies it into the paper artifacts cache:
#   drafts/paper_outputs/e2_scaling_results.json
#
# Then regenerates paper artifacts so `drafts/figures/scaling_time_vs_d.pdf` becomes real.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: scaling (E2)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_scaling_${TS}_${SLURM_JOB_ID:-nojobid}"
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
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true
git status --porcelain 2>/dev/null | tee "${RUN_DIR}/analysis/git_status_porcelain.txt" || true

echo ""
echo "Running scaling benchmark..."
echo ""

SCALING_LOG="${RUN_DIR}/logs/e2_scaling.log"

python drafts/scripts/e2_scaling.py \
  --output-base "${OUTPUT_BASE}" \
  --device cuda \
  --dims 2 5 10 20 50 100 200 500 \
  --n-train 2000 \
  --rho 0.6 \
  --repeats 1 \
  --ours-mode both \
  --ours-edge-batch-size 256 \
  --pyvine both \
  --pyvine-param-max-d 200 \
  --pyvine-nonpar-max-d 200 \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --out-json "${RUN_DIR}/results/e2_scaling_results.json" \
  2>&1 | tee "${SCALING_LOG}"

cp "${RUN_DIR}/results/e2_scaling_results.json" "drafts/paper_outputs/e2_scaling_results.json"

echo ""
echo "Regenerating paper artifacts (force refresh)..."
echo ""

# Avoid OOMs during PNG rasterization for large multi-panel figures.
export FIG_PNG_DPI="${FIG_PNG_DPI:-120}"

python drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
  2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_scaling.log"

echo ""
echo "============================================================================"
echo "DONE: scaling (E2) completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Scaling JSON: ${RUN_DIR}/results/e2_scaling_results.json"
echo "Paper cache:  ${REPO_ROOT}/drafts/paper_outputs/e2_scaling_results.json"
echo ""
