#!/bin/bash
#SBATCH --job-name=vdc_mi_consistency
#SBATCH --output=logs/vdc_mi_consistency_%j.out
#SBATCH --error=logs/vdc_mi_consistency_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#
# ============================================================================
# MI Self-Consistency Tests for ICML 2026 Paper
# ============================================================================
# Tests Data Processing Inequality, Additivity, and Monotone Invariance
# for DCD-Vine vs baseline MI estimators (KSG, etc.)
#
# Output: drafts/tables/tab_self_consistency.tex
# ============================================================================

set -euo pipefail

REPO_ROOT="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"

echo "============================================================================"
echo "MI Self-Consistency Tests"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Create logs directory
mkdir -p "${REPO_ROOT}/logs"

# Environment setup
module purge
module load cuda/12.2.0-fasrc01

# Fix libffi issue
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"
eval "$(conda shell.bash hook)" || true
conda activate vdc 2>/dev/null || conda activate diffuse_vine_cop 2>/dev/null || true

cd "${REPO_ROOT}"

# Find latest checkpoint
CHECKPOINT=""
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
if [ -d "${OUTPUT_BASE}" ]; then
    # Find latest denoiser run
    LATEST_RUN=$(ls -td ${OUTPUT_BASE}/vdc_paper_denoiser_* 2>/dev/null | head -1 || true)
    if [ -n "${LATEST_RUN}" ] && [ -d "${LATEST_RUN}/checkpoints" ]; then
        CHECKPOINT=$(ls -t ${LATEST_RUN}/checkpoints/model_step_*.pt 2>/dev/null | head -1 || true)
    fi
fi

echo "Using checkpoint: ${CHECKPOINT:-none (will use KSG only)}"

# Run self-consistency tests
python scripts/mi_self_consistency_tests.py \
    ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"} \
    --n_samples 10000 \
    --n_trials 5 \
    --seed 42 \
    --output drafts/tables/tab_self_consistency.tex \
    --json_output results/mi_self_consistency.json \
    --device cuda

echo ""
echo "============================================================================"
echo "Self-consistency tests complete!"
echo "Output: drafts/tables/tab_self_consistency.tex"
echo "JSON: results/mi_self_consistency.json"
echo "============================================================================"
