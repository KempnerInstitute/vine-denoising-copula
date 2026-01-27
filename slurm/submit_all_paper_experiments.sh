#!/bin/bash
# ============================================================================
# Master Script: Submit All Paper Experiments for ICML 2026
# ============================================================================
# This script submits all E2-E5 experiments and TC/MI benchmarks that are
# needed to populate the paper tables and figures.
#
# Usage:
#   ./slurm/submit_all_paper_experiments.sh
#
# Dependencies: The denoiser_cond checkpoint must already exist.
# ============================================================================

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"
cd "${REPO_ROOT}"

echo "============================================================================"
echo "Submitting All Paper Experiments for ICML 2026"
echo "============================================================================"
echo "Date: $(date)"
echo ""

# Verify checkpoint exists
CHECKPOINT="/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_20260113_235331_55240324/checkpoints/model_step_150000.pt"
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    echo "Please train the denoiser model first."
    exit 1
fi
echo "Using checkpoint: ${CHECKPOINT}"
echo ""

# Create logs directory
mkdir -p "${REPO_ROOT}/logs"

# Track job IDs for dependencies
declare -a JOB_IDS

# Submit E2: UCI Benchmark
echo "Submitting E2: UCI Benchmark..."
E2_JOB=$(sbatch --parsable slurm/paper_e2_uci.sh)
JOB_IDS+=("${E2_JOB}")
echo "  Job ID: ${E2_JOB}"

# Submit E3: VaR Backtest
echo "Submitting E3: VaR Backtest..."
E3_JOB=$(sbatch --parsable slurm/paper_e3_var.sh)
JOB_IDS+=("${E3_JOB}")
echo "  Job ID: ${E3_JOB}"

# Submit E4: Imputation Benchmark
echo "Submitting E4: Imputation Benchmark..."
E4_JOB=$(sbatch --parsable slurm/paper_e4_imputation.sh)
JOB_IDS+=("${E4_JOB}")
echo "  Job ID: ${E4_JOB}"

# Submit E5: Anomaly Detection
echo "Submitting E5: Anomaly Detection..."
E5_JOB=$(sbatch --parsable slurm/paper_e5_anomaly.sh)
JOB_IDS+=("${E5_JOB}")
echo "  Job ID: ${E5_JOB}"

# Submit TC Benchmark
echo "Submitting TC Benchmark..."
TC_JOB=$(sbatch --parsable slurm/paper_tc_benchmark.sh)
JOB_IDS+=("${TC_JOB}")
echo "  Job ID: ${TC_JOB}"

# Submit MI Benchmark
echo "Submitting MI Benchmark..."
MI_JOB=$(sbatch --parsable slurm/paper_mi_benchmark.sh)
JOB_IDS+=("${MI_JOB}")
echo "  Job ID: ${MI_JOB}"

# Submit MI Self-Consistency
echo "Submitting MI Self-Consistency..."
MI_SC_JOB=$(sbatch --parsable slurm/paper_mi_consistency.sh)
JOB_IDS+=("${MI_SC_JOB}")
echo "  Job ID: ${MI_SC_JOB}"

echo ""
echo "============================================================================"
echo "All jobs submitted!"
echo "============================================================================"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: ${REPO_ROOT}/logs/"
echo ""
echo "After all jobs complete, regenerate paper artifacts with:"
echo "  python drafts/scripts/paper_artifacts.py all --force"
echo "============================================================================"
