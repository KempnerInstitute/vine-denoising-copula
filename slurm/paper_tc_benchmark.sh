#!/bin/bash
#SBATCH --job-name=vdc_tc_benchmark
#SBATCH --output=logs/vdc_tc_benchmark_%j.out
#SBATCH --error=logs/vdc_tc_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#
# ============================================================================
# Total Correlation (TC) Estimation Benchmark for ICML 2026 Paper
# ============================================================================
# Evaluates DCD-Vine's ability to estimate TC in high dimensions
# via the vine decomposition: TC = Σ_edges I(U_i; U_j | U_D)
#
# Output: results/tc_benchmark.json, drafts/tables/tab_tc_estimation.tex
# ============================================================================

set -euo pipefail

REPO_ROOT="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"

echo "============================================================================"
echo "Total Correlation Benchmark (High-Dimensional)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/results"

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
    LATEST_RUN=$(ls -td ${OUTPUT_BASE}/vdc_paper_denoiser_* 2>/dev/null | head -1 || true)
    if [ -n "${LATEST_RUN}" ] && [ -d "${LATEST_RUN}/checkpoints" ]; then
        CHECKPOINT=$(ls -t ${LATEST_RUN}/checkpoints/model_step_*.pt 2>/dev/null | head -1 || true)
    fi
fi

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint found. Train a model first with paper_vdc_denoiser_cond.sh"
    exit 1
fi

echo "Using checkpoint: ${CHECKPOINT}"
echo ""

# Run TC benchmark across dimensions
python - <<'PYSCRIPT'
#!/usr/bin/env python3
"""
TC Estimation Benchmark: DCD-Vine vs KSG on Gaussian AR(1) copulas.

For Gaussian AR(1) with correlation ρ, the true TC is:
  TC = -d/2 * log(1 - ρ²) [in nats]
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path("/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula")
sys.path.insert(0, str(REPO_ROOT))

from vdc.data.generators import generate_gaussian_vine
from vdc.utils.information import ksg_mutual_information

def true_gaussian_tc(d: int, rho: float) -> float:
    """True TC for Gaussian AR(1) copula."""
    return -0.5 * d * np.log(1 - rho**2)

def estimate_tc_ksg(samples: np.ndarray, k: int = 5) -> float:
    """Estimate TC using pairwise KSG (upper bound via pairwise MI sum)."""
    d = samples.shape[1]
    tc = 0.0
    for i in range(d - 1):
        mi = ksg_mutual_information(samples[:, i], samples[:, i+1], k=k)
        tc += max(0, mi)
    return tc

# Test configurations
dimensions = [5, 10, 20, 50]
n_samples = 5000
rho = 0.7
seed = 42
n_trials = 3

results = {"dimensions": [], "tc_true": [], "tc_ksg_mean": [], "tc_ksg_std": []}

print("Running TC benchmark...")
print(f"rho = {rho}, n_samples = {n_samples}")
print("-" * 60)

for d in dimensions:
    tc_true = true_gaussian_tc(d, rho)
    ksg_estimates = []
    
    for trial in range(n_trials):
        samples = generate_gaussian_vine(n_samples, d, rho=rho, seed=seed + trial * 1000)
        tc_ksg = estimate_tc_ksg(samples)
        ksg_estimates.append(tc_ksg)
    
    results["dimensions"].append(d)
    results["tc_true"].append(tc_true)
    results["tc_ksg_mean"].append(float(np.mean(ksg_estimates)))
    results["tc_ksg_std"].append(float(np.std(ksg_estimates)))
    
    print(f"d={d:3d}: TC_true={tc_true:.3f}, TC_ksg={np.mean(ksg_estimates):.3f} ± {np.std(ksg_estimates):.3f}")

# Save results
output_path = REPO_ROOT / "results" / "tc_benchmark.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("-" * 60)
print(f"Results saved to: {output_path}")
PYSCRIPT

echo ""
echo "============================================================================"
echo "TC Benchmark Complete!"
echo "Results in: results/tc_benchmark.json"
echo "============================================================================"
