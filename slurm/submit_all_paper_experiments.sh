#!/bin/bash
# ============================================================================
# Master Script: Submit All Paper Experiments for NeurIPS 2026
# ============================================================================
# This script submits all E2-E5 experiments and TC/MI benchmarks that are
# needed to populate the paper tables and figures.
#
# Usage:
#   ./slurm/submit_all_paper_experiments.sh
#
# Dependencies: At least one completed enhanced-denoiser checkpoint run must
# exist in OUTPUT_BASE (or provide PAPER_CHECKPOINT explicitly).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================================"
echo "Submitting All Paper Experiments for NeurIPS 2026"
echo "============================================================================"
echo "Date: $(date)"
echo ""

# Resolve one shared checkpoint for all paper benchmarks.
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
PARTITION="${PARTITION:-kempner_eng}"
CHECKPOINT="${PAPER_CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
    CHECKPOINT="$(python - <<'PY'
import os
from pathlib import Path
import sys

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))
from vdc.utils.paper import choose_best_checkpoint

output_base = Path(os.environ.get("OUTPUT_BASE", "/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula"))
ckpt = choose_best_checkpoint(
    output_bases=[output_base],
    preferred_methods=["denoiser_cond_enhanced", "denoiser_cond"],
    metric="mean_ise",
    prefer_joint=True,
)
if ckpt is None:
    import torch

    def _is_supported(path: Path) -> bool:
        try:
            obj = torch.load(str(path), map_location="cpu")
            cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
            mt = str(cfg.get("model", {}).get("type", "")) if isinstance(cfg, dict) else ""
            return mt == "denoiser"
        except Exception:
            return False

    local = []
    for root_name in ("checkpoints", "archive/checkpoints", "archive/checkpoints_old"):
        root = repo_root / root_name
        if root.exists():
            local.extend(root.glob("**/model_step_*.pt"))
    for cand in sorted(local, key=lambda p: p.stat().st_mtime, reverse=True):
        if _is_supported(cand):
            ckpt = cand
            break
print("" if ckpt is None else str(ckpt))
PY
)"
fi
if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Could not resolve a valid enhanced-denoiser checkpoint."
    echo "Set PAPER_CHECKPOINT=/path/to/model_step_*.pt or run slurm/paper_train_enhanced.sh first."
    exit 1
fi
echo "Using checkpoint: ${CHECKPOINT}"
echo "Using output base: ${OUTPUT_BASE}"
echo "Using partition: ${PARTITION}"
export OUTPUT_BASE
export PAPER_CHECKPOINT="${CHECKPOINT}"
echo ""

# Create logs directory
mkdir -p "${REPO_ROOT}/logs"

# Track job IDs for dependencies
declare -a JOB_IDS

# Submit E2: UCI Benchmark
echo "Submitting E2: UCI Benchmark..."
E2_JOB=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" slurm/paper_e2_uci.sh)
JOB_IDS+=("${E2_JOB}")
echo "  Job ID: ${E2_JOB}"

# Submit E3: VaR Backtest
echo "Submitting E3: VaR Backtest..."
E3_JOB=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" slurm/paper_e3_var.sh)
JOB_IDS+=("${E3_JOB}")
echo "  Job ID: ${E3_JOB}"

# Submit E4: Imputation Benchmark
echo "Submitting E4: Imputation Benchmark..."
E4_JOB=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" slurm/paper_e4_imputation.sh)
JOB_IDS+=("${E4_JOB}")
echo "  Job ID: ${E4_JOB}"

# Submit E5: Anomaly Detection
echo "Submitting E5: Anomaly Detection..."
E5_JOB=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" slurm/paper_e5_anomaly.sh)
JOB_IDS+=("${E5_JOB}")
echo "  Job ID: ${E5_JOB}"

# Submit TC Benchmark
echo "Submitting TC Benchmark..."
TC_JOB=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" slurm/paper_tc_benchmark.sh)
JOB_IDS+=("${TC_JOB}")
echo "  Job ID: ${TC_JOB}"

# Submit MI Benchmark
echo "Submitting MI Benchmark..."
MI_JOB=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" slurm/paper_mi_benchmark.sh)
JOB_IDS+=("${MI_JOB}")
echo "  Job ID: ${MI_JOB}"

# Submit MI Self-Consistency
echo "Submitting MI Self-Consistency..."
MI_SC_JOB=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" slurm/paper_mi_consistency.sh)
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
