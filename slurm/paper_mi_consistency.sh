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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/mi_self_consistency_tests_v2.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

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

# Resolve checkpoint (explicit env -> best checkpoint auto-selection).
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
CHECKPOINT="${MI_CONSISTENCY_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
if [ -z "${CHECKPOINT}" ]; then
    CHECKPOINT="$("${PYTHON_BIN}" - <<'PY'
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

echo "Using checkpoint: ${CHECKPOINT:-none (will use KSG only)}"

# Run self-consistency tests (v2 is wired to current codebase/checkpoints)
"${PYTHON_BIN}" scripts/mi_self_consistency_tests_v2.py \
    ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"} \
    --n_samples 10000 \
    --n_trials 5 \
    --seed 42 \
    --dcd-diffusion-steps "${MI_CONS_DIFFUSION_STEPS:-4}" \
    --dcd-diffusion-cfg-scale "${MI_CONS_CFG_SCALE:-1.0}" \
    --dcd-pred-noise-clip "${MI_CONS_PRED_NOISE_CLIP:-1.0}" \
    --dcd-truncation-level "${MI_CONS_TRUNCATION_LEVEL:-1}" \
    --output drafts/tables/tab_self_consistency.tex \
    --json_output results/mi_self_consistency.json \
    --device cuda

echo ""
echo "============================================================================"
echo "Self-consistency tests complete!"
echo "Output: drafts/tables/tab_self_consistency.tex"
echo "JSON: results/mi_self_consistency.json"
echo "============================================================================"
