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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/tc_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Total Correlation Benchmark (High-Dimensional)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output base: ${OUTPUT_BASE}"

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/results"

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

# Resolve checkpoint (prefer explicit TC_CHECKPOINT, else auto-select by best mean_ise).
CHECKPOINT="${TC_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
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

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint found. Set TC_CHECKPOINT or run paper training/model-selection jobs first."
    exit 1
fi

echo "Using checkpoint: ${CHECKPOINT}"
echo ""

OUT_JSON="${REPO_ROOT}/results/tc_benchmark.json"
DIMS="${TC_DIMS:-5 10 20 50}"

CMD=(
  "${PYTHON_BIN}" scripts/tc_benchmark.py
  --checkpoint "${CHECKPOINT}"
  --device cuda
  --rho "${TC_RHO:-0.7}"
  --dims ${DIMS}
  --n "${TC_N:-5000}"
  --n-trials "${TC_TRIALS:-3}"
  --seed "${TC_SEED:-42}"
  --ksg-k "${TC_KSG_K:-5}"
  --diffusion-steps "${TC_DIFFUSION_STEPS:-4}"
  --diffusion-cfg-scale "${TC_CFG_SCALE:-1.0}"
  --diffusion-pred-noise-clip "${TC_PRED_NOISE_CLIP:-1.0}"
  --truncation-level "${TC_TRUNCATION_LEVEL:-1}"
  --batch-edges
  --edge-batch-size "${TC_EDGE_BATCH_SIZE:-256}"
  --out-json "${OUT_JSON}"
)

if [ "${TC_INCLUDE_MINE:-0}" = "1" ]; then
  CMD+=(--include-mine)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "============================================================================"
echo "TC Benchmark Complete!"
echo "Results in: ${OUT_JSON}"
echo "============================================================================"
