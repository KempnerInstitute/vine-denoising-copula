#!/bin/bash
#SBATCH --job-name=vdc_paper_denoiser
#SBATCH --output=logs/vdc_paper_denoiser_%j.out
#SBATCH --error=logs/vdc_paper_denoiser_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Diffusion Copula (ICML 2026) - PAPER RESULTS JOB
# ============================================================================
# Variant: Noise-conditioned single-pass denoiser (one-shot at t=0)
#
# Expected runtime: ~4-10 hours on 4x H100 (depends on max_steps + logging)
#
# Output Directory Structure (created under OUTPUT_BASE):
#   vdc_paper_denoiser_cond_{timestamp}_{SLURM_JOB_ID}/
#     results/      - JSON/CSV metrics (model_selection)
#     logs/         - train.log, eval.log, env.txt
#     figures/      - symlink to training visualizations
#     checkpoints/  - model_step_*.pt checkpoints + visualizations/
#     analysis/     - exact config copy + git info + slurm script copy
# ============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT=""
if [ -n "${VDC_REPO_ROOT:-}" ] && [ -f "${VDC_REPO_ROOT}/scripts/train_unified.py" ]; then
  REPO_ROOT="${VDC_REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/train_unified.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -f "${PWD}/scripts/train_unified.py" ]; then
  REPO_ROOT="${PWD}"
elif [ -f "${SCRIPT_DIR}/../scripts/train_unified.py" ]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [ -z "${REPO_ROOT}" ]; then
  echo "ERROR: unable to resolve repository root"
  echo "Set VDC_REPO_ROOT to the repo checkout containing scripts/train_unified.py"
  exit 2
fi
CONFIG_SRC="${CONFIG_SRC:-${REPO_ROOT}/configs/train/denoiser_cond.yaml}"
MODEL_TYPE="${MODEL_TYPE:-denoiser}"
METHOD_TAG="${METHOD_TAG:-denoiser_cond}"

echo "============================================================================"
echo "Vine Diffusion Copula PAPER JOB: ${METHOD_TAG}"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"

# Output base (override by exporting OUTPUT_BASE before sbatch)
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
echo "Output Base: ${OUTPUT_BASE}"

# Repo-local logs directory for SLURM stdout/err targets (must exist)
mkdir -p "${REPO_ROOT}/logs"

# Create paper run directory
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_${METHOD_TAG}_${TS}_${SLURM_JOB_ID:-nojobid}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${RUN_DIR}/"{results,logs,figures,checkpoints,analysis}

echo "Run Dir: ${RUN_DIR}"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01

# Prefer the stable base Mambaforge interpreter used by the successful paper
# benchmark jobs. Some inherited netscratch conda environments fail on H200
# nodes before Python can initialize its codec table.
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
else
  PYTHON_BIN="/n/sw/Mambaforge-23.11.0-0/bin/python"
fi

if [ "${PYTHON_BIN}" != "python" ] && [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: VDC_PYTHON_BIN is not executable: ${PYTHON_BIN}"
  exit 3
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# New env var names (old ones are deprecated in recent PyTorch)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Do not disable the user site by default. Some working paper environments
# (notably vdc_paper) rely on user-site packages such as torch.
if [ -n "${PYTHONNOUSERSITE:-}" ]; then
  export PYTHONNOUSERSITE
else
  unset PYTHONNOUSERSITE || true
fi
unset PYTHONHOME || true

# Some cluster images don't have system libbz2; ensure we can load it from conda.
# (Fixes: ImportError: libbz2.so.1.0: cannot open shared object file)
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

{
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  echo "Python: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version() if torch.cuda.is_available() else None)"
  nvidia-smi || true
} | tee "${RUN_DIR}/logs/env.txt"

# Save reproducibility artifacts
cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true
git status --porcelain 2>/dev/null | tee "${RUN_DIR}/analysis/git_status_porcelain.txt" || true

# Write an exact config copy with a top-level checkpoint_dir (used by scripts/train_unified.py)
CONFIG_OUT="${RUN_DIR}/analysis/train_config.yaml"
export CONFIG_SRC CONFIG_OUT CHECKPOINT_DIR
"${PYTHON_BIN}" - <<PY
import os
from pathlib import Path
import yaml

src = Path(os.environ["CONFIG_SRC"])
dst = Path(os.environ["CONFIG_OUT"])
cfg = yaml.safe_load(src.read_text())
cfg["checkpoint_dir"] = os.environ["CHECKPOINT_DIR"]
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"Wrote config: {dst}")
PY

echo ""
echo "Running training..."
echo ""

GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
TRAIN_LOG="${RUN_DIR}/logs/train.log"

set +e
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPUS}" \
  scripts/train_unified.py \
  --config "${CONFIG_OUT}" \
  --model-type "${MODEL_TYPE}" \
  2>&1 | tee "${TRAIN_LOG}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if [ "${TRAIN_RC}" -ne 0 ]; then
  echo "Training failed with exit code ${TRAIN_RC}" | tee -a "${TRAIN_LOG}"
  exit "${TRAIN_RC}"
fi

# Copy/symlink training visualizations into figures/
if [ -d "${CHECKPOINT_DIR}/visualizations" ]; then
  ln -s "${CHECKPOINT_DIR}/visualizations" "${RUN_DIR}/figures/training_visualizations" 2>/dev/null || true
fi

# Pick latest checkpoint
CKPT="$(ls -1 "${CHECKPOINT_DIR}"/model_step_*.pt 2>/dev/null | sort -V | tail -1)"
if [ -z "${CKPT}" ]; then
  echo "ERROR: no checkpoint found in ${CHECKPOINT_DIR}" | tee -a "${TRAIN_LOG}"
  exit 2
fi
echo "${CKPT}" > "${RUN_DIR}/results/checkpoint_path.txt"
echo "Latest checkpoint: ${CKPT}"

echo ""
echo "Running model selection evaluation..."
echo ""

EVAL_LOG="${RUN_DIR}/logs/model_selection.log"
mkdir -p "${RUN_DIR}/figures/examples"

# NOTE: evaluation is helpful but should not fail the overall training job.
set +e
"${PYTHON_BIN}" scripts/model_selection.py \
  --checkpoints "${CKPT}" \
  --n-samples 2000 \
  --device cuda \
  --diffusion-steps "${DIFFUSION_STEPS:-200}" \
  --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
  --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
  --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
  --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
  --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
  --write-examples \
  --examples-dir "${RUN_DIR}/figures/examples" \
  --out-json "${RUN_DIR}/results/model_selection.json" \
  --out-csv "${RUN_DIR}/results/model_selection.csv" \
  2>&1 | tee "${EVAL_LOG}"
EVAL_RC=${PIPESTATUS[0]}
set -e
if [ "${EVAL_RC}" -ne 0 ]; then
  echo "WARNING: standard-suite model_selection failed (exit ${EVAL_RC}). You can rerun with slurm/paper_rerun_model_selection.sh" | tee -a "${EVAL_LOG}"
fi

echo ""
echo "Running model selection evaluation (COMPLEX suite)..."
echo ""

EVAL_COMPLEX_LOG="${RUN_DIR}/logs/model_selection_complex.log"
mkdir -p "${RUN_DIR}/figures/examples_complex"

set +e
"${PYTHON_BIN}" scripts/model_selection.py \
  --suite complex \
  --checkpoints "${CKPT}" \
  --n-samples 2000 \
  --device cuda \
  --diffusion-steps "${DIFFUSION_STEPS:-200}" \
  --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
  --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
  --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
  --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
  --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
  --write-examples \
  --examples-dir "${RUN_DIR}/figures/examples_complex" \
  --out-json "${RUN_DIR}/results/model_selection_complex.json" \
  --out-csv "${RUN_DIR}/results/model_selection_complex.csv" \
  2>&1 | tee "${EVAL_COMPLEX_LOG}"
EVAL_COMPLEX_RC=${PIPESTATUS[0]}
set -e
if [ "${EVAL_COMPLEX_RC}" -ne 0 ]; then
  echo "WARNING: complex-suite model_selection failed (exit ${EVAL_COMPLEX_RC}). You can rerun with slurm/paper_rerun_model_selection_complex.sh" | tee -a "${EVAL_COMPLEX_LOG}"
fi

echo ""
echo "============================================================================"
echo "DONE: ${METHOD_TAG} completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Results: ${RUN_DIR}/results/model_selection.json"
echo ""
