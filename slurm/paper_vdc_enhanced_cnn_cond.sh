#!/bin/bash
#SBATCH --job-name=vdc_paper_enhcnn
#SBATCH --output=logs/vdc_paper_enhcnn_%j.out
#SBATCH --error=logs/vdc_paper_enhcnn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Diffusion Copula (ICML 2026) - PAPER RESULTS JOB
# ============================================================================
# Variant: Enhanced CNN baseline (single-pass)
#
# Expected runtime: ~4-10 hours on 4x H100 (depends on max_steps + logging)
#
# Output Directory Structure (created under OUTPUT_BASE):
#   vdc_paper_enhanced_cnn_cond_{timestamp}_{SLURM_JOB_ID}/
#     results/      - JSON/CSV metrics (model_selection)
#     logs/         - train.log, eval.log, env.txt
#     figures/      - symlink to training visualizations
#     checkpoints/  - model_step_*.pt checkpoints + visualizations/
#     analysis/     - exact config copy + git info + slurm script copy
# ============================================================================

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"
CONFIG_SRC="${REPO_ROOT}/configs/train/enhanced_cnn_cond.yaml"
MODEL_TYPE="enhanced_cnn"
METHOD_TAG="enhanced_cnn_cond"

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
eval "$(conda shell.bash hook)" || true
conda activate vdc 2>/dev/null || conda activate diffuse_vine_cop 2>/dev/null || true

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

{
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  echo "Python: $(which python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version() if torch.cuda.is_available() else None)"
  nvidia-smi || true
} | tee "${RUN_DIR}/logs/env.txt"

# Save reproducibility artifacts
cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true
git status --porcelain 2>/dev/null | tee "${RUN_DIR}/analysis/git_status_porcelain.txt" || true

# Write an exact config copy with a top-level checkpoint_dir (used by scripts/train_unified.py)
CONFIG_OUT="${RUN_DIR}/analysis/train_config.yaml"
export CONFIG_SRC CONFIG_OUT CHECKPOINT_DIR
python - <<PY
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
torchrun --standalone --nproc_per_node="${GPUS}" \
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
python scripts/model_selection.py \
  --checkpoints "${CKPT}" \
  --n-samples 2000 \
  --device cuda \
  --out-json "${RUN_DIR}/results/model_selection.json" \
  --out-csv "${RUN_DIR}/results/model_selection.csv" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "============================================================================"
echo "DONE: ${METHOD_TAG} completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Results: ${RUN_DIR}/results/model_selection.json"
echo ""

