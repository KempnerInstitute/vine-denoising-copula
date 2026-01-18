#!/bin/bash
#SBATCH --job-name=vdc_train_conditional
#SBATCH --output=logs/vdc_train_conditional_%j.out
#SBATCH --error=logs/vdc_train_conditional_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#
# ============================================================================
# Training with Enhanced Copula Zoo (Conditional + BB Families)
# ============================================================================
# This training run includes:
# - Conditional copulas (for vine higher-tree estimation)
# - BB1/BB7 two-parameter families (better tail dependence)
# - Complex synthetic patterns (X, ring, double-banana)
#
# These additions improve the model's ability to estimate conditional
# pair copulas needed for vine copula construction beyond tree 1.
# ============================================================================

set -euo pipefail

REPO_ROOT="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"
CONFIG_SRC="${REPO_ROOT}/configs/train/denoiser_cond_enhanced.yaml"
MODEL_TYPE="denoiser"
METHOD_TAG="denoiser_cond_enhanced"

echo "============================================================================"
echo "Vine Diffusion Copula: Enhanced Training (Conditional + BB)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"

# Output directory
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_${METHOD_TAG}_${TS}_${SLURM_JOB_ID:-nojobid}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${RUN_DIR}/"{results,logs,figures,checkpoints,analysis}

echo "Output Dir: ${RUN_DIR}"
mkdir -p "${REPO_ROOT}/logs"

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)" || true
conda activate vdc 2>/dev/null || conda activate diffuse_vine_cop 2>/dev/null || true

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Save reproducibility info
cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

# Create enhanced config if not exists
if [ ! -f "${CONFIG_SRC}" ]; then
    echo "Creating enhanced training config..."
    cat > "${CONFIG_SRC}" <<'YAML'
# Enhanced training config with conditional copulas and BB families
experiment:
  name: "copula_denoiser_cond_enhanced"
  description: "Denoiser trained on parametric + conditional + BB + complex copulas"
  seed: 42

model:
  type: "denoiser"
  grid_size: 64
  base_channels: 128
  depth: 4
  blocks_per_level: 2
  dropout: 0.1
  time_emb_dim: 256
  use_coordinates: true
  use_probit_coords: true
  use_log_n: true
  time_conditioning: true
  output_mode: "log"

data:
  m: 64
  n_samples_per_copula:
    mode: "log_uniform"
    min: 200
    max: 5000

  # Enhanced family mix with conditional copulas
  copula_families:
    gaussian: 0.14
    student: 0.06
    clayton: 0.10
    gumbel: 0.10
    frank: 0.06
    joe: 0.06
    bb1: 0.06
    bb7: 0.06
    conditional_gaussian: 0.08
    conditional_clayton: 0.08
    independence: 0.04
    complex_x: 0.06
    complex_ring: 0.05
    complex_double_banana: 0.05

  binning: "uniform"
  num_workers: 4
  mixture_prob: 0.25
  n_mixture_components: [2, 4]

  param_ranges:
    gaussian_rho: [-0.95, 0.95]
    student_rho: [-0.9, 0.9]
    student_df: [2, 30]
    clayton_theta: [0.1, 12.0]
    gumbel_theta: [1.1, 10.0]
    frank_theta: [-15.0, 15.0]
    joe_theta: [1.1, 10.0]
    bb1_theta: [0.1, 5.0]
    bb1_delta: [1.0, 5.0]
    bb7_theta: [1.0, 5.0]
    bb7_delta: [0.1, 5.0]

training:
  max_steps: 250000
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  gradient_clip: 1.0
  
  hist_noise:
    enable: true
    max_strength: 0.75
    power: 1.0

  loss_weights:
    ce: 1.0
    ise: 0.0
    tail: 0.1
    ms: 0.0
    marg_kl: 0.0

  checkpoint_every: 10000
  eval_every: 5000
  log_every: 100
YAML
fi

# Generate final config with checkpoint path
CONFIG_OUT="${RUN_DIR}/analysis/train_config.yaml"
python -c "
import yaml
from pathlib import Path

with open('${CONFIG_SRC}') as f:
    cfg = yaml.safe_load(f)

cfg['checkpoint_dir'] = '${CHECKPOINT_DIR}'
cfg['run_dir'] = '${RUN_DIR}'

with open('${CONFIG_OUT}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f'Config written to: ${CONFIG_OUT}')
"

echo ""
echo "Starting training..."
echo ""

# Run training
python scripts/train_unified.py --config "${CONFIG_OUT}" \
    2>&1 | tee "${RUN_DIR}/logs/train.log"

echo ""
echo "============================================================================"
echo "Training complete!"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "============================================================================"
