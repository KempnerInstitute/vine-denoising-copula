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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/train_unified.py" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
unset PYTHONHOME || true
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

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
"${PYTHON_BIN}" -c "
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
TRAIN_LOG="${RUN_DIR}/logs/train.log"
set +e
"${PYTHON_BIN}" scripts/train_unified.py --config "${CONFIG_OUT}" --model-type "${MODEL_TYPE}" \
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

# Collect checkpoints
mapfile -t CKPTS < <(ls -1 "${CHECKPOINT_DIR}"/model_step_*.pt 2>/dev/null | sort -V)
if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "ERROR: no checkpoint found in ${CHECKPOINT_DIR}" | tee -a "${TRAIN_LOG}"
    exit 2
fi
CKPT="${CKPTS[$((${#CKPTS[@]}-1))]}"
echo "Latest checkpoint: ${CKPT}"

# Optional checkpoint sweeps: select by standard, complex, or joint score.
SELECT_BEST_CKPT="${SELECT_BEST_CKPT:-1}"
SELECT_BEST_MODE="${SELECT_BEST_MODE:-joint}"   # standard|complex|joint
JOINT_COMPLEX_WEIGHT="${JOINT_COMPLEX_WEIGHT:-1.0}"
if [ "${SELECT_BEST_CKPT}" = "1" ] && [ "${#CKPTS[@]}" -gt 1 ]; then
    echo ""
    echo "Running checkpoint sweeps (${#CKPTS[@]} checkpoints) to select best model..."
    echo ""

    SWEEP_STD_LOG="${RUN_DIR}/logs/model_selection_sweep.log"
    SWEEP_STD_JSON="${RUN_DIR}/results/model_selection_sweep.json"
    SWEEP_STD_CSV="${RUN_DIR}/results/model_selection_sweep.csv"
    SWEEP_COMPLEX_LOG="${RUN_DIR}/logs/model_selection_sweep_complex.log"
    SWEEP_COMPLEX_JSON="${RUN_DIR}/results/model_selection_sweep_complex.json"
    SWEEP_COMPLEX_CSV="${RUN_DIR}/results/model_selection_sweep_complex.csv"

    set +e
    "${PYTHON_BIN}" scripts/model_selection.py \
      --checkpoints "${CKPTS[@]}" \
      --n-samples 2000 \
      --device cuda \
      --diffusion-steps "${DIFFUSION_STEPS:-200}" \
      --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
      --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
      --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
      --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
      --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
      --out-json "${SWEEP_STD_JSON}" \
      --out-csv "${SWEEP_STD_CSV}" \
      2>&1 | tee "${SWEEP_STD_LOG}"
    SWEEP_STD_RC=${PIPESTATUS[0]}
    set -e
    if [ "${SWEEP_STD_RC}" -ne 0 ]; then
        echo "WARNING: standard sweep failed (exit ${SWEEP_STD_RC})." | tee -a "${SWEEP_STD_LOG}"
    fi

    set +e
    "${PYTHON_BIN}" scripts/model_selection.py \
      --suite complex \
      --checkpoints "${CKPTS[@]}" \
      --n-samples 2000 \
      --device cuda \
      --diffusion-steps "${DIFFUSION_STEPS:-200}" \
      --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
      --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
      --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
      --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
      --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
      --out-json "${SWEEP_COMPLEX_JSON}" \
      --out-csv "${SWEEP_COMPLEX_CSV}" \
      2>&1 | tee "${SWEEP_COMPLEX_LOG}"
    SWEEP_COMPLEX_RC=${PIPESTATUS[0]}
    set -e
    if [ "${SWEEP_COMPLEX_RC}" -ne 0 ]; then
        echo "WARNING: complex sweep failed (exit ${SWEEP_COMPLEX_RC})." | tee -a "${SWEEP_COMPLEX_LOG}"
    fi

    BEST_CKPT="$("${PYTHON_BIN}" - <<PY
import json
import math
from pathlib import Path

mode = str("${SELECT_BEST_MODE}").strip().lower()
joint_w = float("${JOINT_COMPLEX_WEIGHT}")
std_path = Path("${SWEEP_STD_JSON}")
cx_path = Path("${SWEEP_COMPLEX_JSON}")

def load_rows(path: Path):
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text())
        rows = obj.get("results", [])
        return rows if isinstance(rows, list) else []
    except Exception:
        return []

def f(row, key, default=1e9):
    try:
        v = float(row.get(key, default))
        return v if math.isfinite(v) else default
    except Exception:
        return default

def best_single(rows):
    if not rows:
        return ""
    best = min(
        rows,
        key=lambda r: (
            f(r, "mean_mi_err"),
            f(r, "mean_tau_err"),
            f(r, "mean_hfunc_mae"),
            f(r, "mean_ise"),
        ),
    )
    return str(best.get("checkpoint", ""))

std_rows = load_rows(std_path)
cx_rows = load_rows(cx_path)

if mode == "standard":
    print(best_single(std_rows))
    raise SystemExit(0)
if mode == "complex":
    print(best_single(cx_rows))
    raise SystemExit(0)

if std_rows and cx_rows:
    cx_by_ckpt = {str(r.get("checkpoint", "")): r for r in cx_rows}
    joint_rows = []
    for s in std_rows:
        ck = str(s.get("checkpoint", ""))
        c = cx_by_ckpt.get(ck)
        if c is None:
            continue
        joint_rows.append(
            (
                (
                    f(s, "mean_mi_err") + joint_w * f(c, "mean_mi_err"),
                    f(s, "mean_tau_err") + joint_w * f(c, "mean_tau_err"),
                    f(s, "mean_hfunc_mae") + joint_w * f(c, "mean_hfunc_mae"),
                    f(s, "mean_ise") + joint_w * f(c, "mean_ise"),
                ),
                ck,
            )
        )
    if joint_rows:
        joint_rows.sort(key=lambda x: x[0])
        print(joint_rows[0][1])
        raise SystemExit(0)

if std_rows:
    print(best_single(std_rows))
elif cx_rows:
    print(best_single(cx_rows))
else:
    print("")
PY
)"
    if [ -n "${BEST_CKPT}" ] && [ -f "${BEST_CKPT}" ]; then
        CKPT="${BEST_CKPT}"
        echo "Selected best checkpoint (${SELECT_BEST_MODE}) : ${CKPT}" | tee -a "${SWEEP_STD_LOG}" "${SWEEP_COMPLEX_LOG}"
    else
        echo "WARNING: sweeps did not return a valid checkpoint; using latest checkpoint." | tee -a "${SWEEP_STD_LOG}" "${SWEEP_COMPLEX_LOG}"
    fi
fi

echo "${CKPT}" > "${RUN_DIR}/results/checkpoint_path.txt"
echo "Checkpoint used for paper evaluation: ${CKPT}"

echo ""
echo "Running model selection evaluation (standard + complex)..."
echo ""

EVAL_STD_LOG="${RUN_DIR}/logs/model_selection.log"
EVAL_COMPLEX_LOG="${RUN_DIR}/logs/model_selection_complex.log"
mkdir -p "${RUN_DIR}/figures/examples" "${RUN_DIR}/figures/examples_complex"

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
  2>&1 | tee "${EVAL_STD_LOG}"
EVAL_STD_RC=${PIPESTATUS[0]}
set -e
if [ "${EVAL_STD_RC}" -ne 0 ]; then
    echo "WARNING: standard-suite model_selection failed (exit ${EVAL_STD_RC})." | tee -a "${EVAL_STD_LOG}"
fi

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
    echo "WARNING: complex-suite model_selection failed (exit ${EVAL_COMPLEX_RC})." | tee -a "${EVAL_COMPLEX_LOG}"
fi

# Optional baseline benchmark table for paper claims.
RUN_BASELINES="${RUN_BASELINES:-1}"
if [ "${RUN_BASELINES}" = "1" ]; then
    echo ""
    echo "Running baseline comparisons..."
    echo ""
    BASELINES=(histogram kde_probit kde_logit pyvine_param pyvine_nonpar)

    set +e
    "${PYTHON_BIN}" scripts/model_selection.py \
      --suite standard \
      --checkpoints "${CKPT}" \
      --baselines "${BASELINES[@]}" \
      --n-samples 2000 \
      --device cuda \
      --diffusion-steps "${DIFFUSION_STEPS:-200}" \
      --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
      --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
      --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
      --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
      --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
      --out-json "${RUN_DIR}/results/model_selection_with_baselines.json" \
      --out-csv "${RUN_DIR}/results/model_selection_with_baselines.csv" \
      2>&1 | tee "${RUN_DIR}/logs/model_selection_with_baselines.log"
    BASE_STD_RC=${PIPESTATUS[0]}
    set -e
    if [ "${BASE_STD_RC}" -ne 0 ]; then
        echo "WARNING: standard baseline comparison failed (exit ${BASE_STD_RC})." | tee -a "${RUN_DIR}/logs/model_selection_with_baselines.log"
    fi

    set +e
    "${PYTHON_BIN}" scripts/model_selection.py \
      --suite complex \
      --checkpoints "${CKPT}" \
      --baselines "${BASELINES[@]}" \
      --n-samples 2000 \
      --device cuda \
      --diffusion-steps "${DIFFUSION_STEPS:-200}" \
      --diffusion-cfg-scale "${DIFFUSION_CFG_SCALE:-4.0}" \
      --diffusion-ensemble "${DIFFUSION_ENSEMBLE:-1}" \
      --diffusion-ensemble-mode "${DIFFUSION_ENSEMBLE_MODE:-geometric}" \
      --diffusion-smooth-sigma "${DIFFUSION_SMOOTH_SIGMA:-0.0}" \
      --diffusion-pred-noise-clip "${DIFFUSION_PRED_NOISE_CLIP:-10.0}" \
      --out-json "${RUN_DIR}/results/model_selection_complex_with_baselines.json" \
      --out-csv "${RUN_DIR}/results/model_selection_complex_with_baselines.csv" \
      2>&1 | tee "${RUN_DIR}/logs/model_selection_complex_with_baselines.log"
    BASE_COMPLEX_RC=${PIPESTATUS[0]}
    set -e
    if [ "${BASE_COMPLEX_RC}" -ne 0 ]; then
        echo "WARNING: complex baseline comparison failed (exit ${BASE_COMPLEX_RC})." | tee -a "${RUN_DIR}/logs/model_selection_complex_with_baselines.log"
    fi
fi

echo ""
echo "============================================================================"
echo "DONE: ${METHOD_TAG} completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Results: ${RUN_DIR}/results/model_selection.json"
