#!/bin/bash
#SBATCH --job-name=vdc_paper_theory
#SBATCH --output=logs/vdc_paper_theory_%j.out
#SBATCH --error=logs/vdc_paper_theory_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#
# ============================================================================
# Vine Diffusion Copula (ICML 2026) - E6 SYNTHETIC THEORY BENCHMARK JOB
# ============================================================================
# Produces:
#   RUN_DIR/results/e6_theory_synthetic_results.json
# and copies it to:
#   drafts/paper_outputs/e6_theory_synthetic_results.json
# then regenerates paper artifacts.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e6_theory_synthetic_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "Vine Diffusion Copula PAPER JOB: E6 synthetic theory benchmark"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e6_theory_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01

# Fix libffi issue - use Mambaforge's working libffi
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)" || true
set +u
if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
elif [ -n "${VDC_CONDA_ENV_PATH:-}" ]; then
  conda activate "${VDC_CONDA_ENV_PATH}"
  PYTHON_BIN="python"
elif conda activate diffuse_vine_cop 2>/dev/null; then
  PYTHON_BIN="python"
elif conda activate vdc 2>/dev/null; then
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

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "CONDA_PREFIX: ${CONDA_PREFIX:-}"
echo "Testing Python..."
"${PYTHON_BIN}" -c "import numpy, scipy, torch; print(f'numpy={numpy.__version__}, scipy={scipy.__version__}, torch={torch.__version__}')"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON_NAME="${E6_OUT_JSON_NAME:-e6_theory_synthetic_results.json}"
OUT_JSON="${RUN_DIR}/results/${OUT_JSON_NAME}"
CKPT="${E6_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi

E6_EXTRA_ARGS=()
if [ "${E6_GAUSSIAN_COPULA:-1}" = "1" ]; then
  E6_EXTRA_ARGS+=(--gaussian-copula)
fi
if [ "${E6_FLOW_REALNVP:-0}" = "1" ]; then
  E6_EXTRA_ARGS+=(--flow-realnvp)
  E6_EXTRA_ARGS+=(--flow-num-layers "${E6_FLOW_NUM_LAYERS:-8}")
  E6_EXTRA_ARGS+=(--flow-hidden-dim "${E6_FLOW_HIDDEN_DIM:-128}")
  E6_EXTRA_ARGS+=(--flow-hidden-layers "${E6_FLOW_HIDDEN_LAYERS:-2}")
  E6_EXTRA_ARGS+=(--flow-epochs "${E6_FLOW_EPOCHS:-25}")
  E6_EXTRA_ARGS+=(--flow-batch-size "${E6_FLOW_BATCH_SIZE:-2048}")
  E6_EXTRA_ARGS+=(--flow-eval-batch-size "${E6_FLOW_EVAL_BATCH_SIZE:-4096}")
  E6_EXTRA_ARGS+=(--flow-lr "${E6_FLOW_LR:-1e-3}")
  E6_EXTRA_ARGS+=(--flow-val-fraction "${E6_FLOW_VAL_FRACTION:-0.1}")
  E6_EXTRA_ARGS+=(--flow-patience "${E6_FLOW_PATIENCE:-5}")
fi
if [ "${E6_MI_CLONE_MINDE:-0}" = "1" ]; then
  E6_EXTRA_ARGS+=(--mi-clone-minde)
fi
if [ "${E6_MI_CLONE_MIST:-0}" = "1" ]; then
  E6_EXTRA_ARGS+=(--mi-clone-mist)
fi

"${PYTHON_BIN}" drafts/scripts/e6_theory_synthetic_benchmark.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  "${E6_EXTRA_ARGS[@]}" \
  --device cuda \
  --scenarios ${E6_SCENARIOS:-gaussian_ar1 student_ar1 clayton_vine mixed_vine} \
  --dims ${E6_DIMS:-5 10 20 50} \
  --n-train "${E6_N_TRAIN:-5000}" \
  --n-test "${E6_N_TEST:-2000}" \
  --n-trials "${E6_N_TRIALS:-3}" \
  --seed "${E6_SEED:-42}" \
  --pyvine "${E6_PYVINE:-both}" \
  --mi-methods ${E6_MI_METHODS:-dcd ksg gaussian infonce mine nwj} \
  --mi-pairs-per-case "${E6_MI_PAIRS_PER_CASE:-1}" \
  --mi-n-samples "${E6_MI_N_SAMPLES:-3000}" \
  --mi-steps "${E6_MI_STEPS:-300}" \
  --mi-lr "${E6_MI_LR:-1e-3}" \
  --mi-batch-size "${E6_MI_BATCH_SIZE:-512}" \
  --mi-hidden-dim "${E6_MI_HIDDEN_DIM:-128}" \
  --mi-ksg-k "${E6_MI_KSG_K:-5}" \
  --mi-seed-base "${E6_MI_SEED_BASE:-123}" \
  --mi-dcd-diffusion-steps "${E6_MI_DCD_STEPS:-8}" \
  --mi-dcd-cfg-scale "${E6_MI_DCD_CFG_SCALE:-1.0}" \
  --mi-dcd-ensemble "${E6_MI_DCD_ENSEMBLE:-1}" \
  --mi-dcd-ensemble-mode "${E6_MI_DCD_ENSEMBLE_MODE:-geometric}" \
  --mi-dcd-smooth-sigma "${E6_MI_DCD_SMOOTH_SIGMA:-0.0}" \
  --mi-dcd-pred-noise-clip "${E6_MI_DCD_PRED_NOISE_CLIP:-10.0}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --out-json "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e6_theory.log"

PAPER_COPY_PATH="${E6_PAPER_COPY_PATH:-drafts/paper_outputs/e6_theory_synthetic_results.json}"
cp "${OUT_JSON}" "${PAPER_COPY_PATH}"

if [ "${E6_SKIP_ARTIFACTS:-0}" != "1" ]; then
  echo ""
  echo "Regenerating paper artifacts (force refresh)..."
  echo ""

  export FIG_PNG_DPI="${FIG_PNG_DPI:-120}"
  "${PYTHON_BIN}" drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
    2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_e6_theory.log"
fi

echo ""
echo "============================================================================"
echo "DONE: E6 synthetic theory benchmark completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "E6 JSON: ${OUT_JSON}"
echo "Paper cache: ${REPO_ROOT}/${PAPER_COPY_PATH}"
echo ""
