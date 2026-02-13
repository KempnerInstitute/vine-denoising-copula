#!/bin/bash
#SBATCH --job-name=vdc_paper_biomed
#SBATCH --output=logs/vdc_paper_biomed_%j.out
#SBATCH --error=logs/vdc_paper_biomed_%j.err
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
# Vine Diffusion Copula (ICML 2026) - E7 BIOMEDICAL BENCHMARK JOB
# ============================================================================
# Produces:
#   RUN_DIR/results/e7_biomed_results.json
# and copies it to:
#   drafts/paper_outputs/e7_biomed_results.json
# then regenerates paper artifacts.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e7_biomed_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "Vine Diffusion Copula PAPER JOB: E7 biomedical benchmark"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e7_biomed_${TS}_${SLURM_JOB_ID:-nojobid}"
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

OUT_JSON="${RUN_DIR}/results/e7_biomed_results.json"
CKPT="${E7_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi
E7_EXTRA_ARGS=()
if [ "${E7_MI_CLONE_MINDE:-0}" = "1" ]; then
  E7_EXTRA_ARGS+=(--mi-clone-minde)
fi
if [ "${E7_MI_CLONE_MIST:-0}" = "1" ]; then
  E7_EXTRA_ARGS+=(--mi-clone-mist)
fi

"${PYTHON_BIN}" drafts/scripts/e7_biomed_benchmark.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  "${E7_EXTRA_ARGS[@]}" \
  --device cuda \
  --datasets ${E7_DATASETS:-arrhythmia cardio pima} \
  --seeds ${E7_SEEDS:-42 123 456} \
  --max-train "${E7_MAX_TRAIN:-2000}" \
  --max-test "${E7_MAX_TEST:-1000}" \
  --missing-frac "${E7_MISSING_FRAC:-0.20}" \
  --n-impute-eval "${E7_N_IMPUTE_EVAL:-200}" \
  --candidate-pool "${E7_CANDIDATE_POOL:-5000}" \
  --kernel-h "${E7_KERNEL_H:-0.05}" \
  --mi-methods ${E7_MI_METHODS:-dcd ksg gaussian infonce mine nwj} \
  --mi-pairs-per-dataset "${E7_MI_PAIRS_PER_DATASET:-1}" \
  --mi-n-samples "${E7_MI_N_SAMPLES:-3000}" \
  --mi-steps "${E7_MI_STEPS:-300}" \
  --mi-lr "${E7_MI_LR:-1e-3}" \
  --mi-batch-size "${E7_MI_BATCH_SIZE:-512}" \
  --mi-hidden-dim "${E7_MI_HIDDEN_DIM:-128}" \
  --mi-ksg-k "${E7_MI_KSG_K:-5}" \
  --mi-seed-base "${E7_MI_SEED_BASE:-123}" \
  --mi-dcd-diffusion-steps "${E7_MI_DCD_STEPS:-8}" \
  --mi-dcd-cfg-scale "${E7_MI_DCD_CFG_SCALE:-1.0}" \
  --mi-dcd-ensemble "${E7_MI_DCD_ENSEMBLE:-1}" \
  --mi-dcd-ensemble-mode "${E7_MI_DCD_ENSEMBLE_MODE:-geometric}" \
  --mi-dcd-smooth-sigma "${E7_MI_DCD_SMOOTH_SIGMA:-0.0}" \
  --mi-dcd-pred-noise-clip "${E7_MI_DCD_PRED_NOISE_CLIP:-10.0}" \
  --out-json "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e7_biomed.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e7_biomed_results.json"

echo ""
echo "Regenerating paper artifacts (force refresh)..."
echo ""

export FIG_PNG_DPI="${FIG_PNG_DPI:-120}"
"${PYTHON_BIN}" drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
  2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_e7_biomed.log"

echo ""
echo "============================================================================"
echo "DONE: E7 biomedical benchmark completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "E7 JSON: ${OUT_JSON}"
echo "Paper cache: ${REPO_ROOT}/drafts/paper_outputs/e7_biomed_results.json"
echo ""
