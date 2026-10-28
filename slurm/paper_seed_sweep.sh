#!/bin/bash
#SBATCH --job-name=vdc_seed_sweep
#SBATCH --output=logs/vdc_seed_sweep_%A_%a.out
#SBATCH --error=logs/vdc_seed_sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-5
#
# ============================================================================
# Headline-figure seed sweep for NeurIPS 2026 submission (3 seeds per script).
# ============================================================================
# Covers the headline plots that currently report point estimates:
#   * e22_highdim_scaling.py  -> Fig 4(b) full-vine NLL, Fig 4(c) vine fit time
#   * tc_benchmark.py         -> Fig 3(b) TC scaling, Fig 5(c) TC runtime
#
# Array layout (TASK_ID -> (script, seed)):
#   0  e22  seed=7
#   1  e22  seed=17
#   2  e22  seed=42
#   3  tc   seed=7
#   4  tc   seed=17
#   5  tc   seed=42
#
# After the array completes, run:
#   python drafts/scripts/merge_seed_runs.py
# to aggregate per-seed JSONs into mean/std artifacts that the figure scripts
# will pick up automatically.
#
# Submit:
#   sbatch slurm/paper_seed_sweep.sh
# Resubmit a single failed task (e.g. task 4):
#   sbatch --array=4 slurm/paper_seed_sweep.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/drafts/scripts" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

# Map array index -> (kind, seed). Edit the SEEDS list to add seeds.
SEEDS=(7 17 42)
KINDS=(e22 tc)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS_PER_KIND=${#SEEDS[@]}
KIND_IDX=$(( TASK_ID / SEEDS_PER_KIND ))
SEED_IDX=$(( TASK_ID % SEEDS_PER_KIND ))
KIND="${KINDS[${KIND_IDX}]}"
SEED="${SEEDS[${SEED_IDX}]}"

echo "============================================================================"
echo "VDC SEED SWEEP: kind=${KIND} seed=${SEED} (array task ${TASK_ID})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-} (array ${SLURM_ARRAY_JOB_ID:-?}.${TASK_ID})"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Repo: ${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs"

# --- Environment -----------------------------------------------------------
module purge
module load cuda/12.2.0-fasrc01

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

VDC_PAPER_ENV="${VDC_PAPER_ENV:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2}"
if [ -n "${VDC_PYTHON_BIN:-}" ] && [ -x "${VDC_PYTHON_BIN}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
elif [ -x "${VDC_PAPER_ENV}/bin/python" ]; then
  PYTHON_BIN="${VDC_PAPER_ENV}/bin/python"
  export PATH="${VDC_PAPER_ENV}/bin:${PATH}"
  export LD_LIBRARY_PATH="${VDC_PAPER_ENV}/lib:${LD_LIBRARY_PATH:-}"
else
  echo "ERROR: no usable Python at ${VDC_PAPER_ENV}/bin/python or VDC_PYTHON_BIN."
  exit 3
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -c "import sys, torch; print(sys.executable); print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

# --- Resolve canonical paper checkpoint ------------------------------------
if [ -z "${PAPER_CHECKPOINT:-}" ]; then
  PAPER_CHECKPOINT="$("${PYTHON_BIN}" -c "from vdc.utils.paper import resolve_canonical_paper_checkpoint; p = resolve_canonical_paper_checkpoint(); print(p if p else '')")"
fi
if [ -z "${PAPER_CHECKPOINT}" ] || [ ! -f "${PAPER_CHECKPOINT}" ]; then
  echo "ERROR: cannot resolve canonical paper checkpoint. Set PAPER_CHECKPOINT."
  exit 4
fi
echo "Using checkpoint: ${PAPER_CHECKPOINT}"

# --- Per-seed run dir & output paths ---------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_seed_sweep/${KIND}_seed${SEED}_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}
cp "$0" "${RUN_DIR}/analysis/slurm_script.sh" || true
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

# Per-seed staged artifact directory under drafts/paper_outputs/seed_sweep/
STAGE_DIR="${REPO_ROOT}/drafts/paper_outputs/seed_sweep"
mkdir -p "${STAGE_DIR}"

# --- Dispatch --------------------------------------------------------------
case "${KIND}" in
  e22)
    OUT_JSON="${RUN_DIR}/results/e22_highdim_scaling.json"
    OUT_MD="${RUN_DIR}/results/e22_highdim_scaling_summary.md"
    OUT_TEX="${RUN_DIR}/results/tab_highdim_scaling.tex"
    "${PYTHON_BIN}" drafts/scripts/e22_highdim_scaling.py \
      --dims ${E22_DIMS:-100 200 500} \
      --n-train "${E22_N_TRAIN:-20000}" \
      --n-test "${E22_N_TEST:-5000}" \
      --theta "${E22_THETA:-2.0}" \
      --seed "${SEED}" \
      --device cuda \
      --checkpoint "${PAPER_CHECKPOINT}" \
      --output-base "${OUTPUT_BASE}" \
      --m-grid "${E22_M_GRID:-64}" \
      --num-threads "${SLURM_CPUS_PER_TASK:-8}" \
      --pyvine-timeout-s "${E22_PYVINE_TIMEOUT_S:-2400}" \
      --include-realnvp \
      --include-gaussian \
      --out-json "${OUT_JSON}" \
      --out-md "${OUT_MD}" \
      --out-tex "${OUT_TEX}" \
      2>&1 | tee "${RUN_DIR}/logs/e22_highdim.log"
    cp "${OUT_JSON}" "${STAGE_DIR}/e22_highdim_scaling_seed${SEED}.json"
    ;;

  tc)
    OUT_JSON="${RUN_DIR}/results/tc_benchmark.json"
    TC_INCLUDE_FLAGS=()
    [ "${TC_INCLUDE_MINE:-1}" = "1" ] && TC_INCLUDE_FLAGS+=(--include-mine)
    [ "${TC_INCLUDE_INFONCE:-1}" = "1" ] && TC_INCLUDE_FLAGS+=(--include-infonce)
    [ "${TC_INCLUDE_NWJ:-0}" = "1" ] && TC_INCLUDE_FLAGS+=(--include-nwj)
    "${PYTHON_BIN}" scripts/tc_benchmark.py \
      --checkpoint "${PAPER_CHECKPOINT}" \
      --device cuda \
      --rho "${TC_RHO:-0.7}" \
      --dims ${TC_DIMS:-2 5 10 20 50} \
      --n "${TC_N:-5000}" \
      --n-trials "${TC_N_TRIALS:-3}" \
      --seed "${SEED}" \
      --ksg-k "${TC_KSG_K:-5}" \
      --diffusion-steps "${TC_DIFFUSION_STEPS:-4}" \
      --diffusion-cfg-scale "${TC_CFG_SCALE:-1.0}" \
      --diffusion-pred-noise-clip "${TC_PRED_NOISE_CLIP:-1.0}" \
      --truncation-level "${TC_TRUNCATION_LEVEL:-1}" \
      --batch-edges \
      --edge-batch-size "${TC_EDGE_BATCH_SIZE:-256}" \
      "${TC_INCLUDE_FLAGS[@]}" \
      --out-json "${OUT_JSON}" \
      ${TC_EXTRA_ARGS:-} \
      2>&1 | tee "${RUN_DIR}/logs/tc_benchmark.log"
    cp "${OUT_JSON}" "${STAGE_DIR}/tc_benchmark_seed${SEED}.json"
    ;;

  *)
    echo "ERROR: unknown KIND='${KIND}'."
    exit 5
    ;;
esac

echo ""
echo "============================================================================"
echo "DONE: kind=${KIND} seed=${SEED} at $(date)"
echo "Run dir : ${RUN_DIR}"
echo "Staged  : ${STAGE_DIR}/${KIND}_*_seed${SEED}.json"
echo "============================================================================"
echo ""
echo "After all 6 array tasks complete, aggregate seeds with:"
echo "  python drafts/scripts/merge_seed_runs.py --kind ${KIND}"
echo "or simply:"
echo "  python drafts/scripts/merge_seed_runs.py"
echo "to merge both."
