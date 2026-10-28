#!/bin/bash
#SBATCH --job-name=vdc_d500_sweep
#SBATCH --output=logs/vdc_d500_sweep_%A_%a.out
#SBATCH --error=logs/vdc_d500_sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-2
#
# d=500 only, with pyvine-TLL granted the full 6 h budget so it actually
# completes (the default 2400 s budget in the main sweep timed out on every
# seed, and we want the data point back). Three seeds: 7, 17, 42.
#
# Submit:
#   sbatch slurm/paper_seed_sweep_d500.sh
# Then merge with:
#   python drafts/scripts/patch_d500_pyvine.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/drafts/scripts" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

SEEDS=(7 17 42)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[${TASK_ID}]}"

echo "============================================================================"
echo "VDC D=500 RERUN: seed=${SEED}, pyvine budget = 21600 s"
echo "============================================================================"
echo "Job: ${SLURM_JOB_ID:-} (${SLURM_ARRAY_JOB_ID:-?}.${TASK_ID}) | Node: $(hostname) | Start: $(date)"

mkdir -p "${REPO_ROOT}/logs"

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
  echo "ERROR: no usable Python."
  exit 3
fi

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Resolve canonical paper checkpoint
if [ -z "${PAPER_CHECKPOINT:-}" ]; then
  PAPER_CHECKPOINT="$("${PYTHON_BIN}" -c "from vdc.utils.paper import resolve_canonical_paper_checkpoint; p = resolve_canonical_paper_checkpoint(); print(p if p else '')")"
fi
echo "Using checkpoint: ${PAPER_CHECKPOINT}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_seed_sweep_d500/seed${SEED}_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}
cp "$0" "${RUN_DIR}/analysis/slurm_script.sh" || true
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

STAGE_DIR="${REPO_ROOT}/drafts/paper_outputs/seed_sweep_d500"
mkdir -p "${STAGE_DIR}"

OUT_JSON="${RUN_DIR}/results/e22_d500_only.json"
OUT_MD="${RUN_DIR}/results/e22_d500_only_summary.md"
OUT_TEX_TMP="${RUN_DIR}/results/e22_d500_only.tex"

"${PYTHON_BIN}" drafts/scripts/e22_highdim_scaling.py \
  --dims 500 \
  --n-train "${E22_N_TRAIN:-20000}" \
  --n-test "${E22_N_TEST:-5000}" \
  --theta "${E22_THETA:-2.0}" \
  --seed "${SEED}" \
  --device cuda \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --output-base "${OUTPUT_BASE}" \
  --m-grid "${E22_M_GRID:-64}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --pyvine-timeout-s "${E22_PYVINE_TIMEOUT_S:-21600}" \
  --include-realnvp \
  --include-gaussian \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}" \
  --out-tex "${OUT_TEX_TMP}" \
  2>&1 | tee "${RUN_DIR}/logs/e22_d500.log"

cp "${OUT_JSON}" "${STAGE_DIR}/e22_d500_seed${SEED}.json"

echo "DONE: seed=${SEED} at $(date) -> ${STAGE_DIR}/e22_d500_seed${SEED}.json"
