#!/bin/bash
#SBATCH --job-name=vdc_e24_het
#SBATCH --output=logs/vdc_e24_het_%j.out
#SBATCH --error=logs/vdc_e24_het_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e24_heterogeneous_scaling.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "VDC PAPER JOB: E24 heterogeneous-family vine scaling"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e24_het_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

VDC_PAPER_ENV="${VDC_PAPER_ENV:-/n/netscratch/kempner_dev/hsafaai/conda_envs/vdc_paper}"
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

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e24_heterogeneous_scaling.json"
OUT_MD="${RUN_DIR}/results/e24_heterogeneous_scaling_summary.md"
OUT_TEX="${RUN_DIR}/results/tab_heterogeneous_scaling.tex"

"${PYTHON_BIN}" drafts/scripts/e24_heterogeneous_scaling.py \
  --dims ${E24_DIMS:-100 200} \
  --n-train "${E24_N_TRAIN:-20000}" \
  --n-test "${E24_N_TEST:-5000}" \
  --seed "${E24_SEED:-42}" \
  --device cuda \
  --output-base "${OUTPUT_BASE}" \
  --m-grid "${E24_M_GRID:-64}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --pyvine-timeout-s "${E24_PYVINE_TIMEOUT_S:-21600}" \
  --include-realnvp \
  --include-gaussian \
  ${E24_INCLUDE_PYVINE_PARAM:-} \
  --edge-batch-size "${E24_EDGE_BATCH:-512}" \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}" \
  --out-tex "${OUT_TEX}" \
  2>&1 | tee "${RUN_DIR}/logs/e24_het.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e24_heterogeneous_scaling.json"
cp "${OUT_MD}" "drafts/paper_outputs/e24_heterogeneous_scaling_summary.md"
cp "${OUT_TEX}" "drafts/tables/tab_heterogeneous_scaling.tex"

echo ""
echo "============================================================================"
echo "DONE: E24 at $(date)"
echo "============================================================================"
