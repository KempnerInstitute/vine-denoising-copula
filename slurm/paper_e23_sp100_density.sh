#!/bin/bash
#SBATCH --job-name=vdc_e23_sp100
#SBATCH --output=logs/vdc_e23_sp100_%j.out
#SBATCH --error=logs/vdc_e23_sp100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e23_sp100_density.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "VDC PAPER JOB: E23 S&P100 real-world density benchmark"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e23_sp100_${TS}_${SLURM_JOB_ID:-nojobid}"
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
  echo "ERROR: no usable Python at ${VDC_PAPER_ENV}/bin/python."
  exit 3
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e23_sp100_density.json"
OUT_TEX="${RUN_DIR}/results/tab_sp100_density.tex"

"${PYTHON_BIN}" drafts/scripts/e23_sp100_density.py \
  --train-frac "${E23_TRAIN_FRAC:-0.8}" \
  --seed "${E23_SEED:-42}" \
  --device cuda \
  --output-base "${OUTPUT_BASE}" \
  --m-grid "${E23_M_GRID:-64}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --pyvine-timeout-s "${E23_PYVINE_TIMEOUT_S:-3600}" \
  --n-boot "${E23_N_BOOT:-2000}" \
  --out-json "${OUT_JSON}" \
  --out-tex "${OUT_TEX}" \
  2>&1 | tee "${RUN_DIR}/logs/e23_sp100.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e23_sp100_density.json"
[ -f "${OUT_JSON%.json}.npz" ] && cp "${OUT_JSON%.json}.npz" "drafts/paper_outputs/e23_sp100_density.npz"
cp "${OUT_TEX}" "drafts/tables/tab_sp100_density.tex"

echo ""
echo "============================================================================"
echo "DONE: E23 S&P100 density at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "JSON: ${OUT_JSON}"
echo "Table: ${OUT_TEX}"
