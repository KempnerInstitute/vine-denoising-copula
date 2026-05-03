#!/bin/bash
#SBATCH --job-name=vdc_e21_boot
#SBATCH --output=logs/vdc_e21_boot_%j.out
#SBATCH --error=logs/vdc_e21_boot_%j.err
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
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e21_uci_bootstrap_ci.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "VDC PAPER JOB: E21 paired-bootstrap CIs for UCI density"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Repo: ${REPO_ROOT}"
echo "Output Base: ${OUTPUT_BASE}"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e21_bootstrap_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01

export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

VDC_PAPER_ENV="${VDC_PAPER_ENV:-/n/netscratch/kempner_dev/hsafaai/conda_envs/vdc_paper}"
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -c "import sys, torch, scipy, numpy; print(sys.executable); print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}'); print(f'scipy={scipy.__version__}'); print(f'numpy={numpy.__version__}')"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e21_uci_bootstrap_ci.json"
OUT_NPZ="${RUN_DIR}/results/e21_uci_bootstrap_ci.npz"
OUT_TEX="${RUN_DIR}/results/tab_vine_nll_ci.tex"

CKPT="${E21_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
  CKPT_ARGS=(--checkpoint "${CKPT}")
fi

"${PYTHON_BIN}" drafts/scripts/e21_uci_bootstrap_ci.py \
  --output-base "${OUTPUT_BASE}" \
  "${CKPT_ARGS[@]}" \
  --device cuda \
  --datasets ${E21_DATASETS:-power gas hepmass credit miniboone} \
  --max-train "${E21_MAX_TRAIN:-200000}" \
  --max-test "${E21_MAX_TEST:-50000}" \
  --seed "${E21_SEED:-42}" \
  --n-boot "${E21_N_BOOT:-2000}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --out-json "${OUT_JSON}" \
  --out-npz "${OUT_NPZ}" \
  --out-tex "${OUT_TEX}" \
  2>&1 | tee "${RUN_DIR}/logs/e21_bootstrap.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e21_uci_bootstrap_ci.json"
cp "${OUT_NPZ}" "drafts/paper_outputs/e21_uci_bootstrap_ci.npz"
cp "${OUT_TEX}" "drafts/tables/tab_vine_nll_ci.tex"

echo ""
echo "============================================================================"
echo "DONE: E21 bootstrap CI at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "JSON: ${OUT_JSON}"
echo "NPZ: ${OUT_NPZ}"
echo "Table: ${OUT_TEX}"
