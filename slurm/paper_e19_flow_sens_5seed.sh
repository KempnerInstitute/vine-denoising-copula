#!/bin/bash
#SBATCH --job-name=vdc_e19_5seed
#SBATCH --output=logs/vdc_e19_5seed_%j.out
#SBATCH --error=logs/vdc_e19_5seed_%j.err
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
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e19_flow_sensitivity.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "VDC PAPER JOB: E19 flow sensitivity, 5-seed Clayton-vine"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Repo: ${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e19_5seed_${TS}_${SLURM_JOB_ID:-nojobid}"
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
  echo "ERROR: no usable Python at ${VDC_PAPER_ENV}/bin/python or VDC_PYTHON_BIN."
  exit 3
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -c "import sys, torch, scipy, numpy; print(sys.executable); print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}'); print(f'scipy={scipy.__version__}'); print(f'numpy={numpy.__version__}')"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e19_flow_sensitivity.json"
OUT_MD="${RUN_DIR}/results/e19_flow_sensitivity_summary.md"
OUT_TEX="${RUN_DIR}/results/tab_flow_sensitivity.tex"

"${PYTHON_BIN}" drafts/scripts/e19_flow_sensitivity.py \
  --mode "${E19_MODE:-both}" \
  --device cuda \
  --output-base "${OUTPUT_BASE}" \
  --include-vdc \
  --synthetic-scenarios ${E19_SYNTHETIC_SCENARIOS:-clayton_vine} \
  --synthetic-dims ${E19_SYNTHETIC_DIMS:-20 50} \
  --synthetic-trials "${E19_SYNTHETIC_TRIALS:-5}" \
  --synthetic-n-train "${E19_SYNTHETIC_N_TRAIN:-5000}" \
  --synthetic-n-test "${E19_SYNTHETIC_N_TEST:-2000}" \
  --uci-datasets ${E19_UCI_DATASETS:-power hepmass} \
  --uci-seeds ${E19_UCI_SEEDS:-7 17 42} \
  --uci-max-train "${E19_UCI_MAX_TRAIN:-200000}" \
  --uci-max-test "${E19_UCI_MAX_TEST:-50000}" \
  --batch-size "${E19_BATCH_SIZE:-2048}" \
  --eval-batch-size "${E19_EVAL_BATCH_SIZE:-4096}" \
  --seed "${E19_SEED:-42}" \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}" \
  --out-tex "${OUT_TEX}" \
  2>&1 | tee "${RUN_DIR}/logs/e19_flow_sensitivity.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e19_flow_sensitivity.json"
cp "${OUT_MD}" "drafts/paper_outputs/e19_flow_sensitivity_summary.md"
cp "${OUT_TEX}" "drafts/tables/tab_flow_sensitivity.tex"

echo ""
echo "============================================================================"
echo "DONE: E19 5-seed flow sensitivity at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "JSON: ${OUT_JSON}"
echo "Table: ${OUT_TEX}"
