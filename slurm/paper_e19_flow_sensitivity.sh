#!/bin/bash
#SBATCH --job-name=vdc_e19_flow_sens
#SBATCH --output=logs/vdc_e19_flow_sens_%j.out
#SBATCH --error=logs/vdc_e19_flow_sens_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=192GB
#SBATCH --partition=kempner_h100_priority3
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
echo "VDC PAPER JOB: E19 stronger-flow sensitivity"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Repo: ${REPO_ROOT}"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e19_flow_sensitivity_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01

export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
else
  PYTHON_BIN="/n/sw/Mambaforge-23.11.0-0/bin/python"
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Testing Python..."
"${PYTHON_BIN}" -c "import sys, torch, scipy; print(sys.executable); print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}'); print(f'scipy={scipy.__version__}')"

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
  --synthetic-trials "${E19_SYNTHETIC_TRIALS:-2}" \
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
echo "DONE: E19 stronger-flow sensitivity completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "JSON: ${OUT_JSON}"
echo "Summary: ${OUT_MD}"
echo "Table: ${OUT_TEX}"
