#!/bin/bash
#SBATCH --job-name=vdc_e20_ipfp
#SBATCH --output=logs/vdc_e20_ipfp_%j.out
#SBATCH --error=logs/vdc_e20_ipfp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e20_ipfp_sensitivity.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "VDC PAPER JOB: E20 IPFP sensitivity"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Repo: ${REPO_ROOT}"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e20_ipfp_sensitivity_${TS}_${SLURM_JOB_ID:-nojobid}"
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

OUT_JSON="${RUN_DIR}/results/e20_ipfp_sensitivity.json"
OUT_MD="${RUN_DIR}/results/e20_ipfp_sensitivity_summary.md"
OUT_TEX="${RUN_DIR}/results/tab_ipfp_uci_sensitivity.tex"

"${PYTHON_BIN}" drafts/scripts/e20_ipfp_sensitivity.py \
  --output-base "${OUTPUT_BASE}" \
  --device cuda \
  --datasets ${E20_DATASETS:-power gas hepmass credit miniboone} \
  --ipfp-iters ${E20_IPFP_ITERS:-15 30 50 100} \
  --max-train "${E20_MAX_TRAIN:-200000}" \
  --max-test "${E20_MAX_TEST:-50000}" \
  --seed "${E20_SEED:-42}" \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}" \
  --out-tex "${OUT_TEX}" \
  2>&1 | tee "${RUN_DIR}/logs/e20_ipfp_sensitivity.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e20_ipfp_sensitivity.json"
cp "${OUT_MD}" "drafts/paper_outputs/e20_ipfp_sensitivity_summary.md"
cp "${OUT_TEX}" "drafts/tables/tab_ipfp_uci_sensitivity.tex"

echo ""
echo "============================================================================"
echo "DONE: E20 IPFP sensitivity completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "JSON: ${OUT_JSON}"
echo "Summary: ${OUT_MD}"
echo "Table: ${OUT_TEX}"
