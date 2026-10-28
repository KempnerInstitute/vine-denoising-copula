#!/bin/bash
#SBATCH --job-name=vdc_e22_highd
#SBATCH --output=logs/vdc_e22_highd_%j.out
#SBATCH --error=logs/vdc_e22_highd_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e22_highdim_scaling.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "VDC PAPER JOB: E22 high-dim Clayton-vine scaling"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Repo: ${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e22_highdim_${TS}_${SLURM_JOB_ID:-nojobid}"
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

OUT_JSON="${RUN_DIR}/results/e22_highdim_scaling.json"
OUT_MD="${RUN_DIR}/results/e22_highdim_scaling_summary.md"
OUT_TEX="${RUN_DIR}/results/tab_highdim_scaling.tex"

"${PYTHON_BIN}" drafts/scripts/e22_highdim_scaling.py \
  --dims ${E22_DIMS:-100 200 500} \
  --n-train "${E22_N_TRAIN:-20000}" \
  --n-test "${E22_N_TEST:-5000}" \
  --theta "${E22_THETA:-2.0}" \
  --seed "${E22_SEED:-42}" \
  --device cuda \
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

cp "${OUT_JSON}" "drafts/paper_outputs/e22_highdim_scaling.json"
cp "${OUT_MD}" "drafts/paper_outputs/e22_highdim_scaling_summary.md"
cp "${OUT_TEX}" "drafts/tables/tab_highdim_scaling.tex"

echo ""
echo "============================================================================"
echo "DONE: E22 high-dim scaling at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo "JSON: ${OUT_JSON}"
echo "Table: ${OUT_TEX}"
