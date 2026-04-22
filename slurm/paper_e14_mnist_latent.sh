#!/bin/bash
#SBATCH --job-name=vdc_paper_e14_mnist
#SBATCH --output=logs/vdc_paper_e14_mnist_%j.out
#SBATCH --error=logs/vdc_paper_e14_mnist_%j.err
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
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/e14_mnist_latent_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "Vine Denoising Copula PAPER JOB: E14 MNIST latent benchmark"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e14_mnist_latent_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01

export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
else
  PYTHON_BIN="/n/netscratch/kempner_dev/hsafaai/conda_envs/vdc_paper/bin/python"
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

SEED_TAG="${E14_SEED:-42}"
OUT_JSON="${RUN_DIR}/results/e14_mnist_latent_results_seed${SEED_TAG}.json"
OUT_SUMMARY="${RUN_DIR}/results/e14_mnist_latent_summary_seed${SEED_TAG}.md"

"${PYTHON_BIN}" scripts/e14_mnist_latent_benchmark.py \
  --dataset "${E14_DATASET:-mnist}" \
  --latent-method "${E14_LATENT_METHOD:-pca}" \
  --latent-dim "${E14_LATENT_DIM:-16}" \
  --max-train "${E14_MAX_TRAIN:-20000}" \
  --max-test "${E14_MAX_TEST:-5000}" \
  --download \
  --device cuda \
  --seed "${SEED_TAG}" \
  --ae-epochs "${E14_AE_EPOCHS:-10}" \
  --ae-batch-size "${E14_AE_BATCH_SIZE:-512}" \
  --ae-lr "${E14_AE_LR:-1e-3}" \
  --flow-epochs "${E14_FLOW_EPOCHS:-25}" \
  --flow-batch-size "${E14_FLOW_BATCH_SIZE:-2048}" \
  --flow-eval-batch-size "${E14_FLOW_EVAL_BATCH_SIZE:-4096}" \
  --flow-lr "${E14_FLOW_LR:-1e-3}" \
  --flow-val-fraction "${E14_FLOW_VAL_FRACTION:-0.1}" \
  --flow-patience "${E14_FLOW_PATIENCE:-5}" \
  --flow-num-layers "${E14_FLOW_NUM_LAYERS:-8}" \
  --flow-hidden-dim "${E14_FLOW_HIDDEN_DIM:-128}" \
  --flow-hidden-layers "${E14_FLOW_HIDDEN_LAYERS:-2}" \
  --out-json "${OUT_JSON}" \
  --out-summary "${OUT_SUMMARY}" \
  2>&1 | tee "${RUN_DIR}/logs/e14_mnist_latent.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e14_mnist_latent_results.json"
cp "${OUT_SUMMARY}" "drafts/paper_outputs/e14_mnist_latent_summary.md"

echo ""
echo "============================================================================"
echo "DONE: E14 latent image benchmark completed at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
echo ""
