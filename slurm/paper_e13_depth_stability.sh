#!/bin/bash
#SBATCH --job-name=vdc_paper_e13_depth
#SBATCH --output=logs/vdc_paper_e13_depth_%j.out
#SBATCH --error=logs/vdc_paper_e13_depth_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e13_depth_stability_benchmark.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

mkdir -p "${REPO_ROOT}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e13_depth_${TS}_${SLURM_JOB_ID:-nojobid}"
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

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e13_depth_stability_results.json"
ARGS=()

for spec in ${E13_LABEL_METHODS:-vdc:denoiser_cond_enhanced}; do
  ARGS+=(--label-method "${spec}")
done
for spec in ${E13_LABEL_CHECKPOINTS:-}; do
  ARGS+=(--label-checkpoint "${spec}")
done

"${PYTHON_BIN}" drafts/scripts/e13_depth_stability_benchmark.py \
  --device cuda \
  --seed "${E13_SEED:-0}" \
  --output-base "${OUTPUT_BASE}" \
  "${ARGS[@]}" \
  --out "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e13_depth_stability.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e13_depth_stability_results.json"
"${PYTHON_BIN}" drafts/scripts/paper_artifacts.py all --output-base "${OUTPUT_BASE}" --force \
  2>&1 | tee "${RUN_DIR}/logs/paper_artifacts_after_e13.log"
