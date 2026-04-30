#!/bin/bash
#SBATCH --job-name=vdc_paper_e18_nonsimp
#SBATCH --output=logs/vdc_paper_e18_nonsimp_%j.out
#SBATCH --error=logs/vdc_paper_e18_nonsimp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e18_nonsimplified_stress.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
mkdir -p "${REPO_ROOT}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e18_nonsimplified_${TS}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "${RUN_DIR}/"{results,logs,analysis}

module purge
module load cuda/12.2.0-fasrc01
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

PYTHON_BIN="${VDC_PYTHON_BIN:-/n/home13/hsafaai/.venvs/vdc_paper/bin/python}"

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OUTPUT_BASE="${OUTPUT_BASE}"

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e18_nonsimplified_stress.json"
OUT_TEX="${RUN_DIR}/results/tab_nonsimplified_stress.tex"
OUT_FIG="${RUN_DIR}/results/fig_nonsimplified_stress"

"${PYTHON_BIN}" drafts/scripts/e18_nonsimplified_stress.py \
  --device "${E18_DEVICE:-cuda}" \
  --seed "${E18_SEED:-0}" \
  --n-train "${E18_N_TRAIN:-2000}" \
  --n-test "${E18_N_TEST:-10000}" \
  --rho-low "${E18_RHO_LOW:--0.75}" \
  --rho-high "${E18_RHO_HIGH:-0.75}" \
  --out-json "${OUT_JSON}" \
  --out-tex "${OUT_TEX}" \
  --out-fig "${OUT_FIG}" \
  2>&1 | tee "${RUN_DIR}/logs/e18_nonsimplified_stress.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e18_nonsimplified_stress.json"
cp "${OUT_TEX}" "drafts/tables/tab_nonsimplified_stress.tex"
cp "${OUT_FIG}.pdf" "drafts/figures/fig_nonsimplified_stress.pdf"
cp "${OUT_FIG}.png" "drafts/figures/fig_nonsimplified_stress.png"

echo "DONE: E18 non-simplified stress test at $(date)"
echo "Run Dir: ${RUN_DIR}"
