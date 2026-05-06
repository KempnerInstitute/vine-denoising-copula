#!/bin/bash
#SBATCH --job-name=vdc_seed_merge
#SBATCH --output=logs/vdc_seed_merge_%j.out
#SBATCH --error=logs/vdc_seed_merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#
# ============================================================================
# Follow-on job for paper_seed_sweep.sh.
# Submit with: sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/paper_seed_sweep_merge.sh
# Merges per-seed JSONs and re-renders the three headline figures.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/drafts/scripts" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

echo "============================================================================"
echo "VDC SEED-SWEEP MERGE + FIGURE REGENERATION"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-} | Node: $(hostname) | Start: $(date)"
echo "Repo: ${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs"

module purge

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

VDC_PAPER_ENV="${VDC_PAPER_ENV:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2}"
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
export PYTHONUNBUFFERED=1

echo ""
echo "Staged seed-sweep artifacts:"
ls -1 "${REPO_ROOT}/drafts/paper_outputs/seed_sweep/" 2>/dev/null || true

# ---- 1. Merge per-seed JSONs --------------------------------------------------
echo ""
echo "[1/4] python drafts/scripts/merge_seed_runs.py"
"${PYTHON_BIN}" drafts/scripts/merge_seed_runs.py

# ---- 1b. Regenerate seed-dependent tables -------------------------------------
echo ""
echo "[1b] python drafts/scripts/regen_seed_tables.py"
"${PYTHON_BIN}" drafts/scripts/regen_seed_tables.py

# ---- 2. Re-render Figure 4 (density + runtime composite) ---------------------
echo ""
echo "[2/4] python drafts/scripts/fig4_density_runtime_composite_v2.py"
"${PYTHON_BIN}" drafts/scripts/fig4_density_runtime_composite_v2.py || \
  echo "WARN: fig4 regeneration failed; continuing"

# ---- 3. Re-render Figure 5 (information composite) ---------------------------
echo ""
echo "[3/4] python drafts/scripts/fig5_information_composite.py"
"${PYTHON_BIN}" drafts/scripts/fig5_information_composite.py || \
  echo "WARN: fig5 regeneration failed; continuing"

# ---- 4. Re-render Figure 3 (TC story composite) ------------------------------
echo ""
echo "[4/4] python drafts/scripts/fig_tc_story_composite.py"
"${PYTHON_BIN}" drafts/scripts/fig_tc_story_composite.py --second-dataset gas --third-dataset credit || \
  echo "WARN: fig_tc_story regeneration failed; continuing"

echo ""
echo "============================================================================"
echo "DONE: merge + figure regeneration at $(date)"
echo "============================================================================"
echo ""
echo "Re-compile the paper to pick up the new figures:"
echo "  cd ${REPO_ROOT}/drafts && pdflatex paper_neurips.tex && bibtex paper_neurips && pdflatex paper_neurips.tex && pdflatex paper_neurips.tex"
