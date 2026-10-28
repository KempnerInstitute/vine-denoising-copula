#!/bin/bash
#SBATCH --job-name=vdc_e22_d500
#SBATCH --output=logs/vdc_e22_d500_%j.out
#SBATCH --error=logs/vdc_e22_d500_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# Focused rerun: d=500 only, pyvine wall-clock budget extended to ~6h so the
# pyvine_tll cell gets a real NLL instead of "timeout(2400s)". VDC, RealNVP,
# Gaussian numbers will be re-collected too (cheap) and used for a clean merge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/drafts/scripts/e22_highdim_scaling.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
export OUTPUT_BASE

echo "============================================================================"
echo "VDC PAPER JOB: E22 d=500 focused rerun (long pyvine budget)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p "${REPO_ROOT}/logs"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_e22_d500_${TS}_${SLURM_JOB_ID:-nojobid}"
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
  echo "ERROR: no usable Python."
  exit 3
fi

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cp "$0" "${RUN_DIR}/analysis/slurm_script.sh"
git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/analysis/git_commit.txt" || true

OUT_JSON="${RUN_DIR}/results/e22_d500_only.json"
OUT_MD="${RUN_DIR}/results/e22_d500_only_summary.md"
OUT_TEX_TMP="${RUN_DIR}/results/e22_d500_only.tex"

# 6 hour pyvine budget; 16 threads. d=500 only.
"${PYTHON_BIN}" drafts/scripts/e22_highdim_scaling.py \
  --dims 500 \
  --n-train "${E22_N_TRAIN:-20000}" \
  --n-test "${E22_N_TEST:-5000}" \
  --theta "${E22_THETA:-2.0}" \
  --seed "${E22_SEED:-42}" \
  --device cuda \
  --output-base "${OUTPUT_BASE}" \
  --m-grid "${E22_M_GRID:-64}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --pyvine-timeout-s "${E22_PYVINE_TIMEOUT_S:-21600}" \
  --include-realnvp \
  --include-gaussian \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}" \
  --out-tex "${OUT_TEX_TMP}" \
  2>&1 | tee "${RUN_DIR}/logs/e22_d500.log"

# Merge into the previous full JSON and regenerate the combined table
"${PYTHON_BIN}" - <<'PYEOF'
import json
from pathlib import Path

repo = Path("/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula")
prior = repo / "drafts/paper_outputs/e22_highdim_scaling.json"
new_j = Path("OUT_JSON_PLACEHOLDER".replace("OUT_JSON_PLACEHOLDER", "${OUT_JSON}"))
PYEOF
"${PYTHON_BIN}" - <<PYEOF
import json
from pathlib import Path
import sys

repo = Path("/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula")
prior = json.loads((repo / "drafts/paper_outputs/e22_highdim_scaling.json").read_text())
new = json.loads(Path("${OUT_JSON}").read_text())

# Drop d=500 records from prior (where they had timeout / partial), splice in new ones.
prior_records = [r for r in prior["records"] if int(r["d"]) != 500]
new_records = list(new["records"])
merged = prior_records + new_records

# Re-emit table from merged records
sys.path.insert(0, str(repo / "drafts/scripts"))
from e22_highdim_scaling import emit_tex, emit_summary_md

out_json = repo / "drafts/paper_outputs/e22_highdim_scaling.json"
out_md = repo / "drafts/paper_outputs/e22_highdim_scaling_summary.md"
out_tex = repo / "drafts/tables/tab_highdim_scaling.tex"

merged_payload = dict(prior)
merged_payload["records"] = merged
merged_payload["d500_rerun_settings"] = new.get("settings", {})
out_json.write_text(json.dumps(merged_payload, default=str, indent=2))
emit_summary_md(merged, out_md)
emit_tex(merged, out_tex)
print("Merged d=500 records into", out_json)
print("Updated table at", out_tex)
PYEOF

echo ""
echo "============================================================================"
echo "DONE: E22 d=500 focused rerun + merge at $(date)"
echo "============================================================================"
echo "Run Dir: ${RUN_DIR}"
