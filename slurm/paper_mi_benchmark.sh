#!/bin/bash
#SBATCH --job-name=vdc_mi_benchmark
#SBATCH --output=logs/vdc_mi_benchmark_%j.out
#SBATCH --error=logs/vdc_mi_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#
# ============================================================================
# MI Estimation Benchmark for ICML 2026 Paper
# ============================================================================
# Compares DCD-Vine against MIST, MINDE, MINE, InfoNCE, KSG on bivariate
# copula MI estimation tasks.
#
# Output: results/mi_benchmark_*.json, drafts/tables/tab_mi.tex
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/mi_estimation.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

echo "============================================================================"
echo "MI Estimation Benchmark (All Methods)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/results"
mkdir -p "${REPO_ROOT}/external"

# Environment setup
module purge
module load cuda/12.2.0-fasrc01

# Fix libffi issue
export LD_LIBRARY_PATH="/n/sw/Mambaforge-23.11.0-0/lib:${LD_LIBRARY_PATH:-}"
eval "$(conda shell.bash hook)" || true
set +u
if [ -n "${VDC_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${VDC_PYTHON_BIN}"
elif [ -n "${VDC_CONDA_ENV_PATH:-}" ]; then
  conda activate "${VDC_CONDA_ENV_PATH}"
  PYTHON_BIN="python"
elif conda activate vdc 2>/dev/null; then
  PYTHON_BIN="python"
elif conda activate diffuse_vine_cop 2>/dev/null; then
  PYTHON_BIN="python"
else
  echo "ERROR: failed to activate conda env. Set VDC_CONDA_ENV_PATH or VDC_PYTHON_BIN."
  exit 3
fi
set -u

if [ "${PYTHON_BIN}" != "python" ] && [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: VDC_PYTHON_BIN is not executable: ${PYTHON_BIN}"
  exit 3
fi

cd "${REPO_ROOT}"

N_SAMPLES=5000
SEED=42
DCD_CKPT="${MI_CHECKPOINT:-${PAPER_CHECKPOINT:-}}"
DCD_CKPT_ARGS=()
if [ -n "${DCD_CKPT}" ]; then
  DCD_CKPT_ARGS=(--checkpoint "${DCD_CKPT}")
fi

echo ""
echo "Running MI estimators with n=${N_SAMPLES}, seed=${SEED}"
echo ""

# 1. KSG (fast, baseline)
echo "[1/7] Running KSG..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator ksg \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --out-json results/mi_ksg.json

# 2. DCD-Vine (ours)
echo "[2/7] Running DCD-Vine..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator dcd \
    --output-base "${OUTPUT_BASE}" \
    "${DCD_CKPT_ARGS[@]}" \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --dcd-seed-base "${MI_DCD_SEED_BASE:-5}" \
    --dcd-diffusion-steps "${MI_DCD_DIFFUSION_STEPS:-8}" \
    --dcd-diffusion-cfg-scale "${MI_DCD_CFG_SCALE:-1.0}" \
    --dcd-diffusion-ensemble "${MI_DCD_ENSEMBLE:-1}" \
    --dcd-diffusion-ensemble-mode "${MI_DCD_ENSEMBLE_MODE:-geometric}" \
    --dcd-diffusion-smooth-sigma "${MI_DCD_SMOOTH_SIGMA:-0.0}" \
    --dcd-pred-noise-clip "${MI_DCD_PRED_NOISE_CLIP:-1.0}" \
    --out-json results/mi_dcd.json

# 3. Gaussian MI
echo "[3/7] Running Gaussian MI..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator gaussian \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --out-json results/mi_gaussian.json

# 4. InfoNCE
echo "[4/7] Running InfoNCE..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator infonce \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --mine-steps 2000 \
    --mine-lr 1e-3 \
    --out-json results/mi_infonce.json

# 5. MINE
echo "[5/7] Running MINE..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator mine \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --mine-steps 2000 \
    --mine-lr 1e-3 \
    --out-json results/mi_mine.json

# 6. NWJ
echo "[6/7] Running NWJ..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator nwj \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --mine-steps 2000 \
    --mine-lr 1e-3 \
    --out-json results/mi_nwj.json

# 7. MINDE (slower, needs external repo)
echo "[7/7] Running MINDE..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator minde \
    --clone-minde \
    --minde-repo external/minde \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --minde-max-epochs 50 \
    --out-json results/mi_minde.json || echo "MINDE failed (optional dependency)"

# 8. MIST (optional, external dependency)
if [ "${MI_INCLUDE_MIST:-0}" = "1" ]; then
echo "[8/8] Running MIST..."
"${PYTHON_BIN}" scripts/mi_estimation.py \
    --estimator mist \
    --clone-mist \
    --mist-repo external/mist \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --mist-max-epochs "${MI_MIST_EPOCHS:-100}" \
    --out-json results/mi_mist.json || echo "MIST failed (optional dependency)"
fi

echo ""
echo "============================================================================"
echo "MI Benchmark Complete!"
echo "Results in: results/mi_*.json"
echo "============================================================================"

# Aggregate results into a summary
"${PYTHON_BIN}" -c "
import json
from pathlib import Path

results_dir = Path('results')
methods = ['ksg', 'dcd', 'gaussian', 'infonce', 'mine', 'nwj', 'minde', 'mist']
summary = {}

for m in methods:
    f = results_dir / f'mi_{m}.json'
    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
        mae = data.get('mean_abs_err')
        t = data.get('mean_time_s')
        summary[m] = {
            'mean_abs_err': mae,
            'mean_time_s': t,
        }
        mae_s = f'{float(mae):.4f}' if mae is not None else 'N/A'
        t_s = f'{float(t):.2f}s' if t is not None else 'N/A'
        print(f'{m}: MAE={mae_s}, time={t_s}')

with open(results_dir / 'mi_benchmark_summary.json', 'w') as fp:
    json.dump(summary, fp, indent=2)
print(f'\nSummary saved to: results/mi_benchmark_summary.json')
"
