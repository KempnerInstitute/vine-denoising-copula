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

REPO_ROOT="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula"

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
eval "$(conda shell.bash hook)" || true
conda activate vdc 2>/dev/null || conda activate diffuse_vine_cop 2>/dev/null || true

cd "${REPO_ROOT}"

N_SAMPLES=5000
SEED=42

echo ""
echo "Running MI estimators with n=${N_SAMPLES}, seed=${SEED}"
echo ""

# 1. KSG (fast, baseline)
echo "[1/6] Running KSG..."
python scripts/mi_estimation.py \
    --estimator ksg \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --out-json results/mi_ksg.json

# 2. Gaussian MI
echo "[2/6] Running Gaussian MI..."
python scripts/mi_estimation.py \
    --estimator gaussian \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --out-json results/mi_gaussian.json

# 3. InfoNCE
echo "[3/6] Running InfoNCE..."
python scripts/mi_estimation.py \
    --estimator infonce \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --mine-steps 2000 \
    --mine-lr 1e-3 \
    --out-json results/mi_infonce.json

# 4. MINE
echo "[4/6] Running MINE..."
python scripts/mi_estimation.py \
    --estimator mine \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --mine-steps 2000 \
    --mine-lr 1e-3 \
    --out-json results/mi_mine.json

# 5. NWJ
echo "[5/6] Running NWJ..."
python scripts/mi_estimation.py \
    --estimator nwj \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --mine-steps 2000 \
    --mine-lr 1e-3 \
    --out-json results/mi_nwj.json

# 6. MINDE (slower, needs external repo)
echo "[6/6] Running MINDE..."
python scripts/mi_estimation.py \
    --estimator minde \
    --clone-minde \
    --minde-repo external/minde \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED} \
    --minde-max-epochs 50 \
    --out-json results/mi_minde.json || echo "MINDE failed (optional dependency)"

echo ""
echo "============================================================================"
echo "MI Benchmark Complete!"
echo "Results in: results/mi_*.json"
echo "============================================================================"

# Aggregate results into a summary
python -c "
import json
from pathlib import Path

results_dir = Path('results')
methods = ['ksg', 'gaussian', 'infonce', 'mine', 'nwj', 'minde']
summary = {}

for m in methods:
    f = results_dir / f'mi_{m}.json'
    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
        summary[m] = {
            'mean_abs_err': data.get('mean_abs_err'),
            'mean_time_s': data.get('mean_time_s'),
        }
        print(f'{m}: MAE={data.get(\"mean_abs_err\", \"N/A\"):.4f}, time={data.get(\"mean_time_s\", \"N/A\"):.2f}s')

with open(results_dir / 'mi_benchmark_summary.json', 'w') as fp:
    json.dump(summary, fp, indent=2)
print(f'\nSummary saved to: results/mi_benchmark_summary.json')
"
