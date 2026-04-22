#!/bin/bash
#SBATCH --job-name=vdc_seed_variance
#SBATCH --output=logs/vdc_seed_variance_%A_%a.out
#SBATCH --error=logs/vdc_seed_variance_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_priority
#SBATCH --account=kempner_dev
#SBATCH --array=0-2

# Seed variance array: three seeds in parallel on kempner_priority.
# Each task trains + evaluates RealNVP and runs classical baselines +
# frozen-VDC inference with a distinct seed on all five UCI datasets.

set -euo pipefail

SEEDS=(7 17 42)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula}"
# kempner_project_b on holylfs06 is full; write intermediate artifacts to netscratch.
OUTPUT_BASE="${OUTPUT_BASE:-/n/netscratch/kempner_dev/hsafaai/vdc_paper}"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME PYTHONPATH

PYTHON_BIN="${VDC_PYTHON_BIN:-/n/home13/hsafaai/.venvs/vdc_paper/bin/python}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_BASE}/vdc_paper_seed_variance_${TS}_${SLURM_JOB_ID:-nojobid}_seed${SEED}"
mkdir -p "${RUN_DIR}/"{results,logs}

OUT_JSON="${RUN_DIR}/results/e10_uci_flow_results_seed${SEED}.json"

echo "Running seed variance: seed=${SEED}, task=${SLURM_ARRAY_TASK_ID}"

"${PYTHON_BIN}" drafts/scripts/e10_uci_flow_benchmark.py \
  --device cuda \
  --datasets power gas hepmass credit miniboone \
  --max-train 200000 \
  --max-test 50000 \
  --seed "${SEED}" \
  --epochs 25 \
  --batch-size 2048 \
  --eval-batch-size 4096 \
  --lr 1e-3 \
  --val-fraction 0.1 \
  --patience 5 \
  --num-layers 8 \
  --hidden-dim 128 \
  --hidden-layers 2 \
  --out-json "${OUT_JSON}" \
  2>&1 | tee "${RUN_DIR}/logs/e10_uci_flow.log"

cp "${OUT_JSON}" "drafts/paper_outputs/e10_uci_flow_results_seed${SEED}.json"

echo "Seed ${SEED} complete: ${OUT_JSON}"
