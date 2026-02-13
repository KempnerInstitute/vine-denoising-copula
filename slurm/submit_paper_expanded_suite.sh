#!/bin/bash
# ============================================================================
# Submit expanded paper benchmarking suite (multi-seed + stronger settings).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
PARTITION="${PARTITION:-kempner_eng}"
SEEDS="${SEEDS:-42 123 456}"
CHECKPOINT="${PAPER_CHECKPOINT:-}"
E3_SBATCH_TIME="${E3_SBATCH_TIME:-48:00:00}"
E6_SBATCH_TIME="${E6_SBATCH_TIME:-24:00:00}"
E7_SBATCH_TIME="${E7_SBATCH_TIME:-24:00:00}"
RUN_E6="${RUN_E6:-1}"
RUN_E7="${RUN_E7:-1}"

if [ -z "${CHECKPOINT}" ]; then
  CHECKPOINT="$(python - <<'PY'
import os
from pathlib import Path
import sys
repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))
from vdc.utils.paper import choose_best_checkpoint
output_base = Path(os.environ.get("OUTPUT_BASE", "/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula"))
ckpt = choose_best_checkpoint(
    output_bases=[output_base],
    preferred_methods=["denoiser_cond_enhanced", "denoiser_cond"],
    metric="mean_ise",
    prefer_joint=True,
)
print("" if ckpt is None else str(ckpt))
PY
)"
fi

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: unable to resolve PAPER_CHECKPOINT."
  exit 1
fi

echo "============================================================================"
echo "Submitting expanded paper suite"
echo "Date: $(date)"
echo "Partition: ${PARTITION}"
echo "Output base: ${OUTPUT_BASE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Seeds: ${SEEDS}"
echo "E3 walltime: ${E3_SBATCH_TIME}"
echo "E6 walltime: ${E6_SBATCH_TIME} (run=${RUN_E6})"
echo "E7 walltime: ${E7_SBATCH_TIME} (run=${RUN_E7})"
echo "============================================================================"

declare -a JOB_IDS

submit() {
  local label="$1"
  shift
  local jid
  jid="$(sbatch --parsable --partition="${PARTITION}" "$@")"
  JOB_IDS+=("${jid}")
  printf "  %-30s %s\n" "${label}" "${jid}"
}

echo "Submitting core information benchmarks..."
submit "MI benchmark" \
  --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" \
  slurm/paper_mi_benchmark.sh
submit "MI consistency" \
  --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" \
  slurm/paper_mi_consistency.sh
submit "TC benchmark" \
  --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" \
  slurm/paper_tc_benchmark.sh

if [ "${RUN_E6}" = "1" ]; then
  submit "E6 theory suite" \
    --time="${E6_SBATCH_TIME}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" \
    slurm/paper_e6_theory.sh
fi

if [ "${RUN_E7}" = "1" ]; then
  submit "E7 biomed suite" \
    --time="${E7_SBATCH_TIME}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}" \
    slurm/paper_e7_biomed.sh
fi

echo "Submitting expanded E2/E3/E4/E5 jobs..."
for seed in ${SEEDS}; do
  submit "E2 UCI seed=${seed}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}",E2_SEED="${seed}",E2_GAUSSIAN_COPULA=1,E2_PYVINE=both,E2_MAX_TRAIN=200000,E2_MAX_TEST=50000 \
    slurm/paper_e2_uci.sh

  submit "E3 VaR seed=${seed}" \
    --time="${E3_SBATCH_TIME}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}",E3_SEED="${seed}",E3_WINDOW=252,E3_REFIT_EVERY=5,E3_N_SIM=5000,E3_MAX_DAYS=0 \
    slurm/paper_e3_var.sh

  submit "E4 impute seed=${seed}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}",E4_SEED="${seed}",E4_MISSING_FRAC=0.20,E4_N_EVAL=1000,E4_CANDIDATE_POOL=50000,E4_KERNEL_H=0.05,E4_MAX_TRAIN=200000,E4_MAX_TEST=50000 \
    slurm/paper_e4_imputation.sh

  submit "E5 anomaly seed=${seed}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}",E5_SEED="${seed}",E5_MAX_TRAIN=50000,E5_MAX_TEST=50000,E5_SCORE_MODE=neg_logpdf \
    slurm/paper_e5_anomaly.sh
done

echo ""
echo "Submitted job IDs:"
printf "  %s\n" "${JOB_IDS[@]}"
echo ""
echo "Monitor:"
echo "  squeue -u ${USER}"
echo "  sacct -j $(IFS=,; echo "${JOB_IDS[*]}") --format=JobID,JobName%30,Partition,State,Elapsed,ExitCode -P"
