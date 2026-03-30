#!/bin/bash
set -euo pipefail

# Submit paper-related SLURM jobs for the single-model paper setup
# (enhanced denoiser + downstream benchmarks).
#
# Usage:
#   OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula \
#   ./slurm/submit_all_paper_jobs.sh
#
# Optional:
#   PARTITION=kempner_h100_priority3 ./slurm/submit_all_paper_jobs.sh
#   SUBMIT_TRAINING=0 ./slurm/submit_all_paper_jobs.sh
#   SUBMIT_E1_BASELINES=0 ./slurm/submit_all_paper_jobs.sh
#   SUBMIT_MI=0 ./slurm/submit_all_paper_jobs.sh
#   SUBMIT_MODEL_SELECTION=1 CHECKPOINTS="/path/a.pt /path/b.pt" ./slurm/submit_all_paper_jobs.sh
#
# Notes:
# - Command-line sbatch options override script #SBATCH directives; we still pass --partition
#   to be explicit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PARTITION="${PARTITION:-kempner_h100_priority3}"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

SUBMIT_TRAINING="${SUBMIT_TRAINING:-1}"
SUBMIT_E1_BASELINES="${SUBMIT_E1_BASELINES:-1}"
SUBMIT_MI="${SUBMIT_MI:-1}"
SUBMIT_SCALING="${SUBMIT_SCALING:-1}"
SUBMIT_MODEL_SELECTION="${SUBMIT_MODEL_SELECTION:-0}"
SUBMIT_ARTIFACTS_JOB="${SUBMIT_ARTIFACTS_JOB:-1}"

cd "${REPO_ROOT}"
mkdir -p logs

export OUTPUT_BASE

echo "Submitting paper job suite"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  OUTPUT_BASE=${OUTPUT_BASE}"
echo "  PARTITION=${PARTITION}"
echo "  SUBMIT_TRAINING=${SUBMIT_TRAINING}"
echo "  SUBMIT_E1_BASELINES=${SUBMIT_E1_BASELINES}"
echo "  SUBMIT_MI=${SUBMIT_MI}"
echo "  SUBMIT_SCALING=${SUBMIT_SCALING}"
echo "  SUBMIT_MODEL_SELECTION=${SUBMIT_MODEL_SELECTION}"
echo "  SUBMIT_ARTIFACTS_JOB=${SUBMIT_ARTIFACTS_JOB}"
echo ""

submit() {
  # Print command and submit.
  echo "+ sbatch --partition=${PARTITION} $*" >&2
  sbatch --partition="${PARTITION}" "$@"
}

if [ "${SUBMIT_TRAINING}" = "1" ]; then
  echo "== Training job (single model) =="
  TRAIN_JIDS=()
  OUT=$(submit "${SCRIPT_DIR}/paper_train_enhanced.sh")
  TRAIN_JIDS+=("$(echo "${OUT}" | awk '{print $4}')")
  echo ""
fi

SCALING_JID=""
if [ "${SUBMIT_SCALING}" = "1" ]; then
  echo "== E2 scaling (vine build time vs dimension) =="
  if [ "${SUBMIT_TRAINING}" = "1" ]; then
    DEP="afterok:$(IFS=:; echo "${TRAIN_JIDS[*]}")"
    OUT=$(submit --dependency="${DEP}" "${SCRIPT_DIR}/paper_vdc_scaling.sh")
  else
    OUT=$(submit "${SCRIPT_DIR}/paper_vdc_scaling.sh")
  fi
  SCALING_JID="$(echo "${OUT}" | awk '{print $4}')"
  echo ""
fi

if [ "${SUBMIT_E1_BASELINES}" = "1" ]; then
  echo "== E1 bivariate baselines =="
  for b in histogram kde_probit pyvine_param pyvine_nonpar; do
    submit "${SCRIPT_DIR}/paper_vdc_e1_baseline.sh" "${b}"
  done
  echo ""
fi

if [ "${SUBMIT_MI}" = "1" ]; then
  echo "== MI estimator baselines =="
  for est in ksg dcd gaussian infonce nwj mine minde; do
    submit "${SCRIPT_DIR}/paper_vdc_mi_estimation.sh" "${est}"
  done
  echo ""
fi

if [ "${SUBMIT_MODEL_SELECTION}" = "1" ]; then
  echo "== Model selection (requires CHECKPOINTS env var) =="
  if [ -z "${CHECKPOINTS:-}" ]; then
    echo "ERROR: SUBMIT_MODEL_SELECTION=1 but CHECKPOINTS is empty."
    echo "Example: CHECKPOINTS=\"/path/to/a.pt /path/to/b.pt\" SUBMIT_MODEL_SELECTION=1 ./slurm/submit_all_paper_jobs.sh"
    exit 2
  fi
  # shellcheck disable=SC2206
  CKPTS=(${CHECKPOINTS})
  submit "${SCRIPT_DIR}/paper_vdc_model_selection.sh" "${CKPTS[@]}"
  echo ""
fi

if [ "${SUBMIT_ARTIFACTS_JOB}" = "1" ]; then
  echo "== Paper artifacts generation job =="
  DEP_IDS=()
  if [ "${SUBMIT_TRAINING}" = "1" ]; then
    DEP_IDS+=("${TRAIN_JIDS[@]}")
  fi
  if [ "${SUBMIT_SCALING}" = "1" ] && [ -n "${SCALING_JID}" ]; then
    DEP_IDS+=("${SCALING_JID}")
  fi
  if [ "${#DEP_IDS[@]}" -gt 0 ]; then
    DEP="afterok:$(IFS=:; echo "${DEP_IDS[*]}")"
    submit --dependency="${DEP}" "${SCRIPT_DIR}/paper_generate_artifacts.sh"
  else
    submit "${SCRIPT_DIR}/paper_generate_artifacts.sh"
  fi
  echo ""
fi

echo "Done submitting."
