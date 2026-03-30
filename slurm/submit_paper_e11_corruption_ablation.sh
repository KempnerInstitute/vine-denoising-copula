#!/bin/bash
# Submit corruption-ablation training jobs using the generic denoiser paper script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/paper_vdc_denoiser_cond.sh"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "Missing training script: ${TRAIN_SCRIPT}"
  exit 2
fi

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
PARTITION="${PARTITION:-}"
JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX:-}"

declare -a VARIANTS=(
  "uniform_mix:${REPO_ROOT}/configs/train/denoiser_cond_enhanced.yaml:denoiser_cond_enhanced"
  "direct:${REPO_ROOT}/configs/train/denoiser_cond_enhanced_direct.yaml:denoiser_cond_enhanced_direct"
  "gaussian:${REPO_ROOT}/configs/train/denoiser_cond_enhanced_gaussian.yaml:denoiser_cond_enhanced_gaussian"
  "multinomial:${REPO_ROOT}/configs/train/denoiser_cond_enhanced_multinomial.yaml:denoiser_cond_enhanced_multinomial"
)

for spec in "${VARIANTS[@]}"; do
  IFS=":" read -r label config_src method_tag <<< "${spec}"
  echo "Submitting E11 variant ${label}"
  sbatch_args=()
  if [ -n "${PARTITION}" ]; then
    sbatch_args+=(--partition="${PARTITION}")
  fi
  sbatch \
    "${sbatch_args[@]}" \
    --job-name="vdc_e11_${label}${JOB_NAME_SUFFIX}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",CONFIG_SRC="${config_src}",MODEL_TYPE="denoiser",METHOD_TAG="${method_tag}" \
    "${TRAIN_SCRIPT}"
done
