#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV_PATH="${1:-/n/netscratch/kempner_dev/hsafaai/conda_envs/vdc_paper}"
CKPT_LIST="${2:-${REPO_ROOT}/results/checkpoint_candidates_m64_15.txt}"

if [ ! -x "${CONDA_ENV_PATH}/bin/python" ]; then
  echo "ERROR: conda env python does not exist or is not executable: ${CONDA_ENV_PATH}/bin/python" >&2
  exit 2
fi

if [ ! -f "${CKPT_LIST}" ]; then
  echo "ERROR: checkpoint list not found: ${CKPT_LIST}" >&2
  exit 2
fi

mapfile -t CKPTS < "${CKPT_LIST}"
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "ERROR: checkpoint list is empty: ${CKPT_LIST}" >&2
  exit 2
fi

submit_job() {
  local out
  out=$(sbatch "$@")
  echo "${out}" >&2
  awk '{print $4}' <<<"${out}"
}

echo "Submitting model-improvement jobs with conda env: ${CONDA_ENV_PATH}"

J_TRAIN_ENH=$(submit_job --export=ALL,VDC_CONDA_ENV_PATH="${CONDA_ENV_PATH}",PYTHONNOUSERSITE=0 "${SCRIPT_DIR}/paper_train_enhanced.sh")
J_DENOISER=$(submit_job --export=ALL,VDC_CONDA_ENV_PATH="${CONDA_ENV_PATH}",PYTHONNOUSERSITE=0 "${SCRIPT_DIR}/paper_vdc_denoiser_cond.sh")
J_ENHCNN=$(submit_job --export=ALL,VDC_CONDA_ENV_PATH="${CONDA_ENV_PATH}",PYTHONNOUSERSITE=0 "${SCRIPT_DIR}/paper_vdc_enhanced_cnn_cond.sh")
J_DIFF=$(submit_job --export=ALL,VDC_CONDA_ENV_PATH="${CONDA_ENV_PATH}",PYTHONNOUSERSITE=0 "${SCRIPT_DIR}/paper_vdc_diffusion_cond.sh")

J_SEL_S5=$(submit_job \
  --job-name=vdc_sel_s5 \
  --export=ALL,VDC_CONDA_ENV_PATH="${CONDA_ENV_PATH}",PYTHONNOUSERSITE=0,WRITE_EXAMPLES=0,DIFFUSION_SEED_BASE=5,N_SAMPLES=1000,DIFFUSION_STEPS=200,DIFFUSION_CFG_SCALE=4.0,DIFFUSION_PRED_NOISE_CLIP=10,SUITE=standard \
  "${SCRIPT_DIR}/paper_vdc_model_selection.sh" "${CKPTS[@]}")

J_SEL_S123=$(submit_job \
  --job-name=vdc_sel_s123 \
  --export=ALL,VDC_CONDA_ENV_PATH="${CONDA_ENV_PATH}",PYTHONNOUSERSITE=0,WRITE_EXAMPLES=0,DIFFUSION_SEED_BASE=123,N_SAMPLES=1000,DIFFUSION_STEPS=200,DIFFUSION_CFG_SCALE=4.0,DIFFUSION_PRED_NOISE_CLIP=10,SUITE=standard \
  "${SCRIPT_DIR}/paper_vdc_model_selection.sh" "${CKPTS[@]}")

cat <<EOF
Submitted jobs:
  train_enhanced: ${J_TRAIN_ENH}
  denoiser_cond: ${J_DENOISER}
  enhanced_cnn_cond: ${J_ENHCNN}
  diffusion_cond: ${J_DIFF}
  select_seed5: ${J_SEL_S5}
  select_seed123: ${J_SEL_S123}
EOF
