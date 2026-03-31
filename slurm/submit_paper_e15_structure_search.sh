#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PARTITION="${PARTITION:-kempner_h100_priority3}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
PYTHON_BIN="${VDC_PYTHON_BIN:-/n/netscratch/kempner_dev/hsafaai/conda_envs/vdc_paper/bin/python}"

sbatch \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --chdir="${REPO_ROOT}" \
  --export=ALL,VDC_REPO_ROOT="${REPO_ROOT}",VDC_PYTHON_BIN="${PYTHON_BIN}" \
  "${SCRIPT_DIR}/paper_e15_structure_search.sh"
