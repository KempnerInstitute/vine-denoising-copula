#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

for seed in ${E12_SEEDS:-42}; do
  echo "Submitting E12 seed ${seed}"
  sbatch \
    --job-name="vdc_e12_s${seed}" \
    --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",E12_SEED="${seed}" \
    "${SCRIPT_DIR}/paper_e12_block_mi.sh"
done
