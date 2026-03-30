#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"

sbatch \
  --job-name="vdc_e13_depth" \
  --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}" \
  "${SCRIPT_DIR}/paper_e13_depth_stability.sh"
