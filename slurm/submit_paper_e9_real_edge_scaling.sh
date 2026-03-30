#!/usr/bin/env bash
# Submit real-edge scaling benchmark jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-kempner_h100_priority3}"
SEEDS="${E9_SEEDS:-0 1 2}"
EXCLUDE_NODES="${EXCLUDE_NODES:-holygpu8a11403,holygpu8a11501,holygpu8a13404,holygpu8a15504}"

echo "Submitting E9 real-edge scaling jobs"
echo "Partition: ${PARTITION}"
echo "Seeds: ${SEEDS}"
echo "Datasets: ${E9_DATASETS:-gas hepmass miniboone}"
echo "Epochs: ${E9_EPOCHS:-300}"
echo "Exclude: ${EXCLUDE_NODES}"

for seed in ${SEEDS}; do
  echo "+ seed=${seed}"
  sbatch \
    --partition="${PARTITION}" \
    --exclude="${EXCLUDE_NODES}" \
    --export=ALL,E9_SEED="${seed}" \
    "${SCRIPT_DIR}/paper_e9_real_edge_scaling.sh"
done
