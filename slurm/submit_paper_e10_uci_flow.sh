#!/usr/bin/env bash
# Submit E10 global flow baseline benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-kempner_h100_priority3}"
SEEDS="${E10_SEEDS:-0 1 2}"
EXCLUDE_NODES="${EXCLUDE_NODES:-holygpu8a11403,holygpu8a11501,holygpu8a13404,holygpu8a15504}"

echo "Submitting E10 UCI flow benchmark"
echo "Partition: ${PARTITION}"
echo "Datasets: ${E10_DATASETS:-power gas hepmass miniboone}"
echo "Epochs: ${E10_EPOCHS:-25}"
echo "Seeds: ${SEEDS}"
echo "Exclude: ${EXCLUDE_NODES}"

for seed in ${SEEDS}; do
  echo "+ seed=${seed}"
  sbatch \
    --partition="${PARTITION}" \
    --exclude="${EXCLUDE_NODES}" \
    --export=ALL,E10_SEED="${seed}" \
    "${SCRIPT_DIR}/paper_e10_uci_flow.sh"
done
