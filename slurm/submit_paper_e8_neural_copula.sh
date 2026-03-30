#!/usr/bin/env bash
# Submit multi-seed E8 neural-copula benchmark jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-kempner_h100_priority3}"
SEEDS="${E8_SEEDS:-0 1 2}"

echo "Submitting E8 neural-copula jobs"
echo "Partition: ${PARTITION}"
echo "Seeds: ${SEEDS}"
echo "Families: ${E8_FAMILIES:-all}"
echo "Epochs: ${E8_EPOCHS:-300}"

for seed in ${SEEDS}; do
  echo "+ seed=${seed}"
  sbatch \
    --partition="${PARTITION}" \
    --export=ALL,E8_SEED="${seed}" \
    "${SCRIPT_DIR}/paper_e8_neural_copula.sh"
done
