#!/usr/bin/env bash
# Submit E14 latent-image benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-kempner_h100_priority3}"
EXCLUDE_NODES="${EXCLUDE_NODES:-holygpu8a11403,holygpu8a11501,holygpu8a13404,holygpu8a15504}"

echo "Submitting E14 latent image benchmark"
echo "Partition: ${PARTITION}"
echo "Dataset: ${E14_DATASET:-mnist}"
echo "Latent method: ${E14_LATENT_METHOD:-pca}"
echo "Latent dim: ${E14_LATENT_DIM:-16}"

sbatch \
  --partition="${PARTITION}" \
  --exclude="${EXCLUDE_NODES}" \
  "${SCRIPT_DIR}/paper_e14_mnist_latent.sh"
