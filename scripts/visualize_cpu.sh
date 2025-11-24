#!/bin/bash
# Helper script to run visualization on CPU (for login nodes)
# Usage: bash scripts/visualize_cpu.sh CHECKPOINT_PATH [OUTPUT_DIR]

CHECKPOINT=$1
OUTPUT_DIR=${2:-"${CHECKPOINT%/*}/visualizations/eval"}

echo "Running visualization on CPU..."
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""

# Force CPU execution
export CUDA_VISIBLE_DEVICES=""

python scripts/visualize_diffusion_offline.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --device cpu

echo ""
echo "Visualization complete! Results saved to: $OUTPUT_DIR"
