#!/bin/bash
# Load CUDA module before running PyTorch visualization
module load cuda/12.2.0-fasrc01

python scripts/visualize_results.py \
  --checkpoint checkpoints/enhanced_cnn/model_step_30000.pt \
  --output visualizations/enhanced_cnn_eval \
  --m 128
