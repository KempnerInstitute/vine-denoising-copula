#!/bin/bash
# Load CUDA module before running PyTorch visualization
module load cuda/12.2.0-fasrc01

python scripts/visualize_diffusion_offline.py \
  --checkpoint checkpoints/train_diffusion_no_probit_normalized/model_step_20000.pt \
  --output-dir checkpoints/train_diffusion_no_probit_normalized/visualizations/eval \
  --num-samples 8 \
  --color-scale-mode both \
  --projection-iters 15
