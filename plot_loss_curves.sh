#!/bin/bash
# Generate loss curve plots from saved training history

# For diffusion model
if [ -f "checkpoints/train_diffusion_no_probit_normalized/visualizations/loss_history.csv" ]; then
  echo "Plotting diffusion model loss curves..."
  python scripts/monitor_training.py \
    --log checkpoints/train_diffusion_no_probit_normalized/visualizations/loss_history.csv \
    --plot \
    --plot-output checkpoints/train_diffusion_no_probit_normalized/visualizations/loss_curves.png \
    --no-follow
  echo "✓ Saved: checkpoints/train_diffusion_no_probit_normalized/visualizations/loss_curves.png"
fi

# For enhanced CNN model
if [ -f "checkpoints/enhanced_cnn/visualizations/loss_history.csv" ]; then
  echo "Plotting enhanced CNN loss curves..."
  python scripts/monitor_training.py \
    --log checkpoints/enhanced_cnn/visualizations/loss_history.csv \
    --plot \
    --plot-output checkpoints/enhanced_cnn/visualizations/loss_curves.png \
    --no-follow
  echo "✓ Saved: checkpoints/enhanced_cnn/visualizations/loss_curves.png"
fi

echo ""
echo "Loss curve plotting complete!"
