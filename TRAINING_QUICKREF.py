"""
QUICK REFERENCE: Training Diffusion Copula Networks
====================================================

STEP 1: Generate Training Data
-------------------------------
# Generate 5M training samples
python -m vdc.data.generators \\
    --output data/train \\
    --n-samples 5000000 \\
    --m 64 \\
    --families clayton gumbel frank gaussian t \\
    --tau-range -0.8 0.8

# Generate 500K validation samples
python -m vdc.data.generators \\
    --output data/val \\
    --n-samples 500000 \\
    --m 64

STEP 2: Quick Test (5 minutes, 1 GPU)
--------------------------------------
python scripts/quick_train.py --mode test --data-dir data

STEP 3: Local Training (12 hours, 4 GPUs)
------------------------------------------
python scripts/quick_train.py \\
    --mode local \\
    --gpus 4 \\
    --steps 200000 \\
    --batch-size 64 \\
    --data-dir data \\
    --wandb

STEP 4: Cluster Training (48 hours, 16 GPUs)
---------------------------------------------
python scripts/quick_train.py \\
    --mode cluster \\
    --nodes 4 \\
    --gpus 4 \\
    --steps 500000 \\
    --batch-size 64 \\
    --data-dir /path/to/data \\
    --wandb \\
    --time 72 \\
    --partition gpu

Alternative: Manual Launch
--------------------------
# Edit config
vim configs/train_large.yaml

# Launch training
python scripts/train_large_scale.py --config configs/train_large.yaml

# Or with torchrun (multi-GPU)
torchrun --nproc_per_node=4 \\
    scripts/train_large_scale.py \\
    --config configs/train_large.yaml

# Or submit SLURM job
sbatch scripts/slurm/train_large_ddp.sh

STEP 5: Monitor Training
-------------------------
# View logs
tail -f logs/train_*.out

# Check checkpoints
ls -lh checkpoints/large_scale/

# W&B dashboard (if enabled)
# https://wandb.ai/your-project

STEP 6: Resume Training (if interrupted)
-----------------------------------------
# Automatic (if auto_resume: true in config)
python scripts/train_large_scale.py --config configs/train_large.yaml

# Manual
python scripts/train_large_scale.py \\
    --config configs/train_large.yaml \\
    --resume checkpoints/large_scale/latest.pt

STEP 7: Use Trained Model
--------------------------
from vdc.vine.api import VineCopulaModel

# Load model
model = VineCopulaModel.load('checkpoints/large_scale/best.pt')

# Fit vine to data
model.fit_rvine(X, method='dissmann')
model.fit_dvine(X, optimize_order=True)
model.fit_cvine(X, optimize_order=True)

# Evaluate
log_lik = model.logpdf(X_test)
samples = model.simulate(n=10000)

CONFIG FILES
============

configs/train_large.yaml
------------------------
Production training:
- m: 64
- base_channels: 128
- max_steps: 500000
- batch_size: 64
- Estimated time: 48h on 16 GPUs

configs/train_quick.yaml
------------------------
Quick test:
- m: 32
- base_channels: 64
- max_steps: 50000
- batch_size: 32
- Estimated time: 2h on 4 GPUs

IMPORTANT PATHS TO UPDATE IN CONFIGS
=====================================
data:
  train_root: "/path/to/your/train/data"  # ← UPDATE THIS
  val_root: "/path/to/your/val/data"      # ← UPDATE THIS

wandb:
  project: "your-project-name"            # ← UPDATE THIS

HARDWARE REQUIREMENTS
=====================

Quick Test:
- 1 GPU (16GB VRAM)
- 32GB RAM
- 100GB disk

Local Multi-GPU:
- 4 GPUs (40GB VRAM each)
- 128GB RAM
- 500GB disk

Production Cluster:
- 4 nodes × 4 GPUs (40GB each)
- 256GB RAM per node
- 1TB fast storage (SSD)
- InfiniBand (optional)

EXPECTED PERFORMANCE
====================

After 500k steps on 5M samples:

Metric          | Train | Val
----------------|-------|-----
NLL             | 0.020 | 0.030
ISE             | 0.001 | 0.002
Marginal        | 0.003 | 0.004

Throughput: ~2.8 steps/sec on 16 GPUs
Training time: ~48 hours

TROUBLESHOOTING
===============

Out of Memory:
- Reduce batch_size
- Increase grad_accum_steps
- Reduce base_channels or m

Loss Not Decreasing:
- Check data is correct
- Reduce learning rate
- Increase warmup_steps

NaN Loss:
- Enable gradient clipping (max_grad_norm: 1.0)
- Reduce learning rate
- Check projection_iters

Slow Training:
- Increase num_workers (4-8)
- Enable cache_size (100-500)
- Use SSD for data storage

COMPLETE WORKFLOW EXAMPLE
==========================

# 1. Setup
cd /path/to/vine_diffusion_copula
conda activate vine-copula

# 2. Generate data (2-4 hours)
python -m vdc.data.generators \\
    --output data/train \\
    --n-samples 5000000 \\
    --m 64 \\
    --n-jobs 32

# 3. Update config
sed -i 's|train_root:.*|train_root: "data/train"|' configs/train_large.yaml
sed -i 's|val_root:.*|val_root: "data/val"|' configs/train_large.yaml

# 4. Test (5 min)
python scripts/quick_train.py --mode test

# 5. Train (48 hours)
python scripts/quick_train.py \\
    --mode cluster \\
    --nodes 4 \\
    --gpus 4 \\
    --wandb

# 6. Evaluate
python examples/end_to_end.py \\
    --model checkpoints/large_scale/best.pt \\
    --data data/test/sample.h5

# 7. Use for vine construction
python examples/compare_vines.py \\
    --model checkpoints/large_scale/best.pt \\
    --n-samples 1000

DOCUMENTATION
=============

TRAINING_GUIDE.md      - Complete training documentation
TRAINING_SUMMARY.md    - System overview and architecture
VINE_TYPES.md          - Vine construction guide
IMPLEMENTATION_COMPLETE.md - Full implementation details
CHEATSHEET.py          - Code examples

CONTACT & SUPPORT
=================

For issues, questions, or contributions, see the main README.md
"""

# Print this file for quick reference
if __name__ == "__main__":
    print(__doc__)
