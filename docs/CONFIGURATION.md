# Configuration Guide

This guide explains how to configure the Vine Denoising Copula package for different tasks:
1. **Training** - Train a diffusion model for copula density estimation
2. **Density Estimation** - Use a trained model to estimate copula densities
3. **Vine Sampling** - Fit a vine copula and generate samples

---

## Overview

The package uses YAML configuration files located in `configs/`:

```
configs/
├── default.yaml              # Master config (all settings)
├── train/
│   └── default.yaml          # Training-specific config
└── inference/
    └── default.yaml          # Inference/evaluation config
```

You can override any setting via command line:
```bash
python scripts/train.py --config configs/train/default.yaml training.max_steps=200000
```

---

## Task 1: Training the Copula Model

### Config File: `configs/train/default.yaml`

```yaml
# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

experiment:
  name: "my_copula_experiment"
  seed: 42

# MODEL ARCHITECTURE
model:
  type: "diffusion_unet"     # Model type (don't change)
  grid_size: 64              # Output density grid: 64×64
  in_channels: 1             # Input channels
  base_channels: 64          # Base feature channels (larger = more capacity)
  channel_mults: [1, 2, 3, 4]  # Channel multipliers per level
  num_res_blocks: 2          # Residual blocks per level
  attention_resolutions: [16, 8]  # Add attention at these resolutions
  dropout: 0.1               # Dropout rate

# DIFFUSION PROCESS
diffusion:
  timesteps: 1000            # Diffusion timesteps
  noise_schedule: "cosine"   # "linear" or "cosine" (cosine is better)

# TRAINING DATA
data:
  m: 64                      # Grid resolution (must match model.grid_size)
  n_samples_per_copula: 1000 # Samples to create each histogram
  copula_families:           # Which copula families to train on
    - "gaussian"
    - "student"
    - "clayton"
    - "gumbel"
    - "frank"
    - "joe"
  binning: "uniform"         # "uniform" or "probit" (probit helps tails)
  
  # Parameter ranges for random copula generation during training
  param_ranges:
    gaussian_rho: [-0.95, 0.95]     # Correlation range
    student_rho: [-0.9, 0.9]
    student_df: [2, 30]             # Degrees of freedom
    clayton_theta: [0.1, 10.0]      # Clayton parameter
    gumbel_theta: [1.1, 8.0]        # Gumbel parameter
    frank_theta: [-15.0, 15.0]      # Frank parameter (can be negative)
    joe_theta: [1.1, 8.0]           # Joe parameter

# TRAINING PARAMETERS
training:
  max_steps: 150000          # Total training iterations
  batch_size: 32             # Batch size per GPU
  learning_rate: 1.0e-4      # Learning rate
  weight_decay: 0.01         # AdamW weight decay
  gradient_clip: 1.0         # Gradient clipping
  
  # Loss function weights
  loss_weights:
    ce: 1.0                  # Cross-entropy (primary loss)
    ise: 0.1                 # Integrated squared error
    tail: 0.05               # Tail region emphasis
    ms: 0.1                  # Multi-scale loss
    marg_kl: 0.01            # Marginal uniformity penalty
  
  # Copula projection settings
  projection_iters: 10       # IPFP iterations during training
  detach_projection: true    # Detach projection from gradient
  use_amp: true              # Mixed precision training
  
  # Logging
  log_every: 100             # Log every N steps
  save_every: 10000          # Save checkpoint every N steps
  viz_every: 5000            # Visualize every N steps

# OUTPUT
output:
  base_dir: "results"
  checkpoint_dir: "checkpoints"
  include_timestamp: true    # Add timestamp to folder name
  include_job_id: true       # Add SLURM job ID if available
```

### Running Training

```bash
# Single GPU
python scripts/train.py --config configs/train/default.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --config configs/train/default.yaml

# Quick test run (short training)
python scripts/train.py --config configs/train/default.yaml \
    training.max_steps=1000 \
    training.save_every=500

# Resume from checkpoint
python scripts/train.py --config configs/train/default.yaml \
    --resume checkpoints/model_step_50000.pt
```

---

## Task 2: Density Estimation (Inference)

### Config File: `configs/inference/default.yaml`

For estimating copula densities from data using a trained checkpoint:

```yaml
# =============================================================================
# DENSITY ESTIMATION CONFIGURATION
# =============================================================================

task:
  type: "density"            # Task type: "density", "evaluate", "sample", "visualize"

# TRAINED MODEL
model:
  checkpoint: "checkpoints/model_step_100000.pt"  # Path to trained model
  device: "cuda"             # "cuda" or "cpu"
  grid_size: 64              # Must match training config

# INFERENCE PARAMETERS
inference:
  sampling_steps: 50         # DDIM sampling steps (more = better quality)
  cfg_scale: 2.0             # Classifier-free guidance scale
  projection_iters: 50       # IPFP iterations (more = stricter constraints)
  batch_size: 32             # Batch size for inference

# EVALUATION (for testing on known copulas)
evaluation:
  # Bivariate test copulas to evaluate against
  bivariate_tests:
    - family: "gaussian"
      params: {rho: 0.7}
      name: "Gaussian(ρ=0.7)"
    - family: "clayton"
      params: {theta: 3.0}
      name: "Clayton(θ=3.0)"
    - family: "gumbel"
      params: {theta: 2.5}
      name: "Gumbel(θ=2.5)"

# VISUALIZATION
visualization:
  plot_densities: true
  plot_samples: true
  figure_format: ["png"]
  dpi: 150

output:
  base_dir: "results"
  include_timestamp: true
```

### Running Density Estimation

```bash
# Estimate density from your data
python scripts/infer.py density \
    --checkpoint checkpoints/model_step_100000.pt \
    --data my_samples.npy \
    --output estimated_density.npy

# Evaluate on known copulas (compares to ground truth)
python scripts/evaluate.py \
    --checkpoint checkpoints/model_step_100000.pt \
    --config configs/inference/default.yaml

# Quick evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/model_step_100000.pt \
    --quick

# Visualize density estimation
python scripts/infer.py visualize \
    --checkpoint checkpoints/model_step_100000.pt
```

### Python API for Density Estimation

```python
import torch
import numpy as np
from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.hist import scatter_to_hist

# Load model
checkpoint = torch.load('checkpoints/model.pt', map_location='cuda')
config = checkpoint['config']

model = GridUNet(
    m=config['data']['m'],
    in_channels=config['model'].get('in_channels', 1),
    base_channels=config['model'].get('base_channels', 64),
).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

diffusion = CopulaAwareDiffusion(
    timesteps=config['diffusion']['timesteps'],
    beta_schedule=config['diffusion']['noise_schedule'],
).cuda()

# Your bivariate data (pseudo-observations in [0,1]²)
samples = np.random.uniform(0, 1, (1000, 2))

# Estimate density (simplified - see scripts/infer.py for full implementation)
# ...
```

---

## Task 3: Vine Copula Sampling

### Config File: `configs/inference/default.yaml`

For fitting vine copulas and generating samples:

```yaml
# =============================================================================
# VINE COPULA SAMPLING CONFIGURATION
# =============================================================================

task:
  type: "sample"             # Task: "sample" for vine sampling

model:
  checkpoint: "checkpoints/model_step_100000.pt"
  device: "cuda"
  grid_size: 64

inference:
  sampling_steps: 50
  projection_iters: 50

# VINE COPULA SETTINGS
evaluation:
  # High-dimensional scenarios
  scenarios:
    - name: "gaussian_ar1"
      description: "Gaussian copula with AR(1) correlation"
      params: {rho: 0.6}
    - name: "student_ar1"
      description: "Student-t copula with AR(1) correlation"
      params: {rho: 0.6, df: 5}
    - name: "clayton_vine"
      description: "D-vine with Clayton pair copulas"
      params: {theta: 2.0}
    - name: "mixed_vine"
      description: "Mixed copula families"
      params: {}
  
  dimensions: [3, 5, 10, 15, 20]  # Dimensions to test
  n_train: 2000              # Samples for fitting
  n_test: 500                # Samples for evaluation

# SAMPLING SETTINGS
sampling:
  n_samples: 1000            # Number of samples to generate
  dimension: 5               # Vine dimension
  scenario: "gaussian_ar1"   # Which scenario to use
  seed: 42                   # Random seed for reproducibility
```

### Running Vine Copula Sampling

```bash
# Fit vine and generate samples
python scripts/infer.py sample \
    --checkpoint checkpoints/model_step_100000.pt \
    --dimension 5 \
    --n-samples 1000 \
    --output vine_samples.npy

# Evaluate vine copula performance
python scripts/evaluate.py \
    --checkpoint checkpoints/model_step_100000.pt \
    --mode vine \
    --dimensions 3 5 10
```

### Python API for Vine Sampling

```python
import numpy as np
from vdc.vine.api import VineCopulaModel

# Assuming model is already loaded (see above)

# Your high-dimensional data
U_train = np.random.uniform(0, 1, (2000, 5))  # 5-dimensional

# Fit D-vine
vine = VineCopulaModel(vine_type='dvine', m=64, device='cuda')
vine.fit(U_train, model, diffusion, verbose=True)

# Evaluate log-likelihood
logpdf = vine.logpdf(U_test)
print(f"Mean log-likelihood: {logpdf.mean():.2f}")

# Generate new samples
new_samples = vine.simulate(n=1000, seed=42)
print(f"Generated {new_samples.shape[0]} samples of dimension {new_samples.shape[1]}")

# Save/load vine
vine.save('my_fitted_vine.pkl')
loaded_vine = VineCopulaModel.load('my_fitted_vine.pkl')
```

---

## Quick Reference: Config by Task

| Task | Config File | Script | Key Settings |
|------|-------------|--------|--------------|
| **Training** | `configs/train/default.yaml` | `scripts/train.py` | `training.max_steps`, `model.base_channels`, `data.copula_families` |
| **Density Estimation** | `configs/inference/default.yaml` | `scripts/infer.py density` | `model.checkpoint`, `inference.sampling_steps` |
| **Evaluation** | `configs/inference/default.yaml` | `scripts/evaluate.py` | `evaluation.bivariate_tests`, `evaluation.dimensions` |
| **Vine Sampling** | `configs/inference/default.yaml` | `scripts/infer.py sample` | `sampling.n_samples`, `sampling.dimension` |
| **Visualization** | `configs/inference/default.yaml` | `scripts/infer.py visualize` | `visualization.*` |

---

## Creating Custom Configs

### Example: Training with More Capacity

```yaml
# configs/train/high_capacity.yaml
experiment:
  name: "high_capacity_copula"

model:
  grid_size: 64
  base_channels: 128         # Increased from 64
  num_res_blocks: 3          # Increased from 2

training:
  max_steps: 300000          # Longer training
  batch_size: 16             # Smaller batch for memory
```

Run with:
```bash
python scripts/train.py --config configs/train/high_capacity.yaml
```

### Example: Better Tail Estimation

```yaml
# configs/train/tail_focus.yaml
data:
  binning: "probit"          # Probit binning for better tails

training:
  loss_weights:
    tail: 0.2                # Higher tail weight
```

### Example: Custom Evaluation

```yaml
# configs/inference/custom_eval.yaml
task:
  type: "evaluate"

model:
  checkpoint: "checkpoints/my_model.pt"

evaluation:
  bivariate_tests:
    - family: "gaussian"
      params: {rho: 0.9}     # Strong positive dependence
      name: "Strong Gaussian"
    - family: "gaussian"
      params: {rho: -0.9}    # Strong negative dependence
      name: "Strong Negative"
  
  dimensions: [5, 10, 20, 50]  # Test higher dimensions
  n_train: 5000               # More training data
```

---

## Command-Line Overrides

Override any config value from command line using dot notation:

```bash
# Change training steps
python scripts/train.py --config configs/train/default.yaml \
    training.max_steps=50000

# Change multiple values
python scripts/train.py --config configs/train/default.yaml \
    training.max_steps=100000 \
    training.batch_size=64 \
    model.base_channels=128 \
    data.m=128

# Change nested values
python scripts/train.py --config configs/train/default.yaml \
    training.loss_weights.tail=0.2

# Change evaluation dimensions
python scripts/evaluate.py --checkpoint model.pt \
    evaluation.dimensions="[3, 5, 10, 20]"
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CHECKPOINT` | Path to model checkpoint | `export CHECKPOINT=checkpoints/model.pt` |
| `CONFIG` | Path to config file | `export CONFIG=configs/train/default.yaml` |
| `SLURM_JOB_ID` | Auto-included in output directory | (set by SLURM) |

Example SLURM usage:
```bash
CHECKPOINT=checkpoints/model.pt sbatch slurm/evaluate.sh
CONFIG=configs/train/custom.yaml sbatch slurm/train.sh
```
