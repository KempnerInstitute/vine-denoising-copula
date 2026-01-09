# Getting Started with Vine Diffusion Copula

This guide will help you get up and running with the Vine Diffusion Copula package.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Understanding Copulas](#understanding-copulas)
4. [Basic Usage Examples](#basic-usage-examples)
5. [Training Your Own Model](#training-your-own-model)
6. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, but CPU works for inference)
- 8GB+ GPU memory for training

### Setup

```bash
# Clone the repository
git clone https://github.com/KempnerInstitute/vine_diffusion_copula.git
cd vine_diffusion_copula

# Option 1: Using conda (recommended)
conda env create -f environment.yml
conda activate vdc

# Option 2: Using pip
pip install -e .

# Verify installation
python -c "import vdc; print(f'VDC version: {vdc.__version__}')"
```

---

## Quick Start

### 1. Load a Trained Model

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = "checkpoints/your_model/model_step_100000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cuda')

# Extract model
from vdc.models.unet_grid import GridUNet
config = checkpoint['config']
model = GridUNet(
    m=config['data']['m'],
    in_channels=config['model'].get('in_channels', 1),
    base_channels=config['model'].get('base_channels', 64),
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 2. Estimate a Copula Density

```python
import numpy as np
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project

# Create diffusion process
diffusion = CopulaAwareDiffusion(timesteps=1000, beta_schedule='cosine')

# Your bivariate data (pseudo-observations in [0,1]²)
# Example: Generate Gaussian copula samples
from scipy.stats import norm
rho = 0.7
n = 1000
Z = np.random.randn(n, 2)
Z[:, 1] = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
samples = norm.cdf(Z)  # Transform to uniform [0,1]

# Create histogram from samples
from vdc.data.hist import scatter_to_hist
m = 64
hist = scatter_to_hist(samples, m, reflect=True)
du = 1.0 / m
hist = hist / (hist.sum() * du * du + 1e-12)

# Estimate density using diffusion model
hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).cuda()
with torch.no_grad():
    # Sample from diffusion to get density estimate
    # (See scripts/infer.py for complete implementation)
    pass
```

### 3. Fit a Vine Copula

```python
from vdc.vine.api import VineCopulaModel

# Fit a D-vine to high-dimensional data
U = np.random.uniform(0, 1, (1000, 5))  # 5-dimensional copula samples

vine = VineCopulaModel(vine_type='dvine', m=64, device='cuda')
vine.fit(U, model, diffusion)  # Uses the diffusion model + diffusion process

# Evaluate and sample
logpdf = vine.logpdf(U[:100])  # Log-density at points
new_samples = vine.simulate(n=500)  # Generate new samples

# Save for later use
vine.save('my_vine.pkl')
```

---

## Understanding Copulas

### What is a Copula?

A **copula** is a multivariate distribution function that captures the dependence structure between random variables, separate from their marginal distributions.

For bivariate data (X, Y):
- First transform to uniform margins: U = F_X(X), V = F_Y(Y)
- The copula C(u, v) = P(U ≤ u, V ≤ v) captures the dependence
- The copula density c(u, v) = ∂²C/∂u∂v

### Why Diffusion Models for Copulas?

1. **Nonparametric**: No assumption about copula family (Gaussian, Clayton, etc.)
2. **Constraint Enforcement**: We enforce valid copula constraints:
   - c(u,v) ≥ 0
   - ∫c(u,v)dv = 1 for all u
   - ∫c(u,v)du = 1 for all v
3. **Flexible**: Learns complex, multi-modal dependence structures

### Vine Copulas

For high-dimensional data, we use **vine copulas** which decompose the joint distribution into a sequence of bivariate copulas:

- **D-vine**: Linear chain structure
- **C-vine**: Star structure with central variable
- **R-vine**: General structure

This package learns bivariate copulas with diffusion models, then combines them into vines.

---

## Basic Usage Examples

### Example 1: Estimate Density from Financial Data

```python
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# Load your data
# data = pd.read_csv('stock_returns.csv')
# For this example, simulate some data
np.random.seed(42)
returns = np.random.randn(1000, 2) * 0.02  # Daily returns

# Transform to pseudo-observations (uniform [0,1])
def to_uniform(x):
    """Empirical CDF transformation."""
    return rankdata(x) / (len(x) + 1)

u = np.column_stack([to_uniform(returns[:, 0]), to_uniform(returns[:, 1])])

print(f"Pseudo-observations shape: {u.shape}")
print(f"Range: [{u.min():.3f}, {u.max():.3f}]")

# Now use the diffusion model to estimate the copula density
# ... (see full examples in examples/ folder)
```

### Example 2: Compare with Parametric Copulas

```python
from vdc.data.generators import (
    sample_gaussian_copula,
    sample_clayton_copula,
    gaussian_copula_density,
    clayton_copula_density,
)

# Generate samples from known copulas
gaussian_samples = sample_gaussian_copula(n=2000, rho=0.6)
clayton_samples = sample_clayton_copula(n=2000, theta=3.0)

# Estimate with diffusion model and compare to true density
# ... (see scripts/evaluate.py for complete implementation)
```

### Example 3: High-Dimensional Vine Copula

```python
from vdc.data.generators import generate_gaussian_vine
from vdc.vine.api import VineCopulaModel

# Generate 10-dimensional Gaussian copula data with AR(1) correlation
d = 10
n_train = 2000
n_test = 500
rho = 0.6

U_train = generate_gaussian_vine(n_train, d, rho=rho, seed=42)
U_test = generate_gaussian_vine(n_test, d, rho=rho, seed=123)

# Fit D-vine (need trained diffusion model + diffusion process)
# vine = VineCopulaModel(vine_type='dvine')
# vine.fit(U_train, diffusion_model, diffusion_process)
# loglik = vine.logpdf(U_test)
# print(f"Mean log-likelihood: {loglik.mean():.2f}")
```

---

## Training Your Own Model

### Using the Training Script

```bash
# Basic training
python scripts/train.py --config configs/train/default.yaml

# With custom settings
python scripts/train.py --config configs/train/default.yaml \
    training.max_steps=200000 \
    model.base_channels=128

# Multi-GPU training
torchrun --nproc_per_node=4 scripts/train.py --config configs/train/default.yaml

# Resume from checkpoint
python scripts/train.py --config configs/train/default.yaml \
    --resume checkpoints/my_model/model_step_50000.pt
```

### Configuration Options

Key settings in `configs/train/default.yaml`:

```yaml
model:
  type: "diffusion_unet"
  grid_size: 64           # Density grid resolution
  base_channels: 64       # Model capacity

diffusion:
  timesteps: 1000         # Diffusion steps
  noise_schedule: "cosine"

training:
  max_steps: 150000       # Training iterations
  batch_size: 32
  learning_rate: 1.0e-4
  
  loss_weights:
    ce: 1.0               # Cross-entropy
    ise: 0.1              # Integrated squared error
    tail: 0.05            # Tail region loss
```

### SLURM Cluster Training

```bash
# Submit training job
sbatch slurm/train.sh

# Or with custom config
CONFIG=configs/train/my_experiment.yaml sbatch slurm/train.sh

# Monitor progress
tail -f slurm_logs/vdc_train_*.out
```

---

## Next Steps

1. **Explore Examples**: See `examples/` folder for complete working examples
2. **Read API Documentation**: See `docs/API.md` for detailed API reference
3. **Understand the Math**: See `docs/TECHNICAL_DETAILS.md` for mathematical background
4. **Run Evaluation**: Use `scripts/evaluate.py` to test model quality

### Common Tasks

| Task | Command/Code |
|------|--------------|
| Train model | `python scripts/train.py --config configs/train/default.yaml` |
| Evaluate model | `python scripts/evaluate.py --checkpoint path/to/model.pt` |
| Estimate density | `python scripts/infer.py density --checkpoint path/to/model.pt --data samples.npy` |
| Visualize results | `python scripts/infer.py visualize --checkpoint path/to/model.pt` |

### Getting Help

- Check `docs/` for detailed documentation
- Look at `examples/` for working code
- Review `tests/` for expected behavior
