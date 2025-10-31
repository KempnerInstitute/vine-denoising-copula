# Vine-Diffusion-Copula: High-Dimensional Copula Modeling with Diffusion Networks

A PyTorch-based framework for high-dimensional vine-copula estimation and sampling using diffusion networks to learn bivariate copula densities from scatter data.

## Overview

This package implements a novel approach to vine-copula construction by training a diffusion network to transform 2D histograms of pseudo-observations into valid bivariate copula densities. The key innovation is using a U-Net with Sinkhorn/IPFP projection to ensure copula constraints (uniform marginals, unit integral).

## Key Features

- **Diffusion-based copula estimation**: Neural network learns to map scatter data to copula densities
- **Vine construction**: Automatic structure selection via Dißmann algorithm
- **Multi-GPU training**: Efficient DDP training on HPC clusters
- **Comprehensive baselines**: Integration with pyvinecopulib and kdecopula
- **Full inference**: Both density evaluation and sampling via inverse Rosenblatt

## Installation

```bash
# Create conda environment
conda env create -f env/environment.yml
conda activate vdc

# Install package
pip install -e .
```

## Quick Start

### 1. Generate synthetic training data

```bash
python -m vdc.data.generators \
    --output data/zoo \
    --n-pairs 2000000 \
    --m 64
```

### 2. Train the diffusion copula estimator

```bash
# Single GPU
python -m vdc.train.train_grid \
    data.root=data/zoo \
    model.m=64 \
    train.max_steps=400000

# Multi-GPU (8 GPUs)
torchrun --standalone --nproc-per-node=8 \
    -m vdc.train.train_grid \
    data.root=data/zoo \
    model.m=64 \
    train.max_steps=400000
```

### 3. Build a vine on real data

```python
from vdc.vine import DiffusionVine
from vdc.models import load_trained_estimator
import numpy as np

# Load your data and convert to pseudo-observations
X = ...  # shape (n, d)
U = (np.argsort(np.argsort(X, axis=0), axis=0) + 1) / (X.shape[0] + 1)

# Load trained estimator
estimator = load_trained_estimator("checkpoints/model.pt")

# Build and fit vine
vine = DiffusionVine(estimator, m=64)
vine.fit(U)

# Evaluate log-likelihood
log_lik = vine.logpdf(U_test)

# Generate samples
samples = vine.simulate(n=10000)
```

## Architecture

### Data Pipeline
- Synthetic copula zoo: 20+ families (Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, BB1-8)
- Conditional pairs from R-vines for training on Trees 2+
- Histogram representation with optional tail-biasing

### Model Components
- **U-Net**: Denoising diffusion model for density prediction
- **IPFP/Sinkhorn projection**: Enforces copula constraints (uniform marginals)
- **h-functions**: Computed via cumulative integration on grid
- **Vine recursion**: Standard Dißmann structure + Rosenblatt transforms

### Training
- Multi-GPU DDP with PyTorch
- Losses: NLL on points, ISE vs teacher, marginal penalties, tail weighting
- SLURM-ready with automatic resource allocation

## Project Structure

```
vine_diffusion_copula/
├── vdc/
│   ├── data/           # Data generation and loading
│   ├── models/         # U-Net, projection, h-functions
│   ├── vine/           # Structure selection, recursion, inference
│   ├── baselines/      # kdecopula, pyvinecopulib wrappers
│   ├── losses/         # Training objectives
│   ├── train/          # Training loops
│   ├── eval/           # Evaluation metrics
│   └── utils/          # Interpolation, integration, stats
├── configs/            # Hydra configuration files
├── scripts/            # SLURM job templates
└── tests/              # Unit and integration tests
```

## References

- Aas et al. (2009). "Pair-copula constructions of multiple dependence"
- Dißmann et al. (2013). "Selecting and estimating regular vine copulae"
- Song et al. (2021). "Score-Based Generative Modeling through SDEs"
- Nagler & Czado (2016). "Evading the curse of dimensionality in nonparametric density estimation"

## Citation

```bibtex
@software{vine_diffusion_copula,
  title = {Vine-Diffusion-Copula: High-Dimensional Copula Modeling with Diffusion Networks},
  author = {{Kempner Institute}},
  year = {2025},
  url = {https://github.com/KempnerInstitute/vine-diffusion-copula}
}
```

## License

MIT License - see LICENSE file for details.
