# Vine Diffusion Copula

Nonparametric bivariate copula density estimation using denoising diffusion probabilistic models.

## What This Does

Estimates the copula density function c(u,v) from bivariate data:
- **Input**: Pseudo-observations (u_i, v_i) in [0,1]² (data after marginal transformation)
- **Output**: Copula density c(u,v) on a discrete grid, satisfying copula constraints
- **Method**: Denoising diffusion with iterative refinement over 1000 timesteps

Constraints enforced:
- c(u,v) ≥ 0 (non-negative)
- ∫₀¹ c(u,v) dv = 1 for all u (uniform marginals)
- ∫₀¹ c(u,v) du = 1 for all v
- Unit mass over [0,1]²

## Installation

```bash
git clone https://github.com/your-org/vine_diffusion_copula.git
cd vine_diffusion_copula
conda env create -f environment.yml
conda activate diffuse_vine_cop
pip install -e .
```

## Quick Start

### Using Pretrained Models

```python
from vdc.vine.copula_diffusion import DiffusionCopulaModel
import numpy as np

# Load pretrained model
model = DiffusionCopulaModel.from_checkpoint(
    "checkpoints/diffusion_uniform_m128/model_step_20000.pt"
)

# Estimate copula density from data (pseudo-observations in [0,1]²)
density, u_coords, v_coords = model.estimate_density_from_samples(
    your_data,
    m=128
)

# Compute h-functions for vine construction
h1, h2 = model.h_functions_from_grid(density)

# Sample from estimated copula
samples = model.sample_from_density(density, n_samples=5000)
```

### Training Custom Models

```bash
# Submit training job
sbatch slurm_jobs/train_diffusion_uniform_m128.sh

# Monitor progress
tail -f logs/train_diffusion_uniform_m128_*.out

# Visualize results
python scripts/visualize_diffusion_offline.py \
    --checkpoint checkpoints/diffusion_uniform_m128/model_step_20000.pt
```

## Project Structure

```
vine_diffusion_copula/
├── vdc/                    # Core library
│   ├── vine/              # Main user-facing API
│   ├── models/            # Network architectures
│   ├── data/              # Data generation
│   └── utils/             # Utilities
├── scripts/               # Training and evaluation scripts
├── examples/              # Usage examples
├── docs/                  # Documentation
├── configs/               # Training configurations
├── slurm_jobs/           # Batch job scripts
└── tests/                # Unit tests
```

## Documentation

- **For users**: `docs/API.md` - Complete API reference
- **For researchers**: `QUICK_START.md` - Training workflows
- **Technical details**: `docs/TECHNICAL_DETAILS.md` - Mathematical framework
- **Examples**: `examples/` - Working code samples

## Pretrained Models

Available in `checkpoints/`:
- `validate_no_probit_diffusion_m128` - Reference baseline (5k steps)
- `diffusion_uniform_m128` - Production uniform grid (20k steps)
- `diffusion_probit_m128` - Boundary-focused probit grid (20k steps)

## Key Features

- Nonparametric copula density estimation
- Support for uniform and boundary-focused (probit) grids
- H-function computation for vine copula construction
- Sampling from estimated copulas
- Multiple copula families for training (Gaussian, Clayton, Gumbel, Frank, Joe, Student-t)
- Multi-GPU training support
- Comprehensive evaluation metrics

## Citation

If you use this in your research:

```bibtex
@software{vine_diffusion_copula,
  title={Vine Diffusion Copula: Deep Learning for Copula Density Estimation},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/vine_diffusion_copula}
}
```

## License

MIT License
