# Vine Diffusion Copula - API Documentation

Complete API reference for the Vine Diffusion Copula package.

---

## Table of Contents

1. [Installation](#installation)
2. [Core Classes](#core-classes)
3. [Vine Copula API](#vine-copula-api)
4. [Utility Functions](#utility-functions)
5. [Configuration](#configuration)
6. [Examples](#examples)

---

## Installation

```bash
git clone https://github.com/KempnerInstitute/vine_diffusion_copula.git
cd vine_diffusion_copula
pip install -e .
```

---

## Core Classes

### `GridUNet`

The main diffusion model architecture for copula density estimation.

**Location:** `vdc.models.unet_grid`

```python
from vdc.models.unet_grid import GridUNet

model = GridUNet(
    m=64,                          # Grid resolution
    in_channels=1,                 # Input channels
    base_channels=64,              # Base feature channels
    channel_mults=(1, 2, 3, 4),    # Channel multipliers
    num_res_blocks=2,              # Residual blocks per level
    attention_resolutions=(16, 8), # Resolutions with attention
    dropout=0.1,                   # Dropout rate
    time_emb_dim=256,              # Time embedding dimension
)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(x, t) -> Tensor` | Forward pass with noisy input and time |

**Parameters:**
- `x`: `(B, C, m, m)` - Noisy log-density tensor
- `t`: `(B,)` - Normalized timestep in [0, 1]

**Returns:**
- `(B, 1, m, m)` - Predicted noise

---

### `CopulaAwareDiffusion`

Manages the diffusion process with noise scheduling.

**Location:** `vdc.models.copula_diffusion`

```python
from vdc.models.copula_diffusion import CopulaAwareDiffusion

diffusion = CopulaAwareDiffusion(
    timesteps=1000,              # Total diffusion steps
    beta_schedule='cosine',      # 'linear' or 'cosine'
)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `q_sample` | `(x_0, t, noise=None) -> Tensor` | Add noise at timestep t |
| `alphas_cumprod` | `Tensor` | Cumulative product of (1-beta) |

**Example:**
```python
# Forward diffusion (add noise)
t = torch.randint(0, 1000, (batch_size,))
noise = torch.randn_like(x_0)
x_t = diffusion.q_sample(x_0, t, noise)

# Get alpha values for sampling
alpha_t = diffusion.alphas_cumprod[t]
```

---

### `HFuncLookup`

Computes h-functions (conditional CDFs) from a density grid.

**Location:** `vdc.models.hfunc`

```python
from vdc.models.hfunc import HFuncLookup

hfunc = HFuncLookup(density_grid)  # (m, m) numpy array
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `h_u_given_v` | `(u, v) -> ndarray` | H(u\|v) = ∫₀ᵘ c(s,v)ds |
| `h_v_given_u` | `(u, v) -> ndarray` | H(v\|u) = ∫₀ᵛ c(u,s)ds |
| `h_inv_u_given_v` | `(p, v) -> ndarray` | Inverse of h_u_given_v |
| `h_inv_v_given_u` | `(p, u) -> ndarray` | Inverse of h_v_given_u |
| `pdf` | `(u, v) -> ndarray` | Interpolated density c(u,v) |

**Example:**
```python
u = np.array([0.3, 0.5, 0.7])
v = np.array([0.4, 0.6, 0.8])

# Conditional CDFs
h1 = hfunc.h_u_given_v(u, v)  # P(U ≤ u | V = v)
h2 = hfunc.h_v_given_u(u, v)  # P(V ≤ v | U = u)

# Density at points
density_at_points = hfunc.pdf(u, v)
```

---

### `copula_project`

Projects a density grid to satisfy copula constraints (uniform marginals).

**Location:** `vdc.models.projection`

```python
from vdc.models.projection import copula_project

projected = copula_project(
    density,                    # (B, 1, m, m) tensor
    iters=50,                   # IPFP iterations
    row_target=None,            # Target row marginal (default: uniform)
    col_target=None,            # Target col marginal (default: uniform)
)
```

**Returns:** `(B, 1, m, m)` - Projected density with uniform marginals

---

## Vine Copula API

### `VineCopulaModel`

High-level interface for fitting and using vine copulas.

**Location:** `vdc.vine.api`

```python
from vdc.vine.api import VineCopulaModel

vine = VineCopulaModel(
    vine_type='dvine',          # 'dvine', 'cvine', or 'rvine'
    order=None,                 # Variable ordering (auto if None)
    truncation_level=None,      # Truncate at this tree level
    m=64,                       # Grid resolution
    device='cuda',              # Compute device
)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `(U, model, verbose=True)` | Fit vine to data |
| `logpdf` | `(U) -> ndarray` | Compute log-density |
| `pdf` | `(U) -> ndarray` | Compute density |
| `simulate` | `(n, seed=None) -> ndarray` | Generate samples |
| `rosenblatt` | `(U) -> ndarray` | Forward Rosenblatt transform |
| `save` | `(path)` | Save to pickle file |
| `load` | `(path) -> VineCopulaModel` | Load from pickle file |
| `summary` | `() -> dict` | Get model summary |

**Example:**
```python
# Fit D-vine to 5-dimensional data
U = np.random.uniform(0, 1, (1000, 5))

vine = VineCopulaModel(vine_type='dvine', m=64)
vine.fit(U, diffusion_model, verbose=True)

# Evaluate
logpdf = vine.logpdf(U_test)
print(f"Mean log-likelihood: {logpdf.mean():.2f}")

# Sample
new_samples = vine.simulate(n=500)

# Save/load
vine.save('my_vine.pkl')
loaded = VineCopulaModel.load('my_vine.pkl')
```

---

### Convenience Functions

```python
from vdc.vine.api import fit_dvine, fit_cvine, fit_rvine

# Quick D-vine fitting
dvine = fit_dvine(U, model, order=[0, 2, 1, 3, 4], m=64)

# Quick C-vine fitting
cvine = fit_cvine(U, model, m=64)

# Quick R-vine fitting
rvine = fit_rvine(U, model, truncation_level=3, m=64)
```

---

## Utility Functions

### Data Generation

**Location:** `vdc.data.generators`

```python
from vdc.data.generators import (
    # Bivariate copula samplers
    sample_gaussian_copula,
    sample_student_copula,
    sample_clayton_copula,
    sample_gumbel_copula,
    sample_frank_copula,
    sample_joe_copula,
    
    # High-dimensional generators
    generate_gaussian_vine,
    generate_student_vine,
    generate_clayton_vine,
    generate_mixed_vine,
    
    # Density functions
    gaussian_copula_density,
    clayton_copula_density,
)

# Bivariate sampling
samples = sample_gaussian_copula(n=1000, rho=0.7)
samples = sample_clayton_copula(n=1000, theta=3.0)

# High-dimensional
U = generate_gaussian_vine(n=1000, d=5, rho=0.6, seed=42)
```

### Histogram Creation

**Location:** `vdc.data.hist`

```python
from vdc.data.hist import scatter_to_hist

hist = scatter_to_hist(
    pts,                # (n, 2) points in [0,1]²
    m=64,               # Grid resolution
    reflect=True,       # Boundary reflection
)
```

### Metrics

**Location:** `vdc.utils.metrics`

```python
from vdc.utils.metrics import (
    ise,                # Integrated squared error
    kl_divergence,      # KL divergence
    kendall_tau,        # Kendall's tau from samples
    tail_dependence_from_grid,  # λ_U, λ_L
)
```

---

## Configuration

### `Config` Class

**Location:** `vdc.config`

```python
from vdc.config import Config, get_run_dir

# Load configuration
config = Config.load(
    'configs/train/default.yaml',
    overrides=['training.max_steps=200000'],
)

# Access values
print(config.training.max_steps)
print(config.model.base_channels)

# Save
config.save('my_config.yaml')

# Create run directory
run_dir = get_run_dir(
    base_dir='results',
    experiment_name='my_experiment',
    include_timestamp=True,
    include_job_id=True,
)
# Returns: Path like results/my_experiment_20251209_143022_job12345/
```

---

## Examples

### Complete Bivariate Example

```python
import numpy as np
import torch
from scipy.stats import norm

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.models.hfunc import HFuncLookup
from vdc.data.hist import scatter_to_hist

# 1. Load model
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

# 2. Generate test data
rho = 0.7
n = 1000
Z = np.random.randn(n, 2)
Z[:, 1] = rho * Z[:, 0] + np.sqrt(1-rho**2) * Z[:, 1]
samples = norm.cdf(Z)

# 3. Estimate density (simplified DDIM)
m = 64
hist = scatter_to_hist(samples, m, reflect=True)
hist = hist / (hist.sum() * (1/m)**2 + 1e-12)

with torch.no_grad():
    x_t = torch.randn(1, 1, m, m, device='cuda')
    # ... DDIM sampling loop (see scripts/infer.py for full implementation)
    
# 4. Project to copula constraints
# density = copula_project(density, iters=50)

# 5. Create h-function lookup
# hfunc = HFuncLookup(density.cpu().numpy())
```

### Complete Vine Example

```python
import numpy as np
from vdc.vine.api import VineCopulaModel
from vdc.data.generators import generate_gaussian_vine

# 1. Generate high-dimensional data
d = 5
n_train, n_test = 2000, 500
U_train = generate_gaussian_vine(n_train, d, rho=0.6, seed=42)
U_test = generate_gaussian_vine(n_test, d, rho=0.6, seed=123)

# 2. Load diffusion model (see above)
# model, diffusion = ...

# 3. Fit D-vine
vine = VineCopulaModel(vine_type='dvine', m=64, device='cuda')
vine.fit(U_train, model, verbose=True)

# 4. Evaluate
logpdf = vine.logpdf(U_test)
print(f"Mean log-likelihood: {logpdf.mean():.2f}")

# 5. Sample
new_samples = vine.simulate(n=1000, seed=42)

# 6. Save
vine.save('fitted_dvine.pkl')
```

---

## License

MIT License

