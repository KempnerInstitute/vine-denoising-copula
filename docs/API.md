# Vine Diffusion Copula - API Documentation

## For External Users

If you want to use trained diffusion copula models in your research, this is the main API you need.

---

## Installation

```bash
git clone https://github.com/your-org/vine_diffusion_copula.git
cd vine_diffusion_copula
pip install -e .
```

---

## Core API: `DiffusionCopulaModel`

### Loading a Pretrained Model

```python
from vdc.vine.copula_diffusion import DiffusionCopulaModel

# Load a checkpoint
model = DiffusionCopulaModel.from_checkpoint(
    "path/to/model_step_5000.pt",
    device='cuda'  # or 'cpu'
)
```

### Estimating Copula Density

Given bivariate data **after marginal transformation** (pseudo-observations in [0,1]²):

```python
import numpy as np

# Your data (already transformed to uniform marginals)
u = np.random.rand(10000, 2)  # Example: uniform data

# Estimate copula density on a grid
density, row_coords, col_coords = model.estimate_density_from_samples(
    u,
    m=128,              # Grid resolution
    projection_iters=15 # IPFP iterations for copula constraints
)

# Returns:
#   density: (128, 128) numpy array
#   row_coords, col_coords: (128,) arrays of grid points
```

### Computing H-Functions

For vine copula construction, you need conditional CDFs (h-functions):

```python
# Get h-functions from estimated density
h1, h2 = model.h_functions_from_grid(density)

# h1[i,j] ≈ P(U ≤ u_i | V = v_j) = H(u_i | v_j)
# h2[i,j] ≈ P(V ≤ v_j | U = u_i) = H(v_j | u_i)
```

### Sampling from Estimated Copula

```python
# Generate new samples from the learned copula
samples = model.sample_from_density(
    density,
    n_samples=5000
)

# Returns: (5000, 2) numpy array in [0,1]²
```

---

## Example Workflow

### 1. Load Your Bivariate Data

```python
import pandas as pd
from scipy.stats import norm

# Load your data
data = pd.read_csv('my_data.csv')
X = data[['var1', 'var2']].values  # (n, 2) array

# Transform to uniform marginals using empirical CDFs
def to_uniform(x):
    """Transform to uniform [0,1] using empirical CDF."""
    from scipy.stats import rankdata
    return rankdata(x) / (len(x) + 1)

u1 = to_uniform(X[:, 0])
u2 = to_uniform(X[:, 1])
u = np.column_stack([u1, u2])
```

### 2. Estimate Copula

```python
from vdc.vine.copula_diffusion import DiffusionCopulaModel

# Load pretrained model
model = DiffusionCopulaModel.from_checkpoint(
    "checkpoints/diffusion_uniform_m128/model_step_20000.pt"
)

# Estimate density
density, u_grid, v_grid = model.estimate_density_from_samples(u, m=128)
```

### 3. Visualize (Optional)

```python
import matplotlib.pyplot as plt

# Plot the estimated density
plt.figure(figsize=(10, 8))
plt.imshow(density.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
plt.colorbar(label='Copula Density')
plt.xlabel('u')
plt.ylabel('v')
plt.title('Estimated Copula Density')
plt.scatter(u[:200, 0], u[:200, 1], c='red', s=1, alpha=0.5, label='Data')
plt.legend()
plt.savefig('my_copula_estimate.png', dpi=300, bbox_inches='tight')
print("Saved visualization to: my_copula_estimate.png")
```

### 4. Use in Vine Copula

```python
# Get h-functions for conditional copulas
h1, h2 = model.h_functions_from_grid(density)

# Use h-functions to build higher-dimensional vines
# (integrate with your vine copula library)

# Generate new samples
new_samples = model.sample_from_density(density, n_samples=10000)
```

---

## Training Your Own Model

See `examples/train_custom_copula.py` for a complete example.

Quick start:

```bash
# 1. Create a config (see configs/validate_no_probit_diffusion_m128.yaml as template)

# 2. Train
python scripts/train_unified.py \
    --config configs/my_config.yaml \
    --model-type diffusion_unet

# 3. Use the trained model
# (see steps above)
```

---

## Model Zoo

Pre-trained models available:

| Model | Grid | Training Steps | Use Case |
|-------|------|----------------|----------|
| `validate_no_probit_diffusion_m128` | Uniform 128×128 | 5,000 | Reference baseline |
| `diffusion_uniform_m128` | Uniform 128×128 | 20,000 | Production |
| `diffusion_probit_m128` | Probit 128×128 | 20,000 | Boundary-focused |

---

## Advanced Usage

### Custom Grid Resolution

```python
# Higher resolution for more detail
density_256, u, v = model.estimate_density_from_samples(data, m=256)

# Lower resolution for speed
density_64, u, v = model.estimate_density_from_samples(data, m=64)
```

### Adjusting Projection Strength

```python
# More iterations = stricter copula constraints
density, u, v = model.estimate_density_from_samples(
    data,
    projection_iters=30  # Default is 15
)
```

### Different Diffusion Timesteps

```python
# Denoise from a less noisy state
density, u, v = model.estimate_density_from_samples(
    data,
    noise_step=500  # Default is 999 (maximum noise)
)
```

---

## API Reference

### `DiffusionCopulaModel`

**Methods:**

- `from_checkpoint(path, device='cuda', config_path=None)` → `DiffusionCopulaModel`
  - Load a trained model from checkpoint

- `estimate_density_from_samples(u, m=None, noise_step=None, projection_iters=15)` → `(density, row_coords, col_coords)`
  - Estimate copula density from pseudo-observations

- `h_functions_from_grid(density, row_widths=None, col_widths=None)` → `(h1, h2)`
  - Compute conditional CDFs from density grid

- `sample_from_density(density, n_samples, rng=None)` → `samples`
  - Generate samples from estimated copula


## License

MIT License

