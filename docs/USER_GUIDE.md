# Vine Diffusion Copula - User Guide

A comprehensive guide to using the Vine Diffusion Copula package for copula density estimation and vine copula modeling.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Bivariate Copula Estimation](#bivariate-copula-estimation)
4. [Vine Copula Modeling](#vine-copula-modeling)
5. [Model Training](#model-training)
6. [Evaluation and Diagnostics](#evaluation-and-diagnostics)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Package Structure

```
vdc/
├── models/              # Neural network architectures
│   ├── unet_grid.py    # GridUNet diffusion model
│   ├── copula_diffusion.py  # Diffusion process
│   ├── projection.py   # Copula constraint projection
│   └── hfunc.py        # H-function (conditional CDF)
├── vine/               # Vine copula implementation
│   ├── api.py          # High-level VineCopulaModel
│   ├── structure.py    # Vine structure building
│   └── recursion.py    # Vine recursion
├── data/               # Data generation and loading
│   └── generators.py   # Copula samplers
├── train/              # Training utilities
│   └── unified_trainer.py
├── utils/              # Utilities
│   └── metrics.py      # Evaluation metrics
└── config.py           # Configuration management
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `GridUNet` | Diffusion model for density denoising |
| `CopulaAwareDiffusion` | Diffusion process with scheduling |
| `VineCopulaModel` | High-level vine copula fitting |
| `HFuncLookup` | H-function computation from density grid |
| `Config` | Configuration management |

---

## Data Preparation

### Converting Raw Data to Pseudo-Observations

Copulas work with **pseudo-observations** - data transformed to uniform [0,1] margins:

```python
import numpy as np
from scipy.stats import rankdata

def to_pseudo_observations(X: np.ndarray) -> np.ndarray:
    """
    Transform raw data to pseudo-observations using empirical CDFs.
    
    Args:
        X: (n, d) array of raw observations
        
    Returns:
        U: (n, d) array of pseudo-observations in [0,1]^d
    """
    n, d = X.shape
    U = np.zeros_like(X)
    for j in range(d):
        # Empirical CDF: rank / (n+1) to avoid 0 and 1
        U[:, j] = rankdata(X[:, j]) / (n + 1)
    return U

# Example usage
raw_data = np.random.randn(1000, 5)  # 5-dimensional data
pseudo_obs = to_pseudo_observations(raw_data)
print(f"Range: [{pseudo_obs.min():.4f}, {pseudo_obs.max():.4f}]")
```

### Creating Histograms for Model Input

The diffusion model uses smoothed histograms as conditioning:

```python
from vdc.data.hist import scatter_to_hist

def create_histogram(samples: np.ndarray, m: int = 64) -> np.ndarray:
    """
    Create normalized histogram from bivariate samples.
    
    Args:
        samples: (n, 2) array of points in [0,1]^2
        m: grid resolution
        
    Returns:
        hist: (m, m) normalized histogram
    """
    hist = scatter_to_hist(samples, m, reflect=True)
    du = 1.0 / m
    hist = hist / (hist.sum() * du * du + 1e-12)
    return hist

# Example
pair_data = pseudo_obs[:, :2]  # First two variables
hist = create_histogram(pair_data, m=64)
print(f"Histogram shape: {hist.shape}")
print(f"Total mass: {hist.sum() * (1/64)**2:.4f}")  # Should be ~1.0
```

---

## Bivariate Copula Estimation

### Loading a Trained Model

```python
import torch
from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion

def load_diffusion_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained diffusion model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model_cfg = config.get('model', {})
    m = config.get('data', {}).get('m', 64)
    
    model = GridUNet(
        m=m,
        in_channels=model_cfg.get('in_channels', 1),
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', (16, 8))),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config

# Usage
model, diffusion, config = load_diffusion_model('checkpoints/model.pt')
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Estimating Density with DDIM Sampling

```python
import torch
from vdc.models.projection import copula_project

@torch.no_grad()
def estimate_density(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    samples: np.ndarray,
    m: int = 64,
    device: str = 'cuda',
    num_steps: int = 50,
) -> np.ndarray:
    """
    Estimate copula density from bivariate samples.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        samples: (n, 2) array of pseudo-observations
        m: Grid resolution
        device: Compute device
        num_steps: DDIM sampling steps
        
    Returns:
        density: (m, m) estimated copula density
    """
    # Create histogram
    hist = scatter_to_hist(samples, m, reflect=True)
    du = 1.0 / m
    hist = hist / (hist.sum() * du * du + 1e-12)
    
    # Prepare tensors
    hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # DDIM sampling
    T = diffusion.timesteps
    x_t = torch.randn(1, 1, m, m, device=device)
    
    step_size = max(1, T // num_steps)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        t_normalized = t_tensor.float() / T
        
        # Predict noise
        pred_noise = model(x_t, t_normalized)
        
        # Compute predicted x_0
        alpha_t = alphas_cumprod[t]
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-20, 20)
        
        if t == 0:
            x_t = pred_x0
        else:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            x_t = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
    
    # Convert to density and project to copula constraints
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    density = density / (density.sum() * du * du).clamp_min(1e-12)
    density = copula_project(density, iters=50)
    
    return density[0, 0].cpu().numpy()

# Usage
density = estimate_density(model, diffusion, pair_data, m=64)
print(f"Density shape: {density.shape}")
print(f"Density range: [{density.min():.4f}, {density.max():.4f}]")
```

### Computing H-Functions

H-functions are conditional CDFs needed for vine copulas:

```python
from vdc.models.hfunc import HFuncLookup

def compute_hfunctions(density: np.ndarray):
    """
    Compute h-functions from density grid.
    
    Args:
        density: (m, m) copula density grid
        
    Returns:
        hfunc: HFuncLookup object with methods:
            - h_u_given_v(u, v): H(u|v) = ∫₀ᵘ c(s,v)ds
            - h_v_given_u(u, v): H(v|u) = ∫₀ᵛ c(u,s)ds
            - pdf(u, v): interpolated density
    """
    return HFuncLookup(density)

# Usage
hfunc = compute_hfunctions(density)

# Evaluate at specific points
u_points = np.array([0.3, 0.5, 0.7])
v_points = np.array([0.4, 0.6, 0.8])

h_u_given_v = hfunc.h_u_given_v(u_points, v_points)
h_v_given_u = hfunc.h_v_given_u(u_points, v_points)

print(f"H(u|v): {h_u_given_v}")
print(f"H(v|u): {h_v_given_u}")
```

---

## Vine Copula Modeling

### Fitting a D-Vine

```python
from vdc.vine.api import VineCopulaModel

def fit_dvine(
    U: np.ndarray,
    diffusion_model: torch.nn.Module,
    m: int = 64,
    device: str = 'cuda',
) -> VineCopulaModel:
    """
    Fit a D-vine copula to multivariate data.
    
    Args:
        U: (n, d) pseudo-observations
        diffusion_model: Trained diffusion model for pair copulas
        m: Grid resolution
        device: Compute device
        
    Returns:
        Fitted VineCopulaModel
    """
    d = U.shape[1]
    
    vine = VineCopulaModel(
        vine_type='dvine',
        m=m,
        device=device,
    )
    
    vine.fit(U, diffusion_model, verbose=True)
    
    return vine

# Usage
U = np.random.uniform(0, 1, (1000, 5))  # 5D data
vine = fit_dvine(U, model)

# Print summary
print(vine.summary())
```

### Evaluating Log-Likelihood

```python
def evaluate_vine(vine: VineCopulaModel, U_test: np.ndarray):
    """Evaluate vine copula on test data."""
    logpdf = vine.logpdf(U_test)
    
    print(f"Log-likelihood statistics:")
    print(f"  Mean: {logpdf.mean():.4f}")
    print(f"  Std:  {logpdf.std():.4f}")
    print(f"  Min:  {logpdf.min():.4f}")
    print(f"  Max:  {logpdf.max():.4f}")
    
    return logpdf

# Usage
U_test = np.random.uniform(0, 1, (500, 5))
logpdf = evaluate_vine(vine, U_test)
```

### Generating Samples

```python
def sample_from_vine(vine: VineCopulaModel, n_samples: int, seed: int = None):
    """Generate samples from fitted vine."""
    samples = vine.simulate(n_samples, seed=seed)
    
    print(f"Generated {n_samples} samples")
    print(f"Shape: {samples.shape}")
    print(f"Range: [{samples.min():.4f}, {samples.max():.4f}]")
    
    return samples

# Usage
new_samples = sample_from_vine(vine, n_samples=1000, seed=42)
```

### Saving and Loading Vines

```python
# Save
vine.save('my_fitted_vine.pkl')

# Load
loaded_vine = VineCopulaModel.load('my_fitted_vine.pkl')
print(loaded_vine)
```

---

## Model Training

### Configuration File

Create a custom config `configs/train/my_experiment.yaml`:

```yaml
experiment:
  name: "my_copula_experiment"
  seed: 42

model:
  type: "diffusion_unet"
  grid_size: 64
  in_channels: 1
  base_channels: 64
  channel_mults: [1, 2, 3, 4]
  num_res_blocks: 2
  attention_resolutions: [16, 8]
  dropout: 0.1

diffusion:
  timesteps: 1000
  noise_schedule: "cosine"

data:
  m: 64
  n_samples_per_copula: 1000
  copula_families: ["gaussian", "student", "clayton", "gumbel", "frank", "joe"]
  num_workers: 4

training:
  max_steps: 150000
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  gradient_clip: 1.0
  
  loss_weights:
    ce: 1.0
    ise: 0.1
    tail: 0.05
    marg_kl: 0.01
  
  projection_iters: 10
  use_amp: true
  
  log_every: 100
  save_every: 10000

output:
  base_dir: "results"
  checkpoint_dir: "checkpoints"
  include_timestamp: true
  include_job_id: true
```

### Training Commands

```bash
# Single GPU
python scripts/train.py --config configs/train/my_experiment.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --config configs/train/my_experiment.yaml

# Override specific parameters
python scripts/train.py --config configs/train/default.yaml \
    training.max_steps=200000 \
    training.learning_rate=5e-5 \
    model.base_channels=128

# Resume training
python scripts/train.py --config configs/train/default.yaml \
    --resume checkpoints/model_step_50000.pt

# SLURM submission
sbatch slurm/train.sh
```

---

## Evaluation and Diagnostics

### Running Evaluation

```bash
# Full evaluation
python scripts/evaluate.py --checkpoint checkpoints/model.pt

# Quick evaluation
python scripts/evaluate.py --checkpoint checkpoints/model.pt --quick

# Specific dimensions
python scripts/evaluate.py --checkpoint checkpoints/model.pt \
    --dimensions 3 5 10 15 20
```

### Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| ISE | Integrated Squared Error | Lower is better (< 0.01) |
| Corr(True) | Correlation with true density | Higher is better (> 0.9) |
| Corr(Hist) | Correlation with histogram | Lower means less copying |
| KS p-value | Rosenblatt uniformity test | > 0.05 (not rejected) |
| Mean LogPDF | Average log-likelihood | Higher is better |

### Visualizing Results

```python
import matplotlib.pyplot as plt

def plot_density_comparison(density_pred, density_true, title=""):
    """Plot predicted vs true density."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    vmax = min(50, max(density_pred.max(), density_true.max()))
    
    # Predicted
    im0 = axes[0].imshow(density_pred, origin='lower', cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title('Predicted Density')
    plt.colorbar(im0, ax=axes[0])
    
    # True
    im1 = axes[1].imshow(density_true, origin='lower', cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title('True Density')
    plt.colorbar(im1, ax=axes[1])
    
    # Difference
    diff = density_pred - density_true
    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu', vmin=-vmax/2, vmax=vmax/2)
    axes[2].set_title('Difference')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

# Usage
fig = plot_density_comparison(density_pred, density_true, "Gaussian(ρ=0.7)")
fig.savefig('density_comparison.png', dpi=150)
```

---

## Advanced Topics

### Custom Copula Families

Add new copula families for training:

```python
# In vdc/data/generators.py
def sample_my_copula(n: int, param: float) -> np.ndarray:
    """Sample from custom copula family."""
    # Your implementation
    pass

# Register in CopulaGenerator
CopulaGenerator.FAMILY_SAMPLERS['my_copula'] = sample_my_copula
```

### Probit Space Training

For better tail estimation:

```yaml
# In config
model:
  transform_to_probit_space: true

data:
  binning: "probit"
```

### Custom Loss Functions

```python
# Modify loss weights in training
training:
  loss_weights:
    ce: 1.0           # Cross-entropy (primary)
    ise: 0.2          # Integrated squared error
    tail: 0.1         # Tail region emphasis
    ms: 0.1           # Multi-scale loss
    marg_kl: 0.02     # Marginal uniformity
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python scripts/train.py --config config.yaml training.batch_size=16

# Or use gradient accumulation (not yet implemented)
```

**2. NaN in Training**
```yaml
# Enable gradient sanitation
training:
  grad_sanitation:
    enable: true
    max_grad_value: 10.0
```

**3. Poor Density Estimates**
- Increase training steps
- Check if data is properly in [0,1]²
- Increase projection iterations

**4. Slow Evaluation**
```bash
# Use fewer sampling steps
python scripts/evaluate.py --checkpoint model.pt --quick
```

### Getting Help

- Check existing issues on GitHub
- Review `tests/` for expected behavior
- See `slurm_logs/` for training errors
