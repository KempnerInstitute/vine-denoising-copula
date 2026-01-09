# Vine Diffusion Copula

Nonparametric copula density estimation using denoising-based neural estimators (iterative diffusion and single-pass denoisers), with support for high-dimensional vine copula structures.

## What This Does

Estimates the copula density function c(u,v) from bivariate data:
- **Input**: Pseudo-observations (u_i, v_i) in [0,1]² (data after marginal transformation)
- **Output**: Copula density c(u,v) on a discrete grid, satisfying copula constraints
- **Method**: Supports (a) iterative conditional diffusion (DDIM/CFG) and (b) fast single-pass denoiser/CNN baselines, all with IPFP/Sinkhorn projection to enforce exact copula constraints

The estimated bivariate copulas can be combined into vine structures (D-vine, C-vine, R-vine) for modeling high-dimensional dependence.

### Constraints Enforced
- c(u,v) ≥ 0 (non-negative)
- ∫₀¹ c(u,v) dv = 1 for all u (uniform marginals)
- ∫₀¹ c(u,v) du = 1 for all v
- Unit mass over [0,1]²

## Installation

```bash
git clone https://github.com/KempnerInstitute/vine_diffusion_copula.git
cd vine_diffusion_copula
conda env create -f environment.yml
conda activate vdc
pip install -e .
```

## Quick Start

### 1. Training a Model

```bash
# Recommended (unified trainer) — pick one:
python scripts/train_unified.py --config configs/train/diffusion_cond.yaml --model-type diffusion_unet
python scripts/train_unified.py --config configs/train/denoiser_cond.yaml  --model-type denoiser
python scripts/train_unified.py --config configs/train/enhanced_cnn_cond.yaml --model-type enhanced_cnn

# Multi-GPU (example: 4 GPUs)
torchrun --nproc_per_node=4 scripts/train_unified.py --config configs/train/diffusion_cond.yaml --model-type diffusion_unet

# SLURM (paper-style jobs; write a run directory with checkpoints/results/analysis)
sbatch slurm/paper_vdc_diffusion_cond.sh
```

### 2. Evaluating a Model

```bash
# Full evaluation (bivariate + vine copula)
python scripts/evaluate.py --checkpoint checkpoints/model.pt

# Quick evaluation
python scripts/evaluate.py --checkpoint checkpoints/model.pt --quick

# Standardized comparison across checkpoints (bivariate + small vine tasks)
python scripts/model_selection.py --checkpoints checkpoints/*/model_step_*.pt --n-samples 2000

# SLURM cluster
CHECKPOINT=checkpoints/model.pt sbatch slurm/evaluate.sh
```

### 3. Inference: Density Estimation

```bash
# Estimate density from your data
python scripts/infer.py density --checkpoint checkpoints/model.pt --data your_samples.npy

# Visualize on test copulas
python scripts/infer.py visualize --checkpoint checkpoints/model.pt
```

### 4. Using in Python

```python
import torch
import numpy as np
from pathlib import Path
from vdc.vine.api import VineCopulaModel
from scripts.train_unified import build_model
from vdc.models.copula_diffusion import CopulaAwareDiffusion

# Load checkpoint + build model
ckpt_path = Path("checkpoints/your_run/model_step_100000.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
config = ckpt["config"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = config.get("model", {}).get("type", "diffusion_unet")
model = build_model(model_type, config, device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# Diffusion object only needed for diffusion_unet inference
diffusion = None
if model_type == "diffusion_unet":
    diffusion = CopulaAwareDiffusion(
        timesteps=int(config.get("diffusion", {}).get("timesteps", 1000)),
        beta_schedule=str(config.get("diffusion", {}).get("noise_schedule", "cosine")),
    ).to(device)

# Fit a D-vine copula to high-dimensional data
U = np.random.uniform(0, 1, (1000, 5))  # 5D copula samples
vine = VineCopulaModel(vine_type='dvine', m=64, device='cuda')
vine.fit(U, model, diffusion=diffusion)  # diffusion!=None -> iterative DDIM sampling

# Evaluate density and generate samples
logpdf = vine.logpdf(U_test)
samples = vine.simulate(n=1000)

# Save for later use
vine.save('my_vine.pkl')
```

## Project Structure

```
vine_diffusion_copula/
├── vdc/                          # Core library
│   ├── models/                   # Neural network architectures
│   │   ├── unet_grid.py         # Main UNet model
│   │   ├── copula_diffusion.py  # Diffusion process
│   │   ├── projection.py        # Copula constraint projection
│   │   └── hfunc.py             # H-function (conditional CDF)
│   ├── vine/                     # Vine copula API
│   │   ├── api.py               # High-level VineCopulaModel class
│   │   ├── structure.py         # Vine structure building
│   │   └── recursion.py         # Vine recursion algorithms
│   ├── train/                    # Training utilities
│   │   └── unified_trainer.py   # Main training loop
│   ├── data/                     # Data generation
│   │   ├── generators.py        # Copula sampling
│   │   └── onthefly.py          # On-the-fly dataset
│   ├── utils/                    # Utilities
│   └── config.py                 # Configuration management
├── scripts/                      # Main entry points
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── infer.py                 # Inference script
├── configs/                      # Configuration files
│   ├── train/default.yaml       # Training config
│   └── inference/default.yaml   # Inference config
├── slurm/                        # SLURM job scripts
│   ├── train.sh                 # Training job
│   └── evaluate.sh              # Evaluation job
├── examples/                     # Usage examples
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## Configuration

All settings are controlled via YAML config files:

```yaml
# configs/train/default.yaml
model:
  type: "diffusion_unet"
  grid_size: 64
  base_channels: 64

diffusion:
  timesteps: 1000
  noise_schedule: "cosine"

training:
  max_steps: 150000
  batch_size: 32
  learning_rate: 1.0e-4

output:
  base_dir: "results"
  include_timestamp: true
  include_job_id: true
```

Override via command line:
```bash
python scripts/train.py --config configs/train/default.yaml \
    training.max_steps=200000 model.base_channels=128
```

## Results Organization

Results are automatically organized with timestamps and job IDs:
```
results/
└── evaluation_20251209_143022_job12345/
    ├── checkpoints/           # Model checkpoints
    ├── figures/               # Visualizations
    │   ├── bivariate_results.png
    │   └── vine_results.png
    ├── logs/                  # Training logs
    ├── metrics/               # JSON metrics
    ├── config.yaml            # Config used
    └── results.json           # Evaluation results
```

## Key Features

- **Denoising-based density estimation**: Iterative conditional diffusion and fast single-pass estimators
- **Vine copula support**: Build D-vine, C-vine, R-vine structures for high-dimensional data
- **Automatic constraint enforcement**: IPFP projection ensures valid copula densities
- **Multi-GPU training**: Distributed training with PyTorch DDP
- **Flexible configuration**: YAML configs with command-line overrides
- **Comprehensive evaluation**: Bivariate ISE, vine log-likelihood, Rosenblatt uniformity tests

## Comparing One-Shot vs Diffusion

This repo includes configs for head-to-head comparisons:
- `configs/train/diffusion_cond.yaml` (iterative diffusion, optional histogram conditioning / CFG)
- `configs/train/denoiser_cond.yaml` (single-pass denoiser)
- `configs/train/enhanced_cnn_cond.yaml` (strong CNN baseline)

Use `scripts/model_selection.py` to compare checkpoints on a fixed bivariate suite.

## Paper (ICML 2026)

The paper source and paper-only SLURM jobs live in the separate `drafts/` repository:
- LaTeX: `drafts/vine_diffusion.tex`
- Artifact scripts: `drafts/scripts/paper_artifacts.py`

## Documentation

| Document | Description |
|----------|-------------|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Quick start guide for new users |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | **Complete configuration guide for all tasks** |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Comprehensive usage guide with examples |
| [docs/API.md](docs/API.md) | Complete API reference |
| [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md) | Mathematical framework |
| [docs/WHY_DIFFUSION.md](docs/WHY_DIFFUSION.md) | Design decisions and motivation |

## Examples

| Example | Description |
|---------|-------------|
| [examples/use_pretrained_model.py](examples/use_pretrained_model.py) | Load model, estimate density, compute h-functions |
| [examples/train_custom_copula.py](examples/train_custom_copula.py) | Create custom config, train model |
| [examples/fit_vine_copula.py](examples/fit_vine_copula.py) | Fit D-vine to high-dimensional data |

Run examples:
```bash
# Use pretrained model
python examples/use_pretrained_model.py --checkpoint checkpoints/model.pt

# Create custom training config
python examples/train_custom_copula.py

# Fit vine copula
python examples/fit_vine_copula.py --checkpoint checkpoints/model.pt --dimension 5
```

## Citation

```bibtex
@software{vine_diffusion_copula,
  title={Vine Diffusion Copula: Deep Learning for Copula Density Estimation},
  author={Kempner Institute},
  year={2025},
  url={https://github.com/KempnerInstitute/vine_diffusion_copula}
}
```

## License

MIT License
