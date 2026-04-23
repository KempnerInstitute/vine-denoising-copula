# Vine Denoising Copula (VDC)

<p>
  <a href="https://github.com/KempnerInstitute/vine-denoising-copula/actions/workflows/ci.yml"><img src="https://github.com/KempnerInstitute/vine-denoising-copula/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/KempnerInstitute/vine-denoising-copula"><img src="https://img.shields.io/badge/python-3.9%2B-1f6feb" alt="Python"></a>
  <a href="https://github.com/KempnerInstitute/vine-denoising-copula/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-137333" alt="License"></a>
  <a href="https://huggingface.co/hsafaai/vdc-denoiser-m64-v1"><img src="https://img.shields.io/badge/Hugging%20Face-vdc--denoiser--m64--v1-f9d649" alt="Model"></a>
</p>

**VDC** is a Python package and released model for amortized vine-copula estimation. It trains a single neural edge estimator once on synthetic copulas and reuses it across all $O(d^2)$ vine edges, replacing repeated per-edge optimization with fast forward passes while preserving explicit copula structure.

<p align="center">
  <img src="docs/assets/method_pipeline.png" alt="VDC method pipeline" width="100%">
</p>

## Key Features

- **Amortized bivariate estimation**: train-once denoising edge estimator + IPFP projection to guarantee valid copula densities
- **Fast vine fitting**: GPU-batched edge estimation with cached h-functions for D-vine, C-vine, and R-vine structures
- **Information estimation**: MI and total correlation from explicit copula densities, with edge-wise decomposition
- **Self-consistent estimates**: 0% DPI violations under tested MI consistency protocol

<p align="center">
  <img src="docs/assets/vine_schematic.png" alt="D-vine factorization" width="45%">
</p>

## Quick Start

### Install

```bash
git clone https://github.com/KempnerInstitute/vine-denoising-copula.git vine-denoising-copula
cd vine-denoising-copula
conda env create -f environment.yml
conda activate vdc
pip install -e .
```

### Use the released model

```bash
python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1
python examples/use_pretrained_model.py --model-id vdc-denoiser-m64-v1
```

### Python API

```python
import numpy as np
from vdc import estimate_pair_density_from_samples, load_pretrained_model

# Load pretrained edge estimator
bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")

# Estimate a bivariate copula density
u = np.random.rand(2000, 2)
density = estimate_pair_density_from_samples(bundle, u)
print(density.shape)  # (64, 64)
```

### Fit a vine copula

```python
import numpy as np
from vdc import VineCopulaModel, load_pretrained_model

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
U = np.random.rand(1000, 5)

vine = VineCopulaModel(vine_type="dvine", m=bundle.config["data"]["m"], device="cpu")
vine.fit(U, bundle.model, diffusion=bundle.diffusion)

# Evaluate
loglik = vine.logpdf(U)
samples = vine.simulate(n=500)
```

### Command-line tools

The installed `vdc` command supports the following public commands:

```bash
vdc list-models
vdc resolve-model --model-id vdc-denoiser-m64-v1
vdc estimate-pair data/pair.npy --output results/density.npy
vdc fit-vine data/pseudo_obs.npy --output results/vine.pkl --vine-type dvine
```

## Core Workflow

The inference pipeline consists of four steps:

1. Build a normalized histogram from bivariate pseudo-observations
2. Predict a positive density grid with the frozen pretrained model
3. Project the grid to a valid copula via IPFP (uniform marginals, unit mass)
4. Reuse the resulting pair-copula estimator inside vine recursion and information calculations

## Released Model

The packaged released model is `vdc-denoiser-m64-v1`. The published weights live on [Hugging Face](https://huggingface.co/hsafaai/vdc-denoiser-m64-v1), and the release workflow is documented in [docs/MODEL_RELEASES.md](docs/MODEL_RELEASES.md).

### Verification

```bash
python scripts/verify_pretrained_release.py \
  --model-id vdc-denoiser-m64-v1 \
  --device cpu \
  --out-dir docs/reports/pretrained_release
```

The verification checks analytic bivariate cases (Gaussian, Clayton, Frank, Gumbel), mass preservation after IPFP projection, and MI accuracy from the released checkpoint.

Reports:
- [Pretrained release verification](docs/reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md)
- [MI benchmark](docs/reports/pretrained_release/MI_BENCHMARK_DCD_RELEASE.md)

## Training

Train a denoiser:

```bash
python scripts/train_unified.py --config configs/train/denoiser_cond.yaml --model-type denoiser
```

Train a diffusion-style model:

```bash
python scripts/train_unified.py --config configs/train/diffusion_cond.yaml --model-type diffusion_unet
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --checkpoint checkpoints/model.pt
python scripts/model_selection.py --checkpoints checkpoints/*/model_step_*.pt --n-samples 2000
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API.md)
- [Configuration](docs/CONFIGURATION.md)
- [Model Releases](docs/MODEL_RELEASES.md)

## Associated Paper

The public method paper for VDC is:

- Houman Safaai, *Amortized Vine Copulas for High-Dimensional Density and Information Estimation*, arXiv:2604.20568, 2026
- arXiv: https://arxiv.org/abs/2604.20568
- DOI: https://doi.org/10.48550/arXiv.2604.20568

## Verification

This public repository supports:

- loading and verifying the released checkpoint
- rerunning benchmark scripts that live in the package repo
- regenerating the public verification reports under `docs/reports/`

## Citation

If you use this software, please cite the repository:

```bibtex
@software{safaai2026vdc,
  author  = {Houman Safaai},
  title   = {Vine Denoising Copula (VDC)},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/KempnerInstitute/vine-denoising-copula}
}
```

If you use the VDC method, please also cite the paper:

```bibtex
@article{safaai2026amortized,
  author  = {Houman Safaai},
  title   = {Amortized Vine Copulas for High-Dimensional Density and Information Estimation},
  journal = {arXiv preprint arXiv:2604.20568},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.20568},
  url     = {https://arxiv.org/abs/2604.20568}
}
```

A machine-readable citation record is available in [CITATION.cff](CITATION.cff).

## License

MIT; see [LICENSE](LICENSE).

## Notes

- The released model assumes continuous marginals and pseudo-observations in `[0,1]`.
- Vine fitting uses the simplifying assumption (constant conditional copulas).
- The released checkpoint is frozen and versioned.
- Current grid resolution is m=64. Extensions to larger grids and the probit transform are documented in the configuration guide.
