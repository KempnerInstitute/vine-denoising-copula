# Vine Diffusion Copula

<p>
  <a href="https://github.com/KempnerInstitute/vine-diffusion-copula/actions/workflows/ci.yml"><img src="https://github.com/KempnerInstitute/vine-diffusion-copula/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/KempnerInstitute/vine-diffusion-copula/actions/workflows/pages.yml"><img src="https://github.com/KempnerInstitute/vine-diffusion-copula/actions/workflows/pages.yml/badge.svg" alt="Docs"></a>
  <a href="https://github.com/KempnerInstitute/vine-diffusion-copula"><img src="https://img.shields.io/badge/python-3.9%2B-1f6feb" alt="Python"></a>
  <a href="https://github.com/KempnerInstitute/vine-diffusion-copula/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-137333" alt="License"></a>
  <a href="https://huggingface.co/hsafaai/vdc-denoiser-m64-v1"><img src="https://img.shields.io/badge/Hugging%20Face-vdc--denoiser--m64--v1-f9d649" alt="Model"></a>
</p>

Vine Diffusion Copula, or VDC, is a codebase for pretrained bivariate copula density estimation, vine copula fitting, and information estimation from learned copula densities.

The main released artifact is a frozen pretrained edge estimator, `vdc-denoiser-m64-v1`. Given bivariate pseudo-observations, it predicts a positive density grid, projects that grid to a valid copula with IPFP, and reuses the resulting estimator across vine edges.

<p align="center">
  <img src="docs/assets/method_pipeline.png" alt="VDC method pipeline" width="100%">
</p>

## Quick Start

### Install

```bash
git clone https://github.com/KempnerInstitute/vine_diffusion_copula.git
cd vine_diffusion_copula
conda env create -f environment.yml
conda activate vdc
pip install -e .
```

### Use the released model

```bash
python scripts/download_pretrained.py --list
python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1
python examples/use_pretrained_model.py --model-id vdc-denoiser-m64-v1
```

### Verify the released model

```bash
python scripts/verify_pretrained_release.py \
  --model-id vdc-denoiser-m64-v1 \
  --device cpu \
  --out-dir docs/reports/pretrained_release
```

Reports:
- [docs/reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md](docs/reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md)
- [docs/reports/pretrained_release/MI_BENCHMARK_DCD_RELEASE.md](docs/reports/pretrained_release/MI_BENCHMARK_DCD_RELEASE.md)

### Documentation

- [docs/index.html](docs/index.html)
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- [docs/MODEL_RELEASES.md](docs/MODEL_RELEASES.md)
- [docs/PAPER_REPRODUCIBILITY.md](docs/PAPER_REPRODUCIBILITY.md)

## Main Capabilities

- Estimate valid bivariate copula densities from samples in `[0, 1]^2`
- Fit D-vine, C-vine, and R-vine style dependence models
- Compute mutual information and total correlation from learned copula densities
- Train denoising, CNN, and diffusion-style model variants
- Reproduce the released checkpoint and verification artifacts

## Released Model

The packaged released model id is `vdc-denoiser-m64-v1`.

The repository stores the manifest and loader code, not the large checkpoint itself. The published weights live on Hugging Face, and the release workflow is documented in [docs/MODEL_RELEASES.md](docs/MODEL_RELEASES.md).

Python usage:

```python
import numpy as np
from vdc.pretrained import estimate_pair_density_from_samples, load_pretrained_model

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
u = np.random.rand(2000, 2)
density = estimate_pair_density_from_samples(bundle, u)
print(density.shape)
```

## Core Workflow

The released inference path is:

1. Build a normalized histogram from bivariate pseudo-observations.
2. Predict a positive density grid with the released model or a trained checkpoint.
3. Project the grid to a valid copula with IPFP.
4. Reuse the resulting pair-copula estimator inside vine recursion and information calculations.

## Training And Evaluation

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

Fit a vine model:

```python
import numpy as np
from vdc.pretrained import load_pretrained_model
from vdc.vine.api import VineCopulaModel

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
U = np.random.rand(1000, 5)

vine = VineCopulaModel(vine_type="dvine", m=bundle.config["data"]["m"], device="cpu")
vine.fit(U, bundle.model, diffusion=bundle.diffusion)
```

## Verification Assets

The release verification checks:

- analytic bivariate cases including Gaussian, Clayton, Frank, and Gumbel
- mass preservation after projection
- mutual information accuracy from the released checkpoint
- regeneration of the main qualitative pair-copula figure

Reference artifacts:
- [docs/reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md](docs/reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md)
- [docs/reports/pretrained_release/MI_BENCHMARK_DCD_RELEASE.md](docs/reports/pretrained_release/MI_BENCHMARK_DCD_RELEASE.md)
- [docs/assets/method_pipeline.png](docs/assets/method_pipeline.png)

## Notes

- The released model assumes continuous marginals and pseudo-observations.
- Vine fitting uses the simplifying assumption.
- The released checkpoint is frozen and versioned.
