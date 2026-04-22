# Getting Started

This guide is a short path from a fresh clone to a working copula estimate or vine fit.

## Choose Your Path

Most users want one of these:

1. Use the released pretrained model
2. Verify the released checkpoint
3. Fit a vine copula
4. Train a new model

If you are not sure, start with the pretrained model.

## Installation

```bash
git clone https://github.com/KempnerInstitute/vine-denoising-copula.git vine-denoising-copula
cd vine-denoising-copula
conda env create -f environment.yml
conda activate vdc
pip install -e .
```

Sanity check:

```bash
python -c "import vdc; print(vdc.__version__)"
```

## Path 1: Use The Official Pretrained Model

List the packaged model ids:

```bash
vdc list-models
```

Resolve the released model:

```bash
vdc resolve-model --model-id vdc-denoiser-m64-v1
```

Run the example:

```bash
python examples/use_pretrained_model.py --model-id vdc-denoiser-m64-v1
```

In Python:

```python
import numpy as np
from vdc import estimate_pair_density_from_samples, load_pretrained_model

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
samples = np.random.rand(2000, 2)
density = estimate_pair_density_from_samples(bundle, samples)
print(density.shape)
```

## Path 2: Verify The Released Checkpoint

Run the release verification:

```bash
python scripts/verify_pretrained_release.py \
  --model-id vdc-denoiser-m64-v1 \
  --device cpu \
  --out-dir docs/reports/pretrained_release
```

This produces:

- a bivariate verification grid
- a regenerated qualitative figure
- a JSON summary
- a markdown verification report

See:

- [docs/reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md](reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md)

## Path 3: Fit A Vine Copula

VDC expects pseudo-observations, meaning each marginal has already been mapped into `[0, 1]`.

```python
import numpy as np
from vdc import VineCopulaModel, load_pretrained_model

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
U = np.random.rand(1000, 5)

vine = VineCopulaModel(vine_type="dvine", m=bundle.config["data"]["m"], device="cpu")
vine.fit(U, bundle.model, diffusion=bundle.diffusion)

logpdf = vine.logpdf(U[:50])
print(logpdf[:5])
```

The same flow is available from the command line when your data is already in pseudo-observation space:

```bash
vdc estimate-pair data/pair.npy --output results/density.npy
vdc fit-vine data/pseudo_obs.npy --output results/vine.pkl --vine-type dvine
```

## Path 4: Train A New Model

The repo supports three main training families:

- one-shot denoiser
- enhanced CNN baseline
- diffusion-style iterative model

Examples:

```bash
python scripts/train_unified.py --config configs/train/denoiser_cond.yaml --model-type denoiser
python scripts/train_unified.py --config configs/train/enhanced_cnn_cond.yaml --model-type enhanced_cnn
python scripts/train_unified.py --config configs/train/diffusion_cond.yaml --model-type diffusion_unet
```

For the corruption ablation work:

```bash
python scripts/train_unified.py --config configs/train/denoiser_cond_enhanced_direct.yaml --model-type denoiser
python scripts/train_unified.py --config configs/train/denoiser_cond_enhanced_gaussian.yaml --model-type denoiser
python scripts/train_unified.py --config configs/train/denoiser_cond_enhanced_multinomial.yaml --model-type denoiser
```

## What The Model Consumes

The pretrained model consumes bivariate pseudo-observations:

- input shape `(n, 2)`
- values in `[0, 1]`
- continuous marginals

Internally, VDC:

1. builds a normalized histogram on an `m x m` grid
2. predicts a positive density grid
3. projects the grid with IPFP to enforce discrete copula constraints
4. uses the resulting density inside h-functions and vine recursion

## Common Commands

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --checkpoint checkpoints/model.pt
```

Compare checkpoints:

```bash
python scripts/model_selection.py --checkpoints checkpoints/*/model_step_*.pt --n-samples 2000
```

Estimate MI with the released checkpoint:

```bash
export VDC_CHECKPOINT="$(python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1)"
python scripts/mi_estimation.py \
  --estimator dcd \
  --checkpoint "${VDC_CHECKPOINT}" \
  --device cpu \
  --n-samples 2000 \
  --out-json results/mi_benchmark_dcd.json
```

## Where To Go Next

- [docs/index.html](index.html)
- [docs/USER_GUIDE.md](USER_GUIDE.md)
- [docs/API.md](API.md)
- [docs/MODEL_RELEASES.md](MODEL_RELEASES.md)
