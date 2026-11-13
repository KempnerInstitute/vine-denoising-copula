# Getting started

## Install

Python 3.10 or 3.11 is recommended. From a fresh environment:

```bash
git clone https://github.com/KempnerInstitute/vine-denoising-copula.git
cd vine-denoising-copula
python -m pip install -e .
```

For a CPU-only PyTorch installation, install Torch from its CPU wheel index
before installing VDC:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

GPU users should install the PyTorch build appropriate for their CUDA driver.
The package does not require a GPU. The pretrained network is large; allow
several GB of memory. The optional compiled sampler requires a C++17 compiler
with OpenMP and falls back to NumPy when unavailable.

## First pair-density estimate

```python
import numpy as np
from scipy.stats import norm
from vdc import load_pretrained_model, estimate_pair_density_from_samples

rng = np.random.default_rng(42)
z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=2000)
u = norm.cdf(z)
bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
density = estimate_pair_density_from_samples(bundle, u)
print(density.shape)  # (64, 64)
print(density.sum() / density.size)  # grid-integrated mass, approximately 1
```

The loader verifies the checkpoint hash. Set `VDC_PRETRAINED_CACHE` to choose a
cache directory. To use an already downloaded model without network access,
pass its path to `load_checkpoint_bundle`.

## Run the examples

```bash
python examples/use_pretrained_model.py --device cpu
python examples/fit_vine_copula.py --device cpu
python examples/estimate_mutual_information.py --device cpu
```

Each example uses the same public loader as applications. A local trusted
checkpoint can be supplied with `--checkpoint /path/to/model.pt`.

## Next steps

- [User guide](USER_GUIDE.md): raw-data transforms, fitting, sampling and information.
- [API reference](API.md): supported signatures and defaults.
- [Training configuration](CONFIGURATION.md): train a separate model.
- [Pretrained models](MODEL_RELEASES.md): caching, hashes and artifact versions.
- [Reproducing the paper](../paper/README.md): figure and table generators,
  saved outputs and fixed experimental protocols of the NeurIPS 2026 paper.

The paper material in `paper/` imports the library but is not installed with it.
