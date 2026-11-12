# Vine Denoising Copula (VDC)

[Documentation](https://kempnerinstitute.github.io/vine-denoising-copula/) · [Examples](examples/) ·
[NeurIPS 2026 project](projects/neurips2026/) · [Changelog](CHANGELOG.md)

VDC estimates bivariate copula densities with a reusable pretrained network and
assembles them into simplified vines. The library provides training, pretrained
model loading, density evaluation, vine fitting, sampling, and information
estimation for continuous data.

## Install

```bash
git clone https://github.com/KempnerInstitute/vine-denoising-copula.git
cd vine-denoising-copula
pip install -e .
```

Use `environment.yml` for a Conda environment. Install the `data` extra for the
UCI downloader and the `test` extra for the test suite.

## Estimate a pair density

```python
import numpy as np
from vdc import load_pretrained_model, estimate_pair_density_from_samples

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
u = np.random.default_rng(42).uniform(size=(2000, 2))
density = estimate_pair_density_from_samples(bundle, u)
print(density.shape)  # (64, 64)
```

Inputs are pseudo-observations in `[0, 1]`. Fit marginal transforms on training
data before applying them to held-out observations. The released model is
available from [Hugging Face](https://huggingface.co/hsafaai/vdc-denoiser-m64-v1);
the loader checks its checkpoint hash.

## Fit a vine

```python
from vdc import VineCopulaModel

u = np.random.default_rng(42).uniform(size=(1000, 5))
vine = VineCopulaModel(
    vine_type="dvine", m=64, device="cpu", batch_edges=True, edge_batch_size=16,
)
vine.fit(u, bundle.model, diffusion=bundle.diffusion)
log_density = vine.logpdf(u)
samples = vine.simulate(n=500)
```

D/C-vine fitting can batch edges from the same tree with `batch_edges=True`.
`fast_fit=True` enables
execution optimizations while preserving the estimator's arithmetic settings.
The regular-vine fitting path evaluates edges separately. Inverse-Rosenblatt
sampling supports the implemented D/C-vine paths; arbitrary R-vine sampling is
not implemented.

The grid projection enforces copula marginals to finite numerical tolerance.
Interpolated edge densities and h-functions are approximate counterparts, so
whole-vine normalization and conditional calibration are approximate. These
limits matter for downstream use even when edge density estimates are accurate.

## Commands and training

```bash
vdc list-models
vdc resolve-model --model-id vdc-denoiser-m64-v1
vdc estimate-pair data/pair.npy --output results/density.npy
vdc fit-vine data/pseudo_obs.npy --output results/vine.pkl --vine-type dvine
python scripts/train_unified.py --config configs/train/denoiser_cond.yaml --model-type denoiser
```

See `examples/` for model loading, vine fitting, information estimation and
custom training. The older `scripts/infer.py` and `scripts/evaluate.py` commands target the
GridUNet workflow. `scripts/train.py` is a separate training frontend; use the
pretrained API or `scripts/train_unified.py` for the workflow documented here.

## Repository contents

- `vdc/`: models, training, data generators/loaders, inference, vine operations,
  information metrics and baseline adapters.
- `scripts/`: software entry points, dataset/model downloads and release tools.
- `tests/`: library correctness and numerical identity tests.
- `examples/`, `configs/`, `docs/`: usage examples, training configurations and
  software documentation.

The [NeurIPS 2026 project](projects/neurips2026/) provides portable benchmark
runners, exact protocols, reference results and checkpoint hashes. The
[versioned figure archive](https://github.com/KempnerInstitute/vine-denoising-copula/releases/download/v0.2.6/vdc-neurips2026-figures-v0.2.6.zip)
provides the figure/table generators and their saved inputs. Projects are not
installed as part of the `vdc` Python package.

## Documentation and tests

- [Getting started](docs/GETTING_STARTED.md)
- [User guide](docs/USER_GUIDE.md)
- [API reference](docs/API.md)
- [Training configuration](docs/CONFIGURATION.md)
- [Pretrained models](docs/MODEL_RELEASES.md)
- [Datasets](docs/DATASETS.md)
- [Optimized fitting](docs/fast_fit.md)
- [Compiled sampler and fallback](docs/compiled_sampler.md)

```bash
pip install -e '.[test]'
python -m pytest tests
```

The compiled sampler has an automatic NumPy fallback and a configurable packed
table memory limit. See its documentation before choosing resources for large
vines.

## Citation

Houman Safaai. *Amortized Vine Copulas for High-Dimensional Density and Information Estimation*. NeurIPS, 2026.
[arXiv:2604.20568](https://arxiv.org/abs/2604.20568).

Software version: 0.2.7. Machine-readable citation: [CITATION.cff](CITATION.cff).

## Contributing

`main` is the maintained public library branch. See [CONTRIBUTING.md](CONTRIBUTING.md)
for development checks and [the release guide](docs/RELEASING.md) for publication.

## License

MIT; see [LICENSE](LICENSE).
