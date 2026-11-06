# Getting started

Install the package from the repository root:

```bash
pip install -e .
python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1
python examples/use_pretrained_model.py --model-id vdc-denoiser-m64-v1
```

The loader verifies the downloaded checkpoint against the packaged manifest.
Set `device="cuda"` when using the Python API on a suitable GPU. A CPU is
sufficient to load the model and estimate pair densities, but its memory
requirements still apply.

The [README](../README.md) gives pair-density and vine-fitting examples.
[API](API.md) describes the public classes and functions;
[model releases](MODEL_RELEASES.md) describes checkpoint storage and versioning.

To train a denoiser, use the unified trainer:

```bash
python scripts/train_unified.py --config configs/train/denoiser_cond.yaml --model-type denoiser
```

The `configs/train/` files specify model and data choices. Inspect the selected
configuration before training. The older `train.py`, `infer.py` and `evaluate.py`
commands use the legacy GridUNet workflow and are not interchangeable with
pretrained-denoiser loading.

Paper-specific benchmarks, plotting scripts and saved results are in the
[paper companion](https://github.com/houman1359/diffusion-vine-copula/tree/camera-ready/reproducibility).
They are not needed to use the library.
