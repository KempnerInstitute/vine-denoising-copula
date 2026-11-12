# Pretrained models

## Released estimator

| Property | Value |
|---|---|
| Model ID | `vdc-denoiser-m64-v1` |
| Architecture | Single-pass denoiser, 64 × 64 grid |
| Training step | 190,000 |
| Model host | [Hugging Face](https://huggingface.co/hsafaai/vdc-denoiser-m64-v1) |
| SHA-256 | `8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f` |
| Manifest | [`vdc_denoiser_m64_v1.json`](../vdc/resources/pretrained/vdc_denoiser_m64_v1.json) |

The model ID versions the weights separately from the Python package. A library
update does not silently replace this checkpoint. Later research checkpoints
are not registered public replacements.

## Download and cache

```python
from vdc import load_pretrained_model
bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
```

Or download without loading the network:

```bash
vdc resolve-model --model-id vdc-denoiser-m64-v1
```

The cache defaults to `~/.cache/vdc`; override it with `VDC_PRETRAINED_CACHE` or
`cache_dir=`. The loader checks the checkpoint's SHA-256. An incorrect cached
file is an error, not an accepted replacement. A public download needs no write
token or personal cluster account.

## Local and offline models

```python
from vdc import load_checkpoint_bundle
bundle = load_checkpoint_bundle("/path/to/trusted-model.pt", device="cpu")
```

Only load trusted checkpoints. This reconstructs the model from its embedded
configuration; it does not certify a custom checkpoint against the public model
hash. The released denoiser and iterative diffusion models have different input
and inference conventions, handled by the loader.

## Publishing a separate model

Maintainers can use `scripts/package_pretrained_model.py` and
`scripts/upload_pretrained_to_hf.py`; see each command's `--help` and the
[release guide](RELEASING.md). Give new weights a new model ID, publish their
training/evaluation configuration, pin the host revision and record their hash.
Never overwrite the weights of an existing released model ID.
