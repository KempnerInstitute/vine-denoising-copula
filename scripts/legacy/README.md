# Legacy command-line scripts

These entry points belong to the earlier GridUNet/diffusion workflow of the library and are kept
for backward compatibility. They are not used by the released `vdc-denoiser-m64-v1` model, the
paper's experiments or the documented API.

| Script | Purpose |
|---|---|
| `train.py` | earlier training front end (wraps the unified trainer for GridUNet/diffusion configurations) |
| `infer.py` | density estimation and sampling with a GridUNet/diffusion checkpoint |
| `evaluate.py` | bivariate and vine evaluation of a GridUNet/diffusion checkpoint |

For current work use the `vdc` command, the Python API (`vdc.load_pretrained_model`,
`vdc.VineCopulaModel`) and `scripts/train_unified.py`; see the [documentation](../../docs/GETTING_STARTED.md).
