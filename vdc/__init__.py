"""Public package surface for Vine Denoising Copula (VDC).

The public API is centered on:

- ``load_pretrained_model`` for loading the published checkpoint
- ``estimate_pair_density_from_samples`` for bivariate copula estimation
- ``VineCopulaModel`` for fitting and querying explicit vine models
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core exports
__all__ = [
    "LoadedPretrainedModel",
    "DEFAULT_PRETRAINED_MODEL_ID",
    "estimate_pair_density_from_samples",
    "list_pretrained_models",
    "load_checkpoint_bundle",
    "load_pretrained_manifest",
    "load_pretrained_model",
    "resolve_pretrained_checkpoint",
    "VineCopulaModel",
    "Config",
    "get_run_dir",
    # Advanced / lower-level exports
    "GridUNet",
    "CopulaAwareDiffusion",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "GridUNet":
        from vdc.models.unet_grid import GridUNet
        return GridUNet
    elif name == "CopulaAwareDiffusion":
        from vdc.models.copula_diffusion import CopulaAwareDiffusion
        return CopulaAwareDiffusion
    elif name == "LoadedPretrainedModel":
        from vdc.pretrained import LoadedPretrainedModel
        return LoadedPretrainedModel
    elif name == "VineCopulaModel":
        from vdc.vine.api import VineCopulaModel
        return VineCopulaModel
    elif name == "DEFAULT_PRETRAINED_MODEL_ID":
        from vdc.pretrained import DEFAULT_PRETRAINED_MODEL_ID
        return DEFAULT_PRETRAINED_MODEL_ID
    elif name == "estimate_pair_density_from_samples":
        from vdc.pretrained import estimate_pair_density_from_samples
        return estimate_pair_density_from_samples
    elif name == "list_pretrained_models":
        from vdc.pretrained import list_pretrained_models
        return list_pretrained_models
    elif name == "load_checkpoint_bundle":
        from vdc.pretrained import load_checkpoint_bundle
        return load_checkpoint_bundle
    elif name == "load_pretrained_manifest":
        from vdc.pretrained import load_pretrained_manifest
        return load_pretrained_manifest
    elif name == "load_pretrained_model":
        from vdc.pretrained import load_pretrained_model
        return load_pretrained_model
    elif name == "resolve_pretrained_checkpoint":
        from vdc.pretrained import resolve_pretrained_checkpoint
        return resolve_pretrained_checkpoint
    elif name == "Config":
        from vdc.config import Config
        return Config
    elif name == "get_run_dir":
        from vdc.config import get_run_dir
        return get_run_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
