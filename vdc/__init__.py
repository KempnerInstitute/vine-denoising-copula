"""
Vine-Diffusion-Copula: High-dimensional copula modeling with diffusion networks.

This package provides tools for:
- Generating synthetic copula training data
- Training diffusion-based bivariate copula estimators  
- Building vine-copula structures (D-vine, C-vine, R-vine)
- Evaluating densities and generating samples from high-dimensional distributions

Quick Start:
    # Load a trained model
    from vdc.models.unet_grid import GridUNet
    from vdc.models.copula_diffusion import CopulaAwareDiffusion
    
    # Fit a vine copula
    from vdc.vine.api import VineCopulaModel
    
    # See examples/ for complete usage examples
"""

__version__ = "0.1.0"

# Core exports
__all__ = [
    # Models
    "GridUNet",
    "CopulaAwareDiffusion",
    # Vine API
    "VineCopulaModel",
    # Pretrained
    "DEFAULT_PRETRAINED_MODEL_ID",
    "list_pretrained_models",
    "load_pretrained_manifest",
    "load_pretrained_model",
    "resolve_pretrained_checkpoint",
    # Config
    "Config",
    "get_run_dir",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "GridUNet":
        from vdc.models.unet_grid import GridUNet
        return GridUNet
    elif name == "CopulaAwareDiffusion":
        from vdc.models.copula_diffusion import CopulaAwareDiffusion
        return CopulaAwareDiffusion
    elif name == "VineCopulaModel":
        from vdc.vine.api import VineCopulaModel
        return VineCopulaModel
    elif name == "DEFAULT_PRETRAINED_MODEL_ID":
        from vdc.pretrained import DEFAULT_PRETRAINED_MODEL_ID
        return DEFAULT_PRETRAINED_MODEL_ID
    elif name == "list_pretrained_models":
        from vdc.pretrained import list_pretrained_models
        return list_pretrained_models
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
