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
    elif name == "Config":
        from vdc.config import Config
        return Config
    elif name == "get_run_dir":
        from vdc.config import get_run_dir
        return get_run_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
