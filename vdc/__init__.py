"""
Vine-Diffusion-Copula: High-dimensional copula modeling with diffusion networks.

This package provides tools for:
- Generating synthetic copula training data
- Training diffusion-based bivariate copula estimators
- Building vine-copula structures
- Evaluating densities and generating samples from high-dimensional distributions
"""

__version__ = "0.1.0"

from vdc.models.api import DiffusionCopulaEstimator, load_trained_estimator
from vdc.vine.api import DiffusionVine

__all__ = [
    "DiffusionCopulaEstimator",
    "DiffusionVine",
    "load_trained_estimator",
]
