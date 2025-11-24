"""
Training package for vine diffusion copula models.

Contains the unified training engine and utilities.
"""
from .unified_trainer import (
    setup_distributed,
    cleanup_distributed,
    build_binning,
    build_geometry,
    build_model,
    train,
)

__all__ = [
    'setup_distributed',
    'cleanup_distributed',
    'build_binning',
    'build_geometry',
    'build_model',
    'train',
]

