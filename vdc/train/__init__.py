"""Training entry points; install ``vine-denoising-copula[train]`` to train."""
from vdc.models.factory import build_model

__all__ = [
    "setup_distributed", "cleanup_distributed", "build_binning", "build_geometry",
    "build_model", "train",
]


def __getattr__(name):
    if name in __all__:
        from . import unified_trainer
        value = getattr(unified_trainer, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
