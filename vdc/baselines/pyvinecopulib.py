"""Wrappers around `pyvinecopulib` for paper baselines.

We keep these wrappers optional so the rest of the codebase works without
pyvinecopulib installed.

Baseline mapping (paper-friendly)
---------------------------------
- **pyvinecopulib-parametric**: Fit with `BicopFamily.parametric` (BIC selection).
- **pyvinecopulib-nonparametric**: Fit with `BicopFamily.nonparametric` (includes TLL).

`pyvinecopulib` provides both bivariate (`Bicop`) and multivariate (`Vinecop`) fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def is_available() -> bool:
    try:
        import pyvinecopulib as _  # noqa: F401

        return True
    except Exception:
        return False


def _require() -> Any:
    try:
        import pyvinecopulib as pv

        return pv
    except Exception as e:
        raise RuntimeError(
            "pyvinecopulib is not available. Install it (recommended: via pip/conda env) to run "
            "pyvinecopulib baselines."
        ) from e


@dataclass(frozen=True)
class BicopFitResult:
    """Thin wrapper for a fitted bivariate copula model."""

    model: Any
    family: str

    def loglik_per_sample(self, u: np.ndarray) -> float:
        # pyvinecopulib returns total loglik (sum over samples)
        ll = float(self.model.loglik(u))
        return ll / max(1, int(u.shape[0]))

    def pdf(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.pdf(u), dtype=np.float64)

    def rosenblatt(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.rosenblatt(u), dtype=np.float64)


def fit_bicop(
    u: np.ndarray,
    *,
    mode: str = "parametric",
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
    nonparametric_method: str = "constant",
    nonparametric_mult: float = 1.0,
    nonparametric_grid_size: int = 30,
) -> BicopFitResult:
    """Fit a bivariate copula using pyvinecopulib.

    Args:
        u: (n,2) pseudo-observations in (0,1).
        mode: 'parametric' or 'nonparametric' (TLL).
    """
    pv = _require()
    x = np.asarray(u, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"u must be (n,2), got shape={x.shape}")
    x = np.clip(x, 1e-10, 1.0 - 1e-10)

    mode_l = str(mode).lower()
    if mode_l in {"parametric", "param"}:
        fam_set = pv.BicopFamily.parametric
        family_label = "pyvinecopulib_parametric"
    elif mode_l in {"nonparametric", "nonpar", "tll"}:
        # includes indep + tll
        fam_set = pv.BicopFamily.nonparametric
        family_label = "pyvinecopulib_nonparametric"
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'parametric' or 'nonparametric'.")

    controls = pv.FitControlsBicop(
        family_set=fam_set,
        selection_criterion=str(selection_criterion),
        allow_rotations=bool(allow_rotations),
        num_threads=int(num_threads),
        nonparametric_method=str(nonparametric_method),
        nonparametric_mult=float(nonparametric_mult),
        nonparametric_grid_size=int(nonparametric_grid_size),
    )

    model = pv.Bicop(x, controls=controls)
    return BicopFitResult(model=model, family=family_label)


def fit_vinecop(
    u: np.ndarray,
    *,
    mode: str = "parametric",
    trunc_lvl: Optional[int] = None,
    tree_criterion: str = "tau",
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
    nonparametric_method: str = "constant",
    nonparametric_mult: float = 1.0,
    nonparametric_grid_size: int = 30,
) -> Any:
    """Fit a vine copula using pyvinecopulib (returns a `Vinecop`)."""
    pv = _require()
    x = np.asarray(u, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(f"u must be (n,d), got shape={x.shape}")
    x = np.clip(x, 1e-10, 1.0 - 1e-10)

    mode_l = str(mode).lower()
    if mode_l in {"parametric", "param"}:
        fam_set = pv.BicopFamily.parametric
    elif mode_l in {"nonparametric", "nonpar", "tll"}:
        fam_set = pv.BicopFamily.nonparametric
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'parametric' or 'nonparametric'.")

    if trunc_lvl is None:
        trunc_lvl = x.shape[1] - 1

    controls = pv.FitControlsVinecop(
        family_set=fam_set,
        trunc_lvl=int(trunc_lvl),
        tree_criterion=str(tree_criterion),
        selection_criterion=str(selection_criterion),
        allow_rotations=bool(allow_rotations),
        num_threads=int(num_threads),
        nonparametric_method=str(nonparametric_method),
        nonparametric_mult=float(nonparametric_mult),
        nonparametric_grid_size=int(nonparametric_grid_size),
    )
    return pv.Vinecop(x, controls=controls)

