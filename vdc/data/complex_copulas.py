"""
Complex synthetic copula-like test cases.

Goal: create *challenging* dependence patterns (multi-modal / non-elliptical)
on [0,1]^2, then project them to a valid copula density (uniform marginals,
unit integral) using Sinkhorn/IPFP.

These are intended for evaluation and stress-testing (e.g., "X" shape,
"ring/O", "double banana"), and can optionally be used for training if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vdc.models.projection import copula_project


@dataclass(frozen=True)
class ComplexCopulaSpec:
    kind: str
    params: Dict[str, Any]
    name: str


DEFAULT_COMPLEX_TEST_SUITE: List[ComplexCopulaSpec] = [
    ComplexCopulaSpec(kind="x", params={"sigma": 0.03, "w2": 1.0}, name="Complex-X"),
    ComplexCopulaSpec(kind="ring", params={"r0": 0.30, "sigma": 0.035}, name="Complex-Ring (O)"),
    ComplexCopulaSpec(kind="double_banana", params={"amp": 0.15, "offset": 0.20, "sigma": 0.03}, name="Complex-DoubleBanana"),
]


def _grid_centers(m: int) -> np.ndarray:
    return np.linspace(0.5 / m, 1.0 - 0.5 / m, m, dtype=np.float64)


def _project_to_copula_density(
    density: np.ndarray,
    *,
    device: torch.device,
    projection_iters: int,
) -> np.ndarray:
    """Project a positive density grid to a valid copula density."""
    d = np.asarray(density, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"Expected square 2D grid, got shape {d.shape}")
    m = int(d.shape[0])
    du = 1.0 / m

    d = np.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0)
    d = np.clip(d, 1e-12, 1e6)
    d = d / (float(d.sum()) * du * du + 1e-12)

    t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,m,m)
    t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=1e-12, max=1e6)
    t = t / ((t * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    if int(projection_iters) > 0:
        t = copula_project(t, iters=int(projection_iters))
        t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=1e-12, max=1e6)
        t = t / ((t * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    return t[0, 0].detach().cpu().numpy()


def complex_copula_density_grid(
    kind: str,
    params: Optional[Dict[str, Any]],
    *,
    m: int = 64,
    device: torch.device,
    projection_iters: int = 80,
) -> np.ndarray:
    """
    Create a projected copula density for a given synthetic pattern.

    Args:
        kind: One of {"x", "ring", "double_banana"} (case-insensitive).
        params: Pattern parameters (see DEFAULT_COMPLEX_TEST_SUITE for examples).
        m: Grid resolution.
        device: Torch device for projection.
        projection_iters: Sinkhorn projection iterations.
    """
    kind_l = str(kind).lower().strip()
    p: Dict[str, Any] = {} if params is None else dict(params)

    u = _grid_centers(int(m))
    U, V = np.meshgrid(u, u, indexing="ij")

    # Note: these are *pre-copula* positive patterns; projection enforces copula constraints.
    if kind_l in {"x", "x_shape", "xshape"}:
        sigma = max(1e-4, float(p.get("sigma", 0.03)))
        w2 = float(p.get("w2", 1.0))
        d1 = np.abs(U - V)          # main diagonal
        d2 = np.abs(U + V - 1.0)    # anti-diagonal
        base = np.exp(-0.5 * (d1 / sigma) ** 2) + w2 * np.exp(-0.5 * (d2 / sigma) ** 2) + 1e-6
    elif kind_l in {"ring", "o", "circle"}:
        r0 = float(p.get("r0", 0.30))
        sigma = max(1e-4, float(p.get("sigma", 0.035)))
        r = np.sqrt((U - 0.5) ** 2 + (V - 0.5) ** 2)
        base = np.exp(-0.5 * ((r - r0) / sigma) ** 2) + 1e-6
    elif kind_l in {"double_banana", "doublebanana", "banana2"}:
        amp = float(p.get("amp", 0.15))
        offset = float(p.get("offset", 0.20))
        sigma = max(1e-4, float(p.get("sigma", 0.03)))
        y1 = 0.5 - offset + amp * np.sin(2.0 * np.pi * U)
        y2 = 0.5 + offset - amp * np.sin(2.0 * np.pi * U)
        base = np.exp(-0.5 * ((V - y1) / sigma) ** 2) + np.exp(-0.5 * ((V - y2) / sigma) ** 2) + 1e-6
    else:
        raise ValueError(f"Unknown complex copula kind: {kind!r}")

    return _project_to_copula_density(base, device=device, projection_iters=int(projection_iters))


__all__ = [
    "ComplexCopulaSpec",
    "DEFAULT_COMPLEX_TEST_SUITE",
    "complex_copula_density_grid",
]

