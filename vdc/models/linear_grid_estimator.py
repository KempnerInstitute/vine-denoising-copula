"""Pair-copula estimation with the linear-grid representation.

The ``linear_grid`` estimator is the inference pipeline of the VDC-64 and
VDC-128 models (``vdc-denoiser-m64-v2`` and ``vdc-denoiser-m128-v2``):

1. Pseudo-observations are counted in ``m x m`` cells with uniform edges and
   divided by ``n`` times the cell area.
2. The network input stacks that histogram, ``log n`` and the probit
   coordinates of the cell centers (``scipy.special.ndtri``, rounded to
   float32), and the network is evaluated at ``t = 0``.
3. The predicted log density is clamped to ``[-15, 15]`` and exponentiated.
   Its values are treated as node values of a bilinear density and scaled to
   uniform margins with integrated basis weights (``projection_iters``
   alternating float32 row/column scalings).
4. A :class:`vdc.models.linear_grid_copula.LinearGridCopula` re-projects the
   grid in float64 and provides the matching PDF, h and inverse-h functions.

The released models were evaluated in float32 with TF32 disabled, network
batches of exactly 16 edges (``forward_batch_size``; shorter batches are
padded) and outer edge batches of 64. :func:`estimate_linear_grid_densities`
disables TF32 and cuDNN benchmarking while it runs the network and restores
the previous settings afterwards. Floating-point identity is established only
for the recorded device type and software versions, and for the same grouping
of pairs per call: on GPUs, a pair's 64-cell grid estimated alone differed
from the same pair estimated with others by up to about 1e-6 (relative), so
recorded evaluations are reproduced by passing the same list of pairs (see
``docs/IMPROVED_ESTIMATORS.md``).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.special import ndtr, ndtri

from .linear_grid_copula import LinearGridCopula, basis_weights

LINEAR_GRID_ESTIMATOR = "linear_grid"
_MODEL_METADATA = ("vdc_use_log_n", "vdc_use_coordinates", "vdc_use_probit_coords", "vdc_probit_coord_eps")


def grid_geometry(m: int, spacing: str = "uniform"):
    """Return ``(edges, nodes)`` of an ``m``-cell grid on [0, 1].

    ``"uniform"`` uses equal cells; ``"probit"`` uses normal-quantile edges
    ``ndtr(linspace(-3.5, 3.5, m + 1))`` with the outer edges set to 0 and 1.
    Nodes are cell midpoints. The released models use uniform cells.
    """
    m = int(m)
    if m < 2:
        raise ValueError("Grid size must be at least 2")
    if spacing not in ("uniform", "probit"):
        raise ValueError("spacing must be 'uniform' or 'probit'")
    edges = np.linspace(0, 1, m+1) if spacing == "uniform" else ndtr(np.linspace(-3.5, 3.5, m+1))
    edges[0], edges[-1] = 0., 1.
    return edges, .5*(edges[:-1]+edges[1:])


def project_density_torch(d: torch.Tensor, weights: torch.Tensor, iterations: int = 30) -> torch.Tensor:
    """Differentiable projection under integrated bilinear-basis weights.

    ``d`` has shape ``(B, m, m)``; each iteration rescales rows then columns so
    that ``d @ weights`` and ``weights @ d`` approach one.
    """
    for _ in range(iterations):
        d = d/(d*weights[None,None,:]).sum(-1, keepdim=True).clamp_min(1e-12)
        d = d/(d*weights[None,:,None]).sum(-2, keepdim=True).clamp_min(1e-12)
    return d


def bilinear_density_torch(d: torch.Tensor, nodes: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Evaluate node-value grids bilinearly with constant boundary extension.

    ``d`` has shape ``(B, m, m)``, ``uv`` has shape ``(B, N, 2)``; returns
    ``(B, N)`` densities, differentiable with respect to ``d``.
    """
    m = d.shape[-1]
    u = uv[...,0].clamp(float(nodes[0]),float(nodes[-1]))
    v = uv[...,1].clamp(float(nodes[0]),float(nodes[-1]))
    i = (torch.searchsorted(nodes, u.contiguous())-1).clamp(0,m-2)
    j = (torch.searchsorted(nodes, v.contiguous())-1).clamp(0,m-2)
    a, b = (u-nodes[i])/(nodes[i+1]-nodes[i]), (v-nodes[j])/(nodes[j+1]-nodes[j])
    batch = torch.arange(len(d), device=d.device)[:,None]
    return ((1-a)*((1-b)*d[batch,i,j]+b*d[batch,i,j+1])
            +a*((1-b)*d[batch,i+1,j]+b*d[batch,i+1,j+1]))


class FixedBatchForward(torch.nn.Module):
    """Run a network with one physical batch shape.

    Inputs are split into chunks of ``batch_size``; a short final chunk is
    padded by repeating its first row and the padding outputs are discarded.
    This fixes the execution shape of the network. It does not establish
    invariance across devices, precision settings, physical batch sizes or
    the grouping of pairs in one call.
    """

    def __init__(self, model: torch.nn.Module, batch_size: int = 16):
        super().__init__()
        if batch_size < 1:
            raise ValueError('Physical batch size must be positive')
        self.model = model.eval()
        self.batch_size = batch_size
        for key in _MODEL_METADATA:
            if hasattr(model, key):
                setattr(self, key, getattr(model, key))

    @property
    def in_conv(self):
        return self.model.in_conv

    @torch.no_grad()
    def forward(self, x, t):
        if len(x) != len(t) or not len(x):
            raise ValueError('Invalid inference batch')
        outputs = []
        for start in range(0, len(x), self.batch_size):
            xx, tt = x[start:start+self.batch_size], t[start:start+self.batch_size]
            count = len(xx)
            if count < self.batch_size:
                xx = torch.cat([xx, xx[:1].expand(self.batch_size-count, *xx.shape[1:])])
                tt = torch.cat([tt, tt[:1].expand(self.batch_size-count)])
            out = self.model(xx.contiguous(), tt.contiguous())['log_density']
            outputs.append(out[:count].float().clone())
        return {'log_density': torch.cat(outputs)}


@contextmanager
def float32_network_execution():
    """Disable TF32 and cuDNN benchmarking inside the block, then restore them."""
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = saved


def _check_pair_data(pair_data_list: Sequence[np.ndarray]) -> None:
    for pair in pair_data_list:
        pair = np.asarray(pair)
        if pair.ndim != 2 or pair.shape[1] != 2 or len(pair) < 1:
            raise ValueError(f"Each pair must have shape (n, 2) with n >= 1, got {pair.shape}")
        if not np.isfinite(pair).all() or np.any((pair < 0) | (pair > 1)):
            raise ValueError("Pseudo-observations must be finite and lie in [0, 1]")


@torch.no_grad()
def estimate_linear_grid_densities(
    model: torch.nn.Module,
    pair_data_list: Sequence[np.ndarray],
    *,
    grid_size: int,
    device,
    spacing: str = "uniform",
    edge_batch_size: int = 64,
    projection_iters: int = 30,
    log_density_clamp: float = 15.,
) -> List[np.ndarray]:
    """Estimate projected float32 node-value grids for a list of ``(n, 2)`` pairs.

    ``model`` must accept ``(x, t)`` with ``x`` of shape ``(B, 4, m, m)`` and
    return a dict with ``log_density`` of shape ``(B, 1, m, m)``; wrap it in
    :class:`FixedBatchForward` to reproduce the evaluated execution shape.
    Points must lie in [0, 1]. Returns one ``(m, m)`` float32 array per pair;
    pass each to :class:`LinearGridCopula` for the continuous copula.
    """
    _check_pair_data(pair_data_list)
    m = int(grid_size)
    edges, nodes = grid_geometry(m, spacing)
    densities = []
    area = np.outer(np.diff(edges),np.diff(edges))
    z = ndtri(nodes)
    coord = torch.tensor(np.stack(np.meshgrid(z,z,indexing="ij")),dtype=torch.float32,device=device)
    w = torch.tensor(basis_weights(nodes),dtype=torch.float32,device=device)
    bound = float(log_density_clamp)
    with float32_network_execution():
        for start in range(0,len(pair_data_list),edge_batch_size):
            pairs = pair_data_list[start:start+edge_batch_size]
            hist = np.stack([np.histogram2d(p[:,0],p[:,1],bins=(edges,edges))[0]/(len(p)*area) for p in pairs])
            x = torch.tensor(hist,dtype=torch.float32,device=device)[:,None]
            ln = torch.tensor(np.log([len(p) for p in pairs]),dtype=torch.float32,device=device)[:,None,None,None].expand(-1,1,m,m)
            x = torch.cat([x,ln,coord[None].expand(len(pairs),-1,-1,-1)],1)
            out = model(x,torch.zeros(len(x),device=device))["log_density"][:,0].float().clamp(-bound,bound).exp()
            densities.extend(project_density_torch(out,w,projection_iters).cpu().numpy())
    return densities


@dataclass(frozen=True)
class LinearGridSettings:
    """Inference settings of a linear-grid model (the manifest ``inference`` block)."""

    grid_size: int
    grid_spacing: str = "uniform"
    projection_iters: int = 30
    log_density_clamp: float = 15.
    copula_projection_solver: str = "hybrid"
    conditional_clip_eps: float = 1e-12
    forward_batch_size: Optional[int] = 16
    edge_batch_size: int = 64

    def __post_init__(self):
        if self.grid_spacing not in ("uniform", "probit"):
            raise ValueError("grid_spacing must be 'uniform' or 'probit'")
        if self.copula_projection_solver not in ("ipfp", "hybrid"):
            raise ValueError("copula_projection_solver must be 'ipfp' or 'hybrid'")
        if int(self.grid_size) < 2 or int(self.projection_iters) < 0 or int(self.edge_batch_size) < 1:
            raise ValueError("Invalid grid size, projection count or edge batch size")
        if self.forward_batch_size is not None and int(self.forward_batch_size) < 1:
            raise ValueError("forward_batch_size must be positive or None")
        if not 0 < float(self.conditional_clip_eps) < .5:
            raise ValueError("conditional_clip_eps must lie in (0, .5)")

    @classmethod
    def from_manifest(cls, inference: Dict[str, Any]) -> "LinearGridSettings":
        """Build settings from a pretrained manifest's ``inference`` block."""
        if inference.get("pair_estimator") != LINEAR_GRID_ESTIMATOR:
            raise ValueError("Manifest does not describe a linear-grid estimator")
        names = cls.__dataclass_fields__
        return cls(**{key: inference[key] for key in names if key in inference})

    def to_dict(self) -> Dict[str, Any]:
        return {"pair_estimator": LINEAR_GRID_ESTIMATOR, **asdict(self)}

    def vine_options(self) -> Dict[str, Any]:
        """Keyword arguments for :class:`vdc.vine.api.VineCopulaModel`."""
        if self.grid_spacing != "uniform":
            raise ValueError("VineCopulaModel supports uniform linear grids only")
        return dict(m=int(self.grid_size), projection_iters=int(self.projection_iters),
                    hfunc_use_spline=False, batch_edges=True, edge_batch_size=int(self.edge_batch_size),
                    pair_estimator=LINEAR_GRID_ESTIMATOR,
                    copula_projection_solver=self.copula_projection_solver,
                    conditional_clip_eps=float(self.conditional_clip_eps),
                    forward_batch_size=self.forward_batch_size)


class LinearGridEstimator:
    """Bivariate copula estimator for a loaded linear-grid network.

    Args:
        model: Network returning ``{"log_density": (B, 1, m, m)}`` from
            ``(x, t)``; it is set to evaluation mode.
        settings: :class:`LinearGridSettings` (from the model manifest).
        device: Device holding the network.

    ``estimate_grids`` returns float32 node-value grids; ``copulas`` returns
    :class:`LinearGridCopula` objects with consistent PDF and conditional
    functions. A pair's result can depend on the other pairs in the same call
    at the level of float32 rounding (observed on GPUs for the 64-cell model).
    """

    def __init__(self, model: torch.nn.Module, settings: LinearGridSettings, device=None):
        self.settings = settings
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        model.eval()
        model.to(self.device)
        self.network = model
        self.model = (FixedBatchForward(model, int(settings.forward_batch_size))
                      if settings.forward_batch_size else model)
        _, self.nodes = grid_geometry(settings.grid_size, settings.grid_spacing)

    def estimate_grids(self, pair_data_list: Sequence[np.ndarray]) -> List[np.ndarray]:
        s = self.settings
        return estimate_linear_grid_densities(
            self.model, [np.asarray(p, dtype=np.float64) for p in pair_data_list],
            grid_size=s.grid_size, device=self.device, spacing=s.grid_spacing,
            edge_batch_size=s.edge_batch_size, projection_iters=s.projection_iters,
            log_density_clamp=s.log_density_clamp)

    def copula(self, grid: np.ndarray) -> LinearGridCopula:
        return LinearGridCopula(grid, self.nodes, solver=self.settings.copula_projection_solver)

    def copulas(self, pair_data_list: Sequence[np.ndarray]) -> List[LinearGridCopula]:
        return [self.copula(grid) for grid in self.estimate_grids(pair_data_list)]
