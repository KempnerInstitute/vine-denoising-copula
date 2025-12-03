"""
High-level interface for using a trained diffusion copula model as a bivariate copula.

This module is intentionally lightweight and depends only on existing training/eval
code paths. It provides:

    - DiffusionCopulaModel: load from checkpoint and estimate a copula density
      on a grid from either
        * an analytic truth (for benchmarks), or
        * empirical bivariate samples in [0, 1]^2.
    - Utilities to approximate h-functions and to sample from the estimated copula.

The goal is to give a clean "vine-ready" API without re-implementing training.

NOTE: This wrapper currently targets diffusion UNet checkpoints trained with
      scripts/train_unified.py and evaluated via scripts/visualize_diffusion_offline.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# We rely on the existing offline visualization utilities to avoid code duplication.
from scripts.visualize_diffusion_offline import (  # type: ignore
    build_binning,
    build_diffusion,
    build_model,
    denoise_from_histogram,
    load_checkpoint,
    to_area_tensor,
)
from vdc.data.hist import points_to_histogram
from vdc.models.projection import copula_project


@dataclass
class DiffusionCopulaModel:
    """
    Wrapper around a trained diffusion UNet checkpoint for copula use.

    Typical usage
    -------------
    >>> model = DiffusionCopulaModel.from_checkpoint("checkpoints/diffusion_uniform_m128/model_step_20000.pt")
    >>> u = np.random.rand(10_000, 2)   # pseudo-observations in [0,1]^2
    >>> density_grid, row_coords, col_coords = model.estimate_density_from_samples(u)
    >>> samples = model.sample_from_density(density_grid, n_samples=5000)
    """

    model: torch.nn.Module
    diffusion: object
    config: dict
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        config_path: Optional[str | Path] = None,
    ) -> "DiffusionCopulaModel":
        """
        Load a diffusion UNet copula model from a training checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to `model_step_*.pt` produced by scripts/train_unified.py.
        device:
            'cuda' or 'cpu'. If None, chooses CUDA when available.
        config_path:
            Optional path to a YAML config; if omitted, the config embedded
            in the checkpoint is used (recommended).
        """
        ckpt_path = Path(checkpoint_path)
        if device is None:
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_t = torch.device(device)

        checkpoint, config = load_checkpoint(ckpt_path, device_t, config_path)
        model = build_model(config, device_t)
        state = checkpoint.get("model_state_dict", checkpoint.get("model", None))
        if state is None:
            raise RuntimeError(f"Checkpoint at {ckpt_path} does not contain model weights")
        model.load_state_dict(state, strict=True)
        model.eval()

        diffusion = build_diffusion(config, device_t)

        return cls(model=model, diffusion=diffusion, config=config, device=device_t)

    # ------------------------------------------------------------------
    # Core density estimation
    # ------------------------------------------------------------------
    def _grid_geometry(
        self,
        m: Optional[int] = None,
        binning: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """
        Build 1D binning and 2D cell-area tensor for a given grid resolution.

        Returns
        -------
        u_centers, v_centers, cell_areas_np, area_tensor
        """
        m_cfg = int(self.config.get("data", {}).get("m", 64))
        m_eff = int(m) if m is not None else m_cfg

        mode = binning or self.config.get("data", {}).get("binning", "uniform")
        _, u_centers, du = build_binning(m_eff, mode=mode)
        _, v_centers, dv = build_binning(m_eff, mode=mode)
        cell_areas_np = np.outer(du, dv)
        area_tensor = to_area_tensor(du, dv, self.device, torch.float32)
        return u_centers, v_centers, cell_areas_np, area_tensor

    def _denoise_from_log_density(
        self,
        log_density_grid: np.ndarray,
        noise_step: Optional[int] = None,
        projection_iters: int = 15,
    ) -> np.ndarray:
        """
        Run reverse diffusion denoising starting from a given density.
        
        Uses the denoise_from_histogram approach which treats the input as a
        noisy observation and runs reverse diffusion from an intermediate timestep.
        """
        m = log_density_grid.shape[0]
        assert log_density_grid.shape == (m, m)

        # Normalize to a proper copula density under the chosen geometry.
        u_centers, v_centers, cell_areas_np, area_tensor = self._grid_geometry(m=m)
        density_true = np.exp(np.clip(log_density_grid, -20.0, 20.0))
        density_true = np.clip(density_true, 1e-12, 1e6)
        density_true /= max(1e-20, float((density_true * cell_areas_np).sum()))

        density_tensor = (
            torch.from_numpy(density_true).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Determine starting timestep
        timesteps = int(self.diffusion.timesteps)
        start_t = noise_step if noise_step is not None else min(500, timesteps - 1)

        # Use denoise_from_histogram to run reverse diffusion
        recon_log = denoise_from_histogram(
            self.model, 
            self.diffusion, 
            density_tensor,  # Input is normalized density
            self.device,
            num_steps=50,
            start_t=start_t
        )
        
        recon_density = torch.exp(recon_log).clamp(1e-12, 1e6)
        recon_density = recon_density / (
            (recon_density * area_tensor).sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        )

        # Marginal projection for a valid copula (row/col sums ~1).
        du_vec = torch.from_numpy(cell_areas_np.sum(axis=1)).float().to(self.device)
        dv_vec = torch.from_numpy(cell_areas_np.sum(axis=0)).float().to(self.device)
        if projection_iters > 0:
            recon_density = copula_project(
                recon_density,
                iters=projection_iters,
                row_target=du_vec,
                col_target=dv_vec,
            )
            recon_density = recon_density.clamp(1e-12, 1e6)
            recon_density = recon_density / (
                (recon_density * area_tensor).sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
            )

        return recon_density[0, 0].detach().cpu().numpy()

    # Public API --------------------------------------------------------
    def estimate_density_from_samples(
        self,
        u: np.ndarray,
        m: Optional[int] = None,
        noise_step: Optional[int] = None,
        projection_iters: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate a copula density from empirical pseudo-observations u in [0,1]^2.

        Parameters
        ----------
        u:
            Array of shape (n, 2) with values in [0, 1]^2 (after marginal CDFs).
        m:
            Grid resolution. Defaults to the training config's `data.m`.
        noise_step:
            Diffusion timestep to denoise from. Defaults to T-1.
        projection_iters:
            Number of copula projection iterations for final normalization.

        Returns
        -------
        density_pred, row_coords, col_coords
        """
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError(f"u must have shape (n, 2), got {u.shape}")

        m_cfg = int(self.config.get("data", {}).get("m", 64))
        m_eff = int(m) if m is not None else m_cfg

        # Histogram on [0,1]^2; we use a uniform grid here on purpose.
        hist = points_to_histogram(u, m=m_eff)
        hist = np.clip(hist, 1e-12, 1e6)
        hist /= max(1e-20, hist.sum())  # integral ≈ 1 on uniform grid

        log_density_grid = np.log(hist)
        density_pred = self._denoise_from_log_density(
            log_density_grid, noise_step=noise_step, projection_iters=projection_iters
        )

        # Coordinates under the evaluation geometry.
        row_coords, col_coords, _, _ = self._grid_geometry(m=m_eff)
        return density_pred, row_coords, col_coords

    # ------------------------------------------------------------------
    # h-functions and sampling on an estimated grid
    # ------------------------------------------------------------------
    @staticmethod
    def h_functions_from_grid(
        density: np.ndarray,
        row_widths: Optional[np.ndarray] = None,
        col_widths: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate conditional CDFs (h-functions) from a copula density grid.

        h1(u|v) ≈ ∫_0^u c(t, v) dt   (integrate over rows)
        h2(v|u) ≈ ∫_0^v c(u, s) ds   (integrate over columns)

        Returns
        -------
        h1, h2 : arrays of shape (m, m)
        """
        m = density.shape[0]
        if density.shape != (m, m):
            raise ValueError(f"density must be (m, m), got {density.shape}")

        if row_widths is None:
            row_widths = np.full(m, 1.0 / m)
        if col_widths is None:
            col_widths = np.full(m, 1.0 / m)

        # h1(u_i | v_j) = Σ_{k <= i} c(u_k, v_j) Δu_k
        h1 = np.cumsum(density * row_widths[:, None], axis=0)
        # h2(v_j | u_i) = Σ_{k <= j} c(u_i, v_k) Δv_k
        h2 = np.cumsum(density * col_widths[None, :], axis=1)
        # Clamp numerically
        h1 = np.clip(h1, 0.0, 1.0)
        h2 = np.clip(h2, 0.0, 1.0)
        return h1, h2

    @staticmethod
    def sample_from_density(
        density: np.ndarray,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample (u, v) pairs from a discretized copula density on [0,1]^2.

        We treat the grid as a categorical distribution over cells and then
        jitter uniformly inside each selected cell.
        """
        m = density.shape[0]
        if density.shape != (m, m):
            raise ValueError(f"density must be (m, m), got {density.shape}")

        if rng is None:
            rng = np.random.default_rng()

        flat = density.reshape(-1).astype(np.float64)
        flat = np.clip(flat, 0.0, None)
        s = flat.sum()
        if s <= 0:
            raise ValueError("Density grid has non-positive total mass; cannot sample.")
        flat /= s

        # Draw cell indices
        idx = rng.choice(flat.size, size=n_samples, p=flat)
        i = idx // m
        j = idx % m

        # Uniform jitter inside each cell on a uniform grid.
        du = 1.0 / m
        dv = 1.0 / m
        u = (i + rng.random(size=n_samples)) * du
        v = (j + rng.random(size=n_samples)) * dv

        return np.stack([u, v], axis=1)


