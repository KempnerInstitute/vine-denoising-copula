"""
High-level interface for using a trained diffusion copula model as a bivariate copula.

This wrapper is intended to be **importable as a library module** (e.g. from tests),
so it must not depend on ad-hoc script modules that may be missing in some checkouts.

It provides:
- `DiffusionCopulaModel.from_checkpoint`: load a `GridUNet` + `CopulaAwareDiffusion`
  from a training checkpoint produced by `scripts/train.py` / `scripts/train_unified.py`.
- `estimate_density_from_samples`: estimate a bivariate copula density grid from
  pseudo-observations in [0,1]^2 via the shared reverse diffusion routine
  `vdc.inference.density.sample_density_grid`.

Note: advanced CFG / custom binning utilities previously lived in a separate script;
this module focuses on a stable, self-contained API that matches the rest of the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from vdc.inference.density import sample_density_grid
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.unet_grid import GridUNet
from vdc.models.projection import copula_project
from vdc.utils.smoothing import smooth_density_gaussian


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
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if device is None:
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_t = torch.device(device)

        # We ignore config_path for now and rely on the config embedded in the checkpoint.
        checkpoint = torch.load(ckpt_path, map_location=device_t, weights_only=False)
        config = checkpoint.get("config", {})

        m = int(config.get("data", {}).get("m", 64))
        model_cfg = config.get("model", {})

        state = checkpoint.get("model_state_dict", checkpoint.get("model", None))
        if state is None:
            raise RuntimeError(f"Checkpoint at {ckpt_path} does not contain model weights")

        # Robustly infer in_channels from the checkpoint to support older/newer variants.
        if isinstance(state, dict) and "conv_in.weight" in state:
            in_channels = int(state["conv_in.weight"].shape[1])
        else:
            in_channels = int(model_cfg.get("in_channels", 1))

        model = GridUNet(
            m=m,
            in_channels=in_channels,
            base_channels=int(model_cfg.get("base_channels", 64)),
            channel_mults=tuple(model_cfg.get("channel_mults", (1, 2, 3, 4))),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
            attention_resolutions=tuple(model_cfg.get("attention_resolutions", (16, 8))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        ).to(device_t)

        # Allow missing keys for backward compatibility (e.g., older checkpoints without log_n embedding).
        model.load_state_dict(state, strict=False)
        if not any(str(k).startswith("logn_embed") for k in state.keys()):
            # Make log_n conditioning a no-op if the checkpoint did not include it.
            for p in model.logn_embed.parameters():
                torch.nn.init.zeros_(p)
        model.eval()

        diff_cfg = config.get("diffusion", {})
        diffusion = CopulaAwareDiffusion(
            timesteps=int(diff_cfg.get("timesteps", 1000)),
            beta_schedule=str(diff_cfg.get("noise_schedule", "cosine")),
        ).to(device_t)

        return cls(model=model, diffusion=diffusion, config=config, device=device_t)

    # ------------------------------------------------------------------
    # Core density estimation
    # ------------------------------------------------------------------
    def _grid_centers(self, m: int) -> Tuple[np.ndarray, np.ndarray]:
        u = np.linspace(0.5 / m, 1.0 - 0.5 / m, m)
        v = np.linspace(0.5 / m, 1.0 - 0.5 / m, m)
        return u, v

    # Public API --------------------------------------------------------
    def estimate_density_from_samples(
        self,
        u: np.ndarray,
        m: Optional[int] = None,
        projection_iters: int = 15,
        smooth_sigma: float = 0.0,
        num_diffusion_steps: int = 50,
        cfg_scale: float = 2.0,
        adaptive_cfg: bool = False,
        num_ensemble: int = 1,
        ensemble_mode: str = "geometric",
        return_std: bool = False,
        use_cfg: bool = True,
        # Legacy parameters (ignored when use_cfg=True)
        noise_step: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate a copula density from empirical pseudo-observations u in [0,1]^2.

        Parameters
        ----------
        u:
            Array of shape (n, 2) with values in [0, 1]^2 (after marginal CDFs).
        m:
            Grid resolution. Defaults to the training config's `data.m`.
        projection_iters:
            Number of copula projection iterations for final normalization.
        smooth_sigma:
            Gaussian smoothing sigma (in grid units) applied after generation.
            Set to 0 to disable smoothing. Default 0 (CFG produces smooth outputs).
        num_diffusion_steps:
            Number of reverse diffusion steps (more = better quality but slower).
            Default 50 is usually sufficient with CFG.
        cfg_scale:
            Classifier-Free Guidance scale. Higher values (>1) produce outputs
            more strongly conditioned on the input histogram. Default 2.0.
            Set to 1.0 for no guidance (purely conditional).
            Ignored if adaptive_cfg=True.
        adaptive_cfg:
            If True, automatically adjust CFG scale based on histogram properties:
            - Symmetric histograms (Gaussian, Frank) → CFG ~2.0
            - Asymmetric/peaked histograms (rotated Clayton) → CFG ~5.0
            This helps handle both smooth and peaked copulas optimally.
        num_ensemble:
            Number of independent inference runs to average. Higher values give
            smoother, more robust estimates at the cost of compute. Default 1.
            Recommended: 3-5 for production, 1 for fast iteration.
        ensemble_mode:
            How to aggregate ensemble predictions:
            - "geometric": Average in log-space (preserves relative magnitudes, recommended)
            - "arithmetic": Average in density space (smooths peaks more)
            - "median": Median in density space (robust to outliers)
        return_std:
            If True, also return the standard deviation across ensemble samples
            (useful for uncertainty quantification). Only valid if num_ensemble > 1.
        use_cfg:
            If True (default), use CFG sampling from pure noise. This is the
            recommended method for V2 models trained with CFG dropout.
            If False, use the legacy denoising method (for older models).

        Returns
        -------
        density_pred, row_coords, col_coords
        If return_std=True: density_pred, row_coords, col_coords, density_std
        """
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError(f"u must have shape (n, 2), got {u.shape}")

        m_cfg = int(self.config.get("data", {}).get("m", 64))
        m_eff = int(m) if m is not None else m_cfg

        # This library wrapper currently uses the shared `sample_density_grid()` path.
        # It does not require special histogram-conditioning channels.

        ensemble = []
        # Heuristic: if the UNet expects 2 channels, treat it as histogram-conditioned.
        use_histogram_conditioning = bool(getattr(self.model, "conv_in").in_channels > 1)
        for i in range(max(1, int(num_ensemble))):
            torch.manual_seed(42 + i * 1000)
            density_i = sample_density_grid(
                model=self.model,
                diffusion=self.diffusion,
                samples=u,
                m=m_eff,
                device=self.device,
                num_steps=num_diffusion_steps,
                cfg_scale=cfg_scale,
                use_histogram_conditioning=use_histogram_conditioning,
                projection_iters=projection_iters,
            )
            ensemble.append(density_i)

        ensemble_arr = np.stack(ensemble, axis=0)  # (E,m,m)
        if ensemble_arr.shape[0] == 1:
            density_pred = ensemble_arr[0]
            density_std = None
        else:
            if ensemble_mode == "geometric":
                density_pred = np.exp(np.mean(np.log(np.clip(ensemble_arr, 1e-12, None)), axis=0))
            elif ensemble_mode == "arithmetic":
                density_pred = np.mean(ensemble_arr, axis=0)
            elif ensemble_mode == "median":
                density_pred = np.median(ensemble_arr, axis=0)
            else:
                raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}")
            density_std = np.std(ensemble_arr, axis=0) if return_std else None

        if smooth_sigma > 0:
            t = torch.from_numpy(density_pred).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t = smooth_density_gaussian(t, sigma=float(smooth_sigma), preserve_mass=True)
            density_pred = t[0, 0].detach().cpu().numpy()

        # Final projection (idempotent if already projected in sample_density_grid)
        if projection_iters > 0:
            t = torch.from_numpy(density_pred).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t = copula_project(t.clamp_min(1e-12), iters=int(projection_iters))
            density_pred = t[0, 0].detach().cpu().numpy()

        row_coords, col_coords = self._grid_centers(m_eff)
        if return_std and density_std is not None:
            return density_pred, row_coords, col_coords, density_std
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


