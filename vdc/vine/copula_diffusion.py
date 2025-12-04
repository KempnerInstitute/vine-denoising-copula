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
from vdc.utils.smoothing import smooth_density_gaussian


def compute_corner_concentration(hist: np.ndarray, corner_size: int = 8) -> Tuple[float, int]:
    """
    Compute how much mass is concentrated in each corner.
    
    Returns:
        (concentration_score, dominant_corner)
        
    concentration_score: 0-1, how much mass is in the most concentrated corner
    dominant_corner: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    """
    m = hist.shape[0]
    hist_norm = hist / (hist.sum() + 1e-12)
    
    # Define corner regions
    corners = {
        0: hist_norm[:corner_size, :corner_size],       # Top-left (0,0)
        1: hist_norm[:corner_size, -corner_size:],      # Top-right (0, m-1)
        2: hist_norm[-corner_size:, :corner_size],      # Bottom-left (m-1, 0)
        3: hist_norm[-corner_size:, -corner_size:],     # Bottom-right (m-1, m-1)
    }
    
    corner_masses = {k: v.sum() for k, v in corners.items()}
    dominant = max(corner_masses, key=corner_masses.get)
    max_mass = corner_masses[dominant]
    
    # Expected mass in corner if uniform
    expected_mass = (corner_size / m) ** 2
    
    # Concentration score: how much more than expected
    concentration = min(1.0, max_mass / expected_mass / 10)  # Scale so ~10x expected = 1.0
    
    return concentration, dominant


def adaptive_cfg_scale(hist: np.ndarray, base_cfg: float = 2.0, max_cfg: float = 5.0) -> float:
    """
    Compute adaptive CFG scale based on histogram properties.
    
    The key insight:
    - Peaked copulas (Clayton, Gumbel, Joe) have mass concentrated in corners
    - Smooth copulas (Gaussian, Frank) have more diffuse mass
    
    Higher CFG is needed for peaked copulas to capture the sharp structure.
    Lower CFG works better for smooth copulas to avoid over-sharpening.
    
    Args:
        hist: 2D histogram array
        base_cfg: Minimum CFG scale for smooth copulas (default 2.0)
        max_cfg: Maximum CFG scale for peaked copulas (default 5.0)
        
    Returns:
        Adaptive CFG scale in [base_cfg, max_cfg]
    """
    hist_norm = hist / (hist.sum() + 1e-12)
    
    # 1. Corner concentration score
    corner_conc, _ = compute_corner_concentration(hist)
    
    # 2. Peakedness: max / median ratio
    # Peaked copulas have high max relative to median
    sorted_vals = np.sort(hist_norm.flatten())
    median_val = sorted_vals[len(sorted_vals) // 2]
    max_val = hist_norm.max()
    
    if median_val > 1e-12:
        peak_ratio = max_val / median_val
        # Log scale: ratio of 10 → 0.5, ratio of 100 → 1.0
        peakedness = min(1.0, np.log10(max(1, peak_ratio)) / 2)
    else:
        peakedness = 1.0  # Very sparse histogram
    
    # 3. Combine scores: use weighted average
    # Corner concentration is more reliable than peakedness
    combined = 0.6 * corner_conc + 0.4 * peakedness
    
    # 4. Interpolate CFG scale
    cfg = base_cfg + (max_cfg - base_cfg) * combined
    
    return cfg


def sample_with_cfg(
    model: torch.nn.Module,
    diffusion,
    histogram: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
    cfg_scale: float = 2.0,
) -> torch.Tensor:
    """
    Sample from the conditional diffusion model with Classifier-Free Guidance.
    
    This starts from pure noise and generates a clean density conditioned on the
    input histogram.
    
    Args:
        model: The diffusion UNet model
        diffusion: CopulaAwareDiffusion instance
        histogram: (B, 1, m, m) normalized density histogram as conditioning
        device: torch device
        num_steps: Number of reverse diffusion steps (more = better quality)
        cfg_scale: Guidance scale (>1 emphasizes conditioning, 1 = no guidance)
        
    Returns:
        (B, 1, m, m) generated density
    """
    B = histogram.shape[0]
    m = histogram.shape[-1]
    T = diffusion.timesteps
    
    # Normalize histogram
    du = dv = 1.0 / m
    mass = (histogram * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    histogram = histogram / mass
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    # Start from pure noise
    x_t = torch.randn(B, 1, m, m, device=device)
    
    # Timestep schedule
    if num_steps >= T:
        timesteps = list(range(T - 1, -1, -1))
    else:
        step_size = T // num_steps
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)
    
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_normalized = t_tensor.float() / T
            
            # Conditional prediction (with histogram)
            model_input_cond = torch.cat([x_t, log_histogram], dim=1)
            pred_noise_cond = model(model_input_cond, t_normalized)
            
            # Unconditional prediction (without histogram) for CFG
            log_histogram_uncond = torch.zeros_like(log_histogram)
            model_input_uncond = torch.cat([x_t, log_histogram_uncond], dim=1)
            pred_noise_uncond = model(model_input_uncond, t_normalized)
            
            # CFG: pred = uncond + scale * (cond - uncond)
            pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
            
            # DDIM update
            alpha_t = alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
            pred_x0 = pred_x0.clamp(-20, 20)
            
            if t == 0:
                x_t = pred_x0
            else:
                if i + 1 < len(timesteps):
                    t_prev = timesteps[i + 1]
                else:
                    t_prev = 0
                
                alpha_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
                dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
                x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
    
    # Convert to density
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return density / mass


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
        num_diffusion_steps: int = 100,
    ) -> np.ndarray:
        """
        Run reverse diffusion denoising starting from a given density.
        
        Uses the denoise_from_histogram approach which treats the input as a
        noisy observation and runs reverse diffusion from an intermediate timestep.
        
        Args:
            log_density_grid: (m, m) log-density array
            noise_step: Starting timestep for denoising (lower = preserve more input)
            projection_iters: IPFP iterations for copula constraints
            num_diffusion_steps: Number of reverse diffusion steps (more = smoother)
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

        # Determine starting timestep - default to moderate noise level
        timesteps = int(self.diffusion.timesteps)
        start_t = noise_step if noise_step is not None else min(300, timesteps - 1)

        # Use denoise_from_histogram to run reverse diffusion
        recon_log = denoise_from_histogram(
            self.model, 
            self.diffusion, 
            density_tensor,  # Input is normalized density
            self.device,
            num_steps=num_diffusion_steps,
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

        # Histogram on [0,1]^2; we use a uniform grid here on purpose.
        hist = points_to_histogram(u, m=m_eff)
        hist = np.clip(hist, 1e-12, 1e6)
        du = dv = 1.0 / m_eff
        hist = hist / (hist.sum() * du * dv)  # Normalize to proper density (integral = 1)

        # Determine CFG scale (adaptive or fixed)
        if adaptive_cfg:
            effective_cfg = adaptive_cfg_scale(hist, base_cfg=2.0, max_cfg=5.0)
        else:
            effective_cfg = cfg_scale

        # Prepare histogram tensor for conditioning
        hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Ensemble inference: run multiple times with different random seeds
        ensemble_densities = []
        for i in range(num_ensemble):
            # Set different random seed for each ensemble member
            torch.manual_seed(42 + i * 1000)
            
            if use_cfg:
                # Use CFG sampling from pure noise (recommended for V2 models)
                density_tensor = sample_with_cfg(
                    self.model,
                    self.diffusion,
                    hist_tensor,
                    self.device,
                    num_steps=num_diffusion_steps,
                    cfg_scale=effective_cfg,
                )
                density_i = density_tensor[0, 0].cpu().numpy()
            else:
                # Legacy method: denoise from histogram
                log_density_grid = np.log(hist)
                step = noise_step if noise_step is not None else 300
                density_i = self._denoise_from_log_density(
                    log_density_grid, 
                    noise_step=step, 
                    projection_iters=0,
                    num_diffusion_steps=num_diffusion_steps,
                )
            
            ensemble_densities.append(density_i)
        
        # Aggregate ensemble predictions
        ensemble_densities = np.stack(ensemble_densities, axis=0)  # (num_ensemble, m, m)
        
        if num_ensemble > 1:
            if ensemble_mode == "geometric":
                # Geometric mean (average in log-space) - preserves relative magnitudes
                log_ensemble = np.log(np.clip(ensemble_densities, 1e-12, None))
                density_pred = np.exp(np.mean(log_ensemble, axis=0))
            elif ensemble_mode == "arithmetic":
                # Arithmetic mean - smooths more aggressively
                density_pred = np.mean(ensemble_densities, axis=0)
            elif ensemble_mode == "median":
                # Median - robust to outliers
                density_pred = np.median(ensemble_densities, axis=0)
            else:
                raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}")
            
            # Compute std for uncertainty quantification
            if return_std:
                density_std = np.std(ensemble_densities, axis=0)
        else:
            density_pred = ensemble_densities[0]
            density_std = None

        # Apply Gaussian smoothing before projection
        if smooth_sigma > 0:
            density_t = torch.from_numpy(density_pred).float().unsqueeze(0).unsqueeze(0).to(self.device)
            smoothed_t = smooth_density_gaussian(density_t, sigma=smooth_sigma, preserve_mass=True)
            density_pred = smoothed_t[0, 0].cpu().numpy()
        
        # Now apply copula projection
        if projection_iters > 0:
            density_t = torch.from_numpy(density_pred).float().unsqueeze(0).unsqueeze(0).to(self.device)
            density_t = density_t.clamp(min=1e-12)
            projected_t = copula_project(density_t, iters=projection_iters)
            density_pred = projected_t[0, 0].cpu().numpy()

        # Coordinates under the evaluation geometry.
        row_coords, col_coords, _, _ = self._grid_geometry(m=m_eff)
        
        if return_std and num_ensemble > 1:
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


