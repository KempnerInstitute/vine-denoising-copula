#!/usr/bin/env python
"""Offline visualizations for diffusion UNet checkpoints.

Uses existing evaluation helpers under ``vdc.eval.visualize`` to render
predicted vs. target copula densities, error heatmaps, and marginal plots.

This script reconstructs copula densities using the full reverse diffusion
process (multi-step denoising from t=T-1 down to t=0), which produces smooth,
coherent density estimates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.generators import analytic_logpdf_grid, sample_bicop
from vdc.eval.visualize import plot_comparison, plot_marginals
from vdc.utils.metrics import aggregate_metrics


def build_binning(
    m: int,
    mode: str = "uniform",
    z_max: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct bin edges, centers, and widths on [0, 1].
    """
    if mode == "uniform":
        edges = np.linspace(0.0, 1.0, m + 1, dtype=np.float64)
    elif mode == "probit":
        z_edges = torch.linspace(-z_max, z_max, m + 1, dtype=torch.float64)
        edges = (0.5 * (1.0 + torch.erf(z_edges / np.sqrt(2.0)))).numpy()
        edges[0] = 0.0
        edges[-1] = 1.0
        np.clip(edges, 0.0, 1.0, out=edges)
        edges = np.maximum.accumulate(edges)
        edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown binning mode: {mode}")
    widths = np.diff(edges)
    centers = edges[:-1] + 0.5 * widths
    return edges, centers, widths


def to_area_tensor(
    du: np.ndarray,
    dv: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create (1,1,m,m) tensor of cell areas for broadcasting."""
    areas = np.outer(du, dv)
    return torch.from_numpy(areas).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize diffusion UNet checkpoints")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model_step_*.pt produced by scripts/train_unified.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write visualizations (defaults to <checkpoint_dir>/visualizations/eval)",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of diffusion sampling steps (DDIM-style subsampling). More steps = smoother results.",
    )
    parser.add_argument(
        "--projection-iters",
        type=int,
        default=10,
        help="Number of IPFP projection iterations applied to the reconstructed density",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="How many predefined copula cases to visualize",
    )
    parser.add_argument(
        "--binning",
        choices=["uniform", "probit"],
        default=None,
        help="Override grid binning (defaults to config value if omitted).",
    )
    parser.add_argument(
        "--color-scale-mode",
        choices=["independent", "shared", "both"],
        default="independent",
        help=(
            "Color scaling for predicted vs true panels. "
            "'shared' enforces identical colorbars, "
            "'independent' maximizes contrast per panel, "
            "and 'both' saves one figure of each."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for noise draws and sample generation",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit YAML config (used only if missing from checkpoint)",
    )
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: Path, device: torch.device, explicit_config: Optional[Path] = None
) -> Tuple[Dict, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        if explicit_config is None:
            raise ValueError(
                "Checkpoint did not store the training config. Provide it via --config."
            )
        with explicit_config.open("r") as handle:
            config = yaml.safe_load(handle)
    return checkpoint, config


def build_model(config: Dict, device: torch.device) -> GridUNet:
    model_cfg = config.get("model", {})
    if model_cfg.get("type") != "diffusion_unet":
        raise ValueError(
            "This helper only supports diffusion_unet checkpoints. "
            f"Got model type: {model_cfg.get('type')}"
        )
    m = config.get("data", {}).get("m", 64)
    # Use in_channels from config, defaulting to 1 for unconditional or 2 for conditional
    in_channels = model_cfg.get("in_channels", 1)
    model = GridUNet(
        m=m,
        in_channels=in_channels,
        base_channels=model_cfg.get("base_channels", 64),
        channel_mults=tuple(model_cfg.get("channel_mults", (1, 2, 3, 4))),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", (16, 8))),
        dropout=model_cfg.get("dropout", 0.0),
    )
    model.to(device)
    model.eval()
    return model


def build_diffusion(config: Dict, device: torch.device) -> CopulaAwareDiffusion:
    diff_cfg = config.get("diffusion", {})
    diffusion = CopulaAwareDiffusion(
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("noise_schedule", "linear"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
    )
    diffusion.to(device)
    diffusion.eval()
    return diffusion


@torch.no_grad()
def sample_diffusion_unconditional(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: int = 50,
    eta: float = 0.0,  # DDIM parameter (0 = deterministic)
) -> torch.Tensor:
    """
    Full reverse diffusion sampling (DDIM-style with optional subsampling).
    
    Starts from pure noise and iteratively denoises to produce a clean log-density.
    NOTE: This is UNCONDITIONAL sampling - the model generates a random copula
    from the learned distribution, not a specific target copula.
    
    Args:
        model: Trained denoising UNet
        diffusion: CopulaAwareDiffusion with noise schedule
        shape: Output shape (B, 1, m, m)
        device: torch device
        num_steps: Number of sampling steps (can be less than training timesteps)
        eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
    
    Returns:
        Denoised log-density tensor
    """
    T = diffusion.timesteps
    
    # Create timestep schedule (evenly spaced, reversed)
    if num_steps >= T:
        timesteps = list(range(T - 1, -1, -1))
    else:
        # DDIM-style subsampling
        step_size = T // num_steps
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)
    
    # Start from pure noise
    x_t = torch.randn(shape, device=device)
    
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        t_normalized = t_tensor.float() / T
        
        # Predict noise
        pred_noise = model(x_t, t_normalized)
        
        # Get alpha values
        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Predict x_0
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        pred_x0 = pred_x0.clamp(-20, 20)  # Prevent extreme values
        
        if t == 0:
            x_t = pred_x0
        else:
            # Get next timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0
            
            alpha_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            
            # DDIM update
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_noise
            
            # Compute x_{t-1}
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            
            if eta > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + sigma_t * noise
    
    return x_t


@torch.no_grad()
def denoise_from_histogram(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    histogram: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
    start_t: int = 500,  # Start from intermediate noise level, not full noise
    conditional: bool = None,  # Auto-detect if None
) -> torch.Tensor:
    """
    Denoise a histogram into a clean copula density using reverse diffusion.
    
    The key insight: treat the histogram as a noisy version of the true density
    at some intermediate timestep, then run the reverse diffusion from there.
    
    This is similar to SDEdit / image-to-image diffusion where we:
    1. Add noise to the input to reach timestep start_t
    2. Denoise from start_t back to t=0
    
    Args:
        model: Trained denoising UNet
        diffusion: CopulaAwareDiffusion with noise schedule
        histogram: Input histogram (B, 1, m, m), normalized
        device: torch device
        num_steps: Number of denoising steps
        start_t: Starting timestep (higher = more denoising, lower = preserve input)
        conditional: If True, model expects 2-channel input [noisy, histogram]. Auto-detect if None.
    
    Returns:
        Denoised log-density tensor
    """
    T = diffusion.timesteps
    start_t = min(start_t, T - 1)
    
    # Auto-detect if model is conditional by checking input channels
    if conditional is None:
        # Try to infer from model architecture
        try:
            in_channels = model.conv_in.weight.shape[1]
            conditional = (in_channels == 2)
        except:
            conditional = False
    
    # Normalize histogram
    m = histogram.shape[-1]
    B = histogram.shape[0]
    du = dv = 1.0 / m
    hist_sum = (histogram * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    histogram = histogram / hist_sum
    
    # Convert to log-density
    log_hist = torch.log(histogram.clamp(min=1e-12))
    
    # Add noise to reach start_t (forward diffusion)
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    alpha_start = alphas_cumprod[start_t]
    noise = torch.randn_like(log_hist)
    x_t = torch.sqrt(alpha_start) * log_hist + torch.sqrt(1 - alpha_start) * noise
    
    # Create timestep schedule from start_t down to 0
    if num_steps >= start_t:
        timesteps = list(range(start_t, -1, -1))
    else:
        step_size = max(1, start_t // num_steps)
        timesteps = list(range(start_t, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)
    
    # Reverse diffusion
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        t_normalized = t_tensor.float() / T
        
        # Prepare model input
        if conditional:
            # Conditional model: concatenate noisy log-density with conditioning histogram
            model_input = torch.cat([x_t, log_hist], dim=1)
        else:
            # Unconditional model: just the noisy log-density
            model_input = x_t
        
        # Predict noise
        pred_noise = model(model_input, t_normalized)
        
        # Get alpha values
        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Predict x_0
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        pred_x0 = pred_x0.clamp(-20, 20)
        
        if t == 0:
            x_t = pred_x0
        else:
            # Get next timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0
            
            alpha_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            
            # DDIM update (deterministic)
            dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
    
    return x_t


@torch.no_grad()
def predict_density_direct(
    model: torch.nn.Module,
    histogram: torch.Tensor,
    device: torch.device,
    t_value: float = 0.25,
) -> torch.Tensor:
    """
    Direct density prediction using the model in 'legacy' mode.
    
    The legacy training approach had the model directly predict clean log-density
    from noisy density input. This function emulates that inference pattern.
    
    Args:
        model: Trained UNet (predicts log-density or noise depending on training)
        histogram: Input histogram (B, 1, m, m), normalized
        device: torch device
        t_value: Normalized timestep (0=clean, 1=full noise)
    
    Returns:
        Predicted log-density tensor
    """
    # Normalize histogram
    m = histogram.shape[-1]
    du = dv = 1.0 / m
    hist_sum = (histogram * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    histogram = histogram / hist_sum
    
    # The model takes (input, t) and outputs something
    # For noise-prediction models, we interpret output as noise and reconstruct
    # For direct models, we interpret output as log-density
    t_tensor = torch.full((histogram.shape[0],), t_value, device=device)
    
    # Convert histogram to log-density for input
    log_hist = torch.log(histogram.clamp(min=1e-12))
    
    # Get model output
    output = model(log_hist, t_tensor)
    
    # The model was trained to predict noise, so output is noise estimate
    # At low t, the "clean" signal dominates, so subtracting small noise gives clean estimate
    # At high t, the noise dominates, so we're essentially generating from noise
    
    # For t_value=0, the model should output ~0 noise, so pred = input
    # For t_value>0, we subtract scaled noise estimate
    pred_log = log_hist - output * (t_value ** 0.5)  # Scale by sqrt(t) for reasonable behavior
    
    return pred_log.clamp(-20, 20)


def prepare_density(density: np.ndarray) -> np.ndarray:
    """Clamp density to a safe numeric range without renormalizing."""
    return np.clip(density, 1e-12, 1e6)


def get_test_suite(num_cases: int) -> List[Tuple[str, Dict[str, float], str]]:
    """Return a deterministic suite of copula families for evaluation."""
    catalog: List[Tuple[str, Dict[str, float], str]] = [
        ("gaussian", {"rho": 0.75}, "gaussian_rho0.75"),
        ("gaussian", {"rho": -0.65}, "gaussian_rho-0.65"),
        ("clayton", {"theta": 2.5}, "clayton_theta2.5"),
        ("gumbel", {"theta": 3.0}, "gumbel_theta3.0"),
        ("frank", {"theta": 5.0}, "frank_theta5.0"),
        ("joe", {"theta": 2.0}, "joe_theta2.0"),
        ("student", {"rho": 0.7, "nu": 6.0}, "student_rho0.7_nu6"),
        ("student", {"rho": -0.5, "nu": 8.0}, "student_rho-0.5_nu8"),
    ]
    return catalog[:max(1, num_cases)]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, config = load_checkpoint(args.checkpoint, device, args.config)

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    diffusion = build_diffusion(config, device)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.checkpoint.parent / "visualizations" / "eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    m = config.get("data", {}).get("m", 64)
    binning_mode = args.binning or config.get("data", {}).get("binning", "uniform")
    _, u_centers, du_vec = build_binning(m, mode=binning_mode)
    _, v_centers, dv_vec = build_binning(m, mode=binning_mode)
    cell_areas_np = np.outer(du_vec, dv_vec)
    du_t = torch.from_numpy(du_vec).float().to(device)
    dv_t = torch.from_numpy(dv_vec).float().to(device)
    area_tensor = to_area_tensor(du_vec, dv_vec, device, torch.float32)

    summary_rows: List[Tuple[str, Dict[str, float]]] = []

    print(f"Loaded diffusion UNet checkpoint from step {checkpoint.get('step', 'unknown')}")
    print(f"Evaluating {len(get_test_suite(args.num_samples))} copula cases")
    print(f"Using {args.sampling_steps} diffusion sampling steps\n")

    for family, params, name in get_test_suite(args.num_samples):
        print(f"Case: {name}")
        
        # Generate ground truth density
        log_density = analytic_logpdf_grid(
            family,
            params,
            m,
            rotation=0,
        )
        density_true = prepare_density(np.exp(np.clip(log_density, -20, 20)))
        density_true /= np.maximum(1e-20, (density_true * cell_areas_np).sum())

        # Generate samples from the copula and create histogram
        samples = sample_bicop(family, params, n=5000, rotation=0)
        
        # Create histogram from samples (this is what the model sees during training)
        from vdc.data.hist import scatter_to_hist
        histogram = scatter_to_hist(samples, m, reflect=True)
        histogram = histogram / (histogram.sum() * (1.0/m) * (1.0/m))  # Normalize
        hist_tensor = torch.from_numpy(histogram).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Use model to refine histogram into density prediction via reverse diffusion
        # SDEdit-style: add noise to histogram, then denoise
        start_t = 200  # Start from moderate noise level (out of 1000)
        print(f"  Denoising histogram (start_t={start_t}, {args.sampling_steps} steps)...")
        sampled_log = denoise_from_histogram(
            model,
            diffusion,
            hist_tensor,
            device=device,
            num_steps=args.sampling_steps,
            start_t=start_t,
        )
        
        # Convert to density
        recon_density = torch.exp(sampled_log).clamp(1e-12, 1e6)
        recon_density = recon_density / (
            (recon_density * area_tensor)
            .sum(dim=(2, 3), keepdim=True)
            .clamp_min(1e-12)
        )
        
        # Apply copula projection
        if args.projection_iters > 0:
            recon_density = copula_project(
                recon_density,
                iters=args.projection_iters,
                row_target=du_t,
                col_target=dv_t,
            )
            recon_density = recon_density.clamp(1e-12, 1e6)
            recon_density = recon_density / (
                (recon_density * area_tensor)
                .sum(dim=(2, 3), keepdim=True)
                .clamp_min(1e-12)
            )

        density_pred = recon_density[0, 0].cpu().numpy()

        # samples already generated above for histogram

        metrics = aggregate_metrics(
            torch.from_numpy(density_pred).float().view(1, 1, m, m),
            torch.from_numpy(density_true).float().view(1, 1, m, m),
            samples=torch.from_numpy(samples).float(),
        )
        metrics.update(
            {
                "density_mse": float(np.mean((density_pred - density_true) ** 2)),
                "density_mae": float(np.mean(np.abs(density_pred - density_true))),
            }
        )

        def _copula_stats(
            arr: np.ndarray,
            label: str,
            row_widths: np.ndarray,
            col_widths: np.ndarray,
            cell_areas: np.ndarray,
        ) -> Dict[str, float]:
            stats = {
                f"{label}_mass": float(np.sum(arr * cell_areas)),
                f"{label}_max": float(arr.max()),
                f"{label}_min": float(arr.min()),
            }
            row_marg = (arr * col_widths[None, :]).sum(axis=1)
            col_marg = (arr * row_widths[:, None]).sum(axis=0)
            stats[f"{label}_row_max_err"] = float(np.max(np.abs(row_marg - 1.0)))
            stats[f"{label}_col_max_err"] = float(np.max(np.abs(col_marg - 1.0)))
            return stats

        metrics.update(_copula_stats(density_true, "true", du_vec, dv_vec, cell_areas_np))
        metrics.update(_copula_stats(density_pred, "pred", du_vec, dv_vec, cell_areas_np))
        summary_rows.append((name, metrics))

        scale_modes = [args.color_scale_mode]
        if args.color_scale_mode == "both":
            scale_modes = ["independent", "shared"]

        def _comparison_path(mode: str) -> Path:
            if len(scale_modes) == 1 and mode == "independent":
                return output_dir / f"{name}_comparison.png"
            suffix = mode if mode != "independent" else "independent"
            return output_dir / f"{name}_comparison_{suffix}.png"

        marginals_path = output_dir / f"{name}_marginals.png"

        for mode in scale_modes:
            out_path = _comparison_path(mode)
            plot_comparison(
                density_pred,
                density_true,
                title=f"{name}",
                save_path=str(out_path),
                scale_mode=mode,
            )
            print(f"  Saved: {out_path}")

        plot_marginals(
            density_pred,
            title=f"{name} marginals (predicted)",
            save_path=str(marginals_path),
        )
        print(f"  Saved: {marginals_path}")

        print(f"  Saved:")
        for mode in scale_modes:
            print(f"    - {_comparison_path(mode)}")
        print(f"    - {marginals_path}")
        print(f"  Key metrics:")
        for k, v in sorted(metrics.items()):
            print(f"        {k:>16}: {v:.6e}")
        print()

    # Summary statistics
    if summary_rows:
        print("Summary (means across cases):")
        agg: Dict[str, List[float]] = {}
        for _, m_dict in summary_rows:
            for k, v in m_dict.items():
                agg.setdefault(k, []).append(v)
        for k, vals in sorted(agg.items()):
            print(f"  {k:>16}: {np.mean(vals):.6e}")

    print(f"\nVisualizations available under: {output_dir}")


if __name__ == "__main__":
    main()
