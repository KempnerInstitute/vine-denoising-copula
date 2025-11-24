#!/usr/bin/env python
"""Offline visualizations for diffusion UNet checkpoints.

Uses existing evaluation helpers under ``vdc.eval.visualize`` to render
predicted vs. target copula densities, error heatmaps, and marginal plots.

The script reconstructs copula densities by denoising noisy log-density
realizations at a chosen diffusion timestep. This mirrors the supervision
signal used during training and provides a direct look at how well the model
recovers clean densities from high-noise inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.generators import analytic_logpdf_grid, sample_bicop
from vdc.eval.visualize import plot_comparison, plot_marginals
from src.utils.metrics import aggregate_metrics


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
        "--noise-step",
        type=int,
        default=None,
        help="Diffusion timestep to denoise from (0 <= t < T). Defaults to the noisiest step (T-1).",
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    model = GridUNet(
        m=m,
        in_channels=1,
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


def denoise_log_density(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    target_log: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Denoise a noisy log-density realization drawn at a specific timestep."""
    step = int(np.clip(step, 0, diffusion.timesteps - 1))
    t = torch.full((target_log.shape[0],), step, device=target_log.device, dtype=torch.long)
    noise = torch.randn_like(target_log)
    noisy = diffusion.q_sample(target_log, t, noise)

    with torch.no_grad():
        t_normalized = t.float() / diffusion.timesteps
        pred_noise = model(noisy, t_normalized)

    alpha_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    recon_log = (noisy - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
    return recon_log


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
    noise_step = args.noise_step or (diffusion.timesteps - 1)

    summary_rows: List[Tuple[str, Dict[str, float]]] = []

    print(f"Loaded diffusion UNet checkpoint from step {checkpoint.get('step', 'unknown')}")
    print(f"Evaluating {len(get_test_suite(args.num_samples))} copula cases at timestep t={noise_step}\n")

    for family, params, name in get_test_suite(args.num_samples):
        print(f"Case: {name}")
        # NOTE: analytic_logpdf_grid expects the same family strings used in training
        # (e.g., 'gaussian', 'student', 'clayton', ...), so we pass `family` directly.
        log_density = analytic_logpdf_grid(
            family,
            params,
            m,
            rotation=0,
        )
        density_true = prepare_density(np.exp(np.clip(log_density, -20, 20)))
        density_true /= np.maximum(1e-20, (density_true * cell_areas_np).sum())

        density_tensor = torch.from_numpy(density_true).float().unsqueeze(0).unsqueeze(0).to(device)
        target_log = torch.log(density_tensor.clamp(min=1e-12))

        recon_log = denoise_log_density(model, diffusion, target_log, noise_step)
        recon_density = torch.exp(recon_log).clamp(1e-12, 1e6)
        recon_density = recon_density / (
            (recon_density * area_tensor)
            .sum(dim=(2, 3), keepdim=True)
            .clamp_min(1e-12)
        )
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

        samples = sample_bicop(family, params, n=2000, rotation=0)

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
            comparison_path = _comparison_path(mode)
            plot_comparison(
                density_pred,
                density_true,
                title=f"{name} | t={noise_step} | scale={mode}",
                points=samples,
                save_path=comparison_path,
                metrics=metrics,
                scale_mode=mode,
            )
        plot_marginals(
            density_pred,
            title=f"Marginals | {name}",
            save_path=marginals_path,
            row_coords=u_centers,
            col_coords=v_centers,
            row_widths=du_vec,
            col_widths=dv_vec,
        )

        print("  Saved:")
        for mode in scale_modes:
            print(f"    - {_comparison_path(mode)}")
        print(f"    - {marginals_path}")
        print("  Key metrics:")
        for k in sorted(metrics.keys()):
            print(f"    {k:>15}: {metrics[k]:.6e}")
        print()

    print("Summary (means across cases):")
    if summary_rows:
        aggregated: Dict[str, List[float]] = {}
        for _, metrics in summary_rows:
            for key, value in metrics.items():
                aggregated.setdefault(key, []).append(value)
        for key, values in aggregated.items():
            print(f"  {key:>15}: {np.mean(values):.6e}")

    print(f"\n✓ Visualizations available under: {output_dir}")


if __name__ == "__main__":
    main()
