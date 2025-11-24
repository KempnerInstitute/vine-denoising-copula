#!/usr/bin/env python
"""
Sanity test for a trained diffusion copula model on synthetic bivariate data.

Workflow:
  1. Generate samples from an analytic copula (ground truth).
  2. Treat them as data in [0,1]^2 (marginals already uniform).
  3. Use DiffusionCopulaModel to estimate the copula density from samples.
  4. Compare predicted vs. true density and marginals; save standard figures.

This is intended as a small, repeatable check after training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vdc.data.generators import analytic_logpdf_grid, sample_bicop
from vdc.eval.visualize import plot_comparison, plot_marginals
from vdc.vine.copula_diffusion import DiffusionCopulaModel


def run_test(
    checkpoint: Path,
    output_dir: Path,
    family: str,
    params: dict,
    m: int = 128,
    n_samples: int = 5000,
    device: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 1) Generate synthetic data from analytic copula
    print(f"Generating {n_samples} samples from {family} with params={params}...")
    samples = sample_bicop(family, params, n=n_samples, rotation=0)

    # 2) Build ground-truth density grid on uniform grid (legacy evaluation style)
    print("Computing analytic ground-truth density on grid...")
    log_true = analytic_logpdf_grid(family, params, m=m, rotation=0)
    density_true = np.exp(log_true)
    du = 1.0 / m
    mass_true = float(density_true.sum() * du * du)
    if abs(mass_true - 1.0) > 1e-3:
        density_true /= mass_true

    # 3) Load diffusion copula model and estimate density from samples
    print(f"Loading diffusion copula model from {checkpoint}...")
    model = DiffusionCopulaModel.from_checkpoint(checkpoint, device=device)
    print("Estimating copula density from samples using diffusion model...")
    density_pred, row_coords, col_coords = model.estimate_density_from_samples(
        samples, m=m, projection_iters=15
    )

    # 4) Save comparison and marginals plots
    print("Creating comparison plot...")
    comp_path = output_dir / f"{family}_comparison.png"
    plot_comparison(
        density_pred,
        density_true,
        title=f"{family} diffusion vs analytic",
        points=samples,
        save_path=comp_path,
        metrics={},
    )

    print("Creating marginals plot (predicted)...")
    marg_path = output_dir / f"{family}_marginals.png"
    plot_marginals(
        density_pred,
        title=f"{family} - Predicted marginals",
        save_path=marg_path,
        row_coords=row_coords,
        col_coords=col_coords,
    )

    print("\n✓ Bivariate diffusion copula test complete.")
    print(f"  Comparison : {comp_path}")
    print(f"  Marginals  : {marg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick bivariate test for a trained diffusion copula model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to diffusion checkpoint (model_step_*.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/diffusion_copula_test"),
        help="Directory to store figures.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="clayton",
        help="Copula family for the synthetic test (e.g., gaussian, clayton, gumbel).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=2.5,
        help="Shape parameter theta (for Clayton/Gumbel/Frank/Joe). Ignored for Gaussian/Student.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.7,
        help="Correlation rho for Gaussian/Student copulas.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=6.0,
        help="Degrees of freedom for Student copulas.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=128,
        help="Grid resolution for evaluation.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of synthetic data points.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model evaluation.",
    )

    args = parser.parse_args()

    if args.family.lower() in {"gaussian", "student"}:
        params = {"rho": args.rho}
        if args.family.lower() == "student":
            params["nu"] = args.nu
    else:
        params = {"theta": args.theta}

    run_test(
        checkpoint=args.checkpoint,
        output_dir=args.output,
        family=args.family.lower(),
        params=params,
        m=args.m,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()


