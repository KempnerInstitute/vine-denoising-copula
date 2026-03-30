#!/usr/bin/env python3
# ruff: noqa: E402
"""Example: load the official pretrained model and use it for pair or vine inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.hfunc import HFuncLookup
from vdc.pretrained import (
    DEFAULT_PRETRAINED_MODEL_ID,
    estimate_pair_density_from_samples,
    load_checkpoint_bundle,
    load_pretrained_model,
)


def generate_pair_data(n: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1.0 - rho ** 2) * z[:, 1]
    return norm.cdf(z)


def generate_high_dim_data(n: int, d: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)], dtype=float)
    z = rng.multivariate_normal(np.zeros(d), sigma, size=n)
    return norm.cdf(z)


def main() -> None:
    parser = argparse.ArgumentParser(description="Use the official pretrained VDC model")
    parser.add_argument("--model-id", default=DEFAULT_PRETRAINED_MODEL_ID)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override with a local checkpoint path")
    parser.add_argument("--repo-id", type=str, default=None, help="Override Hugging Face repo id for download")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pair-samples", type=int, default=2000)
    parser.add_argument("--pair-rho", type=float, default=0.7)
    parser.add_argument("--fit-vine", action="store_true")
    parser.add_argument("--vine-dim", type=int, default=5)
    parser.add_argument("--vine-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.checkpoint is not None:
        bundle = load_checkpoint_bundle(args.checkpoint, device=args.device)
    else:
        bundle = load_pretrained_model(args.model_id, device=args.device, repo_id=args.repo_id)

    print("=" * 72)
    print("VDC pretrained model example")
    print("=" * 72)
    print(f"Model id:      {bundle.model_id}")
    print(f"Checkpoint:    {bundle.checkpoint_path}")
    print(f"Model type:    {bundle.config.get('model', {}).get('type', bundle.config.get('model_type', 'unknown'))}")
    print(f"Grid size:     {bundle.config.get('data', {}).get('m', 'unknown')}")
    print(f"Device:        {bundle.device}")
    print()

    pair_data = generate_pair_data(args.pair_samples, args.pair_rho, args.seed)
    density = estimate_pair_density_from_samples(bundle, pair_data)
    hfunc = HFuncLookup(density)
    m = density.shape[0]
    mass = float(density.sum() * (1.0 / m) ** 2)
    test_u = np.array([0.25, 0.5, 0.75])
    test_v = np.array([0.25, 0.5, 0.75])
    print("Pair-copula demo")
    print(f"  Input samples: {pair_data.shape[0]}")
    print(f"  Density range: [{density.min():.4f}, {density.max():.4f}]")
    print(f"  Total mass:    {mass:.4f}")
    print(f"  h(u|v):        {np.round(hfunc.h_u_given_v(test_u, test_v), 4)}")
    print(f"  h(v|u):        {np.round(hfunc.h_v_given_u(test_u, test_v), 4)}")

    if args.fit_vine:
        from vdc.vine.api import VineCopulaModel

        print()
        print("Vine demo")
        train_u = generate_high_dim_data(args.vine_samples, args.vine_dim, 0.5, args.seed)
        test_u = generate_high_dim_data(max(200, args.vine_samples // 2), args.vine_dim, 0.5, args.seed + 1)
        vine = VineCopulaModel(
            vine_type="dvine",
            m=int(bundle.config.get("data", {}).get("m", 64)),
            device=str(bundle.device),
            diffusion_steps=50,
            cfg_scale=1.0,
            batch_edges=(bundle.diffusion is None),
        )
        vine.fit(train_u, bundle.model, diffusion=bundle.diffusion, verbose=False)
        loglik = vine.logpdf(test_u).mean()
        print(f"  Fitted D-vine in dimension {args.vine_dim}")
        print(f"  Mean held-out log-likelihood: {loglik:.4f}")


if __name__ == "__main__":
    main()
