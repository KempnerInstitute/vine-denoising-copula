#!/usr/bin/env python3
"""Estimate Gaussian-copula MI with the public loader and compare with exact truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from vdc import estimate_pair_density_from_samples, load_checkpoint_bundle, load_pretrained_model
from vdc.utils.metrics import mutual_information_from_density_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-id", default="vdc-denoiser-m64-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not -1 < args.rho < 1:
        parser.error("--rho must be strictly between -1 and 1")
    bundle = (load_checkpoint_bundle(args.checkpoint, device=args.device)
              if args.checkpoint else load_pretrained_model(args.model_id, device=args.device))
    rng = np.random.default_rng(args.seed)
    z = rng.multivariate_normal([0, 0], [[1, args.rho], [args.rho, 1]], size=args.samples)
    density = estimate_pair_density_from_samples(bundle, norm.cdf(z))
    estimate = float(mutual_information_from_density_grid(torch.as_tensor(density)))
    truth = float(-0.5 * np.log1p(-args.rho ** 2))
    print(f"Estimated MI: {estimate:.6f} nats")
    print(f"Analytic Gaussian MI: {truth:.6f} nats")
    print(f"Absolute error: {abs(estimate - truth):.6f} nats")


if __name__ == "__main__":
    main()
