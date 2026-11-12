#!/usr/bin/env python3
"""Fit a small D-vine with the released estimator and score held-out data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

from vdc import VineCopulaModel, load_checkpoint_bundle, load_pretrained_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-id", default="vdc-denoiser-m64-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    bundle = (load_checkpoint_bundle(args.checkpoint, device=args.device)
              if args.checkpoint else load_pretrained_model(args.model_id, device=args.device))
    rng = np.random.default_rng(args.seed)
    idx = np.arange(args.dimension)
    covariance = 0.5 ** np.abs(idx[:, None] - idx[None, :])
    u = norm.cdf(rng.multivariate_normal(np.zeros(args.dimension), covariance,
                                       size=args.n_train + args.n_test))
    vine = VineCopulaModel(vine_type="dvine", m=bundle.config["data"]["m"],
                          device=str(bundle.device), batch_edges=bundle.diffusion is None,
                          edge_batch_size=16)
    vine.fit(u[:args.n_train], bundle.model, diffusion=bundle.diffusion, verbose=False)
    print(vine.summary())
    print("Held-out mean log copula density (nats):", float(vine.logpdf(u[args.n_train:]).mean()))
    print("Generated sample shape:", vine.simulate(args.n_test, seed=args.seed + 1).shape)


if __name__ == "__main__":
    main()
