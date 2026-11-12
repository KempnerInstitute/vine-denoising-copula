#!/usr/bin/env python3
"""Write a small denoiser configuration; training is a separate, explicit command."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

import yaml


def make_config():
    return {
        "data": {"m": 32, "n_samples_per_copula": 1000, "num_workers": 0,
                 "copula_families": {"gaussian": 0.5, "clayton": 0.5},
                 "rotation_prob": 0.0, "mixture_prob": 0.0},
        "model": {"type": "denoiser", "base_channels": 16, "depth": 2,
                  "blocks_per_level": 1, "dropout": 0.0,
                  "use_coordinates": True, "use_log_n": False},
        "training": {"batch_size": 4, "max_steps": 1000, "learning_rate": 0.0001,
                     "use_amp": False, "projection_iters": 30, "log_interval": 10,
                     "save_interval": 500, "viz_interval": 1000,
                     "loss_weights": {"ce": 1.0, "ise": 0.1, "marg_kl": 0.1}},
        "checkpoint_dir": "results/custom_denoiser/checkpoints",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/custom_denoiser.yaml"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        yaml.safe_dump(make_config(), handle, sort_keys=False)
    print("Configuration written; no training has been started.")
    print("python scripts/train_unified.py --model-type denoiser --config " + shlex.quote(str(args.output)))


if __name__ == "__main__":
    main()
