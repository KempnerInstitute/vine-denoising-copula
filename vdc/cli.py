"""Stable command-line interface for the public VDC release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vdc.pretrained import (
    DEFAULT_PRETRAINED_MODEL_ID,
    estimate_pair_density_from_samples,
    list_pretrained_models,
    load_pretrained_model,
    resolve_pretrained_checkpoint,
)
from vdc.vine.api import VineCopulaModel


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        arr = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}. Use .npy, .csv, or .txt.")
    return np.asarray(arr, dtype=np.float64)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdc",
        description="Command-line tools for the public Vine Diffusion Copula release.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models", help="List packaged pretrained model ids.")
    list_parser.set_defaults(func=_cmd_list_models)

    resolve_parser = subparsers.add_parser(
        "resolve-model",
        help="Resolve or download a packaged pretrained checkpoint.",
    )
    resolve_parser.add_argument("--model-id", default=DEFAULT_PRETRAINED_MODEL_ID)
    resolve_parser.add_argument("--cache-dir", type=Path, default=None)
    resolve_parser.add_argument("--repo-id", type=str, default=None)
    resolve_parser.add_argument("--revision", type=str, default=None)
    resolve_parser.add_argument("--force-download", action="store_true")
    resolve_parser.add_argument(
        "--no-local",
        action="store_true",
        help="Ignore any configured local checkpoint path and use the cache/download path only.",
    )
    resolve_parser.set_defaults(func=_cmd_resolve_model)

    pair_parser = subparsers.add_parser(
        "estimate-pair",
        help="Estimate a bivariate copula density from pseudo-observations.",
    )
    pair_parser.add_argument("input", type=Path, help="Input pseudo-observations (.npy, .csv, .txt) with shape (n, 2).")
    pair_parser.add_argument("--output", type=Path, required=True, help="Output density grid (.npy).")
    pair_parser.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path.")
    pair_parser.add_argument("--model-id", default=DEFAULT_PRETRAINED_MODEL_ID)
    pair_parser.add_argument("--device", default="cpu")
    pair_parser.add_argument("--repo-id", type=str, default=None)
    pair_parser.add_argument("--m", type=int, default=None)
    pair_parser.add_argument("--projection-iters", type=int, default=50)
    pair_parser.add_argument("--diffusion-steps", type=int, default=None)
    pair_parser.add_argument("--cfg-scale", type=float, default=1.0)
    pair_parser.set_defaults(func=_cmd_estimate_pair)

    vine_parser = subparsers.add_parser(
        "fit-vine",
        help="Fit a vine copula to pseudo-observations and save the fitted model.",
    )
    vine_parser.add_argument("input", type=Path, help="Input pseudo-observations (.npy, .csv, .txt) with shape (n, d).")
    vine_parser.add_argument("--output", type=Path, required=True, help="Output pickle path for the fitted vine.")
    vine_parser.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path.")
    vine_parser.add_argument("--model-id", default=DEFAULT_PRETRAINED_MODEL_ID)
    vine_parser.add_argument("--device", default="cpu")
    vine_parser.add_argument("--repo-id", type=str, default=None)
    vine_parser.add_argument("--vine-type", choices=["dvine", "cvine", "rvine"], default="dvine")
    vine_parser.add_argument("--truncation-level", type=int, default=None)
    vine_parser.add_argument("--quiet", action="store_true")
    vine_parser.set_defaults(func=_cmd_fit_vine)

    return parser


def _cmd_list_models(args: argparse.Namespace) -> int:
    for payload in list_pretrained_models():
        print(f"{payload['model_id']}: {payload.get('display_name', '')}")
    return 0


def _cmd_resolve_model(args: argparse.Namespace) -> int:
    ckpt = resolve_pretrained_checkpoint(
        args.model_id,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        prefer_local=not args.no_local,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(ckpt)
    return 0


def _cmd_estimate_pair(args: argparse.Namespace) -> int:
    pair_data = _load_array(args.input)
    if pair_data.ndim != 2 or pair_data.shape[1] != 2:
        raise SystemExit(f"Expected input with shape (n, 2), found {pair_data.shape}.")

    bundle = load_pretrained_model(args.model_id, device=args.device, repo_id=args.repo_id)
    density = estimate_pair_density_from_samples(
        bundle,
        pair_data,
        m=args.m,
        diffusion_steps=args.diffusion_steps,
        cfg_scale=args.cfg_scale,
        projection_iters=args.projection_iters,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, density)
    print(f"Saved density grid to {args.output}")

    if args.summary_json is not None:
        m = int(density.shape[0])
        du = 1.0 / float(m)
        summary = {
            "model_id": bundle.model_id,
            "input": str(args.input),
            "output": str(args.output),
            "shape": list(density.shape),
            "mass": float(density.sum() * du * du),
            "density_min": float(density.min()),
            "density_max": float(density.max()),
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        _save_json(args.summary_json, summary)
        print(f"Saved summary to {args.summary_json}")
    return 0


def _cmd_fit_vine(args: argparse.Namespace) -> int:
    U = _load_array(args.input)
    if U.ndim != 2:
        raise SystemExit(f"Expected a 2D input array with shape (n, d), found {U.shape}.")

    bundle = load_pretrained_model(args.model_id, device=args.device, repo_id=args.repo_id)
    vine = VineCopulaModel(
        vine_type=args.vine_type,
        truncation_level=args.truncation_level,
        m=int(bundle.config.get("data", {}).get("m", 64)),
        device=str(bundle.device),
        batch_edges=(bundle.diffusion is None),
    )
    vine.fit(U, bundle.model, diffusion=bundle.diffusion, verbose=not args.quiet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vine.save(args.output)
    print(f"Saved fitted vine to {args.output}")

    if args.summary_json is not None:
        summary = vine.summary()
        summary["input"] = str(args.input)
        summary["output"] = str(args.output)
        summary["model_id"] = bundle.model_id
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        _save_json(args.summary_json, summary)
        print(f"Saved summary to {args.summary_json}")
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
