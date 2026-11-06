#!/usr/bin/env python3
"""Reproduce the synchronized projection-iteration ablation on a CUDA GPU.

python paper/neurips2026/runners/ipfp_iterations.py --checkpoint model_step_190000.pt --output ipfp.json
Uses the released model, training-order coordinates, ten warmups per K and 100
randomized-order repetitions with CUDA synchronization before and after each projection.
Outputs include every timing, marginal errors, hardware and checkpoint hash.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdc.data.generators import sample_bicop
from vdc.inference.density import scatter_to_hist
from vdc.models.projection import copula_project
from vdc.train.unified_trainer import build_model

# Coordinate channel builder (same as train_unified)
def build_coordinates(batch, m, device, probit=False):
    import torch as _t
    u = _t.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    X, Y = _t.meshgrid(u, u, indexing="ij")
    if probit:
        from scipy.stats import norm
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        X = _t.from_numpy(norm.ppf(X_np)).float().to(device)
        Y = _t.from_numpy(norm.ppf(Y_np)).float().to(device)
    coords = _t.stack([X, Y], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    return coords


def _marginal_errors(d: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_row_err, max_col_err, total_mass_err) from uniform target."""
    m = d.shape[-1]
    target = 1.0  # Unif[0,1]^2 marginal density
    row = d.sum(dim=-1) / m  # row marginal (integrated over v)
    col = d.sum(dim=-2) / m  # col marginal (integrated over u)
    row_err = float((row - target).abs().max().item())
    col_err = float((col - target).abs().max().item())
    mass = float((d.sum(dim=(-1, -2)) / (m * m)).mean().item())
    mass_err = abs(mass - 1.0)
    return row_err, col_err, mass_err


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This timing protocol requires a CUDA GPU")
    device = torch.device("cuda")
    ckpt_path = args.checkpoint
    expected = "8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f"
    import hashlib
    if hashlib.sha256(ckpt_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint differs from the deployed paper model")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_type = str(config.get("model", {}).get("type", "denoiser"))
    if model_type.startswith("diffusion_unet"):
        model_type = "diffusion_unet"
    model = build_model(model_type, config, device)
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    model.eval()
    m = int(config.get("data", {}).get("m", 64))

    # Generate a small diverse batch of raw network outputs.
    families = [
        ("gaussian", {"rho": 0.6}),
        ("clayton", {"theta": 3.0}),
        ("frank", {"theta": 5.0}),
        ("gumbel", {"theta": 2.5}),
    ]
    raw_list = []
    for fam, params in families:
        samples = sample_bicop(fam, params, n=2000, seed=123)
        hist = scatter_to_hist(samples, m=m, reflect=True)
        x = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
        use_log_n = bool(config.get("model", {}).get("use_log_n", False))
        use_coords = bool(config.get("model", {}).get("use_coordinates", False))
        use_probit_coords = bool(config.get("model", {}).get("use_probit_coords", False))
        if use_log_n:
            ln = float(np.log(2000))
            ln_chan = torch.full((1, 1, m, m), ln, device=device, dtype=x.dtype)
            x = torch.cat([x, ln_chan], dim=1)
        if use_coords:
            coords = build_coordinates(1, m, device, probit=use_probit_coords)
            x = torch.cat([x, coords], dim=1)
        with torch.no_grad():
            if model_type == "denoiser":
                out = model(x, torch.zeros(1, device=device))
            else:
                out = model(x)
            if isinstance(out, dict):
                d = out.get("density", None)
                if d is None and "log_density" in out:
                    d = torch.exp(out["log_density"].clamp(min=-20, max=20))
                if d is None and "residual" in out:
                    d = torch.exp(
                        (torch.log(x[:, :1].clamp_min(1e-12)) + out["residual"]).clamp(min=-20, max=20)
                    )
            else:
                d = out
            d = torch.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=1e-12, max=1e6)
            raw_list.append(d[0, 0])

    raw_batch = torch.stack(raw_list, dim=0).unsqueeze(1)  # [B, 1, m, m]

    import hashlib, platform, subprocess
    torch.cuda.synchronize()
    raw_hash = hashlib.sha256(raw_batch.detach().cpu().numpy().tobytes()).hexdigest()
    iter_sweep = [0, 1, 2, 5, 10, 15, 20, 30, 50, 100]
    def project(K):
        if K == 0:
            du = 1.0 / float(m)
            return raw_batch / (raw_batch.sum(dim=(-1, -2), keepdim=True) * du * du)
        return copula_project(raw_batch.clamp_min(1e-12), iters=K)
    with torch.no_grad():
        for K in iter_sweep:
            for _ in range(10):
                project(K)
        torch.cuda.synchronize()
        timings = {K: [] for K in iter_sweep}
        rng = np.random.default_rng(20261106)
        for repetition in range(100):
            for K in rng.permutation(iter_sweep):
                K = int(K)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                proj = project(K)
                torch.cuda.synchronize()
                timings[K].append((time.perf_counter()-t0)*1000)
        results = []
        for K in iter_sweep:
            proj = project(K)
            row_err, col_err, mass_err = _marginal_errors(proj[:, 0])
            results.append(dict(iters=K, max_row_err=row_err, max_col_err=col_err,
                total_mass_err=mass_err, elapsed_ms=float(np.median(timings[K])),
                mean_ms=float(np.mean(timings[K])), std_ms=float(np.std(timings[K],ddof=1)),
                samples_ms=timings[K]))
            print(K, results[-1]['elapsed_ms'], row_err, col_err)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "raw_batch_sha256": raw_hash,
        "generation_runtime_commit": "5dd66a6c1eb27b87caa00284f0b9e6dd330cee22",
        "library_version": __import__("vdc").__version__,
        "library_source": str(Path(__import__("vdc").__file__).resolve()),
        "hostname": platform.node(), "cpu": subprocess.check_output(["lscpu"],text=True),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "dtype": str(raw_batch.dtype), "batch_shape": list(raw_batch.shape),
        "timing_protocol": "100 randomized-order repetitions; 10 warmups per K; synchronize immediately before and after each whole-batch projection; median elapsed wall time; model loading and inference excluded",
        "threads": torch.get_num_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "m": m,
        "families": [f[0] for f in families],
        "iter_sweep": iter_sweep,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
