#!/usr/bin/env python3
"""
Run scaling experiments with accuracy metrics (TC error, NLL error) vs dimension.

This script generates the data needed for:
  - Figure: TC error vs dimension
  - Figure: NLL error vs dimension
  - Figure: Probit vs non-probit comparison

Usage:
    python scripts/run_scaling_with_accuracy.py \
        --checkpoint /path/to/model.pt \
        --output_dir /path/to/output \
        --dims 2 5 10 20 50 100 200 \
        --n_samples 2000 \
        --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.data.generators import generate_gaussian_vine
from vdc.models.projection import copula_project


def true_tc_gaussian_ar1(d: int, rho: float) -> float:
    """Compute true total correlation for Gaussian AR(1) copula."""
    # TC = (d-1) * MI for AR(1) structure
    # MI = -0.5 * log(1 - rho^2)
    mi = -0.5 * np.log(1 - rho**2)
    return (d - 1) * mi


def true_nll_gaussian_ar1(d: int, rho: float, n_samples: int = 10000, seed: int = 42) -> float:
    """Compute expected NLL for Gaussian AR(1) copula samples."""
    # For AR(1) Gaussian, the expected NLL is related to the entropy
    # H = -0.5 * log(det(Sigma)) - d/2 * log(2*pi*e) 
    # But for copula, we use the copula density which sums MI across edges
    # Expected NLL = sum of MI across all edges = TC for D-vine
    mi = -0.5 * np.log(1 - rho**2)
    return (d - 1) * mi


@dataclass
class ScalingRecord:
    """Record for one scaling experiment."""
    method: str
    checkpoint: Optional[str]
    d: int
    n_samples: int
    rho: float
    repeat: int
    seed: int
    m: int
    tc_true: float
    tc_pred: float
    tc_err: float
    tc_rel_err: float
    nll_true: float
    nll_pred: float
    nll_err: float
    fit_time_s: float
    infer_time_s: float


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[Any, Dict]:
    """Load model from checkpoint."""
    from vdc.models.unet_grid import GridUNet
    from vdc.models.copula_diffusion import CopulaAwareDiffusion
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    
    # Extract model config
    model_cfg = cfg.get('model', {})
    m = cfg.get('data', {}).get('m', 64)
    
    # Check model type
    model_type = model_cfg.get('model_type', 'diffusion_unet')
    in_channels = int(model_cfg.get('in_channels', 1))
    
    # Detect upsample mode from checkpoint keys
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    has_upsample_conv = any("upsample_conv" in k for k in state_dict.keys())
    upsample_mode = "bilinear" if has_upsample_conv else "transpose"
    
    # Build model with correct upsample_mode to match checkpoint
    model = GridUNet(
        m=m,
        in_channels=in_channels,
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', (16, 8))),
        dropout=model_cfg.get('dropout', 0.1),
        upsample_mode=upsample_mode,
    ).to(device)
    
    # Load weights
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    
    # Create diffusion wrapper
    diffusion = CopulaAwareDiffusion(
        timesteps=cfg.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=cfg.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    # Add m to config for later use
    cfg['grid_size'] = m
    cfg['model_type'] = model_type
    
    return (model, diffusion), cfg


def estimate_tc_dvine(
    U: np.ndarray,
    model: Any,
    m: int,
    projection_iters: int,
    device: torch.device,
    use_spline: bool = False,
) -> Tuple[float, float, float]:
    """
    Estimate total correlation via D-vine decomposition using histogram method.
    
    For D-vine with sequential order [1,...,d]:
    - Tree 0: edges (1,2), (2,3), ..., (d-1,d)
    - Tree k: conditional copulas between h-transforms from tree k-1
    
    Returns:
        tc_pred: Predicted total correlation
        nll_pred: Average negative log-likelihood 
        infer_time: Inference time in seconds
    """
    from scipy.stats import norm
    
    n, d = U.shape
    t0 = perf_counter()
    
    tc_sum = 0.0
    nll_sum = 0.0
    n_edges = 0
    du = 1.0 / m
    
    # Store h-function values: V[tree] has shape (n, d-tree)
    # V[0] = original pseudo-observations
    V = [U.copy()]
    
    for tree in range(d - 1):
        n_pairs = d - 1 - tree
        V_curr = V[tree]
        
        # Will store h-transforms for next tree
        if tree < d - 2:
            V_next = np.zeros((n, d - tree - 1))
        
        for i in range(n_pairs):
            u1 = V_curr[:, i]
            u2 = V_curr[:, i + 1]
            
            # Create scatter histogram
            pts = np.stack([u1, u2], axis=1)
            pts = np.clip(pts, 1e-6, 1 - 1e-6)
            
            hist, _, _ = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=m,
                range=[[0, 1], [0, 1]]
            )
            hist = hist.astype(np.float32)
            
            # Normalize
            hist = hist / (hist.sum() * du * du + 1e-12)
            
            # Convert to tensor and project to copula
            hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
            hist_t = copula_project(hist_t, iters=projection_iters)
            
            # Compute MI for this edge
            c = hist_t[0, 0].cpu().numpy()
            c = np.clip(c, 1e-12, None)
            mi = float(np.sum(c * np.log(c)) * du * du)
            tc_sum += mi
            
            # Compute NLL contribution
            log_c = np.log(c + 1e-12)
            u1_idx = np.clip((u1 * m).astype(int), 0, m-1)
            u2_idx = np.clip((u2 * m).astype(int), 0, m-1)
            nll_edge = -np.mean(log_c[u1_idx, u2_idx])
            nll_sum += nll_edge
            n_edges += 1
            
            # Compute h-functions for next tree level
            if tree < d - 2:
                # Use Gaussian h-transform (approximation for computational efficiency)
                # h(u|v) = Φ(Φ^{-1}(u) - ρ*Φ^{-1}(v)) / sqrt(1-ρ^2)
                rho = np.corrcoef(u1, u2)[0, 1]
                rho = np.clip(rho, -0.99, 0.99)
                
                # h_1|2: condition u1 on u2
                z1 = norm.ppf(np.clip(u1, 1e-6, 1-1e-6))
                z2 = norm.ppf(np.clip(u2, 1e-6, 1-1e-6))
                h_1given2 = norm.cdf((z1 - rho * z2) / np.sqrt(1 - rho**2))
                h_1given2 = np.clip(h_1given2, 1e-6, 1-1e-6)
                
                # For D-vine, V_next columns are h-transforms
                # Edge i gives h_{i|i+1} which becomes V_next[:, i]
                V_next[:, i] = h_1given2
        
        if tree < d - 2:
            V.append(V_next)
    
    infer_time = perf_counter() - t0
    
    return tc_sum, nll_sum / n_edges if n_edges > 0 else 0.0, infer_time


def run_one_experiment(
    d: int,
    n_samples: int,
    rho: float,
    repeat: int,
    seed: int,
    model: Any,
    model_cfg: Dict,
    device: torch.device,
    method: str = "ours",
    checkpoint_path: str = None,
) -> ScalingRecord:
    """Run one scaling experiment."""
    np.random.seed(seed)
    
    m = model_cfg.get("grid_size", 64)
    projection_iters = model_cfg.get("projection_iters", 30)
    
    # Generate samples
    t0 = perf_counter()
    U = generate_gaussian_vine(n_samples, d, rho=rho, seed=seed)
    fit_time = perf_counter() - t0
    
    # True values
    tc_true = true_tc_gaussian_ar1(d, rho)
    nll_true = true_nll_gaussian_ar1(d, rho)
    
    # Estimate
    tc_pred, nll_pred, infer_time = estimate_tc_dvine(
        U, model, m, projection_iters, device
    )
    
    tc_err = abs(tc_pred - tc_true)
    tc_rel_err = tc_err / (tc_true + 1e-12)
    nll_err = abs(nll_pred - nll_true)
    
    return ScalingRecord(
        method=method,
        checkpoint=checkpoint_path,
        d=d,
        n_samples=n_samples,
        rho=rho,
        repeat=repeat,
        seed=seed,
        m=m,
        tc_true=tc_true,
        tc_pred=tc_pred,
        tc_err=tc_err,
        tc_rel_err=tc_rel_err,
        nll_true=nll_true,
        nll_pred=nll_pred,
        nll_err=nll_err,
        fit_time_s=fit_time,
        infer_time_s=infer_time,
    )


def run_pyvine_experiment(
    d: int,
    n_samples: int,
    rho: float,
    repeat: int,
    seed: int,
    method: str = "pyvine_param",
) -> Optional[ScalingRecord]:
    """Run pyvine baseline experiment."""
    try:
        import pyvinecopulib as pv
    except ImportError:
        print("pyvinecopulib not installed, skipping baseline")
        return None
    
    np.random.seed(seed)
    
    # Generate samples
    t0 = perf_counter()
    U = generate_gaussian_vine(n_samples, d, rho=rho, seed=seed)
    
    # True values
    tc_true = true_tc_gaussian_ar1(d, rho)
    nll_true = true_nll_gaussian_ar1(d, rho)
    
    # Fit vine
    fit_t0 = perf_counter()
    if method == "pyvine_param":
        controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian])
    else:
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.tll],
            nonparametric_method="constant",
            nonparametric_mult=1.0,
        )
    
    vine = pv.Vinecop.from_data(U, controls=controls)
    fit_time = perf_counter() - fit_t0
    
    # Estimate TC from fitted vine
    infer_t0 = perf_counter()
    tc_pred = 0.0
    d = vine.structure.dim
    for tree in range(d - 1):
        for edge in range(d - 1 - tree):
            bicop = vine.get_pair_copula(tree, edge)
            # MI ≈ -0.5 * log(1 - tau^2) for Gaussian copulas
            tau = bicop.tau  # tau is a property, not a method
            mi = -0.5 * np.log(1 - min(tau**2, 0.999)) if abs(tau) > 1e-6 else 0.0
            tc_pred += mi
    
    # NLL via loglik
    nll_pred = -vine.loglik(U) / n_samples
    
    infer_time = perf_counter() - infer_t0
    
    tc_err = abs(tc_pred - tc_true)
    tc_rel_err = tc_err / (tc_true + 1e-12)
    nll_err = abs(nll_pred - nll_true)
    
    return ScalingRecord(
        method=method,
        checkpoint=None,
        d=d,
        n_samples=n_samples,
        rho=rho,
        repeat=repeat,
        seed=seed,
        m=64,
        tc_true=tc_true,
        tc_pred=tc_pred,
        tc_err=tc_err,
        tc_rel_err=tc_rel_err,
        nll_true=nll_true,
        nll_pred=nll_pred,
        nll_err=nll_err,
        fit_time_s=fit_time,
        infer_time_s=infer_time,
    )


def main():
    parser = argparse.ArgumentParser(description="Scaling experiments with accuracy metrics")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--checkpoint_probit", type=str, default=None,
                        help="Path to probit model checkpoint for comparison")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100, 200],
                        help="Dimensions to test")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of samples per experiment")
    parser.add_argument("--rho", type=float, default=0.6,
                        help="AR(1) correlation coefficient")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeats per configuration")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Also run pyvine baselines")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, model_cfg = load_model(args.checkpoint, device)
    
    # Load probit model if specified
    model_probit = None
    if args.checkpoint_probit:
        print(f"Loading probit model from {args.checkpoint_probit}")
        model_probit, _ = load_model(args.checkpoint_probit, device)
    
    records: List[ScalingRecord] = []
    
    base_seed = 123
    
    for d in args.dims:
        print(f"\n{'='*60}")
        print(f"Dimension d={d}")
        print(f"{'='*60}")
        
        for rep in range(args.repeats):
            seed = base_seed + d * 100 + rep
            
            # Our method
            print(f"  [ours] repeat {rep+1}/{args.repeats}...", end=" ", flush=True)
            rec = run_one_experiment(
                d=d,
                n_samples=args.n_samples,
                rho=args.rho,
                repeat=rep,
                seed=seed,
                model=model,
                model_cfg=model_cfg,
                device=device,
                method="ours",
                checkpoint_path=args.checkpoint,
            )
            records.append(rec)
            print(f"TC_err={rec.tc_err:.4f}, NLL_err={rec.nll_err:.4f}, time={rec.infer_time_s:.2f}s")
            
            # Probit model if available
            if model_probit is not None:
                print(f"  [ours_probit] repeat {rep+1}/{args.repeats}...", end=" ", flush=True)
                rec_p = run_one_experiment(
                    d=d,
                    n_samples=args.n_samples,
                    rho=args.rho,
                    repeat=rep,
                    seed=seed,
                    model=model_probit,
                    model_cfg=model_cfg,
                    device=device,
                    method="ours_probit",
                    checkpoint_path=args.checkpoint_probit,
                )
                records.append(rec_p)
                print(f"TC_err={rec_p.tc_err:.4f}, NLL_err={rec_p.nll_err:.4f}, time={rec_p.infer_time_s:.2f}s")
            
            # Baselines
            if args.run_baselines:
                for method in ["pyvine_param", "pyvine_nonpar"]:
                    print(f"  [{method}] repeat {rep+1}/{args.repeats}...", end=" ", flush=True)
                    rec_b = run_pyvine_experiment(
                        d=d,
                        n_samples=args.n_samples,
                        rho=args.rho,
                        repeat=rep,
                        seed=seed,
                        method=method,
                    )
                    if rec_b:
                        records.append(rec_b)
                        print(f"TC_err={rec_b.tc_err:.4f}, NLL_err={rec_b.nll_err:.4f}, time={rec_b.infer_time_s:.2f}s")
                    else:
                        print("skipped (pyvine not available)")
    
    # Save results
    results = {
        "generated_at": str(np.datetime64('now')),
        "checkpoint": args.checkpoint,
        "checkpoint_probit": args.checkpoint_probit,
        "dims": args.dims,
        "n_samples": args.n_samples,
        "rho": args.rho,
        "repeats": args.repeats,
        "records": [asdict(r) for r in records],
    }
    
    output_path = output_dir / "scaling_accuracy_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Mean TC Error by Dimension")
    print("="*80)
    print(f"{'Method':<20} " + " ".join(f"d={d:>5}" for d in args.dims))
    print("-"*80)
    
    methods = sorted(set(r.method for r in records))
    for method in methods:
        errs = []
        for d in args.dims:
            recs = [r for r in records if r.method == method and r.d == d]
            if recs:
                mean_err = np.mean([r.tc_err for r in recs])
                errs.append(f"{mean_err:>5.3f}")
            else:
                errs.append(f"{'N/A':>5}")
        print(f"{method:<20} " + " ".join(errs))


if __name__ == "__main__":
    main()
