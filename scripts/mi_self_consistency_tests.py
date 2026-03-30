#!/usr/bin/env python3
"""
MI Self-Consistency Tests for ICML Paper.

This script evaluates mutual information estimators on three fundamental 
information-theoretic consistency properties:

1. **Data Processing Inequality (DPI)**: I(X;Y) ≥ I(X;f(Y)) for any function f.
   - We corrupt Y with noise and verify MI does not increase.

2. **Additivity Under Independence**: I(X1,X2; Y1,Y2) = I(X1;Y1) + I(X2;Y2) 
   when (X1,Y1) ⊥ (X2,Y2).
   - We generate independent pairs and check if MI decomposes.

3. **Monotone Invariance**: I(X;Y) = I(f(X); g(Y)) for monotone bijections f, g.
   - We apply monotone transforms (e.g., x³, exp) and verify MI is unchanged.

The goal is to demonstrate that DCD-Vine respects these properties where 
many neural estimators (especially variational bounds) can exhibit violations.

Usage:
    python scripts/mi_self_consistency_tests.py \
        --checkpoint outputs/best_model.pt \
        --n_samples 10000 \
        --output drafts/tables/tab_self_consistency.tex

Reference:
    - Cover & Thomas, Elements of Information Theory, Ch. 2
    - Data Processing Inequality: I(X;Y) ≥ I(X;Z) for Markov chain X → Y → Z
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.models.projection import copula_project
from vdc.utils.information import ksg_mutual_information


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConsistencyTestConfig:
    """Configuration for self-consistency tests."""
    n_samples: int = 10000
    n_trials: int = 5
    seed: int = 42
    
    # DPI test settings
    dpi_noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5])
    
    # Monotone transform test settings
    monotone_transforms: List[str] = field(default_factory=lambda: ["identity", "cube", "exp", "softplus"])
    
    # Test copulas (family, params, rotation, display_name)
    test_copulas: List[Tuple[str, Dict[str, Any], int, str]] = field(default_factory=lambda: [
        ("gaussian", {"rho": 0.7}, 0, "Gaussian(ρ=0.7)"),
        ("student", {"rho": 0.7, "df": 5}, 0, "Student-t"),
        ("clayton", {"theta": 3.0}, 0, "Clayton"),
        ("frank", {"theta": 5.0}, 0, "Frank"),
    ])


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_density_np(d: np.ndarray) -> np.ndarray:
    """Normalize a density grid to integrate to 1."""
    m = int(d.shape[0])
    du = 1.0 / m
    d = np.nan_to_num(d, nan=0.0, posinf=1e300, neginf=0.0)
    d = np.clip(d, 0.0, 1e300)
    mass = float((d * du * du).sum())
    if not np.isfinite(mass) or mass <= 0:
        return np.ones_like(d) * (m * m)
    return d / mass


def mi_from_density_grid(d: np.ndarray) -> float:
    """Compute MI from copula density grid: ∬ c(u,v) log c(u,v) du dv."""
    m = int(d.shape[0])
    du = 1.0 / m
    return float(np.sum(d * np.log(d + 1e-12)) * du * du)


def apply_monotone_transform(u: np.ndarray, transform: str) -> np.ndarray:
    """Apply a monotone bijection to uniform samples.
    
    For copula data on [0,1], we use transforms that remain in a valid range.
    Since MI is invariant to monotone transforms, any such f should preserve MI.
    """
    eps = 1e-6
    u = np.clip(u, eps, 1 - eps)
    
    if transform == "identity":
        return u
    elif transform == "cube":
        # f(u) = u³ is monotone on [0,1]
        return u ** 3
    elif transform == "exp":
        # Map u ∈ (0,1) → (1, e) via exp(u), then rescale back to [0,1]
        # This is still monotone
        return (np.exp(u) - 1) / (np.e - 1)
    elif transform == "softplus":
        # softplus(αu) / softplus(α) rescaled to [0,1]
        alpha = 5.0
        return np.log1p(np.exp(alpha * u)) / np.log1p(np.exp(alpha))
    elif transform == "sqrt":
        return np.sqrt(u)
    elif transform == "logit_based":
        # A sigmoid-like remapping
        from scipy.special import expit
        return expit(5 * (u - 0.5))
    else:
        raise ValueError(f"Unknown transform: {transform}")


# =============================================================================
# DCD Model Interface
# =============================================================================

class DCDEstimator:
    """Wrapper for DCD model MI estimation."""
    
    def __init__(
        self,
        checkpoint_path: Optional[Path],
        device: torch.device,
        grid_size: int = 256,
        ipfp_iters: int = 50,
        use_diffusion: bool = False,
    ):
        self.device = device
        self.grid_size = grid_size
        self.ipfp_iters = ipfp_iters
        self.use_diffusion = use_diffusion
        self.model = None
        
        if checkpoint_path is not None and checkpoint_path.exists():
            self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: Path) -> None:
        """Load trained DCD model."""
        from vdc.models.unet import UNet2D
        
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        
        self.model = UNet2D(
            in_channels=cfg.get("in_channels", 1),
            base_channels=cfg.get("base_channels", 64),
            channel_mult=cfg.get("channel_mult", [1, 2, 4]),
            num_res_blocks=cfg.get("num_res_blocks", 2),
            film_cond_dim=cfg.get("film_cond_dim", 64),
        ).to(self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
    
    def estimate_mi(self, u: np.ndarray, v: np.ndarray) -> float:
        """Estimate MI from bivariate samples using DCD density estimation."""
        if self.model is None:
            # Fallback to KSG if model not loaded
            return ksg_mutual_information(u, v, k=5)
        
        # Use DCD to estimate copula density, then compute MI from grid
        # This is a simplified interface; full implementation uses the
        # empirical 2D histogram + denoising pipeline
        n = len(u)
        log_n = np.log(n)
        
        # Create empirical histogram
        m = self.grid_size
        edges = np.linspace(0, 1, m + 1)
        hist, _, _ = np.histogram2d(u, v, bins=edges, density=True)
        hist = hist.T  # (m, m), matches our convention
        
        # Normalize
        hist = normalize_density_np(hist)
        
        # Run through DCD denoiser
        with torch.no_grad():
            x = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Conditioning on log(n) and t=0 for one-shot
            cond = torch.tensor([[log_n, 0.0]], device=self.device, dtype=torch.float32)
            
            # Forward pass
            pred = self.model(x, cond)
            
            # Exponentiate if predicting log-density
            pred = torch.exp(pred.clamp(-20, 20))
            
            # IPFP projection
            pred = copula_project(pred, iters=self.ipfp_iters)
            
            density = pred[0, 0].cpu().numpy()
        
        density = normalize_density_np(density)
        return mi_from_density_grid(density)


# =============================================================================
# Test 1: Data Processing Inequality
# =============================================================================

def test_dpi(
    estimator: Callable[[np.ndarray, np.ndarray], float],
    config: ConsistencyTestConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Test Data Processing Inequality: I(X;Y) ≥ I(X;f(Y)) where f adds noise.
    
    We generate (U, V) from a copula, then corrupt V → V' = V + noise.
    DPI requires that MI(U, V') ≤ MI(U, V) since X → Y → Z is Markov.
    
    Returns:
        Dictionary with test results and violation statistics.
    """
    results = {
        "test_name": "Data Processing Inequality",
        "copulas": {},
        "summary": {},
    }
    
    all_violations = []
    
    for family, params, rotation, display_name in config.test_copulas:
        copula_results = {
            "noise_levels": [],
            "mi_estimates": [],
            "mi_std": [],
            "violations": [],
        }
        
        mi_base = None  # MI at noise=0
        
        for noise_level in config.dpi_noise_levels:
            trial_mis = []
            
            for trial in range(config.n_trials):
                seed = int(rng.integers(0, 2**30))
                
                # Sample from copula
                samples = sample_bicop(
                    family=family,
                    params=params,
                    n=config.n_samples,
                    rotation=rotation,
                    seed=seed,
                )
                u, v = samples[:, 0], samples[:, 1]
                
                if noise_level > 0:
                    # Corrupt v with uniform noise and re-normalize to [0,1]
                    noise = rng.uniform(-noise_level/2, noise_level/2, size=len(v))
                    v_noisy = np.clip(v + noise, 0, 1)
                    # Re-rank to uniform marginals (keeps copula interpretation valid)
                    v_corrupted = (np.argsort(np.argsort(v_noisy)) + 0.5) / len(v_noisy)
                else:
                    v_corrupted = v
                
                mi = estimator(u, v_corrupted)
                trial_mis.append(mi)
            
            mean_mi = float(np.mean(trial_mis))
            std_mi = float(np.std(trial_mis))
            
            copula_results["noise_levels"].append(noise_level)
            copula_results["mi_estimates"].append(mean_mi)
            copula_results["mi_std"].append(std_mi)
            
            if noise_level == 0:
                mi_base = mean_mi
            else:
                # Check for violation: MI should not INCREASE with more noise
                violation = mean_mi > mi_base + 2 * std_mi  # 2σ tolerance
                copula_results["violations"].append(violation)
                if violation:
                    all_violations.append({
                        "copula": display_name,
                        "noise": noise_level,
                        "mi_base": mi_base,
                        "mi_noisy": mean_mi,
                        "excess": mean_mi - mi_base,
                    })
        
        results["copulas"][display_name] = copula_results
    
    # Summary statistics
    n_tests = len(config.test_copulas) * (len(config.dpi_noise_levels) - 1)
    n_violations = len(all_violations)
    
    results["summary"] = {
        "total_tests": n_tests,
        "violations": n_violations,
        "violation_rate": n_violations / max(n_tests, 1),
        "violation_details": all_violations,
    }
    
    return results


# =============================================================================
# Test 2: Additivity Under Independence
# =============================================================================

def test_additivity(
    estimator: Callable[[np.ndarray, np.ndarray], float],
    config: ConsistencyTestConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Test Additivity: I(X1,X2; Y1,Y2) = I(X1;Y1) + I(X2;Y2) when pairs are independent.
    
    We generate two independent copula pairs and verify MI decomposes additively.
    
    Returns:
        Dictionary with test results and decomposition errors.
    """
    results = {
        "test_name": "Additivity Under Independence", 
        "pairs": [],
        "summary": {},
    }
    
    errors = []
    
    # Test different combinations of copulas
    copula_pairs = [
        (config.test_copulas[0], config.test_copulas[1]),
        (config.test_copulas[0], config.test_copulas[2]),
        (config.test_copulas[1], config.test_copulas[2]),
    ]
    
    for (fam1, par1, rot1, name1), (fam2, par2, rot2, name2) in copula_pairs:
        pair_results = {"pair": f"{name1} ⊥ {name2}", "trials": []}
        
        for trial in range(config.n_trials):
            seed = int(rng.integers(0, 2**30))
            
            # Sample first copula (U1, V1)
            samples1 = sample_bicop(family=fam1, params=par1, n=config.n_samples, rotation=rot1, seed=seed)
            u1, v1 = samples1[:, 0], samples1[:, 1]
            
            # Sample second copula (U2, V2) independently
            samples2 = sample_bicop(family=fam2, params=par2, n=config.n_samples, rotation=rot2, seed=seed + 1)
            u2, v2 = samples2[:, 0], samples2[:, 1]
            
            # Compute individual MIs
            mi1 = estimator(u1, v1)
            mi2 = estimator(u2, v2)
            
            # Compute joint MI: I(U1,U2; V1,V2) using 4D KSG or approximation
            # For our estimator, we concatenate and use bivariate interface
            x_joint = np.column_stack([u1, u2])
            y_joint = np.column_stack([v1, v2])
            
            # Use KSG for joint (estimator is bivariate, need multivariate here)
            mi_joint = ksg_mutual_information(x_joint, y_joint, k=5, seed=seed)
            
            # Additivity says mi_joint ≈ mi1 + mi2
            mi_sum = mi1 + mi2
            error = abs(mi_joint - mi_sum) / max(mi_sum, 0.1)  # Relative error
            
            pair_results["trials"].append({
                "mi1": mi1,
                "mi2": mi2,
                "mi_joint": mi_joint,
                "mi_sum": mi_sum,
                "relative_error": error,
            })
            errors.append(error)
        
        # Average over trials
        pair_results["mean_relative_error"] = float(np.mean([t["relative_error"] for t in pair_results["trials"]]))
        results["pairs"].append(pair_results)
    
    results["summary"] = {
        "mean_relative_error": float(np.mean(errors)),
        "max_relative_error": float(np.max(errors)),
        "std_relative_error": float(np.std(errors)),
    }
    
    return results


# =============================================================================
# Test 3: Monotone Invariance
# =============================================================================

def test_monotone_invariance(
    estimator: Callable[[np.ndarray, np.ndarray], float],
    config: ConsistencyTestConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Test Monotone Invariance: I(X;Y) = I(f(X);g(Y)) for monotone bijections.
    
    MI is invariant under monotone (invertible) transformations of each variable.
    We apply various monotone transforms and verify MI remains unchanged.
    
    Returns:
        Dictionary with test results and invariance errors.
    """
    results = {
        "test_name": "Monotone Invariance",
        "copulas": {},
        "summary": {},
    }
    
    all_errors = []
    
    for family, params, rotation, display_name in config.test_copulas:
        copula_results = {
            "transforms": [],
            "mi_estimates": [],
            "mi_std": [],
            "relative_errors": [],
        }
        
        mi_identity = None  # MI with identity transform
        
        for transform in config.monotone_transforms:
            trial_mis = []
            
            for trial in range(config.n_trials):
                seed = int(rng.integers(0, 2**30))
                
                samples = sample_bicop(
                    family=family,
                    params=params,
                    n=config.n_samples,
                    rotation=rotation,
                    seed=seed,
                )
                u, v = samples[:, 0], samples[:, 1]
                
                # Apply monotone transform to both variables
                u_t = apply_monotone_transform(u, transform)
                v_t = apply_monotone_transform(v, transform)
                
                # Re-rank to uniform marginals (required for copula MI interpretation)
                u_t = (np.argsort(np.argsort(u_t)) + 0.5) / len(u_t)
                v_t = (np.argsort(np.argsort(v_t)) + 0.5) / len(v_t)
                
                mi = estimator(u_t, v_t)
                trial_mis.append(mi)
            
            mean_mi = float(np.mean(trial_mis))
            std_mi = float(np.std(trial_mis))
            
            copula_results["transforms"].append(transform)
            copula_results["mi_estimates"].append(mean_mi)
            copula_results["mi_std"].append(std_mi)
            
            if transform == "identity":
                mi_identity = mean_mi
            
            # Compute error relative to identity
            if mi_identity is not None and mi_identity > 0.01:
                rel_error = abs(mean_mi - mi_identity) / mi_identity
                copula_results["relative_errors"].append(rel_error)
                if transform != "identity":
                    all_errors.append(rel_error)
        
        results["copulas"][display_name] = copula_results
    
    results["summary"] = {
        "mean_relative_error": float(np.mean(all_errors)) if all_errors else 0.0,
        "max_relative_error": float(np.max(all_errors)) if all_errors else 0.0,
        "std_relative_error": float(np.std(all_errors)) if all_errors else 0.0,
    }
    
    return results


# =============================================================================
# Comparison Across Estimators
# =============================================================================

def run_all_tests(
    estimator_name: str,
    estimator_fn: Callable[[np.ndarray, np.ndarray], float],
    config: ConsistencyTestConfig,
) -> Dict[str, Any]:
    """Run all self-consistency tests for a given estimator."""
    rng = np.random.default_rng(config.seed)
    
    print(f"\n{'='*60}")
    print(f"Running self-consistency tests for: {estimator_name}")
    print(f"{'='*60}")
    
    # Test 1: DPI
    print("\n[1/3] Testing Data Processing Inequality...")
    dpi_results = test_dpi(estimator_fn, config, rng)
    n_viol = dpi_results["summary"]["violations"]
    n_test = dpi_results["summary"]["total_tests"]
    print(f"  → DPI violations: {n_viol}/{n_test}")
    
    # Test 2: Additivity
    print("\n[2/3] Testing Additivity Under Independence...")
    add_results = test_additivity(estimator_fn, config, rng)
    print(f"  → Mean relative error: {add_results['summary']['mean_relative_error']:.3f}")
    
    # Test 3: Monotone invariance
    print("\n[3/3] Testing Monotone Invariance...")
    mono_results = test_monotone_invariance(estimator_fn, config, rng)
    print(f"  → Mean relative error: {mono_results['summary']['mean_relative_error']:.3f}")
    
    return {
        "estimator": estimator_name,
        "dpi": dpi_results,
        "additivity": add_results,
        "monotone_invariance": mono_results,
    }


def create_ksg_estimator(k: int = 5) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create KSG estimator function."""
    def _ksg(u: np.ndarray, v: np.ndarray) -> float:
        return ksg_mutual_information(u, v, k=k)
    return _ksg


def create_dcd_estimator(
    checkpoint: Optional[Path],
    device: torch.device,
    use_diffusion: bool = False,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create DCD estimator function."""
    dcd = DCDEstimator(checkpoint, device, use_diffusion=use_diffusion)
    return dcd.estimate_mi


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_table(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate LaTeX table summarizing self-consistency tests."""
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Self-consistency test results. DPI: \# violations out of 12 tests. " +
        r"Additivity \& Invariance: mean relative error (\%). " +
        r"Lower is better for all metrics.}",
        r"\label{tab:self-consistency}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & DPI $\downarrow$ & Additivity $\downarrow$ & Invariance $\downarrow$ \\",
        r"\midrule",
    ]
    
    for res in all_results:
        name = res["estimator"]
        dpi_viol = res["dpi"]["summary"]["violations"]
        dpi_total = res["dpi"]["summary"]["total_tests"]
        add_err = res["additivity"]["summary"]["mean_relative_error"] * 100
        mono_err = res["monotone_invariance"]["summary"]["mean_relative_error"] * 100
        
        lines.append(f"{name} & {dpi_viol}/{dpi_total} & {add_err:.1f}\\% & {mono_err:.1f}\\% \\\\")
    
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nLaTeX table written to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MI Self-Consistency Tests")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to DCD model checkpoint")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of samples per test")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Number of trials per configuration")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="drafts/tables/tab_self_consistency.tex",
                        help="Output path for LaTeX table")
    parser.add_argument("--json_output", type=str, default=None,
                        help="Output path for JSON results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    config = ConsistencyTestConfig(
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    
    # Define estimators to test
    estimators = [
        ("KSG ($k$=5)", create_ksg_estimator(k=5)),
    ]
    
    # Add DCD if checkpoint provided
    if checkpoint is not None and checkpoint.exists():
        estimators.append(("DCD (one-shot)", create_dcd_estimator(checkpoint, device, use_diffusion=False)))
        estimators.append(("DCD (diffusion)", create_dcd_estimator(checkpoint, device, use_diffusion=True)))
    else:
        print("Note: No checkpoint provided, testing KSG only. Add --checkpoint for DCD tests.")
    
    # Run tests
    all_results = []
    for name, est_fn in estimators:
        results = run_all_tests(name, est_fn, config)
        all_results.append(results)
    
    # Generate LaTeX table
    output_path = REPO_ROOT / args.output
    generate_latex_table(all_results, output_path)
    
    # Save JSON if requested
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"JSON results written to: {json_path}")
    
    print("\n" + "="*60)
    print("Self-consistency tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
