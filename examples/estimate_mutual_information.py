#!/usr/bin/env python3
"""
Example: Mutual Information Estimation with DCD-Vine.

This demonstrates how to:
1. Estimate MI from bivariate copula samples using DCD
2. Compare with ground-truth MI for known copulas
3. Estimate Total Correlation (TC) for multivariate data
4. Decompose TC into edge-wise MI contributions via vine structure

Key insight: For any copula C(u,v) with density c(u,v):
    I(U;V) = ∫∫ c(u,v) log c(u,v) du dv

For vine copulas, TC decomposes additively:
    TC(X) = Σ_{edges (i,j;D)} I(X_i; X_j | X_D)

Usage:
    python examples/estimate_mutual_information.py --checkpoint path/to/model.pt

Reference:
    - Cover & Thomas (2006), "Elements of Information Theory"
    - Joe (2015), "Dependence Modeling with Copulas"
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def estimate_mi_from_density_grid(density: np.ndarray) -> float:
    """
    Estimate mutual information from copula density grid.
    
    For copula C with density c:
        I(U;V) = ∫∫ c(u,v) log c(u,v) du dv
        
    This is the negative entropy of the copula density relative to independence.
    
    Args:
        density: (m, m) copula density grid
        
    Returns:
        MI estimate in nats
    """
    m = density.shape[0]
    du = 1.0 / m
    
    # Normalize if needed
    density = np.clip(density, 1e-12, None)
    mass = density.sum() * du * du
    if abs(mass - 1.0) > 0.01:
        density = density / mass
    
    # I(U;V) = ∫∫ c log c du dv
    mi = np.sum(density * np.log(density)) * du * du
    return float(mi)


def true_mi_gaussian(rho: float) -> float:
    """True MI for Gaussian copula: I = -0.5 log(1 - ρ²)."""
    return -0.5 * np.log(1 - rho**2)


def true_mi_student(rho: float, df: int) -> float:
    """True MI for Student-t copula (approximated for df -> ∞ as Gaussian)."""
    # For finite df, MI is slightly higher than Gaussian due to tail dependence
    # Approximate: I ≈ -0.5 log(1 - ρ²) + δ(df) where δ is a correction term
    gaussian_mi = -0.5 * np.log(1 - rho**2)
    # Approximate correction (empirical)
    correction = 0.02 * (30 - df) / 30 if df < 30 else 0
    return gaussian_mi + correction


def true_mi_clayton(theta: float) -> float:
    """
    True MI for Clayton copula.
    
    Kendall's tau for Clayton: τ = θ/(θ+2)
    MI approximation: I ≈ -0.5 log(1 - τ²) (first-order)
    """
    if theta < 0.01:
        return 0.0
    tau = theta / (theta + 2)
    return -0.5 * np.log(1 - tau**2)


def true_tc_gaussian_ar1(d: int, rho: float) -> float:
    """
    True total correlation for Gaussian AR(1) copula.
    
    TC = -0.5 * log det(Σ) where Σ is the correlation matrix.
    For AR(1): Σ[i,j] = ρ^|i-j|, det(Σ) = (1-ρ²)^(d-1)
    
    Thus: TC = -0.5 * (d-1) * log(1 - ρ²)
    """
    return -0.5 * (d - 1) * np.log(1 - rho**2)


class DCDMIEstimator:
    """
    Mutual Information estimator using DCD (Diffusion-Conditioned Denoiser).
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = 'cuda',
        grid_size: int = 64,
        ipfp_iters: int = 50,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.grid_size = grid_size
        self.ipfp_iters = ipfp_iters
        self.model = None
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: Path):
        """Load trained DCD model."""
        from vdc.models.unet_grid import GridUNet
        
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get('config', {})
        model_cfg = cfg.get('model', {})
        
        self.model = GridUNet(
            m=cfg.get('data', {}).get('m', 64),
            in_channels=model_cfg.get('in_channels', 1),
            base_channels=model_cfg.get('base_channels', 64),
            channel_mults=tuple(model_cfg.get('channel_mults', (1, 2, 3, 4))),
            num_res_blocks=model_cfg.get('num_res_blocks', 2),
        ).to(self.device)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")
    
    def estimate_bivariate_mi(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate MI from bivariate samples.
        
        Args:
            u: (n,) first variable (in [0,1])
            v: (n,) second variable (in [0,1])
            
        Returns:
            (mi_estimate, density_grid)
        """
        from vdc.data.generators import scatter_to_hist
        from vdc.models.projection import copula_project
        
        n = len(u)
        m = self.grid_size
        
        # Create empirical histogram
        pts = np.column_stack([u, v])
        pts = np.clip(pts, 1e-6, 1 - 1e-6)
        hist = scatter_to_hist(pts, m, reflect=True)
        
        du = 1.0 / m
        hist = hist / (hist.sum() * du * du + 1e-12)
        
        if self.model is None:
            # No model: use histogram directly (naive estimator)
            density = hist.copy()
        else:
            # Run through DCD denoiser
            with torch.no_grad():
                x = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(self.device)
                log_n = torch.tensor([[np.log(n)]], device=self.device)
                t = torch.zeros((1,), device=self.device)  # t=0 for one-shot
                
                # Conditioning
                cond = torch.cat([log_n, t.unsqueeze(-1)], dim=-1)
                
                # Forward pass (model may output log-density)
                pred = self.model(x, cond)
                
                # Exponentiate if log-domain
                pred = torch.exp(pred.clamp(-20, 20))
                
                # IPFP projection for valid copula
                pred = copula_project(pred, iters=self.ipfp_iters)
                
                density = pred[0, 0].cpu().numpy()
        
        # Normalize
        density = np.clip(density, 1e-12, None)
        density = density / (density.sum() * du * du)
        
        mi = estimate_mi_from_density_grid(density)
        return mi, density
    
    def estimate_tc_dvine(
        self,
        U: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[float, List[float]]:
        """
        Estimate Total Correlation using D-vine decomposition.
        
        TC = Σ_{edges} I(U_i; U_j | U_D)
        
        For a D-vine, this sums MI over all pair copulas across all trees.
        
        Args:
            U: (n, d) multivariate samples in [0,1]^d
            verbose: Print progress
            
        Returns:
            (total_tc, edge_mi_list)
        """
        from vdc.data.conditional_copulas import h_gaussian, h_clayton, H_FUNCTIONS
        
        n, d = U.shape
        
        edge_mis = []
        V = [U.copy()]
        
        for tree in range(d - 1):
            if verbose:
                print(f"  Tree {tree + 1}: ", end='')
            
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                
                # Estimate MI for this pair
                mi, density = self.estimate_bivariate_mi(u_data, v_data)
                edge_mis.append(mi)
                
                if verbose:
                    print(f"e{edge+1}={mi:.3f} ", end='')
                
                # Compute h-transform for next tree (using estimated density)
                # Simple approximation: use KSG or direct h-function if known
                # Here we use a simple Gaussian approximation for h-transform
                rho = np.corrcoef(u_data, v_data)[0, 1]
                rho = np.clip(rho, -0.99, 0.99)
                h_val = h_gaussian(u_data, v_data, rho)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1 - 1e-6)
            
            V.append(V_next)
            if verbose:
                print()
        
        total_tc = sum(edge_mis)
        return total_tc, edge_mis


def demo_bivariate_mi():
    """Demonstrate bivariate MI estimation."""
    print("\n" + "="*60)
    print("Demo: Bivariate Mutual Information Estimation")
    print("="*60)
    
    from vdc.data.generators import sample_gaussian_copula, sample_clayton_copula
    from vdc.utils.information import ksg_mutual_information
    
    n_samples = 5000
    seed = 42
    
    test_cases = [
        ("Gaussian (ρ=0.7)", lambda: sample_gaussian_copula(n_samples, rho=0.7), true_mi_gaussian(0.7)),
        ("Gaussian (ρ=0.3)", lambda: sample_gaussian_copula(n_samples, rho=0.3), true_mi_gaussian(0.3)),
        ("Clayton (θ=3.0)", lambda: sample_clayton_copula(n_samples, theta=3.0), true_mi_clayton(3.0)),
    ]
    
    estimator = DCDMIEstimator(checkpoint_path=None)  # Using histogram only
    
    print(f"\nUsing n={n_samples} samples per case")
    print("-" * 60)
    print(f"{'Copula':<25} {'MI_true':>10} {'MI_hist':>10} {'MI_ksg':>10}")
    print("-" * 60)
    
    np.random.seed(seed)
    for name, sampler, mi_true in test_cases:
        samples = sampler()
        u, v = samples[:, 0], samples[:, 1]
        
        mi_hist, _ = estimator.estimate_bivariate_mi(u, v)
        mi_ksg = ksg_mutual_information(u, v, k=5)
        
        print(f"{name:<25} {mi_true:>10.4f} {mi_hist:>10.4f} {mi_ksg:>10.4f}")
    
    print("-" * 60)


def demo_total_correlation():
    """Demonstrate TC estimation via D-vine decomposition."""
    print("\n" + "="*60)
    print("Demo: Total Correlation Estimation (D-Vine)")
    print("="*60)
    
    from vdc.data.generators import generate_gaussian_vine
    
    d = 5
    n_samples = 5000
    rho = 0.6
    seed = 42
    
    tc_true = true_tc_gaussian_ar1(d, rho)
    
    print(f"\nGaussian AR(1) copula: d={d}, ρ={rho}, n={n_samples}")
    print(f"True TC = {tc_true:.4f} nats")
    print("-" * 60)
    
    np.random.seed(seed)
    U = generate_gaussian_vine(n_samples, d, rho=rho, seed=seed)
    
    estimator = DCDMIEstimator(checkpoint_path=None)
    tc_est, edge_mis = estimator.estimate_tc_dvine(U, verbose=True)
    
    print("-" * 60)
    print(f"Estimated TC = {tc_est:.4f} nats")
    print(f"True TC = {tc_true:.4f} nats")
    print(f"Error = {abs(tc_est - tc_true):.4f} nats ({100*abs(tc_est - tc_true)/tc_true:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="MI Estimation with DCD-Vine")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--demo", type=str, choices=["bivariate", "tc", "all"],
                        default="all", help="Which demo to run")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Mutual Information Estimation with DCD-Vine")
    print("="*60)
    
    if args.demo in ["bivariate", "all"]:
        demo_bivariate_mi()
    
    if args.demo in ["tc", "all"]:
        demo_total_correlation()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
