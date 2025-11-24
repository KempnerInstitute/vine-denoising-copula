"""
Training utility functions for copula denoiser.

This module provides helper functions for the corrected training approach:
- histogram_2d: Convert samples to normalized histograms
- build_coordinates: Generate (u,v) coordinate grids
- tail_loss: Emphasis loss for tail regions
- alpha_bar_cosine: Noise schedule for diffusion-style conditioning
"""

import torch
import torch.nn.functional as F
import numpy as np


def histogram_2d(samples, bins=64, normalize=True):
    """
    Convert pseudo-observations (samples) to 2D histogram on [0,1]^2.
    
    Args:
        samples: (B, N, 2) tensor of samples in [0,1]^2
                 samples[b, :, 0] are u-values
                 samples[b, :, 1] are v-values
        bins: int, number of bins per dimension (m)
        normalize: bool, if True, normalize each histogram to sum to 1 (probability mass)
    
    Returns:
        histogram: (B, 1, m, m) tensor of histograms
                   If normalize=True, each histogram sums to 1
    """
    B, N, _ = samples.shape
    device = samples.device
    
    histograms = []
    
    for b in range(B):
        u = samples[b, :, 0]  # (N,)
        v = samples[b, :, 1]  # (N,)
        
        # Clamp to [0, 1] and convert to bin indices
        u_clipped = u.clamp(0.0, 1.0 - 1e-7)
        v_clipped = v.clamp(0.0, 1.0 - 1e-7)
        
        u_idx = (u_clipped * bins).long()  # (N,) in [0, m-1]
        v_idx = (v_clipped * bins).long()  # (N,) in [0, m-1]
        
        # Count occurrences in each bin
        hist = torch.zeros(bins, bins, device=device, dtype=torch.float32)
        
        # Flatten indices for scatter_add
        flat_idx = u_idx * bins + v_idx  # (N,)
        hist_flat = hist.view(-1)
        hist_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        hist = hist_flat.view(bins, bins)
        
        if normalize:
            # Normalize to sum to 1 (probability mass)
            hist = hist / (hist.sum() + 1e-12)
        
        histograms.append(hist)
    
    # Stack into (B, m, m) and add channel dimension
    histograms = torch.stack(histograms, dim=0).unsqueeze(1)  # (B, 1, m, m)
    
    return histograms


def build_coordinates(B, m, device):
    """
    Build coordinate grids for (u, v) in [0,1]^2.
    
    Args:
        B: batch size
        m: grid resolution
        device: torch device
    
    Returns:
        coords: (B, 2, m, m) tensor where
                coords[:, 0, :, :] is u-coordinate grid
                coords[:, 1, :, :] is v-coordinate grid
                Both in [0, 1]
    """
    # Create 1D grids
    u_grid = torch.linspace(0, 1, m, device=device)
    v_grid = torch.linspace(0, 1, m, device=device)
    
    # Create 2D meshgrids
    uu, vv = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    # Stack and add batch dimension
    coords = torch.stack([uu, vv], dim=0)  # (2, m, m)
    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, m, m)
    
    return coords


def tail_loss(D_pred, D_target, tau=0.15, m=None):
    """
    Tail emphasis loss: emphasize errors in tail regions.
    
    Tail regions are defined as:
    - u < tau or u > 1-tau
    - v < tau or v > 1-tau
    
    Args:
        D_pred: (B, 1, m, m) predicted density
        D_target: (B, 1, m, m) target density
        tau: float, tail threshold (e.g., 0.15 means outer 15% on each side)
        m: int, grid resolution (inferred from shape if None)
    
    Returns:
        loss: scalar, squared log-density error in tail regions
    """
    B, C, m_grid, _ = D_pred.shape
    if m is None:
        m = m_grid
    
    du = dv = 1.0 / m
    device = D_pred.device
    
    # Build coordinate grids
    u_grid = torch.linspace(0, 1, m, device=device)
    v_grid = torch.linspace(0, 1, m, device=device)
    uu, vv = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    # Create tail mask: True in tail regions
    tail_mask = ((uu < tau) | (uu > 1 - tau) | (vv < tau) | (vv > 1 - tau)).float()
    tail_mask = tail_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, m, m)
    
    # Convert to masses for numerical stability
    P_pred = (D_pred * du * dv).clamp_min(1e-12)
    P_target = (D_target * du * dv).clamp_min(1e-12)
    
    # Squared log-error in tail regions
    log_diff = (P_pred.log() - P_target.log()) ** 2
    loss = (tail_mask * log_diff).mean()
    
    return loss


def alpha_bar_cosine(t, s=0.008):
    """
    Cosine noise schedule for diffusion-style conditioning.
    
    Args:
        t: (B,) or scalar, time/noise level in [0, 1]
        s: float, offset parameter for cosine schedule
    
    Returns:
        alpha_bar: same shape as t, values in (0, 1]
                   alpha_bar(0) ≈ 1 (clean)
                   alpha_bar(1) ≈ 0 (noisy)
    """
    # Ensure t is at least 1D
    if t.dim() == 0:
        t = t.unsqueeze(0)
    
    # Cosine schedule: alpha_bar(t) = cos²((t + s) / (1 + s) * π/2)
    inner = (t + s) / (1.0 + s) * (np.pi / 2)
    alpha_bar = torch.cos(inner) ** 2
    
    return alpha_bar


def add_noise_to_histogram(H_clean, t, schedule='cosine'):
    """
    Add noise to histogram for training-time augmentation.
    
    H_noisy = sqrt(alpha_bar) * H_clean + sqrt(1 - alpha_bar) * noise
    
    Args:
        H_clean: (B, 1, m, m) clean histogram (sum to 1)
        t: (B,) noise levels in [0, 1]
        schedule: 'cosine' or 'linear'
    
    Returns:
        H_noisy: (B, 1, m, m) noisy histogram (clamped and renormalized)
    """
    B = H_clean.shape[0]
    
    # Get alpha_bar based on schedule
    if schedule == 'cosine':
        alpha_bar = alpha_bar_cosine(t).view(B, 1, 1, 1)
    elif schedule == 'linear':
        alpha_bar = (1.0 - t).view(B, 1, 1, 1)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    # Add noise
    noise = torch.randn_like(H_clean)
    H_noisy = torch.sqrt(alpha_bar) * H_clean + torch.sqrt(1.0 - alpha_bar) * noise
    
    # Clamp negatives and renormalize to maintain probability mass interpretation
    H_noisy = H_noisy.clamp_min(0.0)
    H_noisy = H_noisy / (H_noisy.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    
    return H_noisy


def check_density_sanity(D, name='density', verbose=True):
    """
    Check if density values are in reasonable range.
    
    Args:
        D: (B, 1, m, m) density tensor
        name: str, name for logging
        verbose: bool, print warnings
    
    Returns:
        dict with diagnostic metrics
    """
    B, C, m, _ = D.shape
    du = dv = 1.0 / m
    
    # Compute statistics
    D_mean = D.mean().item()
    D_max = D.max().item()
    D_min = D.min().item()
    D_std = D.std().item()
    
    # Check total mass
    total_mass = (D * du * dv).sum(dim=(-2, -1)).mean().item()
    
    # Check marginals
    row_marg = (D.sum(dim=-1) * dv).mean().item()
    col_marg = (D.sum(dim=-2) * du).mean().item()
    
    metrics = {
        f'{name}_mean': D_mean,
        f'{name}_max': D_max,
        f'{name}_min': D_min,
        f'{name}_std': D_std,
        f'{name}_total_mass': total_mass,
        f'{name}_row_marg_mean': row_marg,
        f'{name}_col_marg_mean': col_marg,
    }
    
    # Sanity checks
    issues = []
    if D_mean > 100:
        issues.append(f"{name}_mean = {D_mean:.2f} > 100 (scaling issue!)")
    if D_max > 10000:
        issues.append(f"{name}_max = {D_max:.2e} > 10k (scaling issue!)")
    if D_min < 0:
        issues.append(f"{name}_min = {D_min:.2e} < 0 (negative density!)")
    if abs(total_mass - 1.0) > 0.1:
        issues.append(f"{name}_total_mass = {total_mass:.3f} far from 1.0")
    
    if verbose and issues:
        print(f"⚠️  Sanity check warnings for {name}:")
        for issue in issues:
            print(f"   {issue}")
    
    metrics['sanity_ok'] = len(issues) == 0
    
    return metrics


# Convenience function for computing cross-entropy on masses
def cross_entropy_on_masses(D_pred, D_target, m):
    """
    Compute cross-entropy loss between predicted and target densities.
    
    Args:
        D_pred: (B, 1, m, m) predicted density
        D_target: (B, 1, m, m) target density
        m: int, grid resolution
    
    Returns:
        loss: scalar cross-entropy
    """
    du = dv = 1.0 / m
    
    # Convert to probability masses
    P_pred = (D_pred * du * dv)
    P_pred = P_pred / (P_pred.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    
    P_target = (D_target * du * dv)
    P_target = P_target / (P_target.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    
    # Cross-entropy: -E[log P_pred]
    ce_loss = -(P_target * (P_pred + 1e-12).log()).sum(dim=(-2, -1)).mean()
    
    return ce_loss


if __name__ == '__main__':
    """Test utility functions."""
    print("Testing training utility functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, m = 4, 1000, 64
    
    # Test 1: histogram_2d
    print("\n1. Testing histogram_2d...")
    samples = torch.rand(B, N, 2, device=device)  # uniform samples
    hist = histogram_2d(samples, bins=m, normalize=True)
    print(f"   Input: {samples.shape}, Output: {hist.shape}")
    print(f"   Sum per histogram: {hist.sum(dim=(-2, -1))}")
    print(f"   Min: {hist.min():.6f}, Max: {hist.max():.6f}, Mean: {hist.mean():.6f}")
    assert hist.shape == (B, 1, m, m)
    assert torch.allclose(hist.sum(dim=(-2, -1)), torch.ones(B, 1, device=device), atol=1e-5)
    print("   ✓ Passed")
    
    # Test 2: build_coordinates
    print("\n2. Testing build_coordinates...")
    coords = build_coordinates(B, m, device)
    print(f"   Output: {coords.shape}")
    print(f"   U range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    print(f"   V range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
    assert coords.shape == (B, 2, m, m)
    assert coords[:, 0].min() >= 0 and coords[:, 0].max() <= 1
    assert coords[:, 1].min() >= 0 and coords[:, 1].max() <= 1
    print("   ✓ Passed")
    
    # Test 3: tail_loss
    print("\n3. Testing tail_loss...")
    D_pred = torch.ones(B, 1, m, m, device=device) * (m * m)  # uniform density
    D_target = D_pred.clone()
    loss_zero = tail_loss(D_pred, D_target, tau=0.15)
    print(f"   Loss (identical): {loss_zero.item():.6f}")
    assert loss_zero.item() < 1e-6
    
    D_pred_noisy = D_pred + torch.randn_like(D_pred) * 0.1 * (m * m)
    loss_nonzero = tail_loss(D_pred_noisy, D_target, tau=0.15)
    print(f"   Loss (noisy): {loss_nonzero.item():.6f}")
    assert loss_nonzero.item() > 0
    print("   ✓ Passed")
    
    # Test 4: alpha_bar_cosine
    print("\n4. Testing alpha_bar_cosine...")
    t = torch.linspace(0, 1, 10, device=device)
    alpha_bar = alpha_bar_cosine(t)
    print(f"   t: {t}")
    print(f"   alpha_bar: {alpha_bar}")
    assert alpha_bar[0] > 0.99  # t=0 should give alpha_bar ≈ 1
    assert alpha_bar[-1] < 0.1  # t=1 should give alpha_bar ≈ 0
    print("   ✓ Passed")
    
    # Test 5: add_noise_to_histogram
    print("\n5. Testing add_noise_to_histogram...")
    H_clean = histogram_2d(samples, bins=m, normalize=True)
    t = torch.tensor([0.0, 0.5, 1.0], device=device)
    H_noisy = add_noise_to_histogram(H_clean[:3], t, schedule='cosine')
    print(f"   Clean sum: {H_clean[:3].sum(dim=(-2, -1))}")
    print(f"   Noisy sum: {H_noisy.sum(dim=(-2, -1))}")
    print(f"   L2 distance from clean:")
    for i, ti in enumerate(t):
        dist = (H_noisy[i] - H_clean[i]).pow(2).sum().sqrt().item()
        print(f"     t={ti:.1f}: {dist:.6f}")
    print("   ✓ Passed")
    
    # Test 6: check_density_sanity
    print("\n6. Testing check_density_sanity...")
    D_good = torch.ones(B, 1, m, m, device=device) * (m * m)  # uniform copula density
    metrics_good = check_density_sanity(D_good, name='good', verbose=True)
    assert metrics_good['sanity_ok']
    
    D_bad = torch.ones(B, 1, m, m, device=device) * 1e9  # billion-scale (BAD!)
    metrics_bad = check_density_sanity(D_bad, name='bad', verbose=True)
    assert not metrics_bad['sanity_ok']
    print("   ✓ Passed")
    
    # Test 7: cross_entropy_on_masses
    print("\n7. Testing cross_entropy_on_masses...")
    D_pred = torch.ones(B, 1, m, m, device=device) * (m * m)
    D_target = D_pred.clone()
    ce_zero = cross_entropy_on_masses(D_pred, D_target, m)
    print(f"   CE (identical): {ce_zero.item():.6f}")
    assert ce_zero.item() < 0.01  # Should be near 0 for identical distributions
    
    D_pred_diff = torch.rand(B, 1, m, m, device=device) * (2 * m * m)
    ce_nonzero = cross_entropy_on_masses(D_pred_diff, D_target, m)
    print(f"   CE (different): {ce_nonzero.item():.6f}")
    assert ce_nonzero.item() > 0.01
    print("   ✓ Passed")
    
    print("\n✅ All tests passed!")
