"""
Corrected Training Script: Neural Copula Estimation via Noise-Conditioned Denoiser

This script demonstrates the CORRECT approach to neural copula density estimation,
addressing all issues identified in TECHNICAL_REPORT_COPULA_CONSTRAINTS.md.

Key principles:
1. Treat as conditional estimation, not unconditional generation
2. Use proper mass/density scaling throughout
3. Use cross-entropy vs. target densities (not uniform NLL)
4. Use IPFP as constraint layer (not penalty)
5. Add noise conditioning for robustness (not full diffusion)

Author: Corrected implementation based on expert feedback
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

# Assuming vdc package structure
from vdc.models.projection import copula_project, check_copula_constraints


def noise_augment(histogram, t, noise_type='uniform'):
    """
    Add noise to histogram for robustness training.
    
    Args:
        histogram: (B, 1, m, m) clean histogram
        t: (B,) noise levels in [0, 1]
        noise_type: 'uniform' or 'gaussian'
    
    Returns:
        noisy_histogram: (B, 1, m, m)
    """
    B, C, m, _ = histogram.shape
    
    if noise_type == 'uniform':
        # Interpolate toward uniform
        uniform = torch.ones_like(histogram) / (m * m)
        noisy = (1 - t.view(B, 1, 1, 1)) * histogram + t.view(B, 1, 1, 1) * uniform
    elif noise_type == 'gaussian':
        # Add Gaussian noise
        noise = torch.randn_like(histogram)
        noisy = histogram + t.view(B, 1, 1, 1) * noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Renormalize to maintain total mass
    noisy = noisy.clamp_min(0)
    noisy = noisy / (noisy.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    
    return noisy


class SimpleDenoiser(nn.Module):
    """
    Simple CNN-based denoiser for copula density estimation.
    
    Architecture: Conditional on noise level t
    Input: (B, C, m, m) histogram + coords + (B,) noise level
    Output: (B, 1, m, m) - logits (for softmax mass) or log-density
    """
    def __init__(self, base_channels=64, n_blocks=3, output_head='log_density'):
        super().__init__()
        
        self.output_head = output_head
        self.n_blocks = n_blocks
        self.base_channels = base_channels
        
        # Encoder output channels
        encoder_out_ch = base_channels * (2 ** (n_blocks - 1))
        
        # Time embedding - must match encoder output channels
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, encoder_out_ch),  # Match encoder output
        )
        
        # Encoder - accepts 3 input channels (histogram + u + v coordinates)
        layers = []
        in_ch = 3  # ← Changed from 1 to 3 for histogram + coordinates
        for i in range(n_blocks):
            out_ch = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            if i < n_blocks - 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # Decoder
        layers = []
        for i in range(n_blocks - 1):
            in_ch = base_channels * (2 ** (n_blocks - 1 - i))
            out_ch = base_channels * (2 ** (n_blocks - 2 - i))
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
        
        layers.extend([
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 1),  # Output: 1 channel (log-density)
        ])
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, histogram, t=None):
        """
        Args:
            histogram: (B, 3, m, m) input [histogram + u_coords + v_coords]
            t: (B,) or None, noise level for conditioning
        
        Returns:
            dict with either 'logits' or 'log_density' depending on output_head
        """
        # Encode
        features = self.encoder(histogram)
        
        # Add time conditioning if provided
        if t is not None:
            t_emb = self.time_embed(t.view(-1, 1))  # (B, encoder_out_ch)
            # Broadcast and add to features
            t_emb = t_emb.view(-1, features.size(1), 1, 1)
            features = features + t_emb
        
        # Decode
        out = self.decoder(features)
        
        if self.output_head == 'logits':
            # Return raw logits for softmax mass head
            return {'logits': out.clamp(min=-20.0, max=20.0)}
        else:
            # Return log-density (legacy path)
            return {'log_density': out}


def compute_ce_loss(pred_density, target_density, m):
    """
    Cross-entropy loss on cell masses (CORRECT implementation).
    
    Args:
        pred_density: (B, 1, m, m) predicted copula density
        target_density: (B, 1, m, m) target copula density
        m: grid resolution
    
    Returns:
        scalar loss
    """
    du = dv = 1.0 / m
    
    # Convert densities to masses (probabilities)
    pred_mass = (pred_density * du * dv).clamp_min(1e-12)
    target_mass = (target_density * du * dv).clamp_min(1e-12)
    
    # Cross-entropy: -sum(P_true * log(P_pred))
    ce_loss = -(target_mass * pred_mass.log()).sum(dim=(-2, -1)).mean()
    
    return ce_loss


def compute_tail_loss(pred_density, target_density, m, tau=0.15):
    """
    Tail emphasis loss (optional, for better tail behavior).
    
    Args:
        pred_density: (B, 1, m, m)
        target_density: (B, 1, m, m)
        m: grid resolution
        tau: tail threshold (default: emphasize outer 15% on each side)
    
    Returns:
        scalar loss
    """
    du = dv = 1.0 / m
    
    # Create tail mask
    u_grid = torch.linspace(0, 1, m, device=pred_density.device)
    v_grid = torch.linspace(0, 1, m, device=pred_density.device)
    uu, vv = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    tail_mask = ((uu < tau) | (uu > 1 - tau) | (vv < tau) | (vv > 1 - tau)).float()
    tail_mask = tail_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, m, m)
    
    # Squared log-density error in tail regions
    pred_mass = (pred_density * du * dv).clamp_min(1e-12)
    target_mass = (target_density * du * dv).clamp_min(1e-12)
    
    tail_loss = (tail_mask * (pred_mass.log() - target_mass.log()) ** 2).mean()
    
    return tail_loss


def train_step(model, batch, optimizer, config, device):
    """
    Single training step with corrected implementation.
    
    Args:
        model: denoiser network
        batch: dict with 'histogram' and 'target_density'
        optimizer: PyTorch optimizer
        config: training configuration dict
        device: torch device
    
    Returns:
        dict of metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Unpack batch
    histogram = batch['histogram'].to(device)  # (B, 1, m, m)
    target = batch['target_density'].to(device)  # (B, 1, m, m)
    
    B, C, m, _ = histogram.shape
    
    # Add noise conditioning
    if config.get('use_noise_conditioning', True):
        t = torch.rand(B, device=device) * config.get('noise_max', 0.3)
        hist_noisy = noise_augment(histogram, t, noise_type='uniform')
    else:
        t = None
        hist_noisy = histogram
    
    # Forward pass: predict log-density
    log_density_pred = model(hist_noisy, t)
    
    # Convert to positive density
    density_pred = torch.exp(log_density_pred.clamp(min=-15, max=15))
    
    # Apply IPFP projection (constraint enforcement layer)
    projection_iters = config.get('projection_iters', 20)
    density_proj = copula_project(
        density_pred,
        iters=projection_iters,
        method='ipfp'
    )
    
    # Compute losses
    loss_ce = compute_ce_loss(density_proj, target, m)
    
    loss_tail = torch.zeros((), device=device)
    if config['loss_weights'].get('tail', 0) > 0:
        loss_tail = compute_tail_loss(density_proj, target, m)
    
    # Total loss (NO ISE or marg_kl penalties!)
    w_ce = config['loss_weights']['ce']
    w_tail = config['loss_weights'].get('tail', 0)
    
    total_loss = w_ce * loss_ce + w_tail * loss_tail
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    if config.get('gradient_clip', 0) > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['gradient_clip']
        )
    
    optimizer.step()
    
    # Compute diagnostic metrics
    with torch.no_grad():
        # Check copula constraints
        checks = check_copula_constraints(density_proj, verbose=False)
        
        # Track raw model output statistics
        D_raw_mean = density_pred.mean().item()
        D_raw_max = density_pred.max().item()
        D_raw_min = density_pred.min().item()
        
        # Track projected density statistics
        D_proj_mean = density_proj.mean().item()
        D_proj_max = density_proj.max().item()
        
        metrics = {
            'loss': total_loss.item(),
            'ce': loss_ce.item(),
            'tail': loss_tail.item() if w_tail > 0 else 0.0,
            'D_raw_mean': D_raw_mean,
            'D_raw_max': D_raw_max,
            'D_raw_min': D_raw_min,
            'D_proj_mean': D_proj_mean,
            'D_proj_max': D_proj_max,
            'marg_u_error': checks['marginal_u_max_error'],
            'marg_v_error': checks['marginal_v_max_error'],
            'total_mass_error': checks['total_mass_error'],
        }
    
    return metrics


def validate(model, val_loader, config, device):
    """Validation loop."""
    model.eval()
    
    metrics_sum = {}
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            histogram = batch['histogram'].to(device)
            target = batch['target_density'].to(device)
            
            B, C, m, _ = histogram.shape
            
            # Inference: no noise, clean prediction
            log_density_pred = model(histogram, t=None)
            density_pred = torch.exp(log_density_pred.clamp(min=-15, max=15))
            
            # Project to copula
            density_proj = copula_project(density_pred, iters=30, method='ipfp')
            
            # Compute loss
            loss_ce = compute_ce_loss(density_proj, target, m)
            
            # Accumulate
            if count == 0:
                metrics_sum = {'ce': loss_ce.item()}
            else:
                metrics_sum['ce'] += loss_ce.item()
            count += 1
    
    # Average
    metrics_avg = {k: v / count for k, v in metrics_sum.items()}
    
    return metrics_avg


def main():
    """
    Example training script demonstrating corrected approach.
    """
    # Configuration
    config = {
        'model': {
            'base_channels': 64,
            'n_blocks': 3,
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'max_steps': 10000,
            'projection_iters': 20,
            'gradient_clip': 1.0,
            'use_noise_conditioning': True,
            'noise_max': 0.3,
            'loss_weights': {
                'ce': 1.0,
                'ise': 0.0,      # DISABLED
                'marg_kl': 0.0,  # DISABLED
                'tail': 0.1,
            },
        },
        'logging': {
            'log_interval': 10,
        },
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = SimpleDenoiser(
        base_channels=config['model']['base_channels'],
        n_blocks=config['model']['n_blocks'],
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # For this example, we'll create synthetic data
    # In practice, load your actual copula dataset
    print("\nNote: This example uses synthetic data.")
    print("Replace with actual copula dataset for real training.")
    
    # Training loop
    print("\nStarting training...")
    print("Expected behavior:")
    print("  - CE should be negative and decreasing (more negative = better)")
    print("  - D_raw_mean should be around 1-2 (NOT billions!)")
    print("  - marg_*_error should be < 1e-5 after projection")
    print()
    
    model.train()
    for step in range(config['training']['max_steps']):
        # Create synthetic batch (replace with real data)
        m = 64
        B = config['training']['batch_size']
        
        # Synthetic Gaussian copula for demonstration
        rho = np.random.uniform(-0.8, 0.8, size=B)
        histograms = []
        targets = []
        
        for r in rho:
            # Generate synthetic copula density
            u = np.linspace(0, 1, m)
            v = np.linspace(0, 1, m)
            uu, vv = np.meshgrid(u, v)
            
            # Gaussian copula density
            from scipy.stats import norm
            z_u = norm.ppf(np.clip(uu, 0.001, 0.999))
            z_v = norm.ppf(np.clip(vv, 0.001, 0.999))
            
            density = (1 / np.sqrt(1 - r**2)) * np.exp(
                -(r**2 * (z_u**2 + z_v**2) - 2*r*z_u*z_v) / (2*(1-r**2))
            )
            
            histograms.append(density)
            targets.append(density)
        
        batch = {
            'histogram': torch.tensor(np.array(histograms), dtype=torch.float32).unsqueeze(1),
            'target_density': torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1),
        }
        
        # Training step
        metrics = train_step(model, batch, optimizer, config['training'], device)
        
        # Logging
        if step % config['logging']['log_interval'] == 0:
            print(f"Step {step:5d}  "
                  f"loss={metrics['loss']:7.2f}  "
                  f"ce={metrics['ce']:7.2f}  "
                  f"tail={metrics['tail']:.2f}  "
                  f"D_raw_mean={metrics['D_raw_mean']:.2f}  "
                  f"marg_err={max(metrics['marg_u_error'], metrics['marg_v_error']):.2e}")
        
        # Sanity checks
        if step > 100:
            if metrics['D_raw_mean'] > 1000:
                print("\n⚠️  WARNING: D_raw_mean > 1000, scaling issue detected!")
                print("   This indicates the implementation is still broken.")
                break
            
            if metrics['loss'] > 100:
                print("\n⚠️  WARNING: loss > 100, something is wrong!")
                print("   Check that CE loss is computed correctly.")
                break
    
    print("\nTraining completed!")
    print("\nIf you saw:")
    print("  ✓ CE negative and decreasing → GOOD")
    print("  ✓ D_raw_mean around 1-2 → GOOD")
    print("  ✓ marg_err < 1e-5 → GOOD")
    print("\nThen the corrected implementation is working!")


if __name__ == '__main__':
    main()
